"""The recording session.

Runs the subject token by token behind the OpenAI-compatible endpoint, records what
happens inside it, and lines everything up with the conversation and harness events
on one token tape per run.

The record for tape slot t (context position `pos`) describes token `pos` and the
model's prediction *of* that token, made at position pos-1:
  p         final-layer probability the model gave this token
  ent       entropy (bits) of the distribution the token was read or sampled from
  top_*     the five most likely alternatives at that point
  lens_p    logit-lens probability of this token at every layer ("when did it know")
  lens_top  logit-lens top-1 prediction at every layer
  norm      residual norm at every layer at this token's own position
  proj      z-scored projections of this token's residual onto the direction bank
"""

from __future__ import annotations

import bisect
import contextlib
import math
import re
import threading
import time
import uuid
from dataclasses import dataclass, field

import torch

from .anatomy import Anatomy
from .capture import MODES, PHASES, Intervention, LayerTap, predict
from .chat import (THINK_CLOSE, THINK_OPEN, common_prefix, parse_tool_calls, render, split_reasoning,
                   to_template_messages)
from .directions import DirectionBank
from .edits import EditStack
from .events import message_events
from .graphs import DecodeGraphs
from .knobs import KnobBank
from .monitors import MonitorBank
from .neuron_index import NeuronIndex
from .neurons import MAX_DOSE, Neurons, NeuronWatch, label, parse
from .tracing import Tracer


@dataclass
class GenParams:
    messages: list[dict]
    tools: list[dict] | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    stop: list[str] = field(default_factory=list)
    seed: int | None = None
    run_hint: str | None = None
    label: str | None = None
    template_kwargs: dict = field(default_factory=dict)  # e.g. {"enable_thinking": False}


@dataclass
class GenResult:
    run_id: str
    req: int
    text: str
    content: str | None
    tool_calls: list[dict]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning: str | None = None


class OutputStream:
    """Streams reasoning and answer text separately as tokens arrive. When tools are on
    offer the answer is sent once at the end, after tool calls are parsed out of it."""

    HOLD = len(THINK_CLOSE)  # hold back a possibly partial closing tag

    def __init__(self, think_open: bool, buffered: bool, on_text, on_reasoning):
        self.think_open, self.buffered = think_open, buffered
        self.on_text, self.on_reasoning = on_text, on_reasoning
        self.sent_reasoning = self.sent_text = 0

    def update(self, text: str, final: bool = False) -> None:
        reasoning, answer, closed = split_reasoning(text, self.think_open)
        if reasoning is not None and self.on_reasoning is not None:
            safe = reasoning if closed or final else reasoning[:max(0, len(reasoning) - self.HOLD)]
            if len(safe) > self.sent_reasoning:
                self.on_reasoning(safe[self.sent_reasoning:])
                self.sent_reasoning = len(safe)
        if closed and not self.buffered and self.on_text is not None:
            body = answer.lstrip()
            safe = body if final or not body.endswith("�") else body[:-1]
            if len(safe) > self.sent_text:
                self.on_text(safe[self.sent_text:])
                self.sent_text = len(safe)

    def finish(self, content: str | None) -> None:
        if self.buffered and content and self.on_text is not None:
            self.on_text(content)


@dataclass
class RunState:
    id: str
    meta: dict
    tape: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    seen_msgs: int = 0
    last_prompt: list[int] = field(default_factory=list)
    last_msg_count: int = 0
    pos2tape: dict[int, int] = field(default_factory=dict)
    monitor_state: dict = field(default_factory=dict)  # each monitor's live check along this run


LENS_FLOOR = -6.0  # log10 p below this shows as 0 in the scanner


def _lens_byte(p: float) -> int:
    return max(0, min(255, round((math.log10(max(p, 1e-12)) - LENS_FLOOR) / -LENS_FLOOR * 255)))


def compact(r: dict) -> dict:
    """The per-token payload streamed to the scanner UI. Lens values are log10 p from
    1e-6 to 1, quantized to 0-255, so small probabilities keep their resolution."""
    return {"t": r["t"], "pos": r["pos"], "req": r["req"], "tx": r["text"], "r": r["role"], "ph": r["ph"],
            "p": r["p"], "e": r["ent"],
            "lens": None if r["lens_p"] is None else [_lens_byte(x) for x in r["lens_p"]],
            "pj": r["proj"]}


def sample(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits / temperature, -1)
    if top_p < 1.0:
        sorted_p, order = probs.sort(descending=True)
        sorted_p[(sorted_p.cumsum(-1) - sorted_p) > top_p] = 0
        return int(order[torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=generator)])
    return int(torch.multinomial(probs, 1, generator=generator))


def to_host(x):
    """Queue copies of a device tensor (or a nested dict of them) into pinned host memory without
    waiting for the GPU. The copies hold the values only once the stream has been synchronized."""
    if isinstance(x, dict):
        return {k: to_host(v) for k, v in x.items()}
    return x.to("cpu", non_blocking=True)


class ModelSwitched(RuntimeError):
    """The recorder was closed for a model switch while a request waited for it."""


class Cancelled(RuntimeError):
    """The request was cancelled while it waited for the model, so nothing was recorded."""


def run_name(hint: str) -> str:
    """The run id a client's X-Brainscan-Run header (or event run) names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", hint)[:80]


class Recorder:
    def __init__(self, subject, store, settings, publish):
        self.s, self.store, self.settings, self.publish = subject, store, settings, publish
        self.tap = LayerTap(subject)
        self.directions = DirectionBank(store.root / "directions", subject, self.tap)
        self.edits = EditStack(subject)
        self.monitors = MonitorBank(store.root / "monitors" / f"{subject.slug}.json")
        use_graphs = settings.graph_decode and DecodeGraphs.available(subject)
        self.graphs = DecodeGraphs(subject, self.tap) if use_graphs else None
        # Anatomy is an optional feature; a failure to build it must not take the recorder down.
        self.anatomy = None
        if settings.anatomy:
            try:
                self.anatomy = Anatomy(subject, store.root / "anatomy", settings.anatomy_clusters)
            except Exception as exc:
                print(f"[brainscan] weight anatomy disabled ({type(exc).__name__}: {exc}); "
                      "the Weights tab will be unavailable", flush=True)
        self.tracer = Tracer(self) if self.anatomy is not None else None  # needs the unit map
        self.neurons = Neurons(self)
        self.neuron_index = NeuronIndex(self)  # loads this model's index if one was built
        self.knobs = KnobBank(store.root / "knobs" / f"{subject.slug}.json", self.neurons)
        self.watch: NeuronWatch | None = None  # hooks for neuron monitors; there only while one is enabled
        self.monitors.on_change = self._watch_changed
        self.lock = threading.RLock()        # one forward pass at a time
        self.log_lock = threading.Lock()     # events and transcript can arrive mid-generation
        self.runs: dict[str, RunState] = {}
        self._loaded: dict[str, RunState] = {}
        self.interventions: dict[str, Intervention] = {}
        self._iv_epoch = 0
        self._kv = None
        self._kv_ids: list[int] = []
        self._kv_epoch = -1
        self._kv_run: str | None = None
        self.perf = {"decode_tok_s": None, "prefill_tok_s": None, "busy": False}
        self.active: dict | None = None  # the request generating: its run, request number, cancel flag
        self._queued: list[tuple[str | None, threading.Event]] = []  # requests waiting: (run named, cancel)
        self.closed = False

    def close(self) -> None:
        """Let go of the model before another one is loaded. Waits for any forward pass to finish;
        requests still queued for the lock then fail instead of running on a dropped model. The
        caller drops its reference and collects garbage to give the GPU memory back."""
        with self.lock:
            self.closed = True
            for handle in getattr(self.tap, "handles", []):
                handle.remove()
            if self.graphs is not None:
                self.graphs._release()
                self.graphs = None
            if self.tracer is not None:
                self.tracer._cache.clear()
            self.monitors.on_change = None
            if self.watch is not None:
                self.watch.remove()
                self.watch = None
            self._kv, self._kv_ids = None, []
            self.interventions.clear()
            self.tap.interventions = []

    # ---- interventions -------------------------------------------------------------

    def add_intervention(self, direction: str, mode: str, layer: int | None = None, strength: float = 0.0,
                         value: float = 0.0, phase: str = "all", run: str | None = None,
                         seed: int | None = None) -> Intervention:
        if mode not in MODES or phase not in PHASES:
            raise ValueError(f"mode must be one of {MODES}, phase one of {PHASES}")
        if direction == "random":
            vector = torch.randn(self.s.d_model, generator=torch.Generator().manual_seed(seed or 0))
            direction = f"random(seed={seed or 0})"
        else:
            if direction not in self.directions.dirs:
                raise ValueError(f"unknown direction {direction!r}")
            d = self.directions.dirs[direction]
            vector, layer = d.unit, d.layer if layer is None else layer
        if layer is None or not 0 <= int(layer) < self.s.n_layers:
            raise ValueError(f"layer must be in 0..{self.s.n_layers - 1}")
        vector = (vector / vector.norm()).to(self.s.device, torch.float32)
        iv = Intervention(int(layer), mode, direction, vector, float(strength), float(value), phase, run)
        with self.lock:
            self.interventions[iv.id] = iv
            self._sync_interventions()
        return iv

    def _neuron_write(self, layer: int, n: int, dose: float, phase: str, run: str | None = None,
                      knob: str | None = None) -> Intervention:
        """A neuron steered with dose scaling (recorder/neurons.py): a "write" of dose × the layer's
        median residual norm along its output weights."""
        unit, length = self.neurons.write_vector(layer, n, float(dose))
        return Intervention(layer, "write", label(layer, n), unit.to(self.s.device, torch.float32),
                            strength=float(dose), value=length, phase=phase, run=run, knob=knob)

    def steer_neuron(self, neuron: str, dose: float, phase: str = "all", run: str | None = None) -> Intervention:
        """Steer one MLP neuron on every request from now."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        layer, n = parse(neuron)
        iv = self._neuron_write(layer, n, dose, phase, run)
        with self.lock:
            self.interventions[iv.id] = iv
            self._sync_interventions()
        return iv

    def steer_knob(self, name: str, factor: float = 1.0, phase: str = "all") -> list[dict]:
        """Steer a saved knob (recorder/knobs.py) on every request from now: one write per neuron at its
        dose × factor, each built as steer_neuron builds it. Steering a knob again replaces its writes."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        factor = float(factor)
        if not math.isfinite(factor):
            raise ValueError("factor must be a number")
        steers = self.knobs.steers(name)
        for layer, n, dose in steers:
            if abs(dose * factor) > MAX_DOSE:
                raise ValueError(f"{label(layer, n)}: dose {dose:g} × factor {factor:g} is past ±{MAX_DOSE:g}")
        ivs = [self._neuron_write(layer, n, dose * factor, phase, knob=name) for layer, n, dose in steers]
        with self.lock:
            for iid in [i for i, iv in self.interventions.items() if iv.knob == name]:
                del self.interventions[iid]
            self.interventions.update({iv.id: iv for iv in ivs})
            self._sync_interventions()
        return [iv.describe() for iv in ivs]

    def unsteer_knob(self, name: str) -> int:
        """Stop steering a knob; how many of its writes were removed."""
        with self.lock:
            ids = [i for i, iv in self.interventions.items() if iv.knob == name]
            for iid in ids:
                del self.interventions[iid]
            if ids:
                self._sync_interventions()
        return len(ids)

    def set_intervention(self, iid: str, enabled: bool) -> Intervention:
        with self.lock:
            self.interventions[iid].enabled = enabled
            self._sync_interventions()
            return self.interventions[iid]

    def remove_intervention(self, iid: str) -> None:
        with self.lock:
            self.interventions.pop(iid)
            self._sync_interventions()

    def _sync_interventions(self) -> None:
        self.tap.interventions = [iv for iv in self.interventions.values() if iv.enabled]
        self._iv_epoch += 1  # cached keys/values were computed under different edits

    # ---- neuron monitors -----------------------------------------------------------

    def sync_watch(self) -> bool:
        """Make the neuron watch match the enabled neuron monitors: none while there are none (no hooks,
        no cost), else hooks on the layers they name, with each neuron's mean and spread frozen here. It
        doesn't wait for a request in flight: every request starts by calling this, so the change takes
        effect at the next one. True when the watch is in step."""
        if not self.lock.acquire(blocking=False):
            return False
        try:
            if self.closed:
                return False
            items = []
            for m in self.monitors.monitors():
                if m.enabled and m.neuron:
                    try:
                        layer, n = parse(m.direction)
                        self.neurons.check(layer, n)
                    except (KeyError, ValueError):  # saved for a model that had it; it never flags here
                        continue
                    items.append((layer, n))
            items = sorted(set(items))
            watch = self.watch
            if not items:
                if watch is not None:
                    watch.remove()
                    self.watch = None
                return True
            source = self.neurons.stats_source(items)
            if watch is not None and watch.items == tuple(items) and watch.source == source:
                return True
            new = NeuronWatch(self.neurons, items, self.neurons.stats(items, source), source)
            if watch is not None:
                watch.remove()
            self.watch = new
            return True
        finally:
            self.lock.release()

    def _watch_changed(self) -> None:
        try:
            self.sync_watch()
        except Exception as exc:  # the monitor is saved either way; the next request tries again
            print(f"[brainscan] neuron watch not updated ({type(exc).__name__}: {exc})", flush=True)

    def _watch_recording(self):
        return self.watch.recording() if self.watch is not None else contextlib.nullcontext()

    def release_memory(self, below_gb: float | None = None) -> None:
        """Hand cached GPU blocks back to the driver. On Windows/WSL an over-full GPU doesn't raise
        out-of-memory; the driver spills into system RAM and every forward pass crawls. With
        below_gb, only release when free memory has dropped under that many GB."""
        if self.s.device.type != "cuda":
            return
        if below_gb is not None and torch.cuda.mem_get_info()[0] >= below_gb * 2 ** 30:
            return
        torch.cuda.empty_cache()

    def _epoch(self) -> tuple[int, int]:
        """Cached keys/values are only valid under the same steering and the same weights."""
        return self._iv_epoch, self.edits.version

    # ---- runs ----------------------------------------------------------------------

    def _new_run(self, rid: str | None, label: str | None) -> RunState:
        rid = rid or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        if self.store.read_meta(rid) is not None:
            rid = f"{rid}-{uuid.uuid4().hex[:4]}"
        meta = {"id": rid, "label": label, "created": time.time(), "updated": time.time(),
                "model": self.s.info(), "requests": 0, "tokens": 0, "outcome": None,
                "settings": {"prefill_lens": self.settings.prefill_lens,
                             "decision_residuals": self.settings.decision_residuals},
                "interventions": [], "weight_edits": [], "residuals": []}
        run = RunState(rid, meta)
        self.runs[rid] = run
        self.store.write_meta(rid, meta)
        self.publish({"k": "run", "run": rid, "meta": meta})
        return run

    def _resolve_run(self, p: GenParams, ids: list[int]) -> RunState:
        if p.run_hint:
            rid = run_name(p.run_hint)
            return self.runs.get(rid) or self._new_run(rid, p.label)
        best, best_len = None, 0
        for run in self.runs.values():
            if not run.last_prompt or len(p.messages) <= run.last_msg_count:
                continue
            n = common_prefix(run.last_prompt, ids)
            if n >= 0.9 * len(run.last_prompt) and n > best_len:
                best, best_len = run, n
        return best or self._new_run(None, p.label)

    def get_run(self, run_id: str) -> RunState | None:
        if run_id in self.runs:
            return self.runs[run_id]
        if run_id not in self._loaded:
            meta = self.store.read_meta(run_id)
            if meta is None:
                return None
            if len(self._loaded) >= 8:
                self._loaded.pop(next(iter(self._loaded)))
            self._loaded[run_id] = RunState(run_id, meta, tape=self.store.read_tokens(run_id),
                                            events=self.store.read_jsonl(run_id, "events"),
                                            messages=self.store.read_jsonl(run_id, "messages"))
        return self._loaded[run_id]

    def list_runs(self) -> list[dict]:
        metas = {m["id"]: m for m in self.store.list_runs()}
        metas.update({rid: run.meta for rid, run in self.runs.items()})
        return sorted(metas.values(), key=lambda m: m.get("created", 0), reverse=True)

    def _event(self, run: RunState, t: int, etype: str, data: dict, source: str = "recorder") -> dict:
        ev = {"t": int(t), "type": etype, "data": data, "source": source, "ts": time.time(),
              "req": run.meta.get("requests", 0)}
        with self.log_lock:
            run.events.append(ev)
            self.store.append_jsonl(run.id, "events", [ev])
        self.publish({"k": "ev", "run": run.id, "ev": ev})
        return ev

    def _add_message(self, run: RunState, entry: dict) -> None:
        with self.log_lock:
            run.messages.append(entry)
            self.store.append_jsonl(run.id, "messages", [entry])
        self.publish({"k": "msg", "run": run.id, "msg": entry})

    def external_event(self, run_id: str | None, etype: str, data: dict, t: int | None = None) -> dict:
        """Events a harness reports that the transcript can't show (verdicts, file edits)."""
        if run_id:
            run_id = run_name(run_id)
            run = self.runs.get(run_id) or self._new_run(run_id, data.get("task"))
        elif self.runs:
            run = max(self.runs.values(), key=lambda r: r.meta["updated"])
        else:
            raise KeyError("no active run")
        ev = self._event(run, max(len(run.tape) - 1, 0) if t is None else t, etype, data, source="harness")
        if etype in ("verdict", "run_meta"):
            if etype == "verdict":
                run.meta["outcome"] = data
            else:
                run.meta["harness"] = data
                run.meta["label"] = data.get("task") or run.meta.get("label")
            self.store.write_meta(run.id, run.meta)
            self.publish({"k": "run", "run": run.id, "meta": run.meta})
        return ev

    # ---- generation ----------------------------------------------------------------

    def generate(self, p: GenParams, on_text=None, on_reasoning=None,
                 cancel: threading.Event | None = None) -> GenResult:
        """Run one chat request. Setting `cancel` stops it between tokens (or between prefill chunks):
        it then finishes like any other request, with finish_reason "cancelled", and keeps what it
        generated. Set while the request still waits for the model, it raises Cancelled instead."""
        cancel = cancel or threading.Event()
        queued = (run_name(p.run_hint) if p.run_hint else None, cancel)
        self._queued.append(queued)
        try:
            while not self.lock.acquire(timeout=0.2):  # a request cancelled while it waits stops waiting
                if cancel.is_set():
                    raise Cancelled("the request was cancelled before it reached the model")
        finally:
            self._queued.remove(queued)
        try:
            if self.closed:
                raise ModelSwitched("the model was switched while this request waited; send it again")
            if cancel.is_set():
                raise Cancelled("the request was cancelled before it reached the model")
            self.perf["busy"] = True
            self.active = {"named": queued[0], "run": None, "req": None, "cancel": cancel}
            try:
                self.release_memory(below_gb=2.0)
                self._watch_changed()
                with torch.no_grad(), self._anatomy_recording(), self._watch_recording():
                    return self._generate(p, on_text, on_reasoning, cancel)
            finally:
                self.perf["busy"] = False
                self.active = None
        finally:
            self.lock.release()

    def cancel(self, run: str | None = None) -> dict:
        """Stop the request that is generating at its next token. With `run`, only a request of that
        run, and also that run's requests still waiting for the model (they end without recording
        anything). Doesn't wait: the request finishes, and is recorded, in its own thread."""
        out = {"cancelled": False, "run": None, "req": None, "queued": 0}
        active = self.active
        if active is not None and (run is None or run in (active["named"], active["run"])):
            active["cancel"].set()
            out.update(cancelled=True, run=active["run"] or active["named"], req=active["req"])
        if run is not None:
            for named, flag in list(self._queued):
                if named == run_name(run):
                    flag.set()
                    out["queued"] += 1
        return out

    def _anatomy_recording(self):
        return self.anatomy.recording() if self.anatomy is not None else contextlib.nullcontext()

    def _record(self, run, req, pos, tok, role, msg, phase, pred, k, norm, proj, nz=None) -> dict:
        r = {"t": len(run.tape), "req": req, "pos": int(pos), "tok": int(tok), "text": self.s.token_str(int(tok)),
             "role": role, "msg": int(msg), "ph": phase, "p": None, "ent": None, "top_ids": None, "top_p": None,
             "lens_p": None, "lens_top": None,
             "norm": None if norm is None else [round(float(x), 2) for x in norm],
             "proj": {n: round(float(v), 3) for n, v in proj.items()}}
        if nz:  # watched neurons' z-scores, kept apart from "proj": the scanner's z-lanes scale to that
            r["nz"] = {n: round(float(v), 3) for n, v in nz.items()}
        if pred is not None and k is not None:
            r["p"] = round(float(pred["p"][k]), 5)
            r["ent"] = round(float(pred["ent"][k]), 4)
            r["top_ids"] = [int(x) for x in pred["top_ids"][k]]
            r["top_p"] = [round(float(x), 4) for x in pred["top_p"][k]]
            if "lens_p" in pred:
                r["lens_p"] = [round(float(x), 4) for x in pred["lens_p"][k]]
                r["lens_top"] = [int(x) for x in pred["lens_top"][k]]
        run.tape.append(r)
        run.pos2tape[int(pos)] = r["t"]
        for m, flag in self.monitors.feed(run.monitor_state, r):
            self._flag(run, m, flag)
        return r

    def _flag(self, run: RunState, m, flag: dict) -> None:
        """A monitor's rule was met at this token (recorder/monitors.py): an event on the run, live."""
        run.meta["flags"] = run.meta.get("flags", 0) + 1
        self._event(run, flag["raised_t"], "monitor", {
            "monitor": m.id, "name": m.label or m.direction, "direction": m.direction, "rule": m.rule(),
            "z": round(flag["raised_z"], 3), "start": flag["start"]})

    def _prefill_records(self, run, req, rendered, cs, chunk, resid, logits, pending, record_from,
                         new_from, nz=None) -> list[dict]:
        """Records for this chunk's positions at or after `record_from`. Earlier positions were
        recorded on a previous request of this run and are only re-read for context. `nz` is the
        neuron watch's (chunk, neurons) read of the chunk, if there is a watch."""
        first = max(cs, record_from)
        if first >= cs + len(chunk):
            return []
        rows_r, rows_l, targets, target_pos = [], [], [], []
        if first == cs and pending is not None:
            rows_r.append(pending[0][:, None, :])
            rows_l.append(pending[1][None])
            targets.append(chunk[0])
            target_pos.append(cs)
        inner = list(range(max(cs + 1, first), cs + len(chunk)))  # predicted from inside this chunk
        if inner:
            idx = torch.tensor([q - cs - 1 for q in inner], device=self.s.device)
            rows_r.append(resid.index_select(1, idx))
            rows_l.append(logits.index_select(0, idx).float())
            targets += [chunk[q - cs] for q in inner]
            target_pos += inner
        pred = None
        if targets:
            pred = predict(self.s, torch.cat(rows_r, 1), torch.cat(rows_l),
                           torch.tensor(targets, device=self.s.device), with_lens=self.settings.prefill_lens)
        row_of = {pos: i for i, pos in enumerate(target_pos)}
        own = resid[:, first - cs:, :]
        norms = own.float().norm(dim=-1).cpu().numpy()
        proj = self.directions.project(own)
        zs = nz[first - cs:].cpu().numpy() if nz is not None else None
        recs = []
        for k, pos in enumerate(range(first, cs + len(chunk))):
            phase = "replay" if rendered.offsets[pos][0] < new_from else "prefill"
            recs.append(self._record(run, req, pos, chunk[pos - cs], rendered.roles[pos], rendered.msg_index[pos],
                                     phase, pred, row_of.get(pos), norms[:, k], {n: v[k] for n, v in proj.items()},
                                     None if zs is None else dict(zip(self.watch.labels, zs[k]))))
        return recs

    def _generate(self, p: GenParams, on_text, on_reasoning, cancel: threading.Event) -> GenResult:
        s = self.s
        rendered = render(s.tokenizer, p.messages, p.tools, template_kwargs=p.template_kwargs)
        ids = rendered.ids
        room = self.settings.max_context - len(ids)
        if room < 16:
            raise ValueError(f"prompt is {len(ids)} tokens; context limit is {self.settings.max_context}")
        max_tokens = min(p.max_tokens, room)
        run = self._resolve_run(p, ids)
        run.meta["requests"] += 1
        req = run.meta["requests"]
        self.active.update(run=run.id, req=req)
        tmsgs = to_template_messages(p.messages)
        seen = min(run.seen_msgs, len(p.messages))
        new_from = max((e for st, e in rendered.spans[:seen] if st >= 0), default=0)
        active = [iv.describe() for iv in self.tap.interventions if iv.run in (None, run.id)]
        weight_edits = self.edits.active()
        watch = self.watch  # neuron monitors' hooks, fixed for this request (sync_watch runs between requests)

        # Tokens shared with this run's previous request were already recorded. Never share
        # across runs: each run's tape must hold its own full prompt. Attention-only models
        # resume from the cache one token early (so the next prediction is available);
        # models with recurrent layers can't roll their state back and re-read from the start.
        reusable = self._kv is not None and self._kv_epoch == self._epoch() and self._kv_run == run.id
        lcp = min(common_prefix(self._kv_ids, ids), len(ids)) if reusable else 0
        if lcp and s.croppable_cache:
            start = lcp - 1
            if self._kv.get_seq_length() > start:
                self._kv.crop(start - self._kv.get_seq_length())  # transformers 5 crops by a negative count
        else:
            start, self._kv = 0, None
        record_from = lcp
        self._kv_epoch = self._epoch()
        self._kv_run = run.id
        self._kv_ids = ids[:start]
        self.tap.run_id = run.id
        self.tap.phase = "prefill"
        self.publish({"k": "req", "run": run.id, "req": req, "prompt_tokens": len(ids), "cached": start,
                      "already_recorded": record_from, "interventions": active})

        new_records: list[dict] = []
        anat_msgs: list[dict] = []  # sent after the prompt's token records, which the UI needs first
        pending = None
        t0 = time.perf_counter()
        timing = {"prefill_forward": 0.0, "prefill_readout": 0.0, "decode_forward": 0.0, "decode_readout": 0.0}

        def lap(key: str, since: float) -> float:
            if s.device.type == "cuda":
                torch.cuda.synchronize()
            now = time.perf_counter()
            timing[key] += now - since
            return now

        # A cancel between chunks keeps what was read: the cache then holds ids[:read_to] and the
        # next request of the run records on from there.
        read_to, finish = len(ids), "length"
        for cs in range(start, len(ids), s.prefill_chunk):
            if cs > start and cancel.is_set():
                read_to, finish = cs, "cancelled"
                break
            chunk = ids[cs:cs + s.prefill_chunk]
            tick = time.perf_counter()
            out = s.model(input_ids=torch.tensor([chunk], device=s.device), past_key_values=self._kv, use_cache=True)
            tick = lap("prefill_forward", tick)
            self._kv = out.past_key_values
            resid = self.tap.stack()
            logits = out.logits[0]
            nz = watch.read_chunk(len(chunk)) if watch is not None else None
            recs = self._prefill_records(run, req, rendered, cs, chunk, resid, logits, pending,
                                         record_from, new_from, nz=nz)
            new_records += recs
            if self.anatomy is not None:
                codes = self.anatomy.read_chunk(len(chunk))
                if recs:
                    anat_msgs.append(self.anatomy.add(run.id, [r["t"] for r in recs],
                                                      codes[-len(recs):].cpu().numpy()))
            pending = (resid[:, -1, :].clone(), logits[-1].float().clone())
            del out, resid, logits
            lap("prefill_readout", tick)
        self._kv_ids = ids[:read_to]
        prefill_s = time.perf_counter() - t0
        for i in range(0, len(new_records), 256):
            self.publish({"k": "toks", "run": run.id, "items": [compact(r) for r in new_records[i:i + 256]]})
        for msg in anat_msgs:
            self.publish(msg)

        for i in range(seen, len(p.messages)):
            first = rendered.first_token_of(i)
            unread = first is not None and first >= read_to  # a cancelled prefill stopped before it
            t = run.pos2tape.get(first) if first is not None and not unread else None
            if t is None:
                t = new_records[0]["t"] if new_records and not unread else max(len(run.tape) - 1, 0)
            m = tmsgs[i]
            self._add_message(run, {"i": i, "req": req, "t": t, "role": m["role"], "content": m["content"],
                                    "name": m.get("name"), "tool_calls": p.messages[i].get("tool_calls")})
            for etype, data in message_events(m):
                self._event(run, t, etype, data)

        # Decode, one token at a time, recording each step. Tokens inside a <think> block are
        # recorded with role "reasoning". A step queues the forward, then its readout, starts the
        # copies back to the host and waits for the GPU once. While the GPU works, the CPU settles the
        # previous step (record, events, streamed text) and detokenizes. Whether to go on is decided
        # from the text before the next forward, so the steps run exactly as they would one by one.
        self.tap.phase = "gen"
        think_open = rendered.text.rstrip().endswith(THINK_OPEN)
        stream = OutputStream(think_open, bool(p.tools), on_text, on_reasoning)
        pre_gen_t = run.pos2tape.get(len(ids) - 1)
        decision = {"pre_gen": (pre_gen_t, pending[0].to("cpu", torch.float16))}
        gen_ids, gen_tapes, text_lens, gen_states = [], [], [], []
        text, think_end = "", None
        generator = torch.Generator(device=s.device)
        if p.seed is not None:
            generator.manual_seed(int(p.seed))
        else:
            generator.seed()
        msg_idx = len(p.messages)
        # Interventions and the neuron watch's hooks are baked into the decode graphs, so the set that
        # applies and the watch (its key changes whenever the watched set does) key them.
        graph_key = (tuple(iv.id for iv in self.tap.interventions if iv.applies("gen", run.id)),
                     watch.key if watch is not None else None)
        if self.graphs is not None:
            self.graphs.extras = (lambda: (watch.step,)) if watch is not None else None
        cuda = s.device.type == "cuda"
        marks = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if cuda else None

        def settle(st: dict) -> None:
            """Record a decoded step and do its bookkeeping. Runs while the next step's forward is on
            the GPU; what decides whether decoding goes on (stop token, stop strings) is settled earlier."""
            nonlocal think_end
            host = st["host"]
            pred = {k: v.numpy() for k, v in host["pred"].items()}
            norm = host["norm"].numpy() if "norm" in host else None
            proj = {k: v.numpy()[0] for k, v in host["proj"].items()} if "proj" in host else {}
            nz = dict(zip(watch.labels, host["nz"].numpy())) if "nz" in host else None
            rec = self._record(run, req, st["pos"], st["tok"], st["role"], msg_idx, "gen", pred, 0, norm, proj, nz)
            new_records.append(rec)
            self.publish({"k": "toks", "run": run.id, "items": [compact(rec)]})
            if "anat" in host:
                self.publish(self.anatomy.add(run.id, [rec["t"]], host["anat"].numpy()[None]))
            if "state" not in host:  # a stop token: nothing was fed to the model
                return
            gen_states.append((rec["t"], host["state"].to(torch.float16)))
            gen_tapes.append(rec["t"])
            text_lens.append(st["text_len"])
            if st["role"] == "reasoning" and think_end is None and st["closed"]:
                think_end = len(gen_states) - 1
                self._event(run, rec["t"], "think_end", {"reasoning_tokens": st["n_gen"]})
            stream.update(st["text"])

        unsettled, picked = None, None
        t1 = time.perf_counter()
        for _ in range(max_tokens):
            # Checked before a token is picked: every token fed so far is in the cache and is recorded
            # (the last one when it settles below), as if max_tokens had been reached here.
            if cancel.is_set():
                finish = "cancelled"
                break
            next_id = sample(pending[1], p.temperature, p.top_p, generator) if picked is None else picked
            pos = len(self._kv_ids)
            is_stop = next_id in s.stop_ids
            reasoning, _, closed = split_reasoning(text, think_open)
            role = "reasoning" if reasoning is not None and not closed else "assistant"
            tick = time.perf_counter()
            read, dev, forward_s = pending, {}, 0.0
            if not is_stop:  # the forward goes first so the readout's small kernels launch while it runs
                tf = time.perf_counter()
                if cuda:
                    marks[0].record()
                logits = self.graphs.step(next_id, self._kv, graph_key) if self.graphs is not None else None
                if logits is None:
                    out = s.model(input_ids=torch.tensor([[next_id]], device=s.device),
                                  past_key_values=self._kv, use_cache=True)
                    self._kv, logits = out.past_key_values, out.logits
                if cuda:
                    marks[1].record()
                forward_s = time.perf_counter() - tf
                self._kv_ids.append(next_id)
                own = self.tap.stack()[:, 0, :]
                pending = (own.clone(), logits[0, -1].float())  # copies: graph outputs are reused next step
                gen_ids.append(next_id)
            # torch.full, not torch.tensor: a host-to-device copy would wait for the forward.
            target = torch.full((1,), next_id, dtype=torch.long, device=s.device)
            dev["pred"] = predict(s, read[0][:, None, :], read[1][None], target, host=False)
            if not is_stop:
                dev |= {"norm": own.float().norm(dim=-1), "state": own,
                        "proj": self.directions.project(own[:, None, :], host=False)}
                if self.anatomy is not None:
                    dev["anat"] = self.anatomy.read_step()
                if watch is not None:
                    dev["nz"] = watch.read_step()
                if p.temperature <= 0:  # the next greedy pick comes back with this step's copies
                    dev["next"] = pending[1].argmax()
            host = to_host(dev)
            if unsettled is not None:
                settle(unsettled)
            st = {"host": host, "pos": pos, "tok": next_id, "role": role}
            if not is_stop:
                text = s.tokenizer.decode(gen_ids, skip_special_tokens=False)
                st |= {"text_len": len(text), "closed": split_reasoning(text, think_open)[2], "n_gen": len(gen_ids)}
            if cuda:
                torch.cuda.current_stream().synchronize()
                if not is_stop:
                    forward_s = marks[0].elapsed_time(marks[1]) / 1e3
            picked = int(host["next"]) if "next" in host else None
            timing["decode_forward"] += forward_s
            timing["decode_readout"] += time.perf_counter() - tick - forward_s
            unsettled = st
            if is_stop:
                finish = "stop"
                break
            hit = next((x for x in p.stop if x and x in text), None)
            if hit:
                text, finish = text[:text.index(hit)], "stop"
            st["text"] = text
            if hit:
                break
        if unsettled is not None:
            settle(unsettled)
        decode_s = time.perf_counter() - t1
        stream.update(text, final=True)

        reasoning, answer, _ = split_reasoning(text, think_open)
        reasoning = reasoning.strip() if reasoning is not None else None
        # A cancelled reply is returned as the text so far, never as tool calls a harness might run.
        if p.tools and finish != "cancelled":
            content, calls = parse_tool_calls(answer, p.tools)
        else:
            content, calls = answer.strip() or None, []
        stream.finish(content)
        if calls:
            finish = "tool_calls"
        answer_at = len(text) - len(answer)
        first_t = gen_tapes[0] if gen_tapes else (new_records[-1]["t"] if new_records else 0)
        for c in calls:
            at = text.find(c["function"]["name"], answer_at)
            k = min(bisect.bisect_left(text_lens, at + 1), len(gen_tapes) - 1) if at >= 0 and gen_tapes else 0
            self._event(run, gen_tapes[k] if gen_tapes else first_t, "tool_call",
                        {"name": c["function"]["name"], "arguments": c["function"]["arguments"][:400]})
        if finish in ("length", "cancelled") and not calls:
            last_t = gen_tapes[-1] if gen_tapes else first_t
            data = {"tokens": len(gen_ids), "while_reasoning": think_open and not split_reasoning(text, think_open)[2]}
            if finish == "cancelled":
                data["in_prompt"] = read_to < len(ids)
            self._event(run, last_t, "cutoff" if finish == "length" else "cancelled", data)
        elif not calls:
            self._event(run, first_t, "final_answer", {"chars": len(answer), "finish": finish})
        self._add_message(run, {"i": msg_idx, "req": req, "t": first_t, "role": "assistant", "content": content,
                                "reasoning": reasoning, "tool_calls": calls or None, "generated": True,
                                "finish": finish})

        if self.settings.decision_residuals and gen_states:
            lead = answer_at + len(answer) - len(answer.lstrip())
            picks = {"first": 0, "last": len(gen_states) - 1}
            if think_end is not None:
                picks["think_end"] = think_end
            if calls:
                picks["tool_call_start"] = min(bisect.bisect_right(text_lens, lead), len(gen_states) - 1)
            for label, k in picks.items():
                decision[label] = gen_states[k]
            tensors = {f"t{t}": state for t, state in decision.values() if t is not None}
            marks = {label: t for label, (t, _) in decision.items() if t is not None}
            if self.store.write_residuals(run.id, req, tensors, {"marks": marks, "model": s.spec,
                                                                 "shape": [s.n_layers, s.d_model]}):
                run.meta["residuals"] += sorted(set(marks.values()))

        run.seen_msgs = len(p.messages) + 1
        run.last_prompt = ids
        run.last_msg_count = len(p.messages)
        run.meta.update(tokens=len(run.tape), updated=time.time())
        if active:
            run.meta["interventions"].append({"req": req, "active": active})
        if weight_edits:
            run.meta.setdefault("weight_edits", []).append({"req": req, "edits": weight_edits})
        self.store.write_tokens(run.id, req, new_records)
        if self.anatomy is not None:
            try:  # storing activity must never fail the completion (e.g. disk full)
                self.anatomy.flush(self.store, run.id, req)
            except Exception as exc:
                self.anatomy.pending.pop(run.id, None)
                print(f"[brainscan] could not store weight activity for {run.id}: {type(exc).__name__}: {exc}", flush=True)
        self.store.write_meta(run.id, run.meta)
        n_new = read_to - start
        self.perf.update(prefill_tok_s=round(n_new / prefill_s, 1) if prefill_s > 0 else None,
                         decode_tok_s=round(len(gen_ids) / decode_s, 1) if decode_s > 0 and gen_ids else None,
                         seconds={k: round(v, 3) for k, v in timing.items()})
        print(f"[brainscan] {run.id} req {req}: prefill {n_new} tok ({max(0, read_to - record_from)} recorded), "
              f"decode {len(gen_ids)} tok, " + ", ".join(f"{k} {v:.2f}s" for k, v in timing.items())
              + (", cancelled" if finish == "cancelled" else ""), flush=True)
        self.publish({"k": "done", "run": run.id, "req": req, "finish": finish, "generated": len(gen_ids),
                      "perf": dict(self.perf), "meta": run.meta})
        return GenResult(run.id, req, text, content, calls, finish, len(ids), len(gen_ids), start, reasoning)

    # ---- reads ---------------------------------------------------------------------

    def token_detail(self, run_id: str, t: int) -> dict | None:
        run = self.get_run(run_id)
        if run is None or not 0 <= t < len(run.tape):
            return None
        r = dict(run.tape[t])
        r["top"] = [[self.s.token_str(i), p] for i, p in zip(r.get("top_ids") or [], r.get("top_p") or [])]
        r["lens_top_text"] = [self.s.token_str(i) for i in (r.get("lens_top") or [])]
        r["has_residual"] = t in set(run.meta.get("residuals", []))
        r["model_mismatch"] = run.meta.get("model", {}).get("spec") != self.s.spec
        return r

    def status(self) -> dict:
        graphs = self.graphs.describe() if self.graphs else {"on": False, "off": "disabled"}
        active = self.active
        generating = None if active is None else {"run": active["run"] or active["named"], "req": active["req"],
                                                  "cancelling": active["cancel"].is_set()}
        return {"model": self.s.info(), "perf": self.perf | {"decode_graphs": graphs},
                "generating": generating, "queued": len(self._queued),
                "disk": {"used": self.store.used_bytes(), "budget": self.store.budget},
                "interventions": [iv.describe() for iv in self.interventions.values()],
                "directions": self.directions.describe(), "workbench": self.edits.summary(),
                "knobs": self.knobs.names()}

    # ---- workbench -----------------------------------------------------------------

    def add_weight_edit(self, kind: str, direction: str | None = None, seed: int | None = None, **kwargs):
        vector = None
        if kind == "orthogonalize":
            if direction == "random":
                vector = torch.randn(self.s.d_model, generator=torch.Generator().manual_seed(seed or 0))
                direction = f"random(seed={seed or 0})"
            elif direction in self.directions.dirs:
                vector = self.directions.dirs[direction].unit
            else:
                raise ValueError(f"unknown direction {direction!r}")
        with self.lock:
            return self.edits.add(kind, direction=direction, vector=vector, **kwargs)

    def apply_weight_edits(self) -> dict:
        with self.lock, torch.no_grad():
            t0 = time.perf_counter()
            try:
                self.edits.apply()
                if self.anatomy is not None:
                    self.anatomy.refresh()
            finally:
                self.release_memory()
            summary = self.edits.summary() | {"seconds": round(time.perf_counter() - t0, 2)}
        self.publish({"k": "workbench", "summary": summary})
        return summary

    def revert_weight_edits(self) -> dict:
        with self.lock:
            self.edits.revert()
            if self.anatomy is not None:
                self.anatomy.refresh()
            self.release_memory()
            summary = self.edits.summary()
        self.publish({"k": "workbench", "summary": summary})
        return summary

    def analyze_direction(self, positive: list[str], negative: list[str], position: str, chat: bool) -> dict:
        """Directions are measured on the original weights, even while edits are applied."""
        with self.lock:
            try:
                with self.edits.suspended():
                    return self.directions.analyze(positive, negative, position, chat)
            finally:
                self.release_memory()
