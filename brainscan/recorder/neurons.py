"""Neurons: single MLP neurons as units you can read and steer.

A neuron is one input channel of a layer's MLP output projection (down_proj): at every position it
writes act × W_down[:, n] into the residual stream. Neurons are named like "L5 N120" (layer, index).

  steer      dose scaling: a dose is expressed in units of the layer's typical residual size, so
             one dose adds that much along the neuron's output direction at every position. The
             typical size is the median residual norm at the layer's output on the calibration
             texts (workbench.REFERENCE_TEXTS), current weights, without the first token (the
             attention sink). The activation gets dose × that median / ‖W_down[:, n]‖. During a
             chat or an agent run it is a "write" intervention on the layer's output: the same vector.

  watch      a monitor on one neuron (monitors.py): its activation z-scored per token against a mean
             and spread frozen when the watch is built (the neuron index's when it exists, else the
             calibration texts'). NeuronWatch below reads it live, only while such a monitor is on.

  card      what the neuron does, read on the calibration texts:
               logit lens   tokens its write pushes up and down, read straight through the final
                            norm and the unembedding (weights only, nothing run)
               steering effect   what steering it does to the model's predictions: the change in
                            next-token log-probs averaged over every position, at +dose and −dose,
                            with the mean KL from the unsteered predictions
               activity     its activation over the texts, and the contexts where it fires most
               feeds        later neurons whose activation moves most when it is steered at +dose
                            (measured, in units of each one's spread on the texts)
               fed by       earlier neurons whose direct writes drive it where it fires most: each
                            one's activation × its write read through this neuron's input (gate and
                            up, linearized at that position, through the norm in front of the MLP).
                            Attention and paths through other layers are left out.

Everything runs with runtime steering paused, on the weights as they are (edits included).
"""

from __future__ import annotations

import contextlib
import itertools
import math
import re

import torch
import torch.nn.functional as F
from torch import nn

from .anatomy import MLP_NORMS
from .edits import MLP_NAMES, _child, residual_linears
from .tracing import Busy
from .workbench import REFERENCE_TEXTS, _generate, _reference_ids

NAME = re.compile(r"^\s*L(\d+)[\s/:._-]*N(\d+)\s*$", re.IGNORECASE)
LABEL = re.compile(r"^L\d+ N\d+$")  # a neuron as label() writes it; a direction's name can't have the space
SET = re.compile(r"L([0-9]+)\s*N([0-9]+)\s*([+−-]?\s*[0-9]*\.?[0-9]+)?", re.IGNORECASE)  # ui/neurons.js addTyped
TOP = 12            # tokens and neurons listed per side
FIRE = 6            # positions where a neuron fires most, for its contexts and its inputs
MAX_DOSE = 16.0
MIN_STD = 1e-4      # a watched neuron's spread is never taken as smaller than this


def parse(name: str) -> tuple[int, int]:
    m = NAME.match(str(name))
    if not m:
        raise ValueError(f"name a neuron like L5 N120, not {name!r}")
    return int(m[1]), int(m[2])


def label(layer: int, n: int) -> str:
    return f"L{layer} N{n}"


def is_label(text) -> bool:
    return isinstance(text, str) and LABEL.match(text) is not None


def parse_set(text: str) -> list[tuple[int, int, float]]:
    """"L5 N120 +1, L9 N64 -0.5" -> [(5, 120, 1.0), (9, 64, -0.5)]. A neuron with no number
    gets dose 1; the minus sign may be the unicode one."""
    found = list(SET.finditer(str(text or "")))
    if not found:
        raise ValueError("name neurons with a dose like L5 N120 +1, L9 N64 -0.5")
    return [(int(m[1]), int(m[2]), float(re.sub(r"\s", "", m[3].replace("−", "-"))) if m[3] else 1.0) for m in found]


class MLP:
    """One layer's dense MLP: its output projection, its input projections and the norm in front."""

    def __init__(self, block: nn.Module, d: int):
        mlp = _child(block, MLP_NAMES)
        downs = residual_linears(mlp, d, "out")
        if len(downs) != 1:
            raise ValueError("this layer has no single dense MLP (a mixture of experts?)")
        self.down = downs[0]
        readers = {name.rsplit(".", 1)[-1]: m for name, m in mlp.named_modules()
                   if isinstance(m, nn.Linear) and m.in_features == d and m.out_features == self.down.in_features}
        self.gate = readers.get("gate_proj") or readers.get("w1")
        self.up = readers.get("up_proj") or readers.get("w3")
        if self.gate is None or self.up is None:  # a plain MLP: one input projection
            self.gate, self.up = None, next(iter(readers.values()), None)
        self.norm = _child(block, MLP_NORMS)
        self.width = self.down.in_features


class Neurons:
    def __init__(self, recorder):
        self.rec, self.s = recorder, recorder.s
        self._mlps: dict[int, MLP] = {}
        self._cal: dict | None = None

    # ---- addressing --------------------------------------------------------------------

    def mlp(self, layer: int) -> MLP:
        if not 0 <= layer < self.s.n_layers:
            raise KeyError(f"no layer {layer}; layers are 0..{self.s.n_layers - 1}")
        if layer not in self._mlps:
            self._mlps[layer] = MLP(self.s.layers[layer], self.s.d_model)
        return self._mlps[layer]

    def check(self, layer: int, n: int) -> MLP:
        mlp = self.mlp(layer)
        if not 0 <= n < mlp.width:
            raise KeyError(f"no neuron {n} in layer {layer}; it has {mlp.width}")
        return mlp

    def write(self, layer: int, n: int) -> torch.Tensor:
        """W_down[:, n] in fp32."""
        return self.check(layer, n).down.weight[:, n].detach().float()

    @contextlib.contextmanager
    def _model(self):
        rec = self.rec
        if not rec.lock.acquire(timeout=1.0):
            raise Busy("the model is generating; try again when the request finishes")
        try:
            with torch.no_grad(), rec.tap.paused():
                yield
        finally:
            rec.lock.release()

    # ---- the calibration read -----------------------------------------------------------

    def calibration(self) -> dict:
        """One read of the calibration texts on the current weights, kept until they change."""
        version = self.rec.edits.version
        if self._cal is None or self._cal["version"] != version:
            with self._model():
                self._cal = self._calibrate(version)
        return self._cal

    def _calibrate(self, version: int) -> dict:
        s, n_layers = self.s, self.s.n_layers
        dense = [i for i in range(n_layers) if self._try_mlp(i)]
        acts = {i: [] for i in dense}
        xhat = {i: [] for i in dense}
        resid_in = {i: [] for i in dense}
        handles = []

        def grab(store, i):
            def fn(module, args):
                store[i].append(args[0][0].detach())
            return fn

        for i in dense:
            m = self.mlp(i)
            handles.append(m.down.register_forward_pre_hook(grab(acts, i)))
            reader = m.gate or m.up
            handles.append(reader.register_forward_pre_hook(grab(xhat, i)))
            if m.norm is not None:
                handles.append(m.norm.register_forward_pre_hook(grab(resid_in, i)))
        texts, logps, norms = [], [], [[] for _ in range(n_layers)]
        try:
            for text in REFERENCE_TEXTS:
                ids = _reference_ids(s, text)
                x = torch.tensor([ids], device=s.device)
                logits = s.model(input_ids=x, use_cache=False).logits[0].float()
                logps.append(torch.log_softmax(logits, -1).to("cpu", torch.float16))
                stack = self.rec.tap.stack()  # (layers, T, d): the residual leaving each layer
                for i in range(n_layers):
                    norms[i].append(stack[i, 1:].float().norm(dim=-1))
                texts.append(ids)
        finally:
            for h in handles:
                h.remove()
        keep = lambda chunks: torch.cat([c[1:] for c in chunks])  # every text without its first token
        cal = {"version": version, "texts": texts, "logp": logps,
               "pos": [(t, p) for t, ids in enumerate(texts) for p in range(1, len(ids))],
               "median": [float(torch.cat(norms[i]).median()) for i in range(n_layers)],
               "acts": {}, "xhat": {}, "resid": {}, "mean": {}, "std": {}}
        for i in dense:
            a = keep(acts[i]).float()
            cal["acts"][i] = a.half()
            cal["mean"][i], cal["std"][i] = a.mean(0), a.std(0).clamp_min(1e-4)
            cal["xhat"][i] = keep(xhat[i]).float()
            cal["resid"][i] = keep(resid_in[i]).float() if resid_in[i] else None  # in front of the MLP's norm
        return cal

    def _try_mlp(self, layer: int) -> bool:
        try:
            self.mlp(layer)
            return True
        except ValueError:
            return False

    def stats_source(self, items: list[tuple[int, int]]) -> str:
        """Where the mean and spread a watch z-scores against come from: the neuron index when it is
        built (the original weights, over its corpus), else the calibration texts on the current weights."""
        index = getattr(self.rec, "neuron_index", None)
        if index is not None and index.ready() and all(f"mean.{i}" in index.data for i, _ in items):
            return "index"
        return "calibration"

    def stats(self, items: list[tuple[int, int]], source: str) -> dict[tuple[int, int], tuple[float, float]]:
        """(mean, std) of each neuron's activation, read once from `source` (stats_source)."""
        if source == "index":
            data = self.rec.neuron_index.data
            raw = {(i, n): (float(data[f"mean.{i}"][n]), float(data[f"std.{i}"][n])) for i, n in items}
        else:
            cal = self.calibration()
            raw = {(i, n): (float(cal["mean"][i][n]), float(cal["std"][i][n])) for i, n in items}
        return {k: (mean, max(std, MIN_STD)) for k, (mean, std) in raw.items()}

    # ---- steering ------------------------------------------------------------------------

    def boost(self, layer: int, n: int, dose: float) -> float:
        """Dose scaling: what is added to the neuron's activation."""
        if not math.isfinite(dose) or abs(dose) > MAX_DOSE:
            raise ValueError(f"dose must be a number between -{MAX_DOSE:g} and {MAX_DOSE:g}")
        w = self.write(layer, n)
        return dose * self.calibration()["median"][layer] / float(w.norm().clamp_min(1e-8))

    def write_vector(self, layer: int, n: int, dose: float) -> tuple[torch.Tensor, float]:
        """(unit write direction, its length) for a runtime "write" intervention on layer L's output:
        boost × W_down[:, n] = dose × median norm along the unit direction."""
        boost = self.boost(layer, n, dose)
        w = self.write(layer, n)
        return w / w.norm().clamp_min(1e-8), boost * float(w.norm())

    @contextlib.contextmanager
    def steered(self, steers: list[tuple[int, int, float]]):
        """Hooks that add each neuron's boost to its activation at every position."""
        handles = []
        by_layer: dict[int, list[tuple[int, float]]] = {}
        for layer, n, dose in steers:
            by_layer.setdefault(layer, []).append((n, self.boost(layer, n, dose)))
        for layer, items in by_layer.items():
            def fn(module, args, items=items):
                x = args[0].clone()
                for n, b in items:
                    x[..., n] += b
                return (x, *args[1:])
            handles.append(self.mlp(layer).down.register_forward_pre_hook(fn))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def sweep(self, prompt: str, steers: list[tuple[int, int, float]], factors: list[float],
              max_tokens: int = 60) -> list[dict]:
        """The same prompt answered (greedy) with every dose scaled by each factor: a dose response."""
        if not prompt.strip():
            raise ValueError("give a prompt to answer")
        if not steers:
            raise ValueError("pick at least one neuron to steer")
        for layer, n, dose in steers:
            self.boost(layer, n, dose * max(abs(f) for f in factors))  # validates every neuron and dose
        self.calibration()
        out = []
        with self._model():
            for f in factors:
                with self.steered([(layer, n, dose * f) for layer, n, dose in steers]):
                    text = _generate(self.s, prompt, max_tokens, self.rec.settings.template_kwargs())
                out.append({"factor": f, "doses": [round(dose * f, 4) for _, _, dose in steers], "text": text})
        return out

    # ---- the card ------------------------------------------------------------------------

    def card(self, layer: int, n: int, dose: float = 1.0) -> dict:
        mlp = self.check(layer, n)
        cal = self.calibration()
        if layer not in cal["acts"]:
            raise ValueError(f"layer {layer} has no dense MLP")
        s, tok = self.s, self.s.token_str
        w = self.write(layer, n)
        out = {"neuron": label(layer, n), "layer": layer, "index": n, "dose": dose,
               "write_norm": round(float(w.norm()), 5), "median_norm": round(cal["median"][layer], 4),
               "boost": round(self.boost(layer, n, dose), 5), "unit": self._unit_of(layer, n)}
        with self._model():
            logits = s.lm_head(s.norm(w.to(s.lm_head.weight.dtype)[None, None]))[0, 0].float()
            out["logit_lens"] = {"up": self._tokens(logits, True), "down": self._tokens(logits, False)}
            plus, moved = self._steer_effect(cal, layer, n, dose, measure_downstream=True)
            minus, _ = self._steer_effect(cal, layer, n, -dose)
            out["steer_effect"] = {"plus": plus, "minus": minus}
            out["feeds"] = self._feeds(cal, layer, moved)
            out["fed_by"] = self._fed_by(cal, mlp, layer, n)
        out["activity"] = self._activity(cal, layer, n)
        index = getattr(self.rec, "neuron_index", None)
        out["corpus"] = index.contexts(layer, n) if index is not None else None  # its top activations in the index
        return out

    def _unit_of(self, layer: int, n: int) -> dict | None:
        an = getattr(self.rec, "anatomy", None)
        if an is None or an.layers[layer].assign is None:
            return None
        lm = an.layers[layer]
        c = int(lm.assign[n])
        return {"g": lm.offset + lm.n_heads + c, "name": f"L{layer} c{c}"}

    def _tokens(self, scores: torch.Tensor, up: bool, k: int = TOP) -> list[dict]:
        vals, ids = scores.topk(k, largest=up)
        return [{"id": int(i), "text": self.s.token_str(int(i)), "v": round(float(v), 4)} for v, i in zip(vals, ids)]

    def _steer_effect(self, cal: dict, layer: int, n: int, dose: float, measure_downstream: bool = False):
        s = self.s
        delta = torch.zeros(s.vocab_size, device=s.device)
        kl_sum, count = 0.0, 0
        sums: dict[int, torch.Tensor] = {}
        handles = []
        if measure_downstream:
            def acc(i):
                def fn(module, args):
                    a = args[0][0, 1:].float()
                    sums[i] = sums.get(i, 0) + a.sum(0)
                return fn
            handles = [self.mlp(i).down.register_forward_pre_hook(acc(i)) for i in cal["acts"] if i > layer]
        try:
            with self.steered([(layer, n, dose)]):
                for ids, base in zip(cal["texts"], cal["logp"]):
                    x = torch.tensor([ids], device=s.device)
                    logp = torch.log_softmax(s.model(input_ids=x, use_cache=False).logits[0, 1:].float(), -1)
                    lb = base[1:].to(s.device).float()
                    delta += (logp - lb).sum(0)
                    kl_sum += float((lb.exp() * (lb - logp)).sum())
                    count += logp.shape[0]
        finally:
            for h in handles:
                h.remove()
        delta /= max(count, 1)
        lens = {"dose": dose, "kl": round(kl_sum / max(count, 1), 5),
                "up": self._tokens(delta, True), "down": self._tokens(delta, False)}
        moved = {i: v / max(count, 1) for i, v in sums.items()}
        return lens, moved

    def _feeds(self, cal: dict, layer: int, moved: dict) -> dict:
        """Later neurons whose mean activation moves most with this one steered at +dose."""
        rows = []
        for i, mean in moved.items():
            z = (mean - cal["mean"][i]) / cal["std"][i]
            vals, idx = z.abs().topk(min(TOP, len(z)))
            rows += [(float(z[j]), i, int(j)) for j in idx]
        rows.sort(key=lambda r: -abs(r[0]))
        fmt = lambda r: {"neuron": label(r[1], r[2]), "layer": r[1], "index": r[2], "z": round(r[0], 3)}
        return {"toward": [fmt(r) for r in rows if r[0] > 0][:TOP], "away": [fmt(r) for r in rows if r[0] < 0][:TOP],
                "measure": "mean activation with this neuron steered, minus unsteered, in units of its spread"}

    def _fed_by(self, cal: dict, mlp: MLP, layer: int, n: int) -> dict:
        """Earlier neurons whose direct writes drive this one where it fires most."""
        a = cal["acts"][layer][:, n].float()
        top = a.topk(min(FIRE, len(a))).indices
        xh = cal["xhat"][layer][top]
        gain = self._gain(mlp, layer)
        up_w = mlp.up.weight[n].detach().float()
        if mlp.gate is not None:  # act = silu(gate·x̂) × (up·x̂), linearized at each position
            gate_w = mlp.gate.weight[n].detach().float()
            g, u = xh @ gate_w, xh @ up_w
            sg = torch.sigmoid(g)
            dsilu = sg * (1 + g * (1 - sg))
            grad = (g * sg)[:, None] * up_w[None] + (dsilu * u)[:, None] * gate_w[None]
        else:
            grad = up_w[None].expand(len(top), -1)
        J = self._through_norm(mlp, grad * gain[None], cal["resid"][layer][top] if cal["resid"][layer] is not None else None)
        rows = []
        for i in cal["acts"]:
            if i >= layer:
                continue
            down = self.mlp(i).down.weight.detach().float()  # (d, neurons)
            c = (cal["acts"][i][top].float() * (J @ down)).mean(0)  # (neurons,)
            vals, idx = c.abs().topk(min(TOP, len(c)))
            rows += [(float(c[j]), i, int(j)) for j in idx]
        rows.sort(key=lambda r: -abs(r[0]))
        fmt = lambda r: {"neuron": label(r[1], r[2]), "layer": r[1], "index": r[2], "c": round(r[0], 4)}
        return {"drives_up": [fmt(r) for r in rows if r[0] > 0][:TOP], "drives_down": [fmt(r) for r in rows if r[0] < 0][:TOP],
                "positions": len(top), "mean_activation_there": round(float(a[top].mean()), 4),
                "measure": "activation × its write read through this neuron's input, at the positions it fires most"}

    @staticmethod
    def _through_norm(mlp: MLP, g: torch.Tensor, x: torch.Tensor | None) -> torch.Tensor:
        """d act / d residual from d act / d (normed input) × gain, through an RMS norm's scale:
        y = gain ⊙ x / r with r = sqrt(mean(x²) + eps), so dy/dx = diag(gain)/r − (gain ⊙ x) xᵀ / (d r³)."""
        if x is None:
            return g
        eps = float(getattr(mlp.norm, "variance_epsilon", getattr(mlp.norm, "eps", 1e-6)) or 1e-6)
        d = x.shape[-1]
        r = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        return g / r - x * (x * g).sum(-1, keepdim=True) / (d * r ** 3)

    def _gain(self, mlp: MLP, layer: int) -> torch.Tensor:
        d, dev = self.s.d_model, self.s.device
        if mlp.norm is None:
            return torch.ones(d, device=dev)
        p = next(mlp.norm.parameters(), None)
        return mlp.norm(torch.ones(1, d, device=dev, dtype=p.dtype if p is not None else torch.float32)).float()[0]

    def _activity(self, cal: dict, layer: int, n: int) -> dict:
        a = cal["acts"][layer][:, n].float()
        top = a.topk(min(FIRE, len(a))).indices.tolist()
        contexts = []
        for k in top:
            t, p = cal["pos"][k]
            ids = cal["texts"][t]
            contexts.append({"act": round(float(a[k]), 4),
                             "before": "".join(self.s.token_str(i) for i in ids[max(1, p - 10):p]),
                             "token": self.s.token_str(ids[p]),
                             "after": "".join(self.s.token_str(i) for i in ids[p + 1:p + 4])})
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4), "max": round(float(a.max()), 4),
                "min": round(float(a.min()), 4), "positions": len(a), "contexts": contexts}

    def summary(self) -> dict:
        cal = self._cal if self._cal and self._cal["version"] == self.rec.edits.version else None
        widths = []
        for i in range(self.s.n_layers):
            try:
                widths.append(self.mlp(i).width)
            except ValueError:
                widths.append(0)
        return {"layers": self.s.n_layers, "widths": widths,
                "median_norms": [round(x, 3) for x in cal["median"]] if cal else None,
                "calibration": "workbench.REFERENCE_TEXTS, first token skipped",
                "steering": [iv.describe() for iv in self.rec.interventions.values() if iv.mode == "write"]}


class NeuronWatch:
    """The live signal of neuron monitors: each watched neuron's activation, z-scored per token.

    A forward pre-hook on `mlp.down` of every layer with a watched neuron reads the activation (the
    input to down_proj at the neuron's index) and z-scores it against a mean and spread frozen when
    the watch is built. The hooks follow the cluster probe in anatomy.py, so they work in prefill
    chunks and inside decode CUDA graphs: a single-token forward writes into the static buffer `step`
    (a captured graph refreshes it on every replay), a longer forward keeps its rows in `parts`, and
    the recorder copies them out after each forward (read_step, read_chunk). They do nothing outside
    `recording()`. A watch exists only while an enabled neuron monitor does (Recorder.sync_watch).

    `key` names this watch. Decode graphs bake the hooks and their buffers in, so graphs captured
    under another watch are stale: the recorder puts `key` in the graph key.
    """

    _serial = itertools.count(1)

    def __init__(self, nm: Neurons, items, stats: dict, source: str):
        self.items = tuple(sorted(set(items)))
        self.labels = [label(i, n) for i, n in self.items]
        self.stats, self.source = {k: stats[k] for k in self.items}, source  # (mean, std), frozen
        self.key = ("watch", next(self._serial), self.items)
        dev = nm.s.device
        self.step = torch.zeros(len(self.items), device=dev)  # z of the latest single-token forward
        self.parts: dict[int, torch.Tensor] = {}              # layer -> (tokens, its neurons), longer forwards
        self.slices: dict[int, tuple[int, int]] = {}          # layer -> its columns of step
        self.enabled = False
        self.handles = []
        for layer in sorted({i for i, _ in self.items}):
            rows = [j for j, (i, _) in enumerate(self.items) if i == layer]  # items are sorted: contiguous
            a, b = rows[0], rows[-1] + 1
            idx = torch.tensor([self.items[j][1] for j in rows], device=dev)
            mean = torch.tensor([self.stats[self.items[j]][0] for j in rows], device=dev)
            std = torch.tensor([self.stats[self.items[j]][1] for j in rows], device=dev)
            self.slices[layer] = (a, b)
            self.handles.append(nm.mlp(layer).down.register_forward_pre_hook(self._hook(layer, a, b, idx, mean, std)))

    def _hook(self, layer: int, a: int, b: int, idx: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        def fn(module, args):
            x = args[0]
            if not self.enabled or x.dim() != 3 or x.shape[0] != 1:
                return None
            z = (x[0].index_select(-1, idx).float() - mean) / std
            if z.shape[0] == 1:
                self.step[a:b].copy_(z[0])
            else:
                self.parts[layer] = z
            return None
        return fn

    @contextlib.contextmanager
    def recording(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False
            self.parts.clear()

    def read_step(self) -> torch.Tensor:
        """z (neurons,) for the latest single-token forward, as a copy on the device."""
        return self.step.clone()

    def read_chunk(self, n: int) -> torch.Tensor:
        """z (n, neurons) for the latest n-token forward, on the device."""
        if n == 1:
            return self.step[None].clone()
        out = torch.zeros(n, len(self.items), device=self.step.device)
        for layer, z in self.parts.items():
            if z.shape[0] == n:
                a, b = self.slices[layer]
                out[:, a:b] = z
        self.parts.clear()
        return out

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles, self.enabled = [], False
        self.parts.clear()

    def describe(self) -> dict:
        return {"neurons": self.labels, "source": self.source}
