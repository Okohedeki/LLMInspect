"""Monitors: flag a run when a probe fires.

A monitor watches one direction of the bank (a probe: every recorded token's residual is projected
onto it and z-scored, see directions.py) or one neuron (its activation, z-scored against a frozen mean
and spread, see neurons.py NeuronWatch), and fires when that z-score stays past a threshold for some
tokens in a row. A neuron is named like "L5 N120" where a direction would be; its z-scores sit on the
token record under "nz", a direction's under "proj". Each firing is a flag: the token where the rule
was met, where the streak began and ended, and its peak. Flags come out the same two ways, because
both apply one rule to the same recorded numbers:

  live  as each token is recorded; the recorder adds a "monitor" event at the token where the rule
        is met, so the UI and a harness watching the live feed see it while the run goes on
  scan  over a recorded run's tape, which is how runs recorded before a monitor existed are checked

Every scan also applies a direction's rule to the bank's random-control direction on the same tokens.
A rule the random direction trips about as often as the probe is reading noise, not the probe. (A
neuron's rule has no such control.) The z-scores are relative to each direction's own example texts
(a neuron's, to its activations on the index or calibration texts), so a threshold of 2 means "two
spreads past the examples' mean", not a calibrated probability, and a flag is a pointer to tokens
worth reading.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .neurons import is_label, label, parse

SIGNS = ("above", "below", "either")
PHASES = {"gen": ("gen",), "all": ("gen", "prefill")}  # re-read context ("replay") is never counted
MAX_MONITORS = 128  # a knob of 64 neurons is 64 rules


@dataclass
class Monitor:
    direction: str
    threshold: float = 2.0
    sign: str = "above"
    min_tokens: int = 3
    phase: str = "gen"
    label: str = ""
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def check(self) -> Monitor:
        self.threshold, self.min_tokens = float(self.threshold), int(self.min_tokens)
        self.label, self.enabled = str(self.label or "")[:64], bool(self.enabled)
        if self.sign not in SIGNS:
            raise ValueError(f"sign must be one of {SIGNS}")
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {tuple(PHASES)}")
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 20:
            raise ValueError("threshold must be a z-score between 0 and 20")
        if not 1 <= self.min_tokens <= 512:
            raise ValueError("min_tokens must be between 1 and 512")
        return self

    @property
    def neuron(self) -> bool:
        return is_label(self.direction)

    def eligible(self, rec: dict) -> bool:
        return rec.get("ph") in PHASES[self.phase]

    def passes(self, z) -> bool:
        if z is None:
            return False
        return z >= self.threshold if self.sign == "above" else z <= -self.threshold if self.sign == "below" \
            else abs(z) >= self.threshold

    def rule(self) -> str:
        cmp = {"above": f"z ≥ {self.threshold:g}", "below": f"z ≤ −{self.threshold:g}",
               "either": f"|z| ≥ {self.threshold:g}"}[self.sign]
        where = "the model's own tokens" if self.phase == "gen" else "every new token"
        n = self.min_tokens
        return f"{self.direction} {cmp} for {n} token{'' if n == 1 else 's'} in a row, on {where}"

    def describe(self) -> dict:
        return asdict(self) | {"rule": self.rule(), "name": self.label or self.direction,
                               "kind": "neuron" if self.neuron else "direction"}


def z_of(monitor: Monitor, rec: dict, key: str):
    """The z-score a token record holds for `key`: the monitor's own neuron is in "nz", any direction in "proj"."""
    return (rec.get("nz" if monitor.neuron and key == monitor.direction else "proj") or {}).get(key)


class Streak:
    """One rule followed along a tape, token by token: the live check and the scan share it."""

    def __init__(self, monitor: Monitor, key: str):
        self.m, self.key, self.cur = monitor, key, None

    def feed(self, rec: dict) -> tuple[dict | None, dict | None]:
        """(the flag raised at this token, the flag this token ended), either or both None."""
        z = z_of(self.m, rec, self.key) if self.m.eligible(rec) else None
        if self.m.passes(z):
            c = self.cur
            if c is None:
                c = self.cur = {"start": rec["t"], "end": rec["t"], "n": 0, "peak_t": rec["t"], "peak_z": z,
                                "req": rec.get("req"), "raised_t": None}
            c["n"] += 1
            c["end"] = rec["t"]
            if abs(z) > abs(c["peak_z"]):
                c["peak_t"], c["peak_z"] = rec["t"], z
            if c["n"] == self.m.min_tokens:
                c["raised_t"], c["raised_z"] = rec["t"], z
                return c, None
            return None, None
        return None, self.close()

    def close(self) -> dict | None:
        c, self.cur = self.cur, None
        return c if c is not None and c["n"] >= self.m.min_tokens else None


def _flag(c: dict) -> dict:
    return {"t": c["raised_t"], "z": round(c["raised_z"], 3), "start": c["start"], "end": c["end"], "tokens": c["n"],
            "peak_t": c["peak_t"], "peak_z": round(c["peak_z"], 3), "req": c["req"]}


def scan(monitor: Monitor, tape: list[dict], key: str | None = None) -> list[dict]:
    """Every flag the rule raises on a recorded tape, read on direction `key` (the monitor's own
    by default; the random control for the comparison)."""
    s, flags = Streak(monitor, key or monitor.direction), []
    for rec in tape:
        if rec is None:
            continue
        _, ended = s.feed(rec)
        if ended:
            flags.append(_flag(ended))
    last = s.close()
    if last:
        flags.append(_flag(last))
    return flags


def _past(monitor: Monitor, tape: list[dict], key: str) -> tuple[int, float | None]:
    """How many eligible tokens carry this direction, and the share of them past the threshold."""
    zs = [z_of(monitor, r, key) for r in tape if r is not None and monitor.eligible(r)]
    zs = [z for z in zs if z is not None]
    return len(zs), (round(sum(monitor.passes(z) for z in zs) / len(zs), 4) if zs else None)


def control_of(directions: list[dict]) -> str | None:
    """The bank's random-control direction, if it has one."""
    return next((d["name"] for d in directions if d.get("kind") == "random_control"), None)


def scan_run(bank: MonitorBank, tape: list[dict], directions: list[dict]) -> list[dict]:
    """Each monitor on one run: its flags, and the same rule on the random control."""
    control = control_of(directions)
    by_t = {r["t"]: r for r in tape if r is not None}
    out = []
    for m in bank.monitors():
        eligible = sum(1 for r in tape if r is not None and m.eligible(r))
        recorded, past = _past(m, tape, m.direction)
        flags = scan(m, tape) if recorded else []
        for f in flags:  # the words of the streak, for reading the flag at a glance
            f["text"] = "".join(by_t[t].get("text", "") for t in range(f["start"], f["end"] + 1) if t in by_t)[:120]
        row = {"monitor": m.describe(), "tokens": eligible, "recorded": recorded, "past": past, "flags": flags}
        if control and control != m.direction and not m.neuron:  # the random direction is no baseline for a neuron
            c_recorded, c_past = _past(m, tape, control)
            row["control"] = {"direction": control, "recorded": c_recorded, "past": c_past,
                              "flags": len(scan(m, tape, control)) if c_recorded else 0}
        out.append(row)
    return out


def scan_runs(recorder, limit: int = 20) -> dict:
    """Flag counts per monitor across the most recent runs (and the control's count on each)."""
    limit = max(1, min(int(limit), 200))
    control = control_of(recorder.directions.describe())
    monitors = [m for m in recorder.monitors.monitors() if m.enabled]
    runs = []
    for meta in recorder.list_runs()[:limit]:
        run = recorder.get_run(meta["id"])
        if run is None:
            continue
        counts, controls = {}, {}
        for m in monitors:
            counts[m.id] = len(scan(m, run.tape))
            if control and control != m.direction and not m.neuron:
                controls[m.id] = len(scan(m, run.tape, control))
        runs.append({"run": run.id, "label": meta.get("label"), "created": meta.get("created"),
                     "tokens": meta.get("tokens", 0), "flags": counts, "control": controls,
                     "flagged": sum(counts.values())})
    return {"monitors": [m.describe() for m in monitors], "control": control, "runs": runs}


class MonitorBank:
    """The saved monitors of one model, and the live check of each run being recorded."""

    def __init__(self, path: Path):
        self.path = path
        self.items: list[Monitor] = []
        self.on_change = None  # called after a monitor is added, changed or removed (the recorder's neuron watch)
        try:
            raw = json.loads(path.read_text()) if path.exists() else []
            self.items = [Monitor(**d).check() for d in raw]
        except (OSError, ValueError, TypeError):
            self.items = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(m) for m in self.items], indent=2))

    def monitors(self) -> list[Monitor]:
        return list(self.items)

    def list(self) -> list[dict]:
        return [m.describe() for m in self.items]

    def get(self, mid: str) -> Monitor:
        m = next((x for x in self.items if x.id == mid), None)
        if m is None:
            raise KeyError(f"no monitor {mid}")
        return m

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    @staticmethod
    def _neuron(text: str, check) -> str:
        """A neuron named like L5 N120 as label() writes it, if the model has it (`check(layer, n)`)."""
        try:
            layer, n = parse(text)
        except ValueError:
            raise ValueError(f"unknown direction {text!r}: save it to the direction bank first "
                             "(or name a neuron like L5 N120)") from None
        if check is not None:
            try:
                check(layer, n)
            except KeyError as exc:
                raise ValueError(str(exc).strip("'\"")) from exc
        return label(layer, n)

    def add(self, body: dict, directions: set[str], check_neuron=None) -> Monitor:
        """A new monitor, or the one with the same rule. The direction is one of `directions`, or a neuron
        that `check_neuron(layer, n)` accepts (it raises KeyError for one the model doesn't have)."""
        return self.add_many([body], directions, check_neuron)[0]

    def add_many(self, bodies: list[dict], directions: set[str], check_neuron=None) -> list[Monitor]:
        """Several monitors, all or none (a knob's neurons are one rule each)."""
        before, out = list(self.items), []
        try:
            out = [self._add(body, directions, check_neuron) for body in bodies]
        except Exception:
            self.items = before
            raise
        self._save()
        self._changed()
        return out

    def _add(self, body: dict, directions: set[str], check_neuron) -> Monitor:
        direction = str(body.get("direction") or "")
        if direction not in directions:
            direction = self._neuron(direction, check_neuron)
        if len(self.items) >= MAX_MONITORS:
            raise ValueError(f"at most {MAX_MONITORS} monitors")
        keys = ("threshold", "sign", "min_tokens", "phase", "label", "enabled")
        m = Monitor(direction, **{k: body[k] for k in keys if k in body}).check()
        same = next((x for x in self.items if (x.direction, x.threshold, x.sign, x.min_tokens, x.phase)
                     == (m.direction, m.threshold, m.sign, m.min_tokens, m.phase)), None)
        if same is not None:  # the same rule twice is one monitor
            return same
        self.items.append(m)
        return m

    def update(self, mid: str, body: dict) -> Monitor:
        m = self.get(mid)
        before = asdict(m)
        for k in ("threshold", "sign", "min_tokens", "phase", "label", "enabled"):
            if k in body:
                setattr(m, k, body[k])
        try:
            m.check()
        except (TypeError, ValueError):
            for k, v in before.items():
                setattr(m, k, v)
            raise
        self._save()
        self._changed()
        return m

    def remove(self, mid: str) -> list[dict]:
        self.get(mid)
        self.items = [m for m in self.items if m.id != mid]
        self._save()
        self._changed()
        return self.list()

    def feed(self, state: dict, rec: dict) -> list[tuple[Monitor, dict]]:
        """The live check: follow every enabled monitor along a run (state is the run's own) and
        return the flags raised at this token."""
        raised = []
        for m in self.monitors():
            if not m.enabled:
                state.pop(m.id, None)
                continue
            s = state.get(m.id)
            if s is None or s.m is not m:
                s = state[m.id] = Streak(m, m.direction)
            flag, _ = s.feed(rec)
            if flag is not None:
                raised.append((m, flag))
        return raised
