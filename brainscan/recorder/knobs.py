"""Knobs: named, saved sets of neurons with doses, per model.

A knob is a set of neurons found once and reused, e.g. "tone" = L5 N120 +1, L9 N64 -0.5. Each entry is a
neuron (neurons.py) and the dose it is steered at (dose scaling, there). One knob serves four actions:

  steer    Recorder.steer_knob: one runtime "write" per neuron, every request from now, at dose × factor
  ablate   a "units" weight edit that zeroes the knob's neurons' write columns (added, not applied)
  monitor  one single-neuron monitor rule per neuron (monitors.py)
  sweep    Neurons.sweep with the knob's doses scaled by each factor

Saved as <data>/knobs/<model>.json:
  {"version": 1, "shape": [layers, MLP width], "knobs": [{name, description, neurons: [{neuron, dose}],
   created, updated}]}
Models are filed by name, so a file whose shape is not this model's belongs to another model: nothing is
loaded from it, `error` says so, and saving raises instead of writing over it.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

from .directions import NAME_RE
from .neurons import MAX_DOSE, label, parse

VERSION = 1
MAX_KNOBS = 128
MAX_NEURONS = 64
MAX_DESCRIPTION = 300


class Clash(RuntimeError):
    """The knobs file on disk is not this model's (or can't be read), so nothing may be written over it."""


class KnobBank:
    def __init__(self, path: Path, neurons):
        self.path, self.nm = Path(path), neurons
        self.items: list[dict] = []
        self.error: str | None = None
        self._lock = threading.Lock()
        widths = [self._width(i) for i in range(neurons.s.n_layers)]
        self.shape = [neurons.s.n_layers, max(widths, default=0)]
        self._load()

    def _width(self, layer: int) -> int:
        try:
            return self.nm.mlp(layer).width
        except ValueError:
            return 0

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if raw.get("version") != VERSION:
                raise ValueError(f"version {raw.get('version')!r}")
            if raw.get("shape") != self.shape:
                self.error = (f"Two models share the name {self.nm.s.slug!r}: {self.path} holds knobs for a model "
                              f"shaped {raw.get('shape')}, not {self.shape}. Set BRAINSCAN_MODEL_NAME to tell them apart.")
                return
            self.items = [self._entry(k["name"], [(*parse(e["neuron"]), e["dose"]) for e in k["neurons"]],
                                      k.get("description", ""), k["created"], k["updated"]) for k in raw["knobs"]]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            self.items, self.error = [], f"{self.path} could not be read ({type(exc).__name__}: {exc}); fix or remove it"

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"version": VERSION, "shape": self.shape, "knobs": self.items}, indent=2),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ---- reads -----------------------------------------------------------------------------

    def list(self) -> list[dict]:
        return [dict(k) for k in self.items]

    def names(self) -> list[str]:
        return [k["name"] for k in self.items]

    def get(self, name: str) -> dict:
        k = next((x for x in self.items if x["name"] == name), None)
        if k is None:
            raise KeyError(f"no knob {name!r}")
        return dict(k)

    def steers(self, name: str) -> list[tuple[int, int, float]]:
        """(layer, neuron, dose) for each of a knob's neurons."""
        return [(*parse(e["neuron"]), e["dose"]) for e in self.get(name)["neurons"]]

    def units(self, name: str) -> list[dict]:
        """The knob's neurons as the units of a weight edit: one entry per layer, named for the knob."""
        by_layer: dict[int, list[int]] = {}
        for layer, n, _ in self.steers(name):
            by_layer.setdefault(layer, []).append(n)
        return [{"layer": layer, "target": "neuron", "index": sorted(idx), "name": name}
                for layer, idx in sorted(by_layer.items())]

    # ---- writes ----------------------------------------------------------------------------

    def _entry(self, name: str, steers: list, description: str, created: float, updated: float) -> dict:
        """A validated knob. `steers` are (layer, neuron, dose) triples or {neuron, dose} dicts."""
        if not isinstance(name, str) or not NAME_RE.match(name):
            raise ValueError("name: letters, digits, '.', '_' or '-', up to 64 characters")
        description = str(description or "").strip()
        if len(description) > MAX_DESCRIPTION:
            raise ValueError(f"the description is {len(description)} characters; at most {MAX_DESCRIPTION}")
        if not isinstance(steers, (list, tuple)) or not 1 <= len(steers) <= MAX_NEURONS:
            raise ValueError(f"a knob has 1 to {MAX_NEURONS} neurons")
        seen, neurons = set(), []
        for s in steers:
            if isinstance(s, dict):
                layer, n = parse(s.get("neuron"))
                dose = s.get("dose", 1.0)
            elif isinstance(s, (list, tuple)) and len(s) == 3:
                layer, n, dose = s
            else:
                raise ValueError("each neuron is {neuron: \"L5 N120\", dose: 1}")
            try:
                self.nm.check(int(layer), int(n))
            except KeyError as exc:
                raise ValueError(str(exc).strip("'\"")) from exc
            layer, n = int(layer), int(n)
            try:
                if isinstance(dose, bool):
                    raise TypeError("dose")
                dose = float(dose)
            except (TypeError, ValueError):
                raise ValueError(f"{label(layer, n)}: dose must be a number") from None
            if not math.isfinite(dose) or abs(dose) > MAX_DOSE:
                raise ValueError(f"{label(layer, n)}: dose must be a number between -{MAX_DOSE:g} and {MAX_DOSE:g}")
            if dose == 0:
                raise ValueError(f"{label(layer, n)}: a dose of 0 does nothing; remove the neuron instead")
            if (layer, n) in seen:
                raise ValueError(f"{label(layer, n)} is in the knob twice")
            seen.add((layer, n))
            neurons.append({"neuron": label(layer, n), "dose": dose})
        return {"name": name, "description": description, "neurons": neurons, "created": created, "updated": updated}

    def save(self, name: str, neurons: list, description: str = "") -> dict:
        """Add a knob, or replace the one of that name (it keeps its `created`)."""
        if self.error:
            raise Clash(self.error)
        with self._lock:
            now = time.time()
            old = next((i for i, k in enumerate(self.items) if k["name"] == name), None)
            knob = self._entry(name, neurons, description, self.items[old]["created"] if old is not None else now, now)
            if old is None and len(self.items) >= MAX_KNOBS:
                raise ValueError(f"at most {MAX_KNOBS} knobs")
            before = list(self.items)
            if old is None:
                self.items.append(knob)
            else:
                self.items[old] = knob
            try:
                self._write()
            except OSError:
                self.items = before
                raise
            return dict(knob)

    def delete(self, name: str) -> None:
        if self.error:
            raise Clash(self.error)
        with self._lock:
            self.get(name)
            before = list(self.items)
            self.items = [k for k in self.items if k["name"] != name]
            try:
                self._write()
            except OSError:
                self.items = before
                raise
