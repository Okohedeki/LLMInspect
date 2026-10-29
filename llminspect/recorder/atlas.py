"""The neuron atlas: every MLP neuron of the model at once, one value each, so neurons can be found by
looking rather than by name.

  layout     each dense layer's width and, when the weight anatomy is built, which of its neuron clusters
             (the "c" nodes of the Trace graph) each neuron belongs to, so the atlas groups a layer's
             neurons by cluster and a node can be opened as the neurons inside it
  maps       one value per neuron, quantized to a byte for the wire:
               write     ‖W_down[:, n]‖, how much one unit of activation writes (weights only)
               spread    the activation's spread: the neuron index's when it is built, else the
                         calibration texts' (current weights)
               mean      the activation's mean, from the same source (signed)
               changed   how much the neuron's weights differ from the original weights while edits or a
                         checkpoint are applied: ‖Δ‖/‖W‖ over its down_proj column and gate/up rows
  fires      a text of yours read once: each neuron's largest activation on it, as a z-score against
             its usual mean and spread (the same source as spread), with the token where it peaked
  links      the direct weight path between neurons, without running the model: how strongly a
             neuron's write is read by every later neuron's input (W_in · diag(norm gain) · w), and which
             earlier neurons' writes its own input reads most. Cosines, so wide and narrow layers
             compare; activations and attention are left out, as in neurons.py's "fed by".

Values reach the UI as base64 bytes per layer: 0..255 over the map's range (a log scale for
positive maps, symmetric for signed ones), with the range in the response so the legend can say it.
"""

from __future__ import annotations

import base64
import math

import numpy as np
import torch

from .neurons import TOP, label
from .tracing import Busy
from .workbench import _reference_ids

MAPS = ("write", "spread", "mean", "changed")
LINKS = 24        # links listed per direction
FIRES_TOP = 40    # neurons listed by "fires"
MAX_TEXT = 4000   # characters of text "fires" reads


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode("ascii")


def quantize(values: dict[int, torch.Tensor], signed: bool = False, log: bool = True) -> dict:
    """Per layer, the values as bytes over a shared range (robust: the 0.5 and 99.5 percentiles)."""
    flat = torch.cat([v.float().flatten() for v in values.values()]) if values else torch.zeros(1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        flat = torch.zeros(1)
    sample = flat if flat.numel() <= 2_000_000 else flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]]
    if signed:
        hi = float(torch.quantile(sample.abs(), 0.995)) or 1.0
        lo, scale = -hi, "linear"
        enc = lambda v: ((v.clamp(-hi, hi) / hi) * 127.5 + 127.5)
    else:
        pos = sample[sample > 0]
        if log and pos.numel():
            lo, hi = float(torch.quantile(pos, 0.005)), float(torch.quantile(pos, 0.995))
            hi = max(hi, lo * 1.0001)
            scale = "log"
            enc = lambda v: (torch.log(v.clamp(lo, hi) / lo) / math.log(hi / lo)) * 255
        else:
            lo, hi = 0.0, float(torch.quantile(sample, 0.995)) or 1.0
            scale = "linear"
            enc = lambda v: (v.clamp(lo, hi) / hi) * 255
    layers = {str(i): _b64(enc(v.float()).round().clamp(0, 255).to(torch.uint8).cpu().numpy()) for i, v in values.items()}
    return {"lo": lo, "hi": hi, "scale": scale, "signed": signed, "layers": layers}


class Atlas:
    def __init__(self, recorder):
        self.rec, self.s = recorder, recorder.s

    def _dense(self) -> list[int]:
        return [i for i in range(self.s.n_layers) if self.rec.neurons._try_mlp(i)]

    def layout(self) -> dict:
        nm, an = self.rec.neurons, getattr(self.rec, "anatomy", None)
        layers = []
        for i in self._dense():
            row = {"layer": i, "width": nm.mlp(i).width, "clusters": None, "offset": None}
            if an is not None and an.layers[i].assign is not None:
                lm = an.layers[i]
                assign = lm.assign.to("cpu", torch.int64)
                row["clusters"] = _b64(assign.clamp(0, 255).numpy()) if int(assign.max()) < 256 else None
                row["n_clusters"] = int(assign.max()) + 1
                row["offset"] = lm.offset + lm.n_heads  # unit id of cluster 0 in the Trace graph
            layers.append(row)
        return {"layers": layers, "n_layers": self.s.n_layers, "maps": list(MAPS),
                "edited": bool(self.rec.edits.applied), "index": self._stats_source()}

    def _stats_source(self) -> str:
        index = getattr(self.rec, "neuron_index", None)
        if index is not None and index.ready() and all(f"std.{i}" in index.data for i in self._dense()):
            return "index"
        return "calibration"

    def _stats(self) -> tuple[dict, dict, str]:
        source = self._stats_source()
        if source == "index":
            data = self.rec.neuron_index.data
            return ({i: torch.as_tensor(data[f"mean.{i}"]).float() for i in self._dense()},
                    {i: torch.as_tensor(data[f"std.{i}"]).float() for i in self._dense()}, source)
        cal = self.rec.neurons.calibration()
        return ({i: cal["mean"][i].float().cpu() for i in cal["mean"]}, {i: cal["std"][i].float().cpu() for i in cal["std"]}, source)

    @torch.no_grad()
    def map(self, name: str) -> dict:
        if name not in MAPS:
            raise ValueError(f"map must be one of {MAPS}")
        nm = self.rec.neurons
        note, available = "", True
        if name == "write":
            values = {i: nm.mlp(i).down.weight.detach().float().norm(dim=0).cpu() for i in self._dense()}
            out = quantize(values)
            note = "‖W_down[:, n]‖: how much the neuron writes per unit of activation (weights only)"
        elif name in ("spread", "mean"):
            mean, std, source = self._stats()
            out = quantize(std) if name == "spread" else quantize(mean, signed=True)
            note = (f"the activation's {'spread' if name == 'spread' else 'mean'} over "
                    + ("the neuron index's corpus (original weights)" if source == "index" else "the calibration texts (current weights)"))
        else:
            values, available = self._changed()
            out = quantize(values)
            note = ("‖Δ‖/‖W‖ of each neuron's down_proj column and gate/up rows, applied weights against the originals"
                    if available else "no edit or checkpoint is applied, so nothing has changed")
        return out | {"map": name, "note": note, "available": available}

    def _changed(self) -> tuple[dict, bool]:
        stack, nm = self.rec.edits, self.rec.neurons
        names = stack._names
        values, any_changed = {}, False
        for i in self._dense():
            m = nm.mlp(i)
            d2 = w2 = None
            for lin, dim in ((m.down, 0), (m.up, 1), (m.gate, 1)):
                if lin is None:
                    continue
                cur = lin.weight.detach().float()
                orig = stack.originals.get(names.get(id(lin.weight)))
                base = cur if orig is None else orig.to(cur.device, torch.float32)
                dd = torch.linalg.vector_norm(cur - base, dim=dim) ** 2
                ww = torch.linalg.vector_norm(base, dim=dim) ** 2
                d2 = dd if d2 is None else d2 + dd
                w2 = ww if w2 is None else w2 + ww
            rel = (d2.sqrt() / w2.sqrt().clamp_min(1e-12)).cpu()
            any_changed |= bool((rel > 0).any())
            values[i] = rel
        return values, any_changed and bool(stack.applied)

    @torch.no_grad()
    def fires(self, text: str) -> dict:
        text = str(text or "")[:MAX_TEXT]
        if not text.strip():
            raise ValueError("give some text to read")
        s, nm, rec = self.s, self.rec.neurons, self.rec
        mean, std, source = self._stats()
        ids = _reference_ids(s, text)
        peaks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        def hook(i):
            mu, sd = mean[i].to(s.device), std[i].to(s.device).clamp_min(1e-4)

            def fn(module, args):
                z = (args[0][0, 1:].float() - mu) / sd
                v, at = z.max(0)
                peaks[i] = (v.cpu(), at.cpu() + 1)
            return fn

        if not rec.lock.acquire(timeout=1.0):
            raise Busy("the model is generating; try again when the request finishes")
        try:
            handles = [nm.mlp(i).down.register_forward_pre_hook(hook(i)) for i in self._dense()]
            try:
                with rec.tap.paused():
                    s.model(input_ids=torch.tensor([ids], device=s.device), use_cache=False)
            finally:
                for h in handles:
                    h.remove()
        finally:
            rec.lock.release()
        values = {i: v.clamp_min(0) for i, (v, _) in peaks.items()}
        out = quantize(values, log=False)
        rows = []
        for i, (v, at) in peaks.items():
            k = min(FIRES_TOP, len(v))
            top_v, top_i = v.topk(k)
            rows += [(float(x), i, int(n), int(at[n])) for x, n in zip(top_v.tolist(), top_i.tolist())]
        rows.sort(key=lambda r: -r[0])
        tok = s.token_str
        top = [{"neuron": label(i, n), "layer": i, "index": n, "z": round(z, 2), "pos": p, "token": tok(ids[p]),
                "before": "".join(tok(t) for t in ids[max(1, p - 8):p])} for z, i, n, p in rows[:FIRES_TOP]]
        return out | {"map": "fires", "tokens": len(ids) - 1, "source": source, "top": top,
                      "note": "each neuron's largest activation on your text, as a z-score against its usual mean and spread "
                              f"({'the neuron index' if source == 'index' else 'the calibration texts'})"}

    @torch.no_grad()
    def links(self, layer: int, n: int, k: int = LINKS) -> dict:
        """Direct weight-path links of one neuron: later neurons that read its write, earlier ones whose write it reads."""
        nm = self.rec.neurons
        mlp = nm.check(layer, n)
        w = mlp.down.weight[:, n].detach().float()
        w = w / w.norm().clamp_min(1e-12)
        out_rows, in_rows = [], []
        for j in self._dense():
            m = nm.mlp(j)
            if j > layer:
                gain = nm._gain(m, j)
                for lin, kind in ((m.up, "up"), (m.gate, "gate")):
                    if lin is None:
                        continue
                    W = lin.weight.detach().float() * gain[None]
                    c = (W @ w) / W.norm(dim=1).clamp_min(1e-12)
                    v, idx = c.abs().topk(min(TOP * 2, len(c)))
                    out_rows += [(float(c[i]), j, int(i), kind) for i in idx.tolist()]
            elif j < layer:
                gain = nm._gain(mlp, layer)
                D = m.down.weight.detach().float()
                D = D / D.norm(dim=0, keepdim=True).clamp_min(1e-12)
                for lin, kind in ((mlp.up, "up"), (mlp.gate, "gate")):
                    if lin is None:
                        continue
                    r = lin.weight[n].detach().float() * gain
                    r = r / r.norm().clamp_min(1e-12)
                    c = r @ D
                    v, idx = c.abs().topk(min(TOP * 2, len(c)))
                    in_rows += [(float(c[i]), j, int(i), kind) for i in idx.tolist()]

        def best(rows):
            seen, out = set(), []
            for c, j, i, kind in sorted(rows, key=lambda r: -abs(r[0])):
                if (j, i) in seen:
                    continue
                seen.add((j, i))
                out.append({"neuron": label(j, i), "layer": j, "index": i, "cos": round(c, 4), "via": kind})
                if len(out) == k:
                    break
            return out

        return {"neuron": label(layer, n), "reads_it": best(out_rows), "it_reads": best(in_rows),
                "measure": "cosine of the direct weight path: a neuron's write against another's input weights "
                           "(through the norm's gain); the model is not run"}
