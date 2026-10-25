"""Forward hooks on the decoder layers: capture residuals and apply interventions.

One hook per layer serves both tracks: it applies any active interventions to the
layer's output residual, then keeps a reference to the (possibly edited) residual so
the recorder can read it after the forward pass.
"""

from __future__ import annotations

import contextlib
import math
import uuid
from dataclasses import dataclass, field

import torch

LN2 = math.log(2)
MODES = ("add", "ablate", "clamp", "write")
PHASES = ("all", "prefill", "gen")


@dataclass
class Intervention:
    """Edit the residual stream along one unit direction at one layer.

    add:    h += strength * |h| * v   (strength is a fraction of the residual norm)
    ablate: h -= (h . v) v
    clamp:  h += (value - h . v) v
    write:  h += value * v            (a fixed write: a steered neuron, see recorder/neurons.py)

    `knob` names the saved knob (recorder/knobs.py) a steered neuron belongs to, if it was steered with one.
    """
    layer: int
    mode: str
    direction: str
    vector: torch.Tensor
    strength: float = 0.0
    value: float = 0.0
    phase: str = "all"
    run: str | None = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    knob: str | None = None

    def applies(self, phase: str, run_id: str | None) -> bool:
        return (self.enabled and (self.phase == "all" or self.phase == phase)
                and (self.run is None or self.run == run_id))

    def apply(self, h: torch.Tensor) -> torch.Tensor:
        v = self.vector.to(h.dtype)
        if self.mode == "add":
            return h + self.strength * h.norm(dim=-1, keepdim=True) * v
        if self.mode == "write":
            return h + self.value * v
        proj = (h * v).sum(-1, keepdim=True)
        if self.mode == "ablate":
            return h - proj * v
        return h + (self.value - proj) * v

    def describe(self) -> dict:
        return {"id": self.id, "layer": self.layer, "mode": self.mode, "direction": self.direction,
                "strength": self.strength, "value": self.value, "phase": self.phase,
                "run": self.run, "enabled": self.enabled, "knob": self.knob}


class LayerTap:
    def __init__(self, subject):
        self.subject = subject
        self.resid: list[torch.Tensor | None] = [None] * subject.n_layers
        self.interventions: list[Intervention] = []
        self.phase = "prefill"
        self.run_id: str | None = None
        self._paused = False
        self.handles = [layer.register_forward_hook(self._hook(i)) for i, layer in enumerate(subject.layers)]

    def _hook(self, i: int):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            edited = False
            if not self._paused:
                for iv in self.interventions:
                    if iv.layer == i and iv.applies(self.phase, self.run_id):
                        h = iv.apply(h)
                        edited = True
            self.resid[i] = h
            if edited:
                return (h, *output[1:]) if isinstance(output, tuple) else h
            return None
        return fn

    @contextlib.contextmanager
    def paused(self):
        """Run forward passes with interventions switched off (probe building). Nests."""
        before, self._paused = self._paused, True
        try:
            yield
        finally:
            self._paused = before

    def stack(self) -> torch.Tensor:
        """Residuals of the last forward pass for batch item 0, shape (layers, tokens, d_model)."""
        return torch.stack([r[0] for r in self.resid])


@torch.no_grad()
def lens_rows(subject, x: torch.Tensor, targets: torch.Tensor, chunk: int | None = None):
    """Logit lens on rows of residuals: (prob of target token, top-1 token id) per row."""
    chunk = chunk or subject.lens_chunk
    probs, tops = [], []
    for s in range(0, x.shape[0], chunk):
        logits = subject.lm_head(subject.norm(x[s:s + chunk])).float()
        logp = torch.log_softmax(logits, -1)
        tops.append(logits.argmax(-1))
        probs.append(logp.gather(-1, targets[s:s + chunk, None]).squeeze(-1).exp())
    return torch.cat(probs), torch.cat(tops)


@torch.no_grad()
def predict(subject, resid: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor,
            with_lens: bool = True, chunk: int | None = None, host: bool = True) -> dict:
    """Prediction stats for N slots whose next token is known.

    resid: (layers, N, d) residuals at the predicting positions
    logits: (N, vocab) the model's actual output logits at those positions
    targets: (N,) the token that actually came next
    Returns numpy arrays, or with host=False the same values as device tensors, so the caller
    can queue the whole readout without waiting on the GPU.
    """
    chunk = chunk or max(32, subject.lens_chunk // 2)
    out = {"p": [], "ent": [], "top_p": [], "top_ids": []}
    for s in range(0, logits.shape[0], chunk):
        logp = torch.log_softmax(logits[s:s + chunk].float(), -1)
        p = logp.exp()
        out["ent"].append(-(p * logp).sum(-1) / LN2)
        top_p, top_ids = p.topk(5, dim=-1)
        out["top_p"].append(top_p)
        out["top_ids"].append(top_ids)
        out["p"].append(p.gather(-1, targets[s:s + chunk, None]).squeeze(-1))
    result = {k: torch.cat(v) for k, v in out.items()}
    if with_lens:
        n_layers, n, d = resid.shape
        lp, lt = lens_rows(subject, resid.reshape(n_layers * n, d), targets.repeat(n_layers))
        result["lens_p"] = lp.view(n_layers, n).T
        result["lens_top"] = lt.view(n_layers, n).T
    return {k: v.cpu().numpy() for k, v in result.items()} if host else result
