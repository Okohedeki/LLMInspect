"""Cap: keep one direction of the residual stream within the range it has on ordinary text.

A persona or behaviour that drifts during a long conversation often shows up as a direction whose projection
leaves its usual range. Capping clamps that projection, token by token, into bounds measured on ordinary text
and leaves everything orthogonal to it alone:

    p = h . u          (u the unit vector)
    h <- h + (clamp(p, low, high) - p) u

Tokens already in range are returned bit-identical. `layer` is a block number: the clamp acts on that block's
output (the residual after layer + 1 blocks).

calibrate  the projection's per-token range on ordinary text (the tool's neutral reference text by default) and
           bounds at the given quantiles. The first token is left out: it often carries an attention-sink
           activation far outside the range of every other token; it is reported apart.
test       greedy answers and per-token projections with a steering push along the vector (simulated drift),
           with and without the clamp, plus the clamp alone as a control for what capping costs when nothing
           drifts.

The bounds hold the projection in range; whether that keeps the behaviour the direction stands for is a
separate question the answers have to show.
"""

from __future__ import annotations

import contextlib

import torch

from . import blocks, device, encode, generate, hooked, residuals, unit
from ..recorder.scan import NEUTRAL_PROMPTS
from ..recorder.workbench import REFERENCE_TEXTS

QUANTILES = (0.01, 0.99)  # default bounds: these quantiles of the per-token projection on ordinary text
TOKENS = 32               # greedy answer length in test
BINS = 24                 # histogram bins in calibrate


def clamp(hidden: torch.Tensor, unit_vector: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """hidden [..., d] with its component along unit_vector clamped into [low, high]; in-range rows unchanged."""
    u = unit_vector.to(hidden.device, torch.float32)
    p = hidden.float() @ u
    c = p.clamp(low, high)
    moved = (hidden.float() + (c - p)[..., None] * u).to(hidden.dtype)
    return torch.where((c != p)[..., None], moved, hidden)


def _clamp_hook(u: torch.Tensor, low: float, high: float):
    """A layer hook that clamps every token except a prompt's first (the attention sink, left out of calibration)."""
    def fn(h):
        if h.shape[1] > 1:
            return torch.cat([h[:, :1], clamp(h[:, 1:], u, low, high)], 1)
        return clamp(h, u, low, high)
    return fn


def _vector(model, body: dict) -> torch.Tensor:
    v = body.get("vector")
    if not isinstance(v, list) or not v:
        raise ValueError("vector must be a list of numbers")
    v = torch.tensor(v, dtype=torch.float32)
    d = model.get_input_embeddings().weight.shape[1]
    if v.shape != (d,) or not torch.isfinite(v).all() or float(v.norm()) == 0:
        raise ValueError(f"vector must be {d} finite numbers, not all zero")
    return unit(v).to(device(model))


def _layer(model, body: dict) -> int:
    n = len(blocks(model))
    layer = body.get("layer")
    if not isinstance(layer, int) or not 0 <= layer < n:
        raise ValueError(f"layer must be a block number in 0..{n - 1}")
    return layer


def _texts(body: dict, key: str, default: list[str]) -> list[str]:
    items = body.get(key) or default
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError(f"{key} must be a list of texts")
    items = [x for x in items if x.strip()]
    if not items:
        raise ValueError(f"{key} needs at least one text")
    return items


@torch.no_grad()
def calibrate(model, tok, body: dict) -> dict:
    """Per-token projection range at a layer on ordinary text, and bounds at the given quantiles."""
    u, layer = _vector(model, body), _layer(model, body)
    texts = _texts(body, "texts", REFERENCE_TEXTS)
    q = body.get("quantiles") or list(QUANTILES)
    if len(q) != 2 or not 0 <= q[0] < q[1] <= 1:
        raise ValueError("quantiles must be two numbers 0 <= low < high <= 1")
    rows, first = [], []
    for t in texts:
        p = residuals(model, encode(tok, t))[layer + 1] @ u
        first.append(round(float(p[0]), 4))
        rows.append({"text": t[:80], "tokens": len(p) - 1, "min": round(float(p[1:].min()), 4) if len(p) > 1 else None,
                     "max": round(float(p[1:].max()), 4) if len(p) > 1 else None})
        rows[-1]["_p"] = p[1:]
    allp = torch.cat([r.pop("_p") for r in rows]).cpu()
    if len(allp) < 2:
        raise ValueError("the texts are too short to calibrate on")
    low, high = (float(x) for x in torch.quantile(allp, torch.tensor(q, dtype=torch.float32)))
    counts, edges = torch.histogram(allp, bins=BINS)
    return {"layer": layer, "quantiles": q, "low": round(low, 4), "high": round(high, 4),
            "mean": round(float(allp.mean()), 4), "std": round(float(allp.std()), 4),
            "min": round(float(allp.min()), 4), "max": round(float(allp.max()), 4), "n_tokens": len(allp),
            "histogram": {"counts": counts.int().tolist(), "edges": [round(float(e), 4) for e in edges]},
            "texts": rows, "first_token": first,
            "note": "Bounds are quantiles of the projection on these texts, first token left out. Ordinary text "
                    "beyond these texts can fall outside them; a clamp then moves it too."}


@torch.no_grad()
def _projections(model, tok, prompt: str, answer: str, layer: int, u: torch.Tensor) -> list[float]:
    ids = torch.cat([encode(tok, prompt, chat=True), torch.tensor(tok(answer, add_special_tokens=False)["input_ids"],
                                                                  dtype=torch.long)])
    seen = []
    with hooked(model, layer, lambda h: (seen.append((h[0].float() @ u).cpu()), h)[1]):
        model(ids[None].to(device(model)))
    return [round(float(x), 4) for x in seen[0]]


@contextlib.contextmanager
def _condition(model, layer: int, u: torch.Tensor, push: float, push_layer: int, cap: tuple | None):
    with contextlib.ExitStack() as stack:
        if push:
            stack.enter_context(hooked(model, push_layer, lambda h: (h.float() + push * u).to(h.dtype)))
        if cap:
            stack.enter_context(hooked(model, layer, _clamp_hook(u, *cap)))
        yield model


def test(model, tok, body: dict) -> dict:
    """Answers and projections with a steering push, with and without the clamp, beside unpushed controls."""
    u, layer = _vector(model, body), _layer(model, body)
    try:
        low, high = float(body["low"]), float(body["high"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("low and high are required numbers (run calibrate first)") from None
    if not low < high:
        raise ValueError("low must be below high")
    prompts = _texts(body, "prompts", NEUTRAL_PROMPTS[:3])
    push = float(body.get("push", 2 * (high - low)))
    push_layer = body.get("push_layer", layer)
    if not isinstance(push_layer, int) or not 0 <= push_layer <= layer:
        raise ValueError(f"push_layer must be a block number in 0..{layer}")
    tokens = int(body.get("tokens", TOKENS))
    conditions = {"plain": (0.0, None), "clamped": (0.0, (low, high)), "pushed": (push, None),
                  "pushed_clamped": (push, (low, high))}
    out = {name: [] for name in conditions}
    for p in prompts:
        for name, (s, cap) in conditions.items():
            with _condition(model, layer, u, s, push_layer, cap):
                answer = generate(model, tok, p, tokens)
                proj = _projections(model, tok, p, answer, layer, u)
            body_proj = proj[1:]
            inside = sum(low - 1e-4 <= x <= high + 1e-4 for x in body_proj)
            out[name].append({"prompt": p, "answer": answer, "projections": proj,
                              "in_range": round(inside / max(len(body_proj), 1), 4),
                              "max_excursion": round(max([max(x - high, low - x, 0.0) for x in body_proj] or [0.0]), 4)})
    summary = {name: {"in_range": round(sum(r["in_range"] for r in rows) / len(rows), 4),
                      "max_excursion": max(r["max_excursion"] for r in rows),
                      "same_answer_as_plain": sum(r["answer"] == q["answer"] for r, q in zip(rows, out["plain"]))}
               for name, rows in out.items()}
    return {"layer": layer, "push_layer": push_layer, "push": push, "low": low, "high": high, "prompts": len(prompts),
            "conditions": out, "summary": summary,
            "note": "Projections are read at the clamped layer after any clamp, first token (attention sink) left "
                    "unclamped. 'clamped' without a push is the control for what capping changes on its own."}


test.__test__ = False  # not a pytest test when imported by one

STEPS = {"calibrate": calibrate, "test": test}
