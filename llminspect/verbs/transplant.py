"""Transplant: carry a fine-tune, or a direction, from one model into another (a newer release, say).

A fine-tune is its weights minus its base's. Added to another model of the same layout, that difference often
carries the behavior over, as far as the two models still share how they compute (the closer the family, the
better). Tensors are matched by their place in the model (layer and name, without the wrapper prefix) and read one
at a time from safetensors files, so neither checkpoint is ever loaded whole.

  plan       which tensors the fine-tune changed and by how much, whether each fits the loaded model, and how the
             loaded model's weights correlate with the base's: near 1 means the same weights (a fine-tune of it,
             or the base itself), near 0 an unrelated model of the same shape, where a carried delta is unlikely
             to mean the same thing.
  carry      the delta scale * (tuned - base) on the loaded model's parameter names, optionally compressed to a
             low rank per matrix (SVD); tensors that don't fit are skipped and reported.
  direction  a residual direction from another model, mapped into this one. The two embedding matrices are
             aligned by orthogonal Procrustes on the tokens both vocabularies share (each centered and scaled to
             one size), and the vector is sent through that rotation. Quality is the cosine between mapped and
             actual embeddings of held-out shared tokens, next to the same fit with the token pairs shuffled as a
             control. It assumes the layer's residual basis follows the embedding basis, which holds best for
             related models.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .. import verbs

PREFIXES = ("model.language_model.", "language_model.model.", "language_model.", "model.", "transformer.",
            "gpt_neox.", "backbone.")
PLACE = re.compile(r"((?:layers?|h|blocks)\.\d+\..*|(?:embed_tokens|wte|tok_embeddings|word_embeddings|embed_in)\.weight)$")
EMBED = re.compile(r"(?:embed_tokens|wte|tok_embeddings|word_embeddings|embed_in)\.weight$")
SAMPLE = 4096  # values per tensor read for the family correlation
SAME_FAMILY = 0.5  # median correlation above which the loaded model is taken to share the base's weights
CHANGED = 0.0  # a tensor counts as changed when any value differs by more than this
MAX_TOKENS = 50_000  # shared tokens used to align embeddings at most
HELD_OUT = 0.1  # fraction of shared tokens kept out of the fit to measure it
LISTED = 50  # tensors listed per category in a plan


def place(name: str) -> str:
    """A tensor's place in the model without the wrapper prefix, so checkpoints saved apart line up."""
    m = PLACE.search(name)
    if m:
        return m.group(1)
    for p in PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def index(path) -> dict[str, Path]:
    """{tensor name: safetensors file} for a folder of safetensors files (or one file)."""
    if not path:
        raise ValueError("a folder path is required")
    path = Path(path).expanduser()
    files = [path] if path.is_file() else sorted(path.glob("*.safetensors")) if path.is_dir() else []
    if not files:
        raise ValueError(f"no safetensors files at {path}")
    out = {}
    for f in files:
        with safe_open(str(f), "pt", device="cpu") as fh:
            for k in fh.keys():
                out[k] = f
    return out


class Reader:
    """One tensor at a time from a folder's safetensors files, by place."""

    def __init__(self, path):
        self.files = {place(k): (k, f) for k, f in index(path).items()}
        self.handles = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self.handles.values():
            h.__exit__(None, None, None)
        self.handles.clear()

    def _handle(self, f: Path):
        if f not in self.handles:
            h = safe_open(str(f), "pt", device="cpu")
            self.handles[f] = h.__enter__()
        return self.handles[f]

    def shape(self, key: str) -> list[int]:
        name, f = self.files[key]
        return list(self._handle(f).get_slice(name).get_shape())

    def get(self, key: str) -> torch.Tensor:
        name, f = self.files[key]
        return self._handle(f).get_tensor(name)


def _params(model) -> dict[str, tuple[str, torch.nn.Parameter]]:
    return {place(n): (n, p) for n, p in model.named_parameters()}


def _sample(t: torch.Tensor) -> torch.Tensor:
    flat = t.detach().reshape(-1)
    step = max(1, flat.numel() // SAMPLE)
    return flat[::step][:SAMPLE].float().cpu()


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    return (a @ b).item() / den if den else (1.0 if torch.equal(a, b) else 0.0)


def _paths(body: dict) -> tuple[str, str]:
    base, tuned = body.get("base"), body.get("tuned")
    if not base or not tuned:
        raise ValueError("base and tuned (folders of safetensors files) are required")
    return base, tuned


@torch.no_grad()
def plan(model, tok, body: dict) -> dict:
    """What the fine-tune changed, what fits the loaded model, and whether the loaded model is of the base's family."""
    base_path, tuned_path = _paths(body)
    params = _params(model)
    changed, unchanged, mismatched, missing, only_tuned, corrs = [], 0, [], [], [], []
    with Reader(base_path) as base, Reader(tuned_path) as tuned:
        for key in sorted(tuned.files):
            if key not in base.files:
                only_tuned.append(key)
                continue
            b, t = base.get(key).float(), tuned.get(key).float()
            if key in params and list(params[key][1].shape) == list(b.shape):
                corrs.append(_corr(_sample(params[key][1]), _sample(b)))
            if b.shape != t.shape:
                mismatched.append({"tensor": key, "base": list(b.shape), "tuned": list(t.shape)})
                continue
            diff = t - b
            if diff.abs().max().item() <= CHANGED:
                unchanged += 1
                continue
            row = {"tensor": key, "shape": list(b.shape), "delta_norm": diff.norm().item(),
                   "relative": diff.norm().item() / max(b.norm().item(), 1e-12)}
            if key not in params:
                missing.append(key)
                row["fits"] = "not in loaded model"
            elif list(params[key][1].shape) != list(b.shape):
                row["fits"] = f"shape {list(params[key][1].shape)} in loaded model"
                mismatched.append({"tensor": key, "base": list(b.shape), "loaded": list(params[key][1].shape)})
            else:
                row["fits"] = "yes"
            changed.append(row)
    changed.sort(key=lambda r: -r["relative"])
    median = sorted(corrs)[len(corrs) // 2] if corrs else None
    if median is None:
        family = "no tensor of the base matches the loaded model's shapes: another architecture"
    elif median >= 0.99:
        family = "the loaded model's weights are the base's (or a light fine-tune of it)"
    elif median >= SAME_FAMILY:
        family = "the loaded model shares the base's weights, moved by further training"
    else:
        family = ("the loaded model's weights are unrelated to the base's (another training run); a carried delta "
                  "lands on weights it wasn't learned on and may not carry the behavior")
    fits = sum(r["fits"] == "yes" for r in changed)
    return {
        "changed": len(changed), "unchanged": unchanged, "fits": fits,
        "tensors": changed[:LISTED],
        "mismatched": mismatched[:LISTED], "missing_in_model": missing[:LISTED], "only_in_tuned": only_tuned[:LISTED],
        "family": {"median_correlation": median, "tensors_compared": len(corrs), "reading": family},
    }


def _low_rank(d: torch.Tensor, rank: int) -> tuple[torch.Tensor, float]:
    """d's best rank-r approximation and the fraction of its energy kept."""
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    r = min(rank, s.numel())
    energy = (s[:r].pow(2).sum() / s.pow(2).sum().clamp_min(1e-30)).item()
    return (u[:, :r] * s[:r]) @ vh[:r], energy


@torch.no_grad()
def carry(model, tok, body: dict) -> dict:
    """The delta scale * (tuned - base) on the loaded model's parameters."""
    base_path, tuned_path = _paths(body)
    scale = float(body.get("scale", 1.0))
    rank = body.get("low_rank")
    if rank is not None and (not isinstance(rank, int) or rank < 1):
        raise ValueError("low_rank must be a positive integer")
    params = _params(model)
    delta, applied, skipped, energies = {}, [], [], []
    with Reader(base_path) as base, Reader(tuned_path) as tuned:
        for key in sorted(tuned.files):
            if key not in base.files:
                skipped.append({"tensor": key, "why": "not in base"})
                continue
            b, t = base.get(key), tuned.get(key)
            if b.shape != t.shape:
                skipped.append({"tensor": key, "why": f"base {list(b.shape)} vs tuned {list(t.shape)}"})
                continue
            d = t.float() - b.float()
            if d.abs().max().item() <= CHANGED:
                continue
            if key not in params:
                skipped.append({"tensor": key, "why": "not in loaded model"})
                continue
            name, p = params[key]
            if list(p.shape) != list(d.shape):
                skipped.append({"tensor": key, "why": f"shape {list(d.shape)} vs loaded {list(p.shape)}"})
                continue
            row = {"tensor": key, "relative": d.norm().item() * abs(scale) / max(_norm(p), 1e-12)}
            if rank is not None and d.ndim == 2:
                d, row["energy_kept"] = _low_rank(d.to(p.device), rank)
                energies.append(row["energy_kept"])
            delta[name] = (scale * d).to(p.dtype).cpu()
            applied.append(row)
    if not delta:
        raise ValueError("nothing to carry: no changed tensor fits the loaded model")
    return {
        "delta": delta, "applied": len(applied), "tensors": applied[:LISTED], "skipped": skipped[:LISTED],
        "skipped_count": len(skipped), "scale": scale, "low_rank": rank,
        "energy_kept_mean": sum(energies) / len(energies) if energies else None,
        "note": "Measure the result (checks, drift) before keeping it: a delta carried onto other weights is a guess.",
    }


def _norm(p: torch.Tensor) -> float:
    """A parameter's Frobenius norm, read in fp32 rows."""
    flat, total = p.detach().reshape(-1), 0.0
    for i in range(0, flat.numel(), 1 << 24):
        total += flat[i:i + (1 << 24)].float().pow(2).sum().item()
    return math.sqrt(total)


def _embedding_from(path) -> torch.Tensor:
    with Reader(path) as r:
        keys = [k for k in r.files if EMBED.search(k)]
        if not keys:
            raise ValueError(f"no embedding matrix found in {path}")
        return r.get(keys[0]).float()


def _tokenizer(path):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(str(Path(path).expanduser()), local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any failure to read the tokenizer means the same to the user
        raise ValueError(f"can't read a tokenizer from {path}: {e}") from e


def procrustes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The (semi-)orthogonal R minimizing |x R - y| (rows paired)."""
    u, _, vh = torch.linalg.svd(x.T @ y, full_matrices=False)
    return u @ vh


def _normalize(e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    mean = e.mean(0)
    c = e - mean
    size = (c.norm() / math.sqrt(c.shape[0])).item() or 1.0
    return c / size, mean, size


def _quality(x, y, r) -> float:
    return torch.nn.functional.cosine_similarity(x @ r, y, dim=-1).mean().item()


@torch.no_grad()
def direction(model, tok, body: dict) -> dict:
    """A residual direction from another model, mapped into this one's basis through aligned embeddings."""
    vec, layer, src = body.get("vector"), body.get("layer"), body.get("from_model")
    if not isinstance(vec, list) or not vec or not src:
        raise ValueError("vector (list of floats) and from_model (a folder) are required")
    n_layers = len(verbs.blocks(model))
    if not isinstance(layer, int) or not 0 <= layer <= n_layers:
        raise ValueError(f"layer must be an integer in [0, {n_layers}]")
    v = torch.tensor(vec, dtype=torch.float32)
    src_embed = _embedding_from(src)
    if v.numel() != src_embed.shape[1]:
        raise ValueError(f"vector has {v.numel()} values; the source model's width is {src_embed.shape[1]}")
    src_vocab = _tokenizer(src).get_vocab()
    dst_vocab = tok.get_vocab()
    dst_embed = model.get_input_embeddings().weight
    shared = sorted(t for t in src_vocab if t in dst_vocab
                    and src_vocab[t] < src_embed.shape[0] and dst_vocab[t] < dst_embed.shape[0])
    if len(shared) < 2 * min(src_embed.shape[1], dst_embed.shape[1]) // 3:
        raise ValueError(f"only {len(shared)} shared tokens: too few to align the embeddings")
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(shared), generator=g)[:MAX_TOKENS]
    src_ids = torch.tensor([src_vocab[shared[i]] for i in order])
    dst_ids = torch.tensor([dst_vocab[shared[i]] for i in order])
    x, _, sx = _normalize(src_embed[src_ids])
    y, _, sy = _normalize(dst_embed.detach()[dst_ids.to(dst_embed.device)].float().cpu())
    held = max(1, int(len(order) * HELD_OUT))
    fit, test = slice(held, None), slice(None, held)
    r = procrustes(x[fit], y[fit])
    shuffled = procrustes(x[fit][torch.randperm(x[fit].shape[0], generator=g)], y[fit])
    mapped = (v @ r) * (sy / sx)
    quality = _quality(x[test], y[test], r)
    control = _quality(x[test], y[test], shuffled)
    if quality > 0.9:
        reading = "embedding spaces align closely; the mapped direction should mean nearly the same thing"
    elif quality > control + 0.2:
        reading = "embedding spaces align partly; treat the mapped direction as a starting point and measure it"
    else:
        reading = "embedding spaces don't align beyond the shuffled control; the mapped direction is unreliable"
    return {
        "vector": mapped.tolist(), "layer": layer, "norm_in": v.norm().item(), "norm_out": mapped.norm().item(),
        "alignment": {"shared_tokens": len(shared), "used": len(order), "held_out": held,
                      "cosine": quality, "control_shuffled": control, "fit_residual": (x[fit] @ r - y[fit]).norm().item()
                      / max(y[fit].norm().item(), 1e-12)},
        "reading": reading,
        "note": "The mapping is fitted on embeddings; residuals at deeper layers may drift from that basis.",
    }


STEPS = {"plan": plan, "carry": carry, "direction": direction}
