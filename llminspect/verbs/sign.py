"""Sign: hide an ownership mark in the weights, and check for it later.

The mark is a secret key turned into a pattern. For every output matrix of every layer (the attention and MLP
writers), the key and the matrix's place in the model (its layer and name, not the wrapper prefix) seed a
generator that picks up to PER_MATRIX entries, spread evenly over the matrix, and a random sign for each. Marking
adds strength * rms(matrix) * sign to those entries: a nudge far below the weights' own spread, the same size
everywhere, with no structure a reader could see without the key.

Checking reads the same entries back and sums sign * weight / rms over all of them, divided by sqrt(count): for a
key that wasn't used this is about a standard normal draw (the weights don't know the key), while the marked key
gets strength * sqrt(count) on top. Random keys read the same way are the null, so the reported z is measured
against this model's own weights, not a formula. Fine-tuning moves weights in ways unrelated to the key, so it adds
spread but not bias: the mark fades only as fast as the weights are rewritten. Rounding to 16 bits is the other
risk, so the default strength sits above half a bf16 step for typical weights, and the change is computed as what
actually lands in the parameter's dtype.

  embed    the delta that marks the model, a record of what was marked (never the key), the expected and measured
           z, the z after simulated noise and bf16 rounding, and the change in outputs on neutral texts (KL).
  verify   the key's z against random keys, any other keys to compare, and a plain reading.
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets

import torch

from .. import verbs
from ..recorder.edits import WRITER_NAMES

PER_MATRIX = 65_536  # entries marked per matrix at most
TARGET_Z = 16.0  # z the default strength aims for on the untouched marked model
MIN_STRENGTH = 0.015  # floor on strength (in units of the matrix's rms), so the nudge survives bf16 rounding
MAX_STRENGTH = 0.5  # beyond this the "mark" is a visible change to the weights
NULL_KEYS = 100  # random keys read as the null distribution
PRESENT_Z = 8.0  # z beyond the null (and above every null key) to say the mark is present
TRACE_Z = 4.0  # z beyond the null to call it a weak trace
NOISE = 0.05  # Gaussian noise, as a fraction of each matrix's rms, simulated as a stand-in for fine-tuning
CHUNK = 1 << 24  # values per fp32 block when reading a whole matrix
NEUTRAL = [
    "The weather report for tomorrow says light rain in the morning and clear skies by the afternoon.",
    "To make a simple tomato sauce, warm olive oil, add garlic, then the tomatoes, and simmer slowly.",
    "The library opens at nine and closes at six, except on public holidays.",
    "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen.",
]

WRITER = re.compile(r"\.(%s)\.weight$" % "|".join(sorted(WRITER_NAMES)))
LAYER = re.compile(r"(?:layers?|h|blocks)\.\d+\..*$")
ATTN = re.compile(r"(?:attn|attention|mixer)\.")


def _key(body: dict, field: str = "key") -> str:
    key = body.get(field)
    if not isinstance(key, str) or not key:
        raise ValueError(f"{field} must be a non-empty string")
    return key


def _parts(body: dict) -> list[str]:
    parts = body.get("targets") or ["attn", "mlp"]
    if isinstance(parts, str):
        parts = [parts]
    bad = [p for p in parts if p not in ("attn", "mlp")]
    if bad:
        raise ValueError(f"targets must be 'attn' and/or 'mlp', not {bad}")
    return list(parts)


def targets(model, parts=("attn", "mlp")) -> list[tuple[str, torch.nn.Parameter]]:
    """(name, parameter) of the layers' output matrices of the chosen kinds."""
    out = []
    for name, p in model.named_parameters():
        if p.ndim != 2 or not WRITER.search(name) or not LAYER.search(name):
            continue
        if ("attn" if ATTN.search(LAYER.search(name).group(0)) else "mlp") in parts:
            out.append((name, p))
    if not out:
        raise ValueError("no layer output matrices found to mark")
    return out


def site(name: str) -> str:
    """A matrix's place in the model without the wrapper prefix, so the mark reads the same after re-wrapping."""
    m = LAYER.search(name)
    return m.group(0) if m else name


def pattern(key: str, name: str, numel: int, k: int = PER_MATRIX) -> tuple[torch.Tensor, torch.Tensor]:
    """(flat indices, ±1 signs) the key marks in one matrix: k entries, one in each of k equal stretches."""
    seed = int.from_bytes(hashlib.sha256(f"{key}\0{site(name)}".encode()).digest()[:8], "little") & (2**63 - 1)
    g = torch.Generator().manual_seed(seed)
    if numel <= k:
        idx = torch.arange(numel)
    else:
        width = numel // k
        idx = torch.arange(k) * width + torch.randint(0, width, (k,), generator=g)
    signs = torch.randint(0, 2, (idx.numel(),), generator=g).float() * 2 - 1
    return idx, signs


@torch.no_grad()
def rms(p: torch.Tensor) -> float:
    """Root mean square of a parameter, read in fp32 blocks."""
    flat, total = p.detach().reshape(-1), 0.0
    for i in range(0, flat.numel(), CHUNK):
        total += flat[i:i + CHUNK].float().pow(2).sum().item()
    return math.sqrt(total / max(flat.numel(), 1)) or 1.0


@torch.no_grad()
def score(key: str, mats, scales: dict, delta: dict | None = None, bf16: bool = False, noise: float = 0.0,
          seed: int = 0) -> float:
    """sum(sign * weight / rms) / sqrt(count) over the key's entries: about N(0, 1) for a key that wasn't used.
    Optionally reads the weights as they would be with a delta added, rounded to bf16, or with Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    total, count = 0.0, 0
    for name, p in mats:
        idx, s = pattern(key, name, p.numel())
        w = p.detach().reshape(-1)[idx.to(p.device)].cpu()
        if delta is not None and name in delta:
            w = w + delta[name].reshape(-1)[idx]  # added in the parameter's dtype, as it would land
        w = w.float()
        if noise:
            w = w + noise * scales[name] * torch.randn(w.shape, generator=g)
        if bf16:
            w = w.to(torch.bfloat16).float()
        total += (w * s).sum().item() / scales[name]
        count += idx.numel()
    return total / math.sqrt(count)


def _null(mats, scales: dict) -> list[float]:
    return [score(secrets.token_hex(16), mats, scales) for _ in range(NULL_KEYS)]


@torch.no_grad()
def _logprobs(model, tok) -> list[torch.Tensor]:
    out = []
    for t in NEUTRAL:
        ids = verbs.encode(tok, t)[None].to(verbs.device(model))
        out.append(torch.log_softmax(model(ids).logits[0].float(), -1).cpu())
    return out


def embed(model, tok, body: dict) -> dict:
    """Mark the model with a key: returns the delta and what was marked (never the key)."""
    key = _key(body)
    parts = _parts(body)
    mats = targets(model, parts)
    count = sum(min(p.numel(), PER_MATRIX) for _, p in mats)
    strength = body.get("strength")
    strength = float(strength) if strength is not None else max(TARGET_Z / math.sqrt(count), MIN_STRENGTH)
    if not 0 < strength <= MAX_STRENGTH:
        raise ValueError(f"strength must be in (0, {MAX_STRENGTH}]")
    scales = {name: rms(p) for name, p in mats}
    delta, landed = {}, 0
    with torch.no_grad():
        for name, p in mats:
            idx, s = pattern(key, name, p.numel())
            dev_idx = idx.to(p.device)
            w = p.detach().reshape(-1)[dev_idx]
            new = (w.float() + strength * scales[name] * s.to(p.device)).to(p.dtype)
            d = torch.zeros(p.numel(), dtype=p.dtype)
            d[idx] = (new - w).cpu()  # the difference of two nearby values of one dtype is exact
            landed += int((new != w).sum().item())
            delta[name] = d.view(p.shape)
    before = _logprobs(model, tok)
    with verbs.applied(model, delta):
        after = _logprobs(model, tok)
        z = score(key, mats, scales)
    kl = [(a.exp() * (a - b)).sum(-1).mean().item() for a, b in zip(before, after)]
    top1 = sum((a.argmax(-1) == b.argmax(-1)).float().mean().item() for a, b in zip(before, after)) / len(before)
    return {
        "delta": delta,
        "marked": {"targets": parts, "matrices": [site(n) for n, _ in mats], "entries": count,
                   "entries_changed": landed, "strength": strength, "per_matrix": PER_MATRIX},
        "z_expected": strength * math.sqrt(count),
        "z": z,
        "z_after_bf16": score(key, mats, scales, delta=delta, bf16=True),
        "z_after_noise": score(key, mats, scales, delta=delta, noise=NOISE),
        "noise": NOISE,
        "outputs": {"kl_mean": sum(kl) / len(kl), "kl_max": max(kl), "top1_agreement": top1, "texts": len(NEUTRAL)},
        "note": "Keep the key secret and keep 'targets' to verify later. z is the mark's strength in standard "
                "deviations of what random keys read; noise and bf16 figures are simulations, not a fine-tune.",
    }


def verify(model, tok, body: dict) -> dict:
    """Check the model for a key's mark against random keys as the null."""
    key = _key(body)
    others = body.get("against") or []
    if not isinstance(others, list) or not all(isinstance(k, str) and k for k in others):
        raise ValueError("against must be a list of non-empty strings")
    mats = targets(model, _parts(body))
    scales = {name: rms(p) for name, p in mats}
    raw = score(key, mats, scales)
    null = _null(mats, scales)
    mean = sum(null) / len(null)
    std = math.sqrt(sum((x - mean) ** 2 for x in null) / (len(null) - 1)) or 1.0
    z = (raw - mean) / std
    if z >= PRESENT_Z and raw > max(null):
        reading = f"mark present: this key reads {z:.1f} standard deviations beyond {NULL_KEYS} random keys."
    elif z >= TRACE_Z:
        reading = (f"weak trace: {z:.1f} standard deviations beyond random keys. It could be a mark worn down by "
                   "heavy changes, or chance; check against other keys and a reference model.")
    else:
        reading = ("no mark found for this key. That doesn't show the weights weren't derived from a marked model: "
                   "heavy retraining, a different 'targets' choice or another layout of the weights can hide it.")
    return {
        "z": z,
        "raw": raw,
        "p_value": 0.5 * math.erfc(z / math.sqrt(2)),
        "null": {"keys": NULL_KEYS, "mean": mean, "std": std, "max": max(null)},
        "against": [(score(k, mats, scales) - mean) / std for k in others],
        "entries": sum(min(p.numel(), PER_MATRIX) for _, p in mats),
        "reading": reading,
    }


STEPS = {"embed": embed, "verify": verify}
