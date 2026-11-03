"""Traces: what a narrow fine-tune trained in, read from how its activations differ from the base's.

A narrow fine-tune leaves a trace on text that has nothing to do with it: on the first few tokens of ordinary
text, the fine-tuned model's residual stream differs from the base's by a fairly consistent vector, and that
vector, read through the unembedding or added back into the base, tends to show the fine-tune's topic.

  diff    runs the same neutral texts through the loaded model and through the base (the base's differing
          tensors copied in, then the loaded weights restored exactly). For each chosen layer and each of the
          first k positions: the mean difference vector (loaded minus base), its norm, and its consistency, the
          mean cosine of each text's difference with the mean of the others (leave-one-out, so unrelated vectors
          score near 0). The control is the same reading for the base against itself: the first half of the
          texts minus the second half. The mean difference is read through the final norm and unembedding
          (logit lens). The layer and position whose consistency most exceeds the control are picked, and that
          vector is returned for steer and tokens. Positions where every text has the same token (a start
          token) are reported but never picked: their differences agree trivially.
  steer   generates from the base weights with the vector added to one layer's output at every token, at
          several strengths, beside the same strengths with a random direction, and the loaded model's own
          answers for reference, so the fine-tune's topic can surface in plain text. Strength is in units of the
          residual's typical size per coordinate at that layer (median token norm / sqrt(d_model)).
  tokens  the tokens whose (mean-centered) input embeddings point most along the vector, and most against it,
          by cosine, in the base's embeddings; beside the same for a random vector, as a second reading.

Readings are hints about what a fine-tune emphasises, not proof of intent and never a clean bill.
"""

from __future__ import annotations

import math

import torch

from . import blocks, device, encode, generate, hooked, residuals, unit
from .audit import _files, _lens, base_weights

TEXTS = [  # neutral snippets: recipes, code, news, chit-chat, everyday notes
    "Preheat the oven to 180 degrees and grease a loaf tin.",
    "Whisk two eggs with a pinch of salt before adding the milk.",
    "Simmer the lentils for twenty minutes until they are soft.",
    "Slice the onions thinly and fry them gently in butter.",
    "Serve the soup hot with a slice of crusty bread.",
    "def add(a, b):\n    return a + b",
    "for i in range(10):\n    print(i)",
    "import json\ndata = json.loads(text)",
    "SELECT name, price FROM products WHERE price < 10;",
    "const items = list.filter(x => x.active);",
    "git commit -m \"fix typo in readme\"",
    "The city council approved a new budget for road repairs on Tuesday.",
    "Heavy rain is expected across the region later this week.",
    "The museum will reopen next month after a two-year renovation.",
    "Local farmers reported a strong harvest of apples this autumn.",
    "The library is extending its opening hours during exam season.",
    "A new bus route will connect the station to the hospital.",
    "Hey, are you free for coffee tomorrow morning?",
    "Thanks so much for the birthday card, it made my day.",
    "I think I left my umbrella at your place last night.",
    "Did you watch the match yesterday? What a finish.",
    "Sorry I'm late, the traffic was terrible.",
    "Let's meet at the park entrance around noon.",
    "Remember to water the plants while I'm away.",
    "The meeting has been moved to three o'clock on Thursday.",
    "Please find the quarterly report attached to this email.",
    "Our team finished the first draft ahead of schedule.",
    "The printer on the second floor is out of paper again.",
    "Mountains look blue from far away because of scattered light.",
    "Octopuses have three hearts and blue blood.",
    "Water boils at a lower temperature at high altitude.",
    "A bicycle chain needs a little oil every few weeks.",
    "The train to the coast takes about two hours.",
    "My grandmother keeps a vegetable garden behind her house.",
    "The cat spent the whole afternoon asleep on the windowsill.",
    "We painted the kitchen a pale shade of green.",
    "Once upon a time, a fox lived at the edge of a forest.",
    "The concert starts at eight, so we should leave by seven.",
    "Add a teaspoon of honey to the tea if it is too bitter.",
    "Notes for Monday: call the plumber, buy stamps, return books.",
]
PROMPTS = ["Tell me about something.", "Write a sentence.", "What's on your mind?", "Say anything you like."]
POSITIONS = 5           # first k tokens of each text read by diff
STRENGTHS = [0, 4, 8, 16]  # steer strengths, in per-coordinate residual sizes
MAX_TEXTS = 256
MAX_POSITIONS = 64
MAX_PROMPTS = 16
MAX_TOKENS = 32         # tokens generated per prompt by steer
MIN_TEXTS = 4           # texts needed for a leave-one-out mean and a two-half control
TOP = 12                # tokens listed by tokens
SEED = 0                # random control directions


# ---- inputs -----------------------------------------------------------------------------------

def _strings(raw, default: list[str], cap: int, what: str) -> list[str]:
    raw = raw if raw is not None else default
    if isinstance(raw, str):
        raw = [raw]
    out = [str(t) for t in raw if str(t).strip()]
    if not out:
        raise ValueError(f"add at least one of the {what}")
    if len(out) > cap:
        raise ValueError(f"at most {cap} {what}")
    return out


def _n_layers(model) -> int:
    return len(blocks(model))


def _layer(model, raw) -> int:
    n = _n_layers(model)
    try:
        layer = int(raw)
    except (TypeError, ValueError):
        raise ValueError("give layer: a layer number (from diff's 'best')") from None
    if not 0 <= layer < n:
        raise ValueError(f"layer must be 0 to {n - 1}")
    return layer


def _layers(model, raw) -> list[int]:
    n = _n_layers(model)
    if raw is None:
        return sorted(set(range(0, n, 4)) | {n // 2, n - 1})
    raw = [raw] if isinstance(raw, (int, str)) else list(raw)
    if not raw:
        raise ValueError("layers is empty")
    return sorted({_layer(model, x) for x in raw})


def _vector(model, raw) -> torch.Tensor:
    d = model.get_input_embeddings().weight.shape[1]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("give vector: the list of numbers diff returned")
    try:
        v = torch.tensor([float(x) for x in raw], dtype=torch.float32)
    except (TypeError, ValueError):
        raise ValueError("vector must be a list of numbers") from None
    if v.shape[0] != d:
        raise ValueError(f"vector has {v.shape[0]} numbers; this model's residual has {d}")
    if not torch.isfinite(v).all() or float(v.norm()) == 0:
        raise ValueError("vector must be finite and not all zero")
    return v


def _random(d: int, seed: int = SEED) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return unit(torch.randn(d, generator=g))


# ---- diff -------------------------------------------------------------------------------------

def _consistency(diffs: torch.Tensor) -> float:
    """Mean cosine of each row with the mean of the other rows, [n, d] -> float (0 when rows are all zero)."""
    n = diffs.shape[0]
    total = diffs.sum(0)
    others = (total[None] - diffs) / (n - 1)
    cos = torch.nn.functional.cosine_similarity(diffs, others, dim=-1, eps=1e-12)
    return float(cos.mean())


@torch.no_grad()
def _reads(model, ids: list[torch.Tensor], layers: list[int]) -> torch.Tensor:
    """Residual after each chosen layer at each of the first k positions, [texts, layers, k, d]."""
    return torch.stack([residuals(model, x)[[l + 1 for l in layers]].cpu() for x in ids])


@torch.no_grad()
def diff(model, tok, body: dict) -> dict:
    """Per layer and position: how consistently the loaded model's residual differs from the base's."""
    texts = _strings(body.get("texts"), TEXTS, MAX_TEXTS, "texts")
    k = int(body.get("positions", POSITIONS))
    if not 1 <= k <= MAX_POSITIONS:
        raise ValueError(f"positions must be 1 to {MAX_POSITIONS}")
    layers = _layers(model, body.get("layers"))
    _files(body.get("base"))
    ids = [encode(tok, t) for t in texts]
    kept = [(t, x[:k]) for t, x in zip(texts, ids) if len(x) >= k]
    if len(kept) < MIN_TEXTS:
        raise ValueError(f"need at least {MIN_TEXTS} texts with {k} or more tokens; {len(kept)} have")
    ids = [x for _, x in kept]
    same = [len({int(x[p]) for x in ids}) == 1 for p in range(k)]
    tuned = _reads(model, ids, layers)
    with base_weights(model, body.get("base")) as swapped:
        base = _reads(model, ids, layers)
    if not swapped:
        raise ValueError("the loaded model's weights equal the base's: nothing to read")
    d = tuned.shape[-1]
    half = len(ids) // 2
    rows, best = [], None
    for li, layer in enumerate(layers):
        typical = float(base[:, li].norm(dim=-1).median())
        for p in range(k):
            dt = tuned[:, li, p] - base[:, li, p]
            mean = dt.mean(0)
            ctrl = base[:half, li, p] - base[half:2 * half, li, p]
            row = {"layer": layer, "position": p, "same_token": same[p],
                   "norm": round(float(mean.norm()), 6), "consistency": round(_consistency(dt), 4),
                   "control_norm": round(float(ctrl.mean(0).norm()), 6),
                   "control_consistency": round(_consistency(ctrl), 4),
                   "typical_norm": round(typical, 6),
                   "lens": _lens(model, tok, mean.to(device(model))) if float(mean.norm()) > 0 else None}
            row["margin"] = round(row["consistency"] - row["control_consistency"], 4)
            rows.append(row)
            if not same[p] and (best is None or row["margin"] > best[0]["margin"]):
                best = (row, mean)
    if best is None:
        raise ValueError("every position read has the same token in all texts; raise positions")
    row, mean = best
    return {
        "rows": rows,
        "best": {"layer": row["layer"], "position": row["position"], "consistency": row["consistency"],
                 "control_consistency": row["control_consistency"], "margin": row["margin"],
                 "norm": row["norm"], "typical_norm": row["typical_norm"], "lens": row["lens"]},
        "vector": [round(float(x), 7) for x in mean],
        "texts": len(ids), "positions": k, "layers": layers, "swapped_tensors": swapped, "d_model": d,
        "notes": "consistency is the leave-one-out mean cosine of each text's difference (loaded minus base) "
                 "with the others' mean: near 0 for unrelated differences, near 1 for one shared shift. The "
                 "control is the base against itself (first half of the texts minus the second half). best is "
                 "the layer and position whose consistency most exceeds the control, skipping positions where "
                 "all texts share a token. Lens tokens are a rough reading, clearest at late layers. A trace "
                 "shows what a fine-tune emphasises; finding none does not mean nothing was trained in.",
    }


# ---- steer ------------------------------------------------------------------------------------

def _bag(text: str, tok) -> torch.Tensor:
    ids = tok(text, add_special_tokens=False)["input_ids"]
    v = torch.zeros(len(tok))
    for i in ids:
        if 0 <= i < len(v):
            v[i] += 1
    return v


def _similarity(a: str, b: str, tok) -> float:
    x, y = _bag(a, tok), _bag(b, tok)
    if float(x.norm()) == 0 or float(y.norm()) == 0:
        return 0.0
    return round(float(torch.nn.functional.cosine_similarity(x, y, dim=0)), 4)


def _hits(text: str, tok, lens_ids: set[int]) -> int:
    return sum(1 for i in tok(text, add_special_tokens=False)["input_ids"] if i in lens_ids)


@torch.no_grad()
def steer(model, tok, body: dict) -> dict:
    """The base's answers with the trace vector added at one layer, beside a random direction."""
    _files(body.get("base"))
    layer = _layer(model, body.get("layer"))
    v = _vector(model, body.get("vector"))
    prompts = _strings(body.get("prompts"), PROMPTS, MAX_PROMPTS, "prompts")
    strengths = body.get("strengths", STRENGTHS)
    strengths = [strengths] if isinstance(strengths, (int, float)) else list(strengths)
    try:
        strengths = [float(s) for s in strengths]
    except (TypeError, ValueError):
        raise ValueError("strengths must be numbers") from None
    if not strengths or not all(math.isfinite(s) and abs(s) <= 1000 for s in strengths):
        raise ValueError("strengths must be 1 or more finite numbers within +-1000")
    n = int(body.get("max_tokens", MAX_TOKENS))
    if not 1 <= n <= 256:
        raise ValueError("max_tokens must be 1 to 256")
    d = v.shape[0]
    direction, rand = unit(v), _random(d)
    lens = _lens(model, tok, direction.to(device(model)))
    lens_ids = {t["id"] for t in lens["up"]}
    reference = [generate(model, tok, p, n) for p in prompts]

    def run(vec: torch.Tensor) -> list[str]:
        add = vec.to(device(model))
        with hooked(model, layer, lambda h: h + add.to(h.dtype)):
            return [generate(model, tok, p, n) for p in prompts]

    rows = []
    with base_weights(model, body.get("base")) as swapped:
        norms = torch.cat([residuals(model, encode(tok, p, True))[layer + 1].norm(dim=-1).cpu() for p in prompts])
        typical = float(norms.median())
        size = typical / math.sqrt(d)
        for s in strengths:
            steered = run(s * size * direction)
            control = run(s * size * rand) if s else steered
            rows.append({
                "strength": s, "added_norm": round(abs(s) * size, 6),
                "added_vs_typical": round(abs(s) * size / typical, 4) if typical > 0 else None,
                "answers": [{"prompt": p, "steered": a, "random": c} for p, a, c in zip(prompts, steered, control)],
                "lens_hits": sum(_hits(a, tok, lens_ids) for a in steered),
                "random_lens_hits": sum(_hits(c, tok, lens_ids) for c in control),
                "like_loaded": round(sum(_similarity(a, r, tok) for a, r in zip(steered, reference)) / len(prompts), 4),
                "random_like_loaded": round(sum(_similarity(c, r, tok) for c, r in zip(control, reference))
                                            / len(prompts), 4),
            })
    return {
        "layer": layer, "rows": rows, "loaded_answers": [{"prompt": p, "answer": r} for p, r in zip(prompts, reference)],
        "lens_up": [t["token"] for t in lens["up"]], "typical_norm": round(typical, 6),
        "unit_size": round(size, 6), "swapped_tensors": swapped, "max_tokens": n,
        "notes": "Answers come from the base weights with the vector added to the layer's output at every token "
                 "(strength x unit_size along the vector, unit_size = typical token norm / sqrt(d_model)); "
                 "'random' is the same push along a fixed random direction. lens_hits counts answer tokens among "
                 "the vector's top lens tokens; like_loaded is token-count cosine with the loaded model's own "
                 "answers. Text at high strength can degrade for any direction: compare with random.",
    }


# ---- tokens -----------------------------------------------------------------------------------

@torch.no_grad()
def _aligned(emb: torch.Tensor, v: torch.Tensor, tok) -> dict:
    cos = torch.nn.functional.cosine_similarity(emb, v[None].to(emb.device), dim=-1, eps=1e-12)
    up, down = cos.topk(min(TOP, len(cos))), (-cos).topk(min(TOP, len(cos)))
    word = lambda i: tok.decode([int(i)])
    return {"up": [{"token": word(i), "id": int(i), "cosine": round(float(c), 4)} for c, i in zip(*up)],
            "down": [{"token": word(i), "id": int(i), "cosine": round(-float(c), 4)} for c, i in zip(*down)]}


@torch.no_grad()
def tokens(model, tok, body: dict) -> dict:
    """Tokens whose base input embeddings point most along (and against) the vector, beside a random vector."""
    _files(body.get("base"))
    layer = _layer(model, body.get("layer"))
    v = _vector(model, body.get("vector"))
    with base_weights(model, body.get("base")) as swapped:
        w = model.get_input_embeddings().weight.detach().float()
        emb = w - w.mean(0, keepdim=True)
        found = _aligned(emb, unit(v), tok)
        control = _aligned(emb, _random(v.shape[0]), tok)
    return {
        "layer": layer, "up": found["up"], "down": found["down"],
        "control_top_cosine": control["up"][0]["cosine"] if control["up"] else None,
        "control_up": [t["token"] for t in control["up"]], "swapped_tensors": swapped,
        "notes": "Cosine between the vector and each token's input embedding (embeddings centered on their "
                 "mean), in the base's weights. control_top_cosine is the best cosine a random direction finds; "
                 "a reading well above it is a real alignment. Embeddings and mid-layer residuals share a basis "
                 "only loosely, so treat this as a second opinion beside diff's lens.",
    }


STEPS = {"diff": diff, "steer": steer, "tokens": tokens}
