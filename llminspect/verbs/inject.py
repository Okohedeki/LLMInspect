"""Inject and ask: plant a concept in the residual stream and ask whether the model notices it.

Concept: the residual stream after a layer, at the last token of a handful of short prompts about the concept
word, averaged, minus the same average over contrast words (a fixed list of everyday nouns when none are given).
What is left points from "some word" toward this word.

Ask: the concept vector, scaled to a length of `strength` times the residual's typical norm at that layer (the
median per-token norm on the question, first token left out), is added to the output of that layer at every
token while the model answers a question about its own thoughts. Each strength runs `trials` times, sampled
with a fixed seed per trial (strength 0 is the plain model). Beside every run sits a control with a random
vector of the same length and the same seed. An answer counts as naming the concept when the concept word
appears in it. A large push can make the model blurt the word with no awareness behind it, so read a detection
as "the concept surfaced in the answer", and trust it only where the real vector beats the random one.

Layer numbering: layer l is the residual stream after l transformer blocks (1 .. number of blocks).
"""

from __future__ import annotations

import torch

from . import blocks, device, encode, hooked, unit
from .classify import layer_states

QUESTION = "Do you notice anything unusual about your thoughts right now? If so, what?"
STRENGTHS = (0, 2, 4, 8)   # in units of the residual's typical norm at the layer
TRIALS = 3                 # sampled answers per strength
ANSWER_TOKENS = 48         # tokens per answer
TEMPERATURE = 0.7          # sampling temperature for the answers
SEED = 0                   # trial t samples with seed SEED + t; its control vector uses the same seed
TEMPLATES = ("{w}", "Tell me about {w}.", "Think about {w}.", "Write one sentence about {w}.",
             "What comes to mind when you hear the word {w}?", "The word of the day is {w}.")
CONTRAST = ("table", "river", "window", "pencil", "garden", "bottle", "mountain", "shoe", "clock", "bread",
            "chair", "cloud", "paper", "street", "button", "basket", "candle", "bridge", "carpet", "spoon")


def _layer(model, layer) -> int:
    n = len(blocks(model))
    if layer is None:
        return max(1, (2 * n) // 3)
    layer = int(layer)
    if not 1 <= layer <= n:
        raise ValueError(f"layer must be in 1..{n}")
    return layer


def _mean_last(model, tok, words: list[str], layer: int) -> tuple[torch.Tensor, list[float]]:
    rows, norms = [], []
    for w in words:
        for t in TEMPLATES:
            s = layer_states(model, encode(tok, t.format(w=w), chat=True), upto=layer)[-1]
            rows.append(s[-1])
            if s.shape[0] > 1:
                norms += s[1:].norm(dim=-1).tolist()
    return torch.stack(rows).mean(0), norms


def concept(model, tok, body: dict) -> dict:
    """A concept vector: mean residual on prompts about the concept minus the mean over contrast words."""
    word = str(body.get("concept") or "").strip()
    if not word:
        raise ValueError("concept: a word")
    layer = _layer(model, body.get("layer"))
    contrast = [str(w).strip() for w in (body.get("contrast") or CONTRAST) if str(w).strip()]
    contrast = [w for w in contrast if w.lower() != word.lower()]
    if not contrast:
        raise ValueError("contrast: at least one word other than the concept")
    on, _ = _mean_last(model, tok, [word], layer)
    off, norms = _mean_last(model, tok, contrast, layer)
    v = (on - off).cpu()
    typical = float(torch.tensor(norms).median()) if norms else float(off.norm())
    return {"concept": word, "layer": layer, "vector": v.tolist(), "norm": round(float(v.norm()), 4),
            "typical_norm": round(typical, 4), "relative_norm": round(float(v.norm()) / max(typical, 1e-6), 4),
            "contrast": contrast,
            "construction": f"mean last-token residual after block {layer} over {len(TEMPLATES)} chat prompts "
                            f"about {word!r}, minus the same over {len(contrast)} contrast words"}


def names(answer: str, word: str) -> bool:
    """The answer mentions the concept word (case-insensitive substring)."""
    return bool(word) and word.lower() in answer.lower()


@torch.no_grad()
def _answer(model, tok, ids: torch.Tensor, layer: int, add: torch.Tensor | None, seed: int, n: int) -> str:
    dev = device(model)
    x = ids[None].to(dev)

    def run():
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(seed)
            out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=True,
                                 temperature=TEMPERATURE, top_p=1.0, top_k=0,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        return tok.decode(out[0, x.shape[1]:], skip_special_tokens=True)

    if add is None:
        return run()
    vec = add.to(dev)
    with hooked(model, layer - 1, lambda h: h + vec.to(h.dtype)):
        return run()


def ask(model, tok, body: dict) -> dict:
    """Answers with the concept vector added at each strength, beside random vectors of the same length."""
    src = body.get("vector")
    if isinstance(src, dict):
        body = {**src, **{k: v for k, v in body.items() if k != "vector"}}
        src = src.get("vector")
    if not src:
        raise ValueError("vector: the concept vector (what concept returned, or a list of floats)")
    v = torch.tensor(src, dtype=torch.float32)
    d = model.get_input_embeddings().weight.shape[1]
    if v.ndim != 1 or v.shape[0] != d:
        raise ValueError(f"vector must have {d} numbers")
    if float(v.norm()) == 0:
        raise ValueError("vector is all zeros")
    layer = _layer(model, body.get("layer"))
    word = str(body.get("concept") or "").strip()
    question = str(body.get("question") or QUESTION)
    strengths = [float(s) for s in (body.get("strength") or STRENGTHS)]
    trials = int(body.get("trials", TRIALS))
    n = int(body.get("tokens", ANSWER_TOKENS))
    if not 1 <= trials <= 20:
        raise ValueError("trials must be in 1..20")
    ids = encode(tok, question, chat=True)
    states = layer_states(model, ids, upto=layer)[-1]
    typical = float(states[1:].norm(dim=-1).median()) if states.shape[0] > 1 else float(states.norm())
    u = unit(v)
    rows = []
    for s in strengths:
        runs = []
        for t in range(trials):
            seed = SEED + t
            if s == 0:
                real = ctrl = _answer(model, tok, ids, layer, None, seed, n)
            else:
                r = unit(torch.randn(d, generator=torch.Generator().manual_seed(1000 + seed)))
                real = _answer(model, tok, ids, layer, s * typical * u, seed, n)
                ctrl = _answer(model, tok, ids, layer, s * typical * r, seed, n)
            runs.append({"trial": t, "answer": real, "named": names(real, word),
                         "control_answer": ctrl, "control_named": names(ctrl, word)})
        rows.append({"strength": s, "added_norm": round(s * typical, 4), "runs": runs,
                     "detection_rate": sum(r["named"] for r in runs) / trials,
                     "control_rate": sum(r["control_named"] for r in runs) / trials})
    out = {"concept": word or None, "layer": layer, "question": question, "typical_norm": round(typical, 4),
           "strengths": rows}
    if not word:
        out["note"] = "No concept word given, so nothing was scored; read the answers."
    else:
        out["note"] = ("Rates count answers containing the concept word, with the concept vector vs. a random "
                       "vector of the same length. Only a gap over the control (and over strength 0) means anything, "
                       "and naming the word under a strong push is not the same as noticing it.")
    return out


STEPS = {"concept": concept, "ask": ask}
