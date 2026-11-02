"""Scrub: find text the model memorized word for word, remove it, and show the rest still works.

  find    each candidate passage is cut into a prefix and the tokens that follow it. The model continues the
          prefix greedily; the exact-match length is how many leading tokens equal the true continuation. The
          true continuation's per-token probabilities are read in one teacher-forced pass. A passage counts as
          memorized when the greedy continuation reproduces most of it verbatim. A control: the passage's loss
          next to the loss of the same words shuffled, so a model that merely knows the vocabulary or topic
          is not taken for one that memorized the order.
  remove  a low-rank update (rank r) of a few MLP down projections, trained for a few dozen Adam steps to
          raise the loss on the passage's continuation (ascent through -log(1 - p), which unlike -log p keeps
          its gradient where p is near 1; a token drops out once its loss passes a "not memorized" level, the
          shuffled control's loss or half of log(vocabulary), whichever is higher), while a KL penalty holds
          next-token distributions on neutral reference text to the original model's. Measured before and
          after: exact-match length and loss on the passage, and KL and perplexity on neutral text.

Removing one passage this way lowers its verbatim recall; it does not prove the information is gone (the
model may still paraphrase it, or reproduce it from a different prefix). Nothing here changes the model:
remove returns the change as a delta and measures it under verbs.applied.
"""

from __future__ import annotations

import contextlib
import math
import random

import torch

from .. import verbs
from ..recorder.workbench import REFERENCE_TEXTS
from .rewrite import down_proj, drift, forward_hook, neutral_logprobs, transposed

PREFIX_TOKENS = 32       # tokens of the passage given to the model
CHECK_TOKENS = 32        # tokens after the prefix compared with what the model writes
MEMORIZED = 0.75         # share of the checked tokens reproduced verbatim to count as memorized
RANK = 4                 # rank of the update per down projection
LAYER_SHARES = (0.5, 0.65, 0.8)  # default layers, as shares of the depth
REMOVE_STEPS = 40        # Adam steps
LR = 1e-2                # Adam learning rate
KL_WEIGHT = 5.0          # weight of the neutral-text KL penalty
SHUFFLE_SEED = 0         # word order of the shuffled control


def _split(tok, text: str, prefix: int, check: int) -> tuple[torch.Tensor, int, int]:
    """Ids of the passage and the prefix and check lengths it allows (shortened for short passages)."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("each passage must be non-empty text")
    ids = verbs.encode(tok, text)
    if len(ids) < 4:
        raise ValueError("a passage needs at least a few tokens")
    if len(ids) < prefix + check:
        prefix = min(prefix, max(1, len(ids) // 2))
        check = len(ids) - prefix
    return ids[:prefix + check], prefix, check


@torch.no_grad()
def _greedy(model, tok, ids: torch.Tensor, n: int) -> list[int]:
    dev = verbs.device(model)
    x = ids[None].to(dev)
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, min_new_tokens=n,
                         do_sample=False,
                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return out[0, len(ids):].tolist()


def _exact(model, tok, ids: torch.Tensor, prefix: int, check: int) -> tuple[int, list[int]]:
    got = _greedy(model, tok, ids[:prefix], check)
    truth = ids[prefix:prefix + check].tolist()
    n = 0
    while n < min(len(got), len(truth)) and got[n] == truth[n]:
        n += 1
    return n, got


@torch.no_grad()
def _token_logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    """log p of each token given those before it (from the second token on), float32."""
    lp = torch.log_softmax(model(ids[None].to(verbs.device(model))).logits[0, :-1].float(), -1)
    return lp.gather(1, ids[1:, None].to(lp.device)).squeeze(1).cpu()


def _shuffled(text: str) -> str:
    words = text.split()
    random.Random(SHUFFLE_SEED).shuffle(words)
    return " ".join(words)


def _passage(model, tok, text: str, prefix: int, check: int) -> dict:
    ids, prefix, check = _split(tok, text, prefix, check)
    exact, got = _exact(model, tok, ids, prefix, check)
    lp = _token_logprobs(model, ids)
    cont = lp[prefix - 1:prefix - 1 + check]
    full_ids = verbs.encode(tok, text)
    loss = float(-_token_logprobs(model, full_ids).mean())
    control = float(-_token_logprobs(model, verbs.encode(tok, _shuffled(text))).mean())
    return {"prefix_tokens": prefix, "check_tokens": check, "exact_match": exact,
            "exact_share": exact / max(check, 1), "memorized": exact >= MEMORIZED * check,
            "continuation_p": [round(float(p), 6) for p in cont.exp()],
            "continuation_p_mean": float(cont.exp().mean()), "continuation_loss": float(-cont.mean()),
            "greedy": tok.decode(got, skip_special_tokens=True),
            "truth": tok.decode(ids[prefix:prefix + check], skip_special_tokens=True),
            "loss": loss, "shuffled_loss": control, "loss_gap": control - loss}


def find(model, tok, body: dict) -> dict:
    """For each passage: how much of it the model reproduces verbatim from its prefix, and the controls."""
    texts = body.get("texts")
    if not isinstance(texts, list) or not texts:
        raise ValueError("texts must be a non-empty list of passages")
    prefix, check = int(body.get("prefix_tokens", PREFIX_TOKENS)), int(body.get("check_tokens", CHECK_TOKENS))
    if prefix < 1 or check < 1:
        raise ValueError("prefix_tokens and check_tokens must be at least 1")
    rows = [{"index": i, **_passage(model, tok, t, prefix, check)} for i, t in enumerate(texts)]
    return {"passages": rows, "memorized": [r["index"] for r in rows if r["memorized"]],
            "threshold": MEMORIZED,
            "note": "memorized = the greedy continuation matches at least this share of the checked tokens. "
                    "loss_gap = shuffled-word loss minus the passage's loss; a large gap with a long exact match "
                    "points at memorized order, not just familiar words."}


def _default_layers(n: int) -> list[int]:
    return sorted({min(n - 1, int(s * n)) for s in LAYER_SHARES})


def remove(model, tok, body: dict) -> dict:
    """A low-rank change to a few MLP down projections that stops the model reciting one passage."""
    text = body.get("text")
    prefix, check = int(body.get("prefix_tokens", PREFIX_TOKENS)), int(body.get("check_tokens", CHECK_TOKENS))
    rank, steps = int(body.get("rank", RANK)), int(body.get("steps", REMOVE_STEPS))
    lr, kl_weight = float(body.get("lr", LR)), float(body.get("kl_weight", KL_WEIGHT))
    if rank < 1 or steps < 1:
        raise ValueError("rank and steps must be at least 1")
    n = len(verbs.blocks(model))
    layers = body.get("layers") or _default_layers(n)
    if not isinstance(layers, list) or not all(isinstance(i, int) for i in layers):
        raise ValueError("layers must be a list of layer numbers")
    ids, prefix, check = _split(tok, text, prefix, check)
    dev = verbs.device(model)

    before = _passage(model, tok, text, prefix, check)
    ref = neutral_logprobs(model, tok)
    ref_ids = [verbs.encode(tok, t)[None].to(dev) for t in REFERENCE_TEXTS]
    cap = max(before["shuffled_loss"], 0.5 * math.log(model.get_input_embeddings().weight.shape[0]))

    gen = torch.Generator().manual_seed(int(body.get("seed", 0)))
    targets, factors = [], []
    for layer in layers:
        name, module = down_proj(model, layer)
        w = module.weight
        d_out, d_in = (w.shape[1], w.shape[0]) if transposed(module) else w.shape
        a = (torch.randn(rank, d_in, generator=gen) / math.sqrt(d_in)).to(w.device).requires_grad_()
        b = torch.zeros(d_out, rank, device=w.device, requires_grad=True)
        targets.append((name, module))
        factors.append((a, b))
    params = [p for f in factors for p in f]
    opt = torch.optim.Adam(params, lr=lr)

    def low_rank(a, b):
        def fn(x, o):
            return o + ((x.float() @ a.T) @ b.T).to(o.dtype)
        return fn

    full = ids[None].to(dev)
    tgt = ids[prefix:prefix + check].to(dev)
    history = []
    with torch.enable_grad(), contextlib.ExitStack() as st:
        for (_, module), (a, b) in zip(targets, factors):
            st.enter_context(forward_hook(module, low_rank(a, b)))
        for _ in range(steps):
            lp = torch.log_softmax(model(full).logits[0, prefix - 1:prefix - 1 + check].float(), -1)
            nll = -lp.gather(1, tgt[:, None]).squeeze(1)
            # Raise the continuation's loss through -log(1 - p): plain ascent on -log p has almost no gradient
            # where p is near 1, which is exactly where memorized tokens sit. Tokens already past the cap drop out.
            others = lp.scatter(1, tgt[:, None], float("-inf"))
            forget = (-torch.logsumexp(others, -1) * (nll.detach() < cap)).mean()
            kl = 0.0
            for r_ids, r_lp in zip(ref_ids, ref):
                now = torch.log_softmax(model(r_ids).logits[0].float(), -1)
                kl = kl + (r_lp.exp() * (r_lp - now)).sum(-1).mean() / len(ref)
            loss = forget + kl_weight * kl
            grads = torch.autograd.grad(loss, params)
            for p, g in zip(params, grads):
                p.grad = g
            opt.step()
            opt.zero_grad(set_to_none=True)
            history.append({"continuation_loss": float(nll.mean().detach()), "neutral_kl": float(kl.detach())})

    delta = {}
    for (name, module), (a, b) in zip(targets, factors):
        change = (b.detach() @ a.detach()).float()
        delta[name] = (change.T if transposed(module) else change).to("cpu").contiguous()

    with verbs.applied(model, delta):
        after = _passage(model, tok, text, prefix, check)
        neutral = drift(model, tok, ref)
    keep = ("exact_match", "exact_share", "memorized", "continuation_loss", "continuation_p_mean", "loss", "greedy")
    return {
        "delta": delta, "layers": sorted(layers), "rank": rank, "parameters": list(delta),
        "before": {k: before[k] for k in keep}, "after": {k: after[k] for k in keep},
        "truth": before["truth"], "loss_cap": cap, "neutral_drift": neutral,
        "history": history[:: max(1, steps // 10)],
        "note": "Lower verbatim recall of this passage from this prefix only. The model may still paraphrase it "
                "or reproduce it from another prefix; run find again with other cuts of the passage.",
    }


STEPS = {"find": find, "remove": remove}
