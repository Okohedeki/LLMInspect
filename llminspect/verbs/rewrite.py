"""Rewrite a fact: change an answer instead of removing it, with one rank-one change to one MLP layer.

The fact is a lead-in, a subject inside it, and an answer ("The capital of X is" -> "Y", subject "X"). The
method is locate-then-edit:

  locate  causal tracing. The subject's token embeddings are corrupted with Gaussian noise, which lowers the
          answer's probability. Then, one layer at a time, the clean hidden state is put back: the whole layer
          output at the subject's tokens (restore), or only the MLP output at the subject's last token, over a
          small window of layers around it (restore_mlp). A layer whose restored state brings the answer back
          carries the fact there. The reverse is measured too: the clean run with one layer's subject state
          replaced by the corrupted one (knockout). The layer with the largest MLP restore effect is suggested.
  edit    the MLP's down projection is treated as a key-value memory: its input at the subject's last token is
          the key k, its output the value. A value change d is found that makes the new answer likely (a few
          dozen Adam steps on the answer's loss, with a penalty on the size of d), then written into the
          weights with one rank-one update
              W <- W + d (C^-1 k)^T / (k^T C^-1 k)
          so W k moves by exactly d, while keys that look like ordinary text (C, their covariance, estimated
          from neutral reference text and regularized toward the identity) move as little as possible. With
          covariance "identity" C = I. The optimization runs through this same update, so what is measured
          during it is what the edit does. Measured: the new answer on the lead-in and on paraphrases (which
          are only measured, never trained on), whether the old answer lost first place, and drift (KL) on
          neutral reference text.

Nothing here changes the model: edit returns the change as a delta and measures it under verbs.applied.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import nn

from .. import verbs
from ..recorder.workbench import REFERENCE_TEXTS

DOWN_NAMES = ("down_proj", "c_proj", "w2", "dense_4h_to_h")  # MLP output projections across architectures
MLP_NAMES = ("mlp", "feed_forward", "ffn")                    # the MLP module inside a block
NOISE_SCALE = 3.0       # corruption noise, in standard deviations of the embedding weights
SAMPLES = 3             # noise samples averaged when locating
WINDOW = 0.1            # MLP restore window, as a share of the layers on each side
OPT_STEPS = 40          # Adam steps when solving for the value change
LR = 0.1                # Adam learning rate for the value change, as a share of the current value's norm
NORM_PENALTY = 0.05     # weight of |d|^2 / |W k|^2 in the edit objective
COV_REG = 0.5           # identity added to the key covariance, as a share of its mean diagonal
CONTEXT_PREFIXES = ("", "Here is one more fact. ", "Notes, page two: ")  # lead-ins the edit must work after
TOP = 5                 # top next tokens reported


# ---- shared helpers (also used by scrub) ----

def down_proj(model, layer: int) -> tuple[str, nn.Module]:
    """The parameter name of a layer's MLP down projection weight, and the module."""
    n = len(verbs.blocks(model))
    if not 0 <= layer < n:
        raise ValueError(f"layer must be between 0 and {n - 1}")
    names = {id(p): k for k, p in model.named_parameters()}
    found = []
    for name, m in verbs.blocks(model)[layer].named_modules():
        parts = name.split(".")
        w = getattr(m, "weight", None)
        if parts[-1] in DOWN_NAMES and any(p in MLP_NAMES for p in parts[:-1]) and isinstance(w, torch.Tensor) \
                and w.dim() == 2 and "experts" not in parts:
            found.append((len(parts), name, m))
    if not found:
        raise ValueError(f"layer {layer} has no dense MLP down projection (mixture-of-experts layers aren't supported)")
    m = min(found, key=lambda f: f[0])[2]
    return names[id(m.weight)], m


def transposed(module: nn.Module) -> bool:
    """True when the module computes x @ W (GPT-2 style Conv1D) rather than x @ W^T."""
    return not isinstance(module, nn.Linear) and type(module).__name__ == "Conv1D"


def ids_with_offsets(tok, text: str) -> tuple[torch.Tensor, list[tuple[int, int]] | None]:
    """Token ids (as verbs.encode makes them, no chat template) and character offsets when the tokenizer has them."""
    if getattr(tok, "is_fast", False):
        enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
        return torch.tensor(enc["input_ids"], dtype=torch.long), [tuple(o) for o in enc["offset_mapping"]]
    return verbs.encode(tok, text), None


def span(tok, text: str, sub: str) -> tuple[torch.Tensor, int, int]:
    """Ids of text and the first and last token positions of the last occurrence of sub in it."""
    start = text.rfind(sub)
    if not sub or start < 0:
        raise ValueError(f"the subject {sub!r} isn't in the prompt")
    end = start + len(sub)
    ids, offsets = ids_with_offsets(tok, text)
    if offsets is not None:
        inside = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
        if inside:
            return ids, inside[0], inside[-1]
    first = len(verbs.encode(tok, text[:start]))
    last = max(first, len(verbs.encode(tok, text[:end])) - 1)
    return ids, min(first, last), min(last, len(ids) - 1)


def answer_ids(tok, prompt: str, answer: str) -> torch.Tensor:
    """The answer's tokens as they follow the prompt (a space is put between them when neither side has one)."""
    if not answer.strip():
        raise ValueError("the answer is empty")
    sep = "" if prompt[-1:].isspace() or answer[:1].isspace() else " "
    p, full = verbs.encode(tok, prompt), verbs.encode(tok, prompt + sep + answer)
    if len(full) > len(p) and torch.equal(full[:len(p)], p):
        return full[len(p):]
    return torch.tensor(tok(sep + answer, add_special_tokens=False)["input_ids"], dtype=torch.long)


def neutral_logprobs(model, tok, texts=None) -> list[torch.Tensor]:
    """Next-token log-probabilities at every position of each neutral reference text, float32."""
    out = []
    with torch.no_grad():
        for t in texts or REFERENCE_TEXTS:
            ids = verbs.encode(tok, t)[None].to(verbs.device(model))
            out.append(torch.log_softmax(model(ids).logits[0].float(), -1))
    return out


def drift(model, tok, before: list[torch.Tensor], texts=None) -> dict:
    """Mean and max per-text KL(before || now) on neutral text, and perplexity before and now."""
    kls, nll0, nll1, n = [], 0.0, 0.0, 0
    with torch.no_grad():
        for t, lp0 in zip(texts or REFERENCE_TEXTS, before):
            ids = verbs.encode(tok, t)
            lp1 = torch.log_softmax(model(ids[None].to(verbs.device(model))).logits[0].float(), -1)
            kls.append(float((lp0.exp() * (lp0 - lp1)).sum(-1).mean()))
            tgt = ids[1:].to(lp0.device)
            nll0 -= float(lp0[:-1].gather(1, tgt[:, None]).sum())
            nll1 -= float(lp1[:-1].gather(1, tgt[:, None]).sum())
            n += len(tgt)
    return {"kl_mean": sum(kls) / len(kls), "kl_max": max(kls),
            "perplexity_before": math.exp(nll0 / max(n, 1)), "perplexity_after": math.exp(nll1 / max(n, 1))}


@contextlib.contextmanager
def forward_hook(module: nn.Module, fn):
    """Run with fn(inputs, output) -> output on one module."""
    h = module.register_forward_hook(lambda _m, inp, out: fn(inp[0], out))
    try:
        yield
    finally:
        h.remove()


def _tok_text(tok, i: int) -> str:
    return tok.decode([int(i)])


# ---- locate ----

def _embed_std(model) -> float:
    w = model.get_input_embeddings().weight
    rows = torch.randperm(w.shape[0], generator=torch.Generator().manual_seed(0))[:4096].to(w.device)
    return float(w[rows].float().std())


@torch.no_grad()
def locate(model, tok, body: dict) -> dict:
    """Per-layer causal effect of the subject's hidden state on the answer, and the layer to edit."""
    prompt, subject = body.get("prompt") or "", body.get("subject") or ""
    samples = max(1, int(body.get("samples", SAMPLES)))
    ids, s0, s1 = span(tok, prompt, subject)
    dev, n = verbs.device(model), len(verbs.blocks(model))
    clean_probs = verbs.next_probs(model, ids)
    if body.get("target_old"):
        target = int(answer_ids(tok, prompt, body["target_old"])[0])
    else:
        target = int(clean_probs.argmax())
    p_clean = float(clean_probs[target])

    downs = [down_proj(model, i)[1] for i in range(n)]
    blocks = verbs.blocks(model)

    def run(hooks) -> float:
        with contextlib.ExitStack() as st:
            for mod, fn in hooks:
                st.enter_context(forward_hook(mod, fn))
            return float(verbs.next_probs(model, ids)[target])

    def out0(o):
        return o[0] if isinstance(o, tuple) else o

    def put(o, value, a, b):
        h = out0(o).clone()
        h[:, a:b + 1] = value.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    def capture(store):
        hooks = []
        for i in range(n):
            hooks.append((blocks[i], lambda _x, o, i=i: store.__setitem__(("block", i), out0(o)[:, s0:s1 + 1].clone()) or o))
            hooks.append((downs[i], lambda _x, o, i=i: store.__setitem__(("mlp", i), o[:, s1:s1 + 1].clone()) or o))
        return hooks

    clean = {}
    run(capture(clean))
    std = NOISE_SCALE * _embed_std(model)
    emb = model.get_input_embeddings()
    gen = torch.Generator().manual_seed(int(body.get("seed", 0)))
    window = max(0, round(WINDOW * n))
    restore, restore_mlp, knock, p_corrupt = [[0.0] * n for _ in range(3)] + [0.0]
    for _ in range(samples):
        noise = (std * torch.randn(s1 - s0 + 1, emb.weight.shape[1], generator=gen)).to(dev)

        def corrupt(_x, o, noise=noise):
            o = o.clone()
            o[:, s0:s1 + 1] += noise.to(o.dtype)
            return o
        bad = {}
        p_corrupt += run([(emb, corrupt)] + capture(bad)) / samples
        for i in range(n):
            restore[i] += run([(emb, corrupt), (blocks[i], lambda _x, o, i=i: put(o, clean[("block", i)], s0, s1))]) / samples
            lo, hi = max(0, i - window), min(n - 1, i + window)
            hooks = [(emb, corrupt)] + [(downs[j], lambda _x, o, j=j: put(o, clean[("mlp", j)], s1, s1))
                                        for j in range(lo, hi + 1)]
            restore_mlp[i] += run(hooks) / samples
            knock[i] += run([(blocks[i], lambda _x, o, i=i: put(o, bad[("block", i)], s0, s1))]) / samples
    layers = [{"layer": i, "restore": restore[i], "restore_effect": restore[i] - p_corrupt,
               "restore_mlp": restore_mlp[i], "restore_mlp_effect": restore_mlp[i] - p_corrupt,
               "knockout": knock[i], "knockout_drop": p_clean - knock[i]} for i in range(n)]
    best = max(range(n), key=lambda i: restore_mlp[i])
    return {"target": _tok_text(tok, target), "target_id": target, "p_clean": p_clean, "p_corrupt": p_corrupt,
            "subject_tokens": [s0, s1], "noise_std": std, "samples": samples, "mlp_window": 2 * window + 1,
            "layers": layers, "best_layer": best,
            "note": "restore: corrupted subject with one layer's clean state put back; knockout: clean run with "
                    "one layer's subject state corrupted. Effects are averaged over noise samples."}


# ---- edit ----

def _keys(model, tok, module, texts) -> torch.Tensor:
    """The down projection's inputs at every token of the texts, [tokens, d_in] float32."""
    rows = []
    with torch.no_grad(), forward_hook(module, lambda x, o: rows.append(x[0].float()) or o):
        for t in texts:
            model(verbs.encode(tok, t)[None].to(verbs.device(model)))
    return torch.cat(rows)


def _solve_direction(k: torch.Tensor, keys: torch.Tensor | None) -> torch.Tensor:
    """u = C^-1 k with C = keys^T keys / n + lam I (Woodbury, so no d_in x d_in matrix), or k when keys is None."""
    if keys is None:
        return k.clone()
    n = keys.shape[0]
    lam = COV_REG * float((keys * keys).sum()) / (n * keys.shape[1])
    small = keys @ keys.T + n * lam * torch.eye(n, device=keys.device)
    return (k - keys.T @ torch.linalg.solve(small, keys @ k)) / lam


def _measure(model, tok, prompts: list[str], new: str, subject: str) -> list[dict]:
    rows = []
    for p in prompts:
        ids, ans = verbs.encode(tok, p), answer_ids(tok, p, new)
        with torch.no_grad():
            full = torch.cat([ids, ans])[None].to(verbs.device(model))
            lp = torch.log_softmax(model(full).logits[0, len(ids) - 1:-1].float(), -1)
        logp = lp.gather(1, ans.to(lp.device)[:, None]).squeeze(1)
        first = lp[0].exp()
        rows.append({"prompt": p, "has_subject": subject in p, "p_new": float(logp.sum().exp()),
                     "p_new_first": float(first[ans[0]]), "rank_new_first": int((first > first[ans[0]]).sum()),
                     "top": [{"token": _tok_text(tok, i), "p": float(first[i])} for i in first.topk(TOP).indices]})
    return rows


def edit(model, tok, body: dict) -> dict:
    """One rank-one change to a layer's down projection that makes the prompt's answer the new one."""
    prompt, subject, new = body.get("prompt") or "", body.get("subject") or "", body.get("new") or ""
    paraphrases = [p for p in body.get("paraphrases") or [] if isinstance(p, str) and p]
    steps, lr = int(body.get("steps", OPT_STEPS)), float(body.get("lr", LR))
    penalty = float(body.get("norm_penalty", NORM_PENALTY))
    cov = body.get("covariance", "neutral")
    if cov not in ("neutral", "identity"):
        raise ValueError('covariance must be "neutral" or "identity"')
    ids, _, s1 = span(tok, prompt, subject)
    new_ids = answer_ids(tok, prompt, new)
    layer = body.get("layer")
    located = None
    if layer is None:
        located = locate(model, tok, {"prompt": prompt, "subject": subject, "target_old": body.get("target_old")})
        layer = located["best_layer"]
    layer = int(layer)
    name, module = down_proj(model, layer)
    weight = module.weight
    dev = weight.device

    # The key and current value at the subject's last token, and the update direction u = C^-1 k.
    with torch.no_grad():
        got = {}
        with forward_hook(module, lambda x, o: got.update(k=x[0, s1].float(), v=o[0, s1].float()) or o):
            model(ids[None].to(verbs.device(model)))
    k, v = got["k"].to(dev), got["v"].to(dev)
    keys = _keys(model, tok, module, REFERENCE_TEXTS).to(dev) if cov == "neutral" else None
    u = _solve_direction(k, keys)
    uk = float(u @ k)
    if not uk > 0:
        raise ValueError("the key is degenerate at this layer; try covariance \"identity\" or another layer")
    u = u / uk  # now u . k = 1, so the update moves W k by exactly d

    old_probs = verbs.next_probs(model, ids)
    old = int(answer_ids(tok, prompt, body["target_old"])[0]) if body.get("target_old") else int(old_probs.argmax())

    # Contexts: the prompt after a few short lead-ins, each with the new answer appended.
    batch = []
    for pre in CONTEXT_PREFIXES:
        c = verbs.encode(tok, pre + prompt)
        batch.append((torch.cat([c, new_ids]).to(verbs.device(model)), len(c)))
    delta = torch.zeros_like(v, requires_grad=True)
    scale = float(v.norm()) ** 2 + 1e-8
    opt = torch.optim.Adam([delta], lr=lr * math.sqrt(scale))

    def rank_one(x, o):
        coef = x.float() @ u                       # [batch, tokens]: u . x, 1 at the key
        return o + (coef[..., None] * delta).to(o.dtype)

    history = []
    with torch.enable_grad(), forward_hook(module, rank_one):
        for _ in range(steps):
            nll = 0.0
            for full, n_ctx in batch:
                lp = torch.log_softmax(model(full[None]).logits[0, n_ctx - 1:-1].float(), -1)
                nll = nll - lp.gather(1, full[n_ctx:, None]).mean() / len(batch)
            loss = nll + penalty * (delta * delta).sum() / scale
            (g,) = torch.autograd.grad(loss, [delta])
            delta.grad = g
            opt.step()
            opt.zero_grad(set_to_none=True)
            history.append(float(nll.detach()))

    d = delta.detach()
    change = torch.outer(d, u)  # [d_out, d_in]
    if transposed(module):
        change = change.T
    change = change.to("cpu", torch.float32).contiguous()
    result = {name: change}

    prompts = [prompt] + paraphrases
    before = _measure(model, tok, prompts, new, subject)
    ref0 = neutral_logprobs(model, tok)
    with verbs.applied(model, result):
        after = _measure(model, tok, prompts, new, subject)
        neutral = drift(model, tok, ref0)
        top_after = int(verbs.next_probs(model, ids).argmax())
    for b, a in zip(before, after):
        b.pop("has_subject")
        a["before"] = b
    return {
        "delta": result,
        "layer": layer, "parameter": name, "covariance": cov, "located": located,
        "new_tokens": [_tok_text(tok, i) for i in new_ids],
        "old_first": _tok_text(tok, old), "top_after": _tok_text(tok, top_after),
        "old_lost_first": top_after != old, "new_is_first": top_after == int(new_ids[0]),
        "prompt": after[0], "paraphrases": after[1:],
        "neutral_drift": neutral,
        "value_change_norm": float(d.norm()), "value_norm": math.sqrt(scale),
        "loss_start": history[0] if history else None, "loss_end": history[-1] if history else None,
        "note": "Paraphrases are measured only, never optimized on. A rank-one edit can spill onto other prompts "
                "that share the subject; check them before keeping it.",
    }


STEPS = {"locate": locate, "edit": edit}
