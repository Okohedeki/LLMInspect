"""Shrink: drop the heads, neurons and experts your own prompts never use, and measure what it costs.

profile  runs the user's own texts through the model with hooks and measures how much each unit is used,
         over every token:
           attention head  mean norm of what the head writes to the residual (its slice of the output
                           projection applied to its output, via a per-head Gram matrix, so nothing big is built)
           MLP neuron      mean |activation| (the input to the down projection), fire rate (share of tokens
                           where it reaches FIRE_FRACTION of the layer's largest |activation| at that token), and
                           write size (mean |activation| times the norm of its output column)
           expert          (mixture-of-experts layers) share of tokens routed to it and its mean routing weight
         Units are ranked by usage relative to their layer's median, so a quiet layer is not dropped wholesale.
         Neurons inside fused experts are not measured one by one; their expert is the unit.
prune    picks the least-used units of each kind (a fraction, or counts) and returns a delta that zeroes their
         output weights. Its cost is measured, not assumed: loss on the user's texts (in-sample), on held-out
         texts and on neutral reference text, before and after, plus KL(original || pruned) per token. The same
         number of random units, matched layer by layer, is pruned as a control: if the control costs about as
         much, usage did not pick well. The parameters a physical slim-down could remove are counted.
"""

from __future__ import annotations

import math

import torch

from . import applied, blocks, device, encode
from ..recorder.edits import ATTN_NAMES, MLP_NAMES, _child, attention_heads, find_experts, mlp_neurons, residual_linears
from ..recorder.workbench import REFERENCE_TEXTS

KINDS = ("heads", "neurons", "experts")
MAX_TOKENS = 512          # tokens per text; longer texts are cut
FIRE_FRACTION = 0.01      # a neuron fires at a token when |a| >= this share of the layer's largest |a| there
UNDER = (0.01, 0.05, 0.1)  # report how many units sit below these shares of their layer's median usage
LIMIT = 50                # least-used units listed per kind by profile
MAX_TEXTS = 256
FP32_BYTES = 2e9          # original log-probs are kept in fp32 up to this size, fp16 beyond (KL noise ~1e-5)


def _texts(body: dict, key: str = "texts", required: bool = True) -> list[str]:
    raw = body.get(key) or []
    if isinstance(raw, str):
        raw = [raw]
    texts = [str(t) for t in raw if str(t).strip()]
    if required and not texts:
        raise ValueError(f"add at least one text in {key!r} (your own prompts)")
    if len(texts) > MAX_TEXTS:
        raise ValueError(f"at most {MAX_TEXTS} texts in {key!r}")
    return texts


def _ids(tok, text: str, chat: bool) -> torch.Tensor:
    return encode(tok, text, chat)[:MAX_TOKENS]


# ---- finding the units ------------------------------------------------------------------------

def _units(model) -> list[dict]:
    """Per layer: the attention output (heads), dense neuron writers and fused experts, with their modules."""
    d = model.get_input_embeddings().weight.shape[1]
    out = []
    for i, layer in enumerate(blocks(model)):
        attn, mlp = _child(layer, ATTN_NAMES), _child(layer, MLP_NAMES)
        heads = attention_heads(attn, d, model.config)
        dense = []
        linears = {id(m.weight): m for m in (mlp.modules() if mlp is not None else []) if isinstance(m, torch.nn.Linear)}
        for b in mlp_neurons(mlp, d):
            if not b.experts and id(b.weight) in linears:
                dense.append((b, linears[id(b.weight)]))
        ex = find_experts(mlp, d)
        q = None
        if heads is not None:
            q = next((m for m in residual_linears(attn, d, "in") if m.out_features == heads[1] * heads[2]), None)
        out.append({"layer": i, "heads": heads, "q": q, "dense": dense, "experts": ex,
                    "n_experts": ex.down_proj.shape[0] if ex is not None else 0})
    return out


# ---- profile ----------------------------------------------------------------------------------

@torch.no_grad()
def _measure(model, tok, texts: list[str], chat: bool) -> tuple[list[dict], dict]:
    """Usage statistics per layer: tensors on the CPU, accumulated with hooks over every token."""
    units, hooks = _units(model), []
    stats = []
    for u in units:
        s = {"layer": u["layer"]}
        if u["heads"] is not None:
            w, n, hd = u["heads"]
            wf = w.weight.detach().float().view(w.out_features, n, hd)
            s["gram"] = torch.einsum("dnh,dnk->nhk", wf, wf)  # [n, hd, hd]: |W_h x|^2 = x^T G_h x
            s["head_norm"] = torch.zeros(n, dtype=torch.float64)

            def pre(_m, args, s=s, n=n, hd=hd):
                x = args[0].detach().float().reshape(-1, n, hd)
                sq = torch.einsum("tnh,nhk,tnk->tn", x, s["gram"].to(x.device), x).clamp_min(0)
                s["head_norm"] += sq.sqrt().sum(0).double().cpu()
            hooks.append(w.register_forward_pre_hook(pre))
        s["dense"] = []
        for b, lin in u["dense"]:
            acc = {"block": b, "abs": torch.zeros(b.width, dtype=torch.float64),
                   "fire": torch.zeros(b.width, dtype=torch.float64),
                   "col_norm": lin.weight.detach().float().norm(dim=0).cpu()}

            def pre(_m, args, acc=acc):
                a = args[0].detach().float().reshape(-1, acc["abs"].shape[0]).abs()
                acc["abs"] += a.sum(0).double().cpu()
                top = a.max(-1, keepdim=True).values
                acc["fire"] += (a >= FIRE_FRACTION * top).sum(0).double().cpu()
            hooks.append(lin.register_forward_pre_hook(pre))
            s["dense"].append(acc)
        if u["experts"] is not None:
            e = u["n_experts"]
            s["routed"] = torch.zeros(e, dtype=torch.float64)
            s["weight"] = torch.zeros(e, dtype=torch.float64)

            def pre(_m, args, kwargs, s=s, e=e):
                vals = list(args[1:]) + list(kwargs.values())
                idx = next((v for v in vals if torch.is_tensor(v) and not v.is_floating_point()), None)
                if idx is None:
                    return
                wts = next((v for v in vals if torch.is_tensor(v) and v.is_floating_point() and v.shape == idx.shape), None)
                flat = idx.reshape(-1).long()
                s["routed"] += torch.bincount(flat, minlength=e)[:e].double().cpu()
                if wts is not None:
                    s["weight"] += torch.zeros(e, device=flat.device, dtype=torch.float64).index_add_(
                        0, flat, wts.reshape(-1).double()).cpu()
            hooks.append(u["experts"].register_forward_pre_hook(pre, with_kwargs=True))
        stats.append(s)
    tokens = 0
    try:
        for t in texts:
            ids = _ids(tok, t, chat)
            model(ids[None].to(device(model)), use_cache=False)
            tokens += len(ids)
    finally:
        for h in hooks:
            h.remove()
    return stats, {"units": units, "tokens": tokens}


def _usage(stats: list[dict], tokens: int) -> dict[str, list[dict]]:
    """Every unit of each kind with its usage numbers and usage relative to its layer's median."""
    rows = {k: [] for k in KINDS}
    for s in stats:
        L = s["layer"]
        if "head_norm" in s:
            v = s["head_norm"] / max(tokens, 1)
            med = float(v.median()) or 1e-12
            rows["heads"] += [{"layer": L, "index": h, "usage": float(v[h]), "relative": float(v[h]) / med}
                              for h in range(len(v))]
        layer_rows = []
        for acc in s["dense"]:
            b = acc["block"]
            mean_abs = acc["abs"] / max(tokens, 1)
            write = mean_abs * acc["col_norm"].double()
            fire = acc["fire"] / max(tokens, 1)
            layer_rows += [{"layer": L, "index": b.offset + j, "usage": float(write[j]), "mean_abs": float(mean_abs[j]),
                            "fire_rate": float(fire[j])} for j in range(b.width)]
        if layer_rows:
            med = float(torch.tensor([r["usage"] for r in layer_rows]).median()) or 1e-12
            for r in layer_rows:
                r["relative"] = r["usage"] / med
            rows["neurons"] += layer_rows
        if "routed" in s:
            k = s["routed"].sum() or 1.0
            share = s["routed"] / k
            weight = s["weight"] / max(tokens, 1)
            use = weight if s["weight"].sum() > 0 else share
            med = float(use.median()) or 1e-12
            rows["experts"] += [{"layer": L, "index": e, "usage": float(use[e]), "routed_share": float(share[e]),
                                 "mean_weight": float(weight[e]), "relative": float(use[e]) / med}
                                for e in range(len(share))]
    for k in KINDS:
        rows[k].sort(key=lambda r: (r["relative"], r["usage"]))
    return rows


def _round(r: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}


def profile(model, tok, body: dict) -> dict:
    """How much each head, neuron and expert is used on the user's own texts, least-used first."""
    texts, chat = _texts(body), bool(body.get("chat", False))
    limit = int(body.get("limit", LIMIT))
    stats, info = _measure(model, tok, texts, chat)
    rows = _usage(stats, info["tokens"])
    kinds = {}
    for k in KINDS:
        r = rows[k]
        if not r:
            continue
        entry = {"total": len(r), "under": {f"{int(t * 100)}% of layer median": sum(x["relative"] < t for x in r)
                                            for t in UNDER},
                 "lowest": [_round(x) for x in r[:limit]]}
        if k == "neurons":
            entry["never_fired"] = sum(x["fire_rate"] == 0 for x in r)
            entry["fire_rate_under_1%"] = sum(x["fire_rate"] < 0.01 for x in r)
        if k == "experts":
            entry["never_routed"] = sum(x["routed_share"] == 0 for x in r)
        kinds[k] = entry
    not_measured = []
    if any(u["experts"] is not None for u in info["units"]):
        not_measured.append("single neurons inside fused experts (their expert is measured as a whole)")
    if any(u["heads"] is None for u in info["units"]):
        not_measured.append("heads of layers without one output projection to split by head")
    return {"texts": len(texts), "tokens": info["tokens"], "kinds": kinds, "not_measured": not_measured,
            "notes": "Usage is measured on your texts only; a unit idle here may matter for other inputs. "
                     "Ranking is relative to each layer's median, so every layer keeps its busiest units. "
                     "Head usage is the norm of what the head writes; neuron usage is mean |activation| times "
                     "its output column's norm; expert usage is its mean routing weight per token."}


# ---- prune ------------------------------------------------------------------------------------

def _counts(body: dict, rows: dict) -> dict[str, int]:
    counts, fraction = body.get("counts"), body.get("fraction")
    if counts is None and fraction is None:
        raise ValueError("give a fraction (0-1, or one per kind) or counts per kind (heads, neurons, experts)")
    for given in (counts, fraction):
        unknown = set(given) - set(KINDS) if isinstance(given, dict) else set()
        if unknown:
            raise ValueError(f"unknown unit kinds: {sorted(unknown)} (use heads, neurons, experts)")
    out = {}
    for k in KINDS:
        total = len(rows[k])
        if counts is not None:
            n = int((counts or {}).get(k, 0))
        else:
            f = fraction.get(k, 0) if isinstance(fraction, dict) else fraction
            f = float(f)
            if not 0 <= f < 1:
                raise ValueError("fraction must be at least 0 and below 1")
            n = int(math.floor(f * total))
        if n < 0 or n > total:
            raise ValueError(f"can't prune {n} {k}: the model has {total} measured")
        if n:
            out[k] = n
    if not out:
        raise ValueError("that prunes nothing: raise the fraction or the counts")
    return out


def _random_matched(rows: list[dict], chosen: list[dict], gen: torch.Generator) -> list[dict]:
    """As many random units as chosen, in the same layers, layer by layer."""
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r)
    out = []
    for L in sorted({r["layer"] for r in chosen}):
        n = sum(r["layer"] == L for r in chosen)
        pool = by_layer[L]
        out += [pool[i] for i in torch.randperm(len(pool), generator=gen)[:n].tolist()]
    return out


def _delta(model, units: list[dict], picked: dict[str, list[dict]]) -> tuple[dict, dict]:
    """A delta zeroing the picked units' output weights, and the parameters a slim-down could remove."""
    names = {id(p): n for n, p in model.named_parameters()}
    d = model.get_input_embeddings().weight.shape[1]
    delta: dict[str, torch.Tensor] = {}
    removable = {k: 0 for k in picked}

    def zero(p: torch.Tensor, index) -> None:
        n = names[id(p)]
        if n not in delta:
            delta[n] = torch.zeros(p.shape, dtype=p.dtype)
        sl = delta[n]
        sl[index] = -p.detach()[index].cpu()

    for r in picked.get("heads", []):
        u = units[r["layer"]]
        w, n, hd = u["heads"]
        cols = slice(r["index"] * hd, (r["index"] + 1) * hd)
        zero(w.weight, (slice(None), cols))
        removable["heads"] += d * hd + (d * hd if u["q"] is not None else 0)
    for r in picked.get("neurons", []):
        u = units[r["layer"]]
        b = next(b for b, _ in u["dense"] if b.offset <= r["index"] < b.offset + b.count)
        j = r["index"] - b.offset
        zero(b.weight, (slice(None), j))
        removable["neurons"] += d + sum(x.shape[-1] for x in b.readers if x.ndim == 2)
    for r in picked.get("experts", []):
        ex = units[r["layer"]]["experts"]
        e = r["index"]
        zero(ex.down_proj, e)
        bias = ex._parameters.get("down_proj_bias")
        if bias is not None:
            zero(bias, e)
        removable["experts"] += sum(p[e].numel() for p in ex._parameters.values() if p is not None and p.ndim >= 2) + d
    return delta, removable


@torch.no_grad()
def _logprobs(model, ids: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(model(ids[None].to(device(model)), use_cache=False).logits[0, :-1].float(), -1)


@torch.no_grad()
def _cost(model, seqs: dict[str, list[torch.Tensor]], base: dict[str, list[torch.Tensor]]) -> dict:
    """Loss and KL(original || now) per token on each set of texts."""
    out = {}
    for name, ids_list in seqs.items():
        nll = kl = 0.0
        n = 0
        for ids, lp_o in zip(ids_list, base[name]):
            lp = _logprobs(model, ids)
            lp_o = lp_o.to(lp.device).float()
            nxt = ids[1:].to(lp.device)
            nll -= float(lp.gather(-1, nxt[:, None]).sum())
            kl += float((lp_o.exp() * (lp_o - lp)).sum())
            n += len(nxt)
        out[name] = {"loss": nll / max(n, 1), "kl_per_token": kl / max(n, 1), "tokens": n}
    return out


@torch.no_grad()
def _baseline(model, seqs: dict[str, list[torch.Tensor]]) -> tuple[dict, dict]:
    """The original log-probs of every text (on the CPU) and the original loss per set."""
    vocab = model.get_output_embeddings().weight.shape[0]
    size = sum(len(ids) for v in seqs.values() for ids in v) * vocab * 4
    keep = torch.float32 if size <= FP32_BYTES else torch.float16
    base, loss = {}, {}
    for name, ids_list in seqs.items():
        base[name], nll, n = [], 0.0, 0
        for ids in ids_list:
            lp = _logprobs(model, ids)
            nll -= float(lp.gather(-1, ids[1:, None].to(lp.device)).sum())
            n += len(ids) - 1
            base[name].append(lp.to("cpu", keep))
        loss[name] = nll / max(n, 1)
    return base, loss


def prune(model, tok, body: dict) -> dict:
    """Zero the least-used units' output weights (as a delta) and measure the cost against a random control."""
    texts, chat = _texts(body), bool(body.get("chat", False))
    heldout = _texts(body, "heldout", required=False)
    stats, info = _measure(model, tok, texts, chat)
    rows = _usage(stats, info["tokens"])
    counts = _counts(body, rows)
    picked = {k: rows[k][:n] for k, n in counts.items()}
    gen = torch.Generator().manual_seed(int(body.get("seed", 0)))
    control = {k: _random_matched(rows[k], picked[k], gen) for k in picked}
    units = info["units"]
    delta, removable = _delta(model, units, picked)
    control_delta, _ = _delta(model, units, control)

    seqs = {"texts": [_ids(tok, t, chat) for t in texts], "reference": [_ids(tok, t, False) for t in REFERENCE_TEXTS]}
    if heldout:
        seqs["heldout"] = [_ids(tok, t, chat) for t in heldout]
    seqs = {k: [s for s in v if len(s) > 1] for k, v in seqs.items()}
    base, before = _baseline(model, seqs)
    with applied(model, delta):
        after = _cost(model, seqs, base)
    with applied(model, control_delta):
        ctrl = _cost(model, seqs, base)
    cost = {k: {"loss_before": round(before[k], 5), "loss_after": round(after[k]["loss"], 5),
                "kl_per_token": round(after[k]["kl_per_token"], 6),
                "control_loss_after": round(ctrl[k]["loss"], 5),
                "control_kl_per_token": round(ctrl[k]["kl_per_token"], 6), "tokens": after[k]["tokens"]}
            for k in seqs}
    total = sum(p.numel() for p in model.parameters())
    return {
        "pruned": {k: [_round(r) for r in v] for k, v in picked.items()},
        "control": {k: [{"layer": r["layer"], "index": r["index"]} for r in v] for k, v in control.items()},
        "counts": counts,
        "cost": cost,
        "removable_parameters": {**removable, "total": sum(removable.values()), "model_parameters": total,
                                 "share": round(sum(removable.values()) / max(total, 1), 6)},
        "delta": delta,
        "notes": "The delta zeroes output weights only; the units still compute. Loss on 'texts' is in-sample "
                 "(the same texts chose the units); held-out and reference text show what else it costs. The "
                 "control prunes as many random units in the same layers. Removable parameters count output "
                 "columns plus input rows (query rows for heads; key/value heads shared by grouped attention "
                 "are not counted) and are what a later physical slim-down could drop.",
    }


STEPS = {"profile": profile, "prune": prune}
