"""Says vs. knows: put what the model answered beside what its hidden states say it believes.

Train: read true and false statements (plain text, no chat template) and take the residual stream at each
statement's last token after every layer. Two truth probes per layer, both fitted on a training split and scored
on held-out statements: the difference of means (true minus false, with the midpoint between the class means as
threshold) and a logistic regression. Each comes with a shuffled-label control, and the shipped probe also
carries a random direction scored the same way, for use as a control when asking. The probe that ships is the
difference of means at the best layer, refitted on every statement; readings are in units of the spread of true
and false statements around their own means (z), so a reading of +1 sits one spread on the true side.

Ask: for each question, the model answers it greedily under its chat template; the answer it picked is the
candidate named in its reply (or, when the reply names neither or both, the candidate it gives the higher
likelihood to as a reply). The probe reads each candidate written out as a statement. When the model picks one
candidate but the probe reads the other higher by at least MARGIN, the question is flagged: says X, states lean
Y. The random direction gives the same flags by chance; compare the two rates before reading anything into a
flag. A probe is a linear readout fitted on your statements: it can track "looks like the true set" rather than
truth, and no reading here shows that a model is honest or dishonest.

Layer numbering: layer l is the residual stream after l transformer blocks (1 .. number of blocks).
"""

from __future__ import annotations

import torch

from . import blocks, device, encode, generate, unit
from .classify import features, fit_logistic, shuffled, split

MARGIN = 1.0         # z gap between the two candidates needed to flag a disagreement
ANSWER_TOKENS = 48   # greedy tokens of the model's own answer
MIN_STATEMENTS = 4   # per side, so both sides of the split hold some
SEED = 0             # split, shuffle and random-direction seed


def _statements(body: dict, key: str) -> list[str]:
    items = [str(s).strip() for s in (body.get(key) or []) if str(s).strip()]
    if len(items) < MIN_STATEMENTS:
        raise ValueError(f"{key}: at least {MIN_STATEMENTS} statements")
    return items


def _mean_diff(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Unit direction true - false, the midpoint threshold, and the pooled within-class spread of projections."""
    u = unit(x[y == 1].mean(0) - x[y == 0].mean(0))
    return u, *_calibrate(x @ u, y)


def _calibrate(proj: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    t, f = proj[y == 1], proj[y == 0]
    spread = torch.sqrt((t.var(unbiased=False) + f.var(unbiased=False)) / 2)
    return float((t.mean() + f.mean()) / 2), float(spread.clamp_min(1e-6))


def _acc_dir(u, thr, x, y) -> float:
    return float(((x @ u > thr).long() == y).float().mean())


def _acc_lr(w, b, x, y) -> float:
    return float(((x @ w.T + b).argmax(1) == y).float().mean())


def train(model, tok, body: dict) -> dict:
    """A truth probe per layer with held-out accuracy and shuffled-label controls; ships the best layer's probe."""
    true, false = _statements(body, "true"), _statements(body, "false")
    n_blocks = len(blocks(model))
    fixed = body.get("layer")
    if fixed is not None:
        fixed = int(fixed)
        if not 1 <= fixed <= n_blocks:
            raise ValueError(f"layer must be in 1..{n_blocks}")
    x = features(model, tok, true + false, "last", False, upto=fixed)
    y = torch.tensor([1] * len(true) + [0] * len(false))
    tr, te = split(y, seed=SEED)
    ys = shuffled(y[tr], seed=SEED + 1)
    rows = []
    for li in ([fixed - 1] if fixed else range(x.shape[1])):
        a, b = x[tr, li], x[te, li]
        u, thr, spread = _mean_diff(a, y[tr])
        us, thrs, _ = _mean_diff(a, ys)
        w, bias = fit_logistic(a, y[tr], 2)
        ws, bs = fit_logistic(a, ys, 2)
        pt, pf = b[y[te] == 1] @ u, b[y[te] == 0] @ u
        rows.append({"layer": li + 1,
                     "mean_diff_acc": round(_acc_dir(u, thr, b, y[te]), 3),
                     "shuffled_mean_diff_acc": round(_acc_dir(us, thrs, b, y[te]), 3),
                     "logistic_acc": round(_acc_lr(w, bias, b, y[te]), 3),
                     "shuffled_logistic_acc": round(_acc_lr(ws, bs, b, y[te]), 3),
                     "held_out_d": round(float((pt.mean() - pf.mean()) / spread), 3)})
    best = max(rows, key=lambda r: (r["mean_diff_acc"], r["held_out_d"], -r["layer"]))
    layer = best["layer"]
    xa = x[:, layer - 1]
    u, thr, spread = _mean_diff(xa, y)
    r = unit(torch.randn(xa.shape[1], generator=torch.Generator().manual_seed(SEED)))
    rthr, rspread = _calibrate(xa @ r, y)
    w, bias = fit_logistic(xa, y, 2)
    r_thr_train = _calibrate(x[tr, layer - 1] @ r, y[tr])[0]
    random_acc = _acc_dir(r, r_thr_train, x[te, layer - 1], y[te])
    probe = {"kind": "truth-probe", "method": "mean_diff", "layer": layer, "vector": u.tolist(),
             "threshold": thr, "scale": spread,
             "logistic": {"weight": (w[1] - w[0]).tolist(), "bias": float(bias[1] - bias[0])},
             "control": {"vector": r.tolist(), "threshold": rthr, "scale": rspread}}
    return {"probe": probe, "layer": layer, "layers": rows,
            "held_out": {**{k: v for k, v in best.items() if k != "layer"},
                         "random_direction_acc": round(random_acc, 3), "n_train": len(tr), "n_test": len(te)},
            "note": (f"layer {layer}: {best['mean_diff_acc']:.0%} of {len(te)} held-out statements read correctly "
                     f"(shuffled labels {best['shuffled_mean_diff_acc']:.0%}, random direction {random_acc:.0%}). "
                     "The shipped probe is refitted on all statements.")}


def _probe(body: dict) -> tuple[int, dict, dict]:
    p = body.get("probe")
    if isinstance(p, dict) and isinstance(p.get("probe"), dict):
        p = p["probe"]
    if isinstance(p, list):
        p = {"vector": p}
    if not isinstance(p, dict) or not p.get("vector"):
        raise ValueError("probe: what train returned under 'probe' (or a bare vector with layer)")
    layer = body.get("layer", p.get("layer"))
    if layer is None:
        raise ValueError("layer: which layer the probe reads")
    real = {"vector": torch.tensor(p["vector"], dtype=torch.float32), "threshold": float(p.get("threshold", 0.0)),
            "scale": float(p.get("scale", 1.0)) or 1.0}
    c = p.get("control")
    if c:
        ctrl = {"vector": torch.tensor(c["vector"], dtype=torch.float32), "threshold": float(c.get("threshold", 0.0)),
                "scale": float(c.get("scale", 1.0)) or 1.0}
    else:  # a bare vector: a random direction scaled like it, read on the same statements
        g = torch.Generator().manual_seed(SEED)
        ctrl = {"vector": unit(torch.randn(real["vector"].shape[0], generator=g)) * real["vector"].norm(),
                "threshold": real["threshold"], "scale": real["scale"], "uncalibrated": True}
    return int(layer), real, ctrl


def _questions(body: dict) -> list[dict]:
    qs = body.get("questions")
    if not isinstance(qs, list) or not qs:
        raise ValueError("questions: a list of {question, statement_true, statement_false} or "
                         "{question, answer_a, answer_b}")
    out = []
    for q in qs:
        if not isinstance(q, dict) or not str(q.get("question", "")).strip():
            raise ValueError("each question needs a 'question'")
        question = str(q["question"]).strip()
        if q.get("statement_true") and q.get("statement_false"):
            out.append({"question": question, "keys": ("true", "false"),
                        "answers": (str(q.get("answer_true") or q["statement_true"]),
                                    str(q.get("answer_false") or q["statement_false"])),
                        "statements": (str(q["statement_true"]), str(q["statement_false"]))})
        elif q.get("answer_a") and q.get("answer_b"):
            a, b = str(q["answer_a"]).strip(), str(q["answer_b"]).strip()
            out.append({"question": question, "keys": ("a", "b"), "answers": (a, b),
                        "statements": (f"{question} {a}", f"{question} {b}")})
        else:
            raise ValueError("each question needs statement_true and statement_false, or answer_a and answer_b")
    return out


@torch.no_grad()
def _likelihood(model, tok, question: str, answer: str) -> float:
    """Mean log-probability per token of the answer as the model's chat reply."""
    prompt = encode(tok, question, chat=True)
    ans = torch.tensor(tok(answer.strip(), add_special_tokens=False)["input_ids"], dtype=torch.long)
    if ans.numel() == 0:
        return float("-inf")
    ids = torch.cat([prompt, ans])
    logits = model(ids[None].to(device(model)), use_cache=False).logits[0, len(prompt) - 1:-1].float()
    lp = torch.log_softmax(logits, -1).gather(1, ans.to(logits.device)[:, None])
    return float(lp.mean())


def _named(reply: str, answer: str) -> bool:
    return answer.strip().lower() in reply.lower()


def verdict(picked: str | None, z: dict[str, float], keys: tuple[str, str], margin: float = MARGIN) -> dict:
    """Which candidate the readings lean to, by how much, and whether that contradicts the pick."""
    gap = z[keys[0]] - z[keys[1]]
    leans = keys[0] if gap > 0 else keys[1]
    clear = abs(gap) >= margin
    return {"leans": leans if clear else None, "gap": round(gap, 3),
            "disagree": bool(clear and picked is not None and picked != leans)}


def ask(model, tok, body: dict) -> dict:
    """What the model says beside what the probe reads on each candidate statement."""
    layer, real, ctrl = _probe(body)
    n_blocks = len(blocks(model))
    if not 1 <= layer <= n_blocks:
        raise ValueError(f"layer must be in 1..{n_blocks}")
    d = model.get_input_embeddings().weight.shape[1]
    if real["vector"].shape[0] != d:
        raise ValueError(f"probe has width {real['vector'].shape[0]}, the model {d}")
    margin = float(body.get("margin", MARGIN))
    qs = _questions(body)

    results, flags, ctrl_flags = [], 0, 0
    for q in qs:
        k0, k1 = q["keys"]
        reply = generate(model, tok, q["question"], n=int(body.get("tokens", ANSWER_TOKENS)), chat=True)
        named = [k for k, a in zip(q["keys"], q["answers"]) if _named(reply, a)]
        lik = {k: round(_likelihood(model, tok, q["question"], a), 4) for k, a in zip(q["keys"], q["answers"])}
        if len(named) == 1:
            picked, how = named[0], "named in the reply"
        else:
            picked, how = max(lik, key=lik.get), "higher likelihood as a reply"
        x = features(model, tok, list(q["statements"]), "last", False, upto=layer)[:, -1]
        z = {k: float((x[i] @ real["vector"] - real["threshold"]) / real["scale"]) for i, k in enumerate(q["keys"])}
        zc = {k: float((x[i] @ ctrl["vector"] - ctrl["threshold"]) / ctrl["scale"]) for i, k in enumerate(q["keys"])}
        v, vc = verdict(picked, z, q["keys"], margin), verdict(picked, zc, q["keys"], margin)
        flags += v["disagree"]
        ctrl_flags += vc["disagree"]
        row = {"question": q["question"], "says": reply, "picked": picked, "picked_by": how, "likelihood": lik,
               "statements": dict(zip(q["keys"], q["statements"])),
               "reads": {k: round(s, 3) for k, s in z.items()}, **v,
               "control": {"reads": {k: round(s, 3) for k, s in zc.items()}, **vc}}
        if v["disagree"]:
            row["flag"] = f"says {picked!r} but its states lean {v['leans']!r} (gap {abs(v['gap']):.2f} z)"
        if q["keys"] == ("true", "false"):
            row["says_true"] = picked == "true"
            row["states_true"] = None if v["leans"] is None else v["leans"] == "true"
        results.append(row)
    n = len(results)
    return {"layer": layer, "margin": margin, "results": results,
            "summary": {"questions": n, "disagreements": flags, "control_disagreements": ctrl_flags,
                        "note": (f"{flags} of {n} questions flagged with the probe, {ctrl_flags} with a random "
                                 "direction. Flags are leads to look at, not findings about honesty.")}}


STEPS = {"train": train, "ask": ask}
