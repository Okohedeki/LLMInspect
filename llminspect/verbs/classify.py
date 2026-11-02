"""Classify: train a tiny probe on the model's own activations and ship it as a head that runs for free.

Give a few texts per label. Every text is read once; the residual stream after each layer is pooled (the last
token, or the mean over tokens) and a multinomial logistic regression is fitted on it, layer by layer. Each
label's texts are split into a training part and a held-out part (a quarter, at least one text), and the score is
accuracy on the held-out part. Beside it sits a control: the same fit on training labels shuffled at random,
scored on the true held-out labels. A layer where the shuffled fit scores as well as the real one has nothing
to say about the labels.

The head that ships is refitted on every example at the chosen layer, with its weights in the residual's own
units (standardization folded in), so it is a plain softmax(W x + b). Running it costs one forward pass that
stops right after the chosen layer.

Layer numbering: layer l is the residual stream after l transformer blocks (1 .. number of blocks).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from . import blocks, device, encode

HELD_OUT = 0.25      # share of each label's texts held out for scoring
L2 = 1e-2            # ridge penalty on the standardized weights
ITERS = 100          # L-BFGS iterations per fit
SEED = 0             # split and shuffle seed, so a rerun gives the same numbers
MAX_TEXTS = 4000     # texts per call, all labels together
POOLINGS = ("last", "mean")


class _Stop(Exception):
    """Raised by the capture hook to end a forward pass after the layer we need."""


@torch.no_grad()
def layer_states(model, ids: torch.Tensor, upto: int | None = None) -> torch.Tensor:
    """The residual stream after each block, [layers, tokens, d_model] float32, index i = after block i+1.
    With upto, the pass stops after block upto (so only upto rows come back)."""
    stack = blocks(model)
    n = len(stack) if upto is None else upto
    if not 1 <= n <= len(stack):
        raise ValueError(f"layer must be in 1..{len(stack)}")
    got: list[torch.Tensor] = []

    def grab(_m, _inp, out):
        got.append((out[0] if isinstance(out, tuple) else out)[0].float())
        if len(got) == n:
            raise _Stop

    hooks = [stack[i].register_forward_hook(grab) for i in range(n)]
    try:
        model(ids[None].to(device(model)), use_cache=False)
    except _Stop:
        pass
    finally:
        for h in hooks:
            h.remove()
    return torch.stack(got)


def pool(states: torch.Tensor, pooling: str) -> torch.Tensor:
    """[layers, tokens, d] -> [layers, d]. 'mean' skips the first token (it often carries an outsized norm)."""
    if pooling == "last":
        return states[:, -1]
    return states[:, 1:].mean(1) if states.shape[1] > 1 else states[:, 0]


def features(model, tok, texts: list[str], pooling: str = "last", chat: bool = False,
             upto: int | None = None) -> torch.Tensor:
    """Pooled residuals for each text, [texts, layers, d] float32 on the CPU."""
    out = []
    for t in texts:
        ids = encode(tok, t, chat)
        if ids.numel() == 0:
            raise ValueError(f"text {t!r} has no tokens")
        out.append(pool(layer_states(model, ids, upto), pooling).cpu())
    return torch.stack(out)


def fit_logistic(x: torch.Tensor, y: torch.Tensor, classes: int, l2: float = L2,
                 iters: int = ITERS) -> tuple[torch.Tensor, torch.Tensor]:
    """Multinomial logistic regression on [n, d] features. Returns (W [classes, d], b [classes]) in raw units."""
    x = x.float().cpu()
    mu, sd = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mu) / sd
    w = torch.zeros(z.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(z @ w + b, y) + l2 * (w * w).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    w_raw = (w.detach() / sd[:, None]).T.contiguous()
    return w_raw, (b.detach() - mu @ w_raw.T)


def split(y: torch.Tensor, share: float = HELD_OUT, seed: int = SEED) -> tuple[torch.Tensor, torch.Tensor]:
    """Stratified train/held-out indices: each class keeps at least one example on each side."""
    g = torch.Generator().manual_seed(seed)
    train, test = [], []
    for c in y.unique().tolist():
        idx = (y == c).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)]
        k = max(1, round(share * len(idx)))
        test.append(idx[:k])
        train.append(idx[k:])
    return torch.cat(train), torch.cat(test)


def shuffled(y: torch.Tensor, seed: int = SEED + 1) -> torch.Tensor:
    return y[torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))]


def _examples(body: dict) -> tuple[list[str], list[str], torch.Tensor]:
    ex = body.get("examples")
    if not isinstance(ex, dict):
        raise ValueError("examples: an object of {label: [texts]}")
    labels, texts, ys = [], [], []
    for label, items in ex.items():
        items = [str(t) for t in (items or []) if str(t).strip()]
        if not items:
            continue
        if len(items) < 2:
            raise ValueError(f"label {label!r} needs at least two texts (one to train on, one to hold out)")
        labels.append(str(label))
        texts += items
        ys += [len(labels) - 1] * len(items)
    if len(labels) < 2:
        raise ValueError("give at least two labels with texts")
    if len(texts) > MAX_TEXTS:
        raise ValueError(f"at most {MAX_TEXTS} texts")
    return labels, texts, torch.tensor(ys)


def _acc(w, b, x, y) -> float:
    return float(((x @ w.T + b).argmax(1) == y).float().mean())


def train(model, tok, body: dict) -> dict:
    """Fit a head per layer, score it on held-out texts beside a shuffled-label control, and ship the best one."""
    labels, texts, y = _examples(body)
    pooling = body.get("pooling", "last")
    if pooling not in POOLINGS:
        raise ValueError("pooling must be 'last' or 'mean'")
    chat = bool(body.get("chat", False))
    n_blocks = len(blocks(model))
    fixed = body.get("layer")
    if fixed is not None:
        fixed = int(fixed)
        if not 1 <= fixed <= n_blocks:
            raise ValueError(f"layer must be in 1..{n_blocks}")
    x = features(model, tok, texts, pooling, chat, upto=fixed)  # [n, layers, d]
    tr, te = split(y)
    y_shuf = shuffled(y[tr])
    C = len(labels)
    scores = []
    for li in ([fixed - 1] if fixed else range(x.shape[1])):
        w, b = fit_logistic(x[tr, li], y[tr], C)
        ws, bs = fit_logistic(x[tr, li], y_shuf, C)
        scores.append({"layer": li + 1, "acc": round(_acc(w, b, x[te, li], y[te]), 3),
                       "shuffled_acc": round(_acc(ws, bs, x[te, li], y[te]), 3),
                       "train_acc": round(_acc(w, b, x[tr, li], y[tr]), 3), "_wb": (w, b)})
    best = max(scores, key=lambda s: (s["acc"], s["acc"] - s["shuffled_acc"], -s["layer"]))
    layer = best["layer"]
    w, b = best["_wb"]
    pred = (x[te, layer - 1] @ w.T + b).argmax(1)
    confusion = [[int(((y[te] == i) & (pred == j)).sum()) for j in range(C)] for i in range(C)]
    W, B = fit_logistic(x[:, layer - 1], y, C)  # the shipped head sees every example
    majority = float(torch.bincount(y[te]).max()) / len(te)
    return {
        "head": {"kind": "linear-probe", "layer": layer, "pooling": pooling, "chat": chat, "labels": labels,
                 "d_model": int(W.shape[1]), "weight": W.tolist(), "bias": B.tolist()},
        "layer": layer, "labels": labels,
        "layers": [{k: v for k, v in s.items() if k != "_wb"} for s in scores],
        "held_out": {"acc": best["acc"], "shuffled_acc": best["shuffled_acc"], "majority": round(majority, 3),
                     "n_train": len(tr), "n_test": len(te)},
        "confusion": {"rows": "true label", "cols": "predicted label", "labels": labels, "matrix": confusion},
        "note": (f"held-out accuracy {best['acc']:.0%} at layer {layer} vs {best['shuffled_acc']:.0%} with shuffled "
                 f"labels and {majority:.0%} for always guessing the commonest label, on {len(te)} held-out texts. "
                 "The shipped head is refitted on all texts; scores come from the split."),
    }


def _head(head) -> tuple[int, str, bool, list[str], torch.Tensor, torch.Tensor]:
    if not isinstance(head, dict):
        raise ValueError("head: what train returned under 'head'")
    if "head" in head and isinstance(head["head"], dict):
        head = head["head"]
    try:
        w = torch.tensor(head["weight"], dtype=torch.float32)
        b = torch.tensor(head["bias"], dtype=torch.float32)
        layer, labels = int(head["layer"]), [str(s) for s in head["labels"]]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"head is missing or has a bad field: {e}") from None
    pooling = head.get("pooling", "last")
    if pooling not in POOLINGS:
        raise ValueError("head pooling must be 'last' or 'mean'")
    if w.ndim != 2 or w.shape[0] != len(labels) or b.shape != (len(labels),):
        raise ValueError("head weight must be [labels, d_model] and bias [labels]")
    return layer, pooling, bool(head.get("chat", False)), labels, w, b


def run(model, tok, body: dict) -> dict:
    """Label probabilities per text, from a forward pass that stops after the head's layer."""
    layer, pooling, chat, labels, w, b = _head(body.get("head"))
    texts = body.get("texts")
    if isinstance(texts, str):
        texts = [texts]
    if not texts:
        raise ValueError("texts: a list of strings")
    d = model.get_input_embeddings().weight.shape[1]
    if w.shape[1] != d:
        raise ValueError(f"head was trained on a model of width {w.shape[1]}, this one is {d}")
    dev = device(model)
    w, b = w.to(dev), b.to(dev)
    out = []
    for t in texts:
        x = pool(layer_states(model, encode(tok, str(t), chat), upto=layer), pooling)[-1]
        p = torch.softmax(x @ w.T + b, -1).cpu()
        out.append({"text": t, "label": labels[int(p.argmax())],
                    "probs": {lab: round(float(v), 5) for lab, v in zip(labels, p)}})
    return {"layer": layer, "labels": labels, "results": out}


def export(model, tok, body: dict) -> dict:
    """Write the head as <path>.safetensors (weight, bias) and <path>.json (how to use it)."""
    from safetensors.torch import save_file

    layer, pooling, chat, labels, w, b = _head(body.get("head"))
    if not body.get("path"):
        raise ValueError("path: where to write the head (a file name without extension is fine)")
    p = Path(str(body["path"])).expanduser()
    if p.suffix in (".safetensors", ".json"):
        p = p.with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    st, js = p.parent / f"{p.name}.safetensors", p.parent / f"{p.name}.json"
    save_file({"weight": w.contiguous(), "bias": b.contiguous()}, str(st))
    name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
    meta = {"kind": "linear-probe", "model": name, "layer": layer, "pooling": pooling, "chat": chat,
            "labels": labels, "d_model": int(w.shape[1]), "weights": st.name,
            "usage": (f"Run the model on the text{' under its chat template' if chat else ''}, take the output of "
                      f"transformer block {layer} (counting from 1; hidden_states[{layer}] in Hugging Face, except "
                      f"for the last block, whose hidden state there has the final norm applied), "
                      f"{'at the last token' if pooling == 'last' else 'averaged over all tokens but the first'}, "
                      "in float32 as x; label probabilities are softmax(weight @ x + bias), in the order of labels.")}
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8", newline="\n")
    return {"paths": {"weights": str(st), "meta": str(js)},
            "bytes": st.stat().st_size + js.stat().st_size}


STEPS = {"train": train, "run": run, "export": export}
