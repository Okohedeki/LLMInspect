"""Features: read, compare and steer a model through a sparse dictionary of its residual stream.

A sparse dictionary (a sparse autoencoder) rewrites the residual stream x at one layer as a few active features:
f = act((x - b_dec if subtracted) W_enc + b_enc) and x ~ f W_dec + b_dec, where act is a ReLU ("standard"), a
per-feature threshold that keeps or zeroes each pre-activation ("jumprelu"), or a ReLU over only the k largest
("topk"). Each feature has a decoder direction (its row of W_dec) and, in many releases, a text label. This verb
only reads a dictionary that is already in a local folder; it never downloads one.

info    what the dictionary is and how well it fits this model: the fraction of the residual's variance its
        reconstruction explains, and the next-token loss with the residual replaced by the reconstruction, beside
        two controls: the residual plus an error of the same size in a random direction, and the residual zeroed.
        loss_recovered = (zero - reconstruction) / (zero - clean). A dictionary made for another model or layer
        shows as low variance explained and a loss jump far above the random-error control.
read    per token, the most active features; over all texts, the features most active on average, each with how
        often it fires (a feature that fires on every token says little about any one).
diff    dictionary-based model diffing: the same texts through the loaded model and through its base (the base's
        tensors copied in, the loaded ones restored exactly after); per feature, the change in mean activation.
        Controls: a bootstrap standard error over texts, and a noise floor from shuffling which model is which
        text by text (random signs on the paired differences): the 95th percentile of the largest |change| any
        feature reaches by chance.
steer   greedy generations with a feature's decoder direction added to the residual at its layer, at each
        strength (in units of the residual's mean norm there), beside a random other feature added at the same
        norm, and the unsteered answer.
"""

from __future__ import annotations

import contextlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from . import blocks, device, encode, generate, unit
from .audit import base_weights

NEUTRAL = [
    "The library opens at nine and closes early on Sundays.",
    "She put the kettle on and looked out at the rain.",
    "To make a paper plane, fold the sheet in half along its length.",
    "The train was ten minutes late, so we missed the first act.",
    "Tomatoes grow best with plenty of sun and regular watering.",
    "He wrote a short note to thank his neighbour for the help.",
    "A for loop repeats a block of code a set number of times.",
    "The river runs through the valley and into a wide lake.",
    "Our meeting moved to Thursday afternoon because of the holiday.",
    "Bread rises when yeast turns sugar into gas.",
    "The museum has a new room of old maps and globes.",
    "Remember to save your work before closing the program.",
]
PROMPTS = ["Tell me about your weekend.", "Describe a city you like.", "What should I cook tonight?"]
TOP = 10            # features listed per token by read
MAX_TOP = 100
MOST = 20           # features listed over all texts (read) and per side (diff)
MAX_TEXTS = 64
MAX_TOKENS = 256    # tokens per text; longer texts are cut
BOOT = 200          # bootstrap resamples of the texts (diff)
SHUFFLES = 200      # random-sign shuffles for diff's noise floor
FLOOR_Q = 0.95      # noise-floor quantile
STRENGTHS = [2.0, 4.0, 8.0]  # default steering strengths, in mean residual norms
MAX_STRENGTHS = 8
GEN_TOKENS = 32     # default tokens generated per steered answer
SEED = 0
CFG_NAMES = ("cfg.json", "config.json")
LABEL_NAMES = ("labels.json", "explanations.json", "labels.jsonl", "explanations.jsonl")
LABEL_TEXT = ("label", "description", "explanation", "text")
LABEL_INDEX = ("index", "feature", "idx", "id")


# ---- the dictionary ---------------------------------------------------------------------------

@dataclass
class Dictionary:
    """A single-layer residual dictionary; layer is the block whose output it reads (-1: the embeddings)."""
    W_enc: torch.Tensor      # [d_model, n]
    b_enc: torch.Tensor      # [n]
    W_dec: torch.Tensor      # [n, d_model]
    b_dec: torch.Tensor      # [d_model]
    architecture: str        # standard | jumprelu | topk
    threshold: torch.Tensor | None = None
    k: int | None = None
    subtract_b_dec: bool = False
    layer: int | None = None
    hook: str | None = None
    labels: dict[int, str] = field(default_factory=dict)
    format: str = ""

    @property
    def d_model(self) -> int:
        return self.W_enc.shape[0]

    @property
    def n(self) -> int:
        return self.W_enc.shape[1]

    def to(self, dev) -> "Dictionary":
        for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(dev, torch.float32))
        return self

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Feature activations for residuals x [tokens, d_model], float32."""
        pre = (x - self.b_dec if self.subtract_b_dec else x) @ self.W_enc + self.b_enc
        if self.architecture == "jumprelu":
            return pre * (pre > self.threshold)
        if self.architecture == "topk":
            v, i = pre.topk(min(self.k, self.n), -1)
            return torch.zeros_like(pre).scatter(-1, i, v.relu())
        return pre.relu()

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec


def _read_json(f: Path):
    try:
        if f.suffix == ".jsonl":
            return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ValueError(f"can't read {f.name}: {e}") from None


def _cfg(folder: Path) -> dict:
    for name in CFG_NAMES:
        if (folder / name).is_file():
            cfg = _read_json(folder / name)
            return cfg if isinstance(cfg, dict) else {}
    return {}


def _labels(folder: Path) -> dict[int, str]:
    """Feature index -> label from labels.json / explanations.json(l): {index: text}, [text, ...] or
    [{index|feature: i, label|description|explanation: text}, ...]."""
    for name in LABEL_NAMES:
        f = folder / name
        if not f.is_file():
            continue
        raw, out = _read_json(f), {}
        items = raw.items() if isinstance(raw, dict) else enumerate(raw if isinstance(raw, list) else [])
        for key, val in items:
            if isinstance(val, dict):
                key = next((val[k] for k in LABEL_INDEX if k in val), key)
                val = next((val[k] for k in LABEL_TEXT if val.get(k)), None)
            try:
                if isinstance(val, str) and val.strip():
                    out[int(key)] = val.strip()
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _site(text: str) -> int | None:
    """The block whose output a hook name or folder name points at, or None."""
    m = re.search(r"(\d+)\.hook_resid_(pre|post|mid)", text)
    if m:
        n, kind = int(m.group(1)), m.group(2)
        if kind == "mid":
            raise ValueError(f"{text}: a dictionary of the mid-layer residual (between attention and MLP) isn't "
                             "supported, only a layer's input or output")
        return n - 1 if kind == "pre" else n
    if re.search(r"hook_(mlp_out|attn_out|z|q|k|v|post)\b|mlp_out|attn_out|\.mlp\b|\.attn\b|self_attn", text):
        raise ValueError(f"{text}: this dictionary reads an attention or MLP output, not the residual stream")
    m = re.search(r"(?:^|[./_\-])(?:layers?|blocks?|resid(?:ual)?)[._\-]?(\d+)(?!\d)", text, re.I)
    return int(m.group(1)) if m else None


def _layer(cfg: dict, folder: Path, override) -> tuple[int | None, str | None]:
    if override is not None:
        try:
            return int(override), None
        except (TypeError, ValueError):
            raise ValueError("layer must be a whole number (the block whose output the dictionary reads)") from None
    for key in ("hook_name", "hook_point", "hookpoint", "hook"):
        if isinstance(cfg.get(key), str):
            site = _site(cfg[key])
            if site is not None:
                return site, cfg[key]
    for key in ("layer", "hook_layer"):
        if isinstance(cfg.get(key), int):
            return cfg[key], None
    for part in [folder, *list(folder.parents)[:3]]:
        site = _site(part.name)
        if site is not None:
            return site, part.name
    return None, None


def _tensors(p: Path) -> tuple[dict[str, torch.Tensor], str, Path]:
    folder = p if p.is_dir() else p.parent
    if p.is_file():
        files = [p]
    elif (folder / "sae_weights.safetensors").is_file():
        files = [folder / "sae_weights.safetensors"]
    elif (folder / "params.npz").is_file():
        files = [folder / "params.npz"]
    else:
        files = sorted(folder.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"{folder}: expected sae_weights.safetensors + cfg.json, params.npz, or one "
                             f".safetensors file with encoder/decoder weights; found {len(files)} .safetensors files")
    f = files[0]
    try:
        if f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as z:
                raw = {k: torch.from_numpy(np.asarray(z[k], dtype=np.float32)) for k in z.files}
            return raw, "params.npz", folder
        if f.suffix == ".safetensors":
            raw = {k: v.float() for k, v in load_file(str(f)).items()}
            return raw, ("cfg.json + sae_weights.safetensors" if f.name == "sae_weights.safetensors"
                         else "safetensors (encoder/decoder)"), folder
    except Exception as e:  # noqa: BLE001 - a corrupt file should read as a plain message
        raise ValueError(f"can't read {f.name}: {e}") from None
    raise ValueError(f"{f.name}: expected a .safetensors or .npz file")


def _get(raw: dict, *names):
    for n in names:
        if n in raw:
            return raw[n], n
    return None, None


def load(path, layer=None) -> Dictionary:
    """Read a single-layer residual dictionary from a local folder (or file). Layouts:
    cfg.json + sae_weights.safetensors   W_enc [d, n], b_enc, W_dec [n, d], b_dec, threshold (jumprelu);
                                         cfg: d_in, d_sae, hook_name, architecture, k, apply_b_dec_to_input
    params.npz                           W_enc, W_dec, b_enc, b_dec, threshold
    one *.safetensors                    encoder.weight [n, d], encoder.bias, decoder.weight [d, n],
                                         decoder.bias or b_dec, threshold; an optional cfg.json beside it (k, ...)
    The layer comes from `layer`, else the cfg's hook name or layer, else a folder name like layer_12."""
    if not path:
        raise ValueError("give path: the local folder of the dictionary")
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise ValueError(f"{p} doesn't exist (dictionaries are read from local folders only)")
    raw, fmt, folder = _tensors(p)
    cfg = _cfg(folder)
    enc, enc_name = _get(raw, "W_enc", "encoder.weight", "enc.weight")
    dec, dec_name = _get(raw, "W_dec", "decoder.weight", "dec.weight")
    if enc is None or dec is None or enc.ndim != 2 or dec.ndim != 2:
        raise ValueError(f"{p}: no 2-D encoder and decoder weights (W_enc/W_dec or encoder.weight/decoder.weight)")
    if enc_name != "W_enc":
        enc = enc.T          # stored as a linear layer, [n, d]
    if dec_name != "W_dec":
        dec = dec.T          # stored as a linear layer, [d, n]
    d_in = cfg.get("d_in")
    if isinstance(d_in, int) and enc.shape[0] != d_in and enc.shape[1] == d_in:
        enc = enc.T
    if tuple(dec.shape) != (enc.shape[1], enc.shape[0]) and tuple(dec.T.shape) == (enc.shape[1], enc.shape[0]):
        dec = dec.T
    d, n = enc.shape
    if tuple(dec.shape) != (n, d):
        raise ValueError(f"{p}: encoder {list(enc.shape)} and decoder {list(dec.shape)} don't match")
    b_enc, _ = _get(raw, "b_enc", "encoder.bias", "enc.bias")
    b_dec, b_dec_name = _get(raw, "b_dec", "pre_bias", "decoder.bias", "dec.bias")
    thr, thr_name = _get(raw, "threshold", "log_threshold", "jumprelu.threshold")
    if thr is not None and thr_name == "log_threshold":
        thr = thr.exp()
    b_enc = torch.zeros(n) if b_enc is None else b_enc.reshape(-1)
    b_dec = torch.zeros(d) if b_dec is None else b_dec.reshape(-1)
    if b_enc.shape[0] != n or b_dec.shape[0] != d or (thr is not None and thr.numel() not in (1, n)):
        raise ValueError(f"{p}: bias or threshold sizes don't match {n} features of width {d}")
    act = cfg.get("activation_fn") or cfg.get("activation_fn_str")
    kwargs = cfg.get("activation_fn_kwargs") if isinstance(cfg.get("activation_fn_kwargs"), dict) else {}
    k = cfg.get("k") or kwargs.get("k")
    arch = str(cfg.get("architecture") or "").lower()
    if arch not in ("standard", "jumprelu", "topk"):
        arch = "topk" if (k or act == "topk") else "jumprelu" if thr is not None else "standard"
    if arch == "topk" and not k:
        raise ValueError(f"{p}: a topk dictionary needs k in its cfg.json")
    if arch == "jumprelu" and thr is None:
        raise ValueError(f"{p}: a jumprelu dictionary needs a threshold tensor")
    if "apply_b_dec_to_input" in cfg:
        sub = bool(cfg["apply_b_dec_to_input"])
    elif fmt == "params.npz":
        sub = False
    elif fmt.startswith("cfg.json"):
        sub = arch != "jumprelu"
    else:
        sub = b_dec_name in ("b_dec", "pre_bias")
    site, hook = _layer(cfg, folder, layer)
    return Dictionary(W_enc=enc.contiguous(), b_enc=b_enc, W_dec=dec.contiguous(), b_dec=b_dec,
                      architecture=arch, threshold=None if thr is None else thr.reshape(-1).expand(n).clone(),
                      k=int(k) if arch == "topk" else None, subtract_b_dec=sub, layer=site, hook=hook,
                      labels=_labels(folder), format=fmt)


def _open(model, body: dict) -> Dictionary:
    D = load(body.get("path"), body.get("layer"))
    d = model.get_input_embeddings().weight.shape[1]
    if D.d_model != d:
        raise ValueError(f"the dictionary is {D.d_model} wide but the model's residual stream is {d}: "
                         "it was made for another model")
    if D.layer is None:
        raise ValueError("can't tell which layer the dictionary reads; give layer (the block whose output it reads)")
    if not -1 <= D.layer < len(blocks(model)):
        raise ValueError(f"layer {D.layer} is outside the model's {len(blocks(model))} layers")
    return D.to(device(model))


# ---- running the model ------------------------------------------------------------------------

@contextlib.contextmanager
def _at(model, layer: int, fn):
    """fn(hidden [batch, tokens, d]) -> hidden applied to a block's output (layer -1: the embeddings)."""
    module = model.get_input_embeddings() if layer < 0 else blocks(model)[layer]

    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    h = module.register_forward_hook(hook)
    try:
        yield model
    finally:
        h.remove()


def _ids(tok, text: str, chat: bool = False) -> tuple[torch.Tensor, int]:
    """Token ids (cut to MAX_TOKENS) and the first position read. The first token is skipped whether or not it is a
    BOS token: it is where attention sinks, and its residual is an outlier hundreds of times the usual norm that
    dictionaries are not made for (models without a BOS token sink on the first word instead)."""
    ids = encode(tok, text, chat)[:MAX_TOKENS]
    return ids, 1 if len(ids) > 1 else 0


@torch.no_grad()
def _resid(model, D: Dictionary, ids: torch.Tensor) -> torch.Tensor:
    got = {}

    def keep(h):
        got["x"] = h[0].float()
        return h
    with _at(model, D.layer, keep):
        model(ids[None].to(device(model)), use_cache=False)
    return got["x"]


@torch.no_grad()
def _loss(model, D: Dictionary, ids: torch.Tensor, s: int, rep: torch.Tensor | None) -> float:
    """Summed next-token loss over the positions read, with their residual replaced by rep when given."""
    def swap(h):
        h = h.clone()
        h[0, s:s + rep.shape[0]] = rep.to(h.dtype)
        return h
    ctx = _at(model, D.layer, swap) if rep is not None else contextlib.nullcontext()
    with ctx:
        logits = model(ids[None].to(device(model)), use_cache=False).logits[0, s:-1].float()
    return float(torch.nn.functional.cross_entropy(logits, ids[s + 1:].to(logits.device), reduction="sum"))


def _texts(body: dict, key: str, default: list[str]) -> list[str]:
    items = body.get(key) or default
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
        raise ValueError(f"{key} must be a list of texts")
    items = [x for x in items if x.strip()]
    if not items:
        raise ValueError(f"{key} needs at least one text")
    if len(items) > MAX_TEXTS:
        raise ValueError(f"at most {MAX_TEXTS} {key}")
    return items


def _feat(D: Dictionary, i: int, **kw) -> dict:
    row = {"index": int(i), **kw}
    if int(i) in D.labels:
        row["label"] = D.labels[int(i)]
    return row


def _about(D: Dictionary) -> dict:
    return {"architecture": D.architecture, "format": D.format, "d_model": D.d_model, "features": D.n,
            "layer": D.layer, "reads": "the embeddings" if D.layer < 0 else f"the output of block {D.layer}",
            "hook": D.hook, "k": D.k, "subtract_b_dec": D.subtract_b_dec, "labels": bool(D.labels),
            "labeled": len(D.labels)}


# ---- steps ------------------------------------------------------------------------------------

@torch.no_grad()
def info(model, tok, body: dict) -> dict:
    """What the dictionary is, and how well it reconstructs this model's residual on neutral texts."""
    D = _open(model, body)
    texts = _texts(body, "texts", NEUTRAL)
    gen = torch.Generator().manual_seed(SEED)
    total = torch.zeros(D.d_model, dtype=torch.float64)
    sumsq = sse = 0.0
    tokens = predicted = 0
    l0 = 0.0
    seen = torch.zeros(D.n, dtype=torch.bool, device=D.W_enc.device)
    loss = dict.fromkeys(("clean", "reconstruction", "random_error", "zero"), 0.0)
    for t in texts:
        ids, s = _ids(tok, t)
        x = _resid(model, D, ids)[s:]
        f = D.encode(x)
        xhat = D.decode(f)
        err = x - xhat
        total += x.double().sum(0).cpu()
        sumsq += float(x.double().pow(2).sum())
        sse += float(err.double().pow(2).sum())
        tokens += x.shape[0]
        l0 += float((f > 0).sum())
        seen |= (f > 0).any(0)
        if len(ids) - s < 2:
            continue
        r = torch.randn(x.shape, generator=gen).to(x.device)
        noisy = x + unit_rows(r) * err.norm(dim=-1, keepdim=True)
        for name, rep in (("clean", None), ("reconstruction", xhat), ("random_error", noisy),
                          ("zero", torch.zeros_like(x))):
            loss[name] += _loss(model, D, ids, s, rep)
        predicted += len(ids) - s - 1
    var = sumsq - float(total.pow(2).sum()) / max(tokens, 1)
    loss = {k: round(v / predicted, 6) for k, v in loss.items()} if predicted else None
    gap = loss and loss["zero"] - loss["clean"]
    return {
        **_about(D), "texts": len(texts), "tokens": tokens,
        "variance_explained": round(1 - sse / var, 6) if var > 0 else None,
        "loss": loss,
        "loss_added": loss and round(loss["reconstruction"] - loss["clean"], 6),
        "random_error_added": loss and round(loss["random_error"] - loss["clean"], 6),
        "loss_recovered": round((loss["zero"] - loss["reconstruction"]) / gap, 6) if gap else None,
        "mean_active_per_token": round(l0 / max(tokens, 1), 3),
        "features_seen_active": int(seen.sum()),
        "notes": "variance_explained is 1 - |x - reconstruction|^2 / |x - mean|^2 over all tokens read (a leading "
                 "first token, the attention sink, is skipped). Losses are mean next-token loss with the layer's residual replaced: by the "
                 "reconstruction, by the residual plus an error of the same size in a random direction (control), "
                 "and by zeros. A good fit adds little loss and recovers most of the zero-ablation gap; neutral "
                 "texts only, so a domain the texts don't touch can fit worse.",
    }


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def read(model, tok, body: dict) -> dict:
    """Per token the top active features; over all texts the features most active on average."""
    D = _open(model, body)
    texts = _texts(body, "texts", [])
    top = int(body.get("top", TOP))
    if not 1 <= top <= MAX_TOP:
        raise ValueError(f"top must be 1 to {MAX_TOP}")
    dev = D.W_enc.device
    sums = torch.zeros(D.n, device=dev)
    fires = torch.zeros(D.n, device=dev)
    best = torch.full((D.n,), -math.inf, device=dev)
    where = torch.zeros(D.n, 2, dtype=torch.long, device=dev)
    rows, tokens, words = [], 0, []
    for ti, t in enumerate(texts):
        ids, s = _ids(tok, t)
        f = D.encode(_resid(model, D, ids)[s:])
        word = [tok.decode([int(i)]) for i in ids[s:]]
        words.append(word)
        vals, idx = f.topk(min(top, D.n), -1)
        rows.append({"text": t, "tokens": [
            {"token": w, "active": int((f[j] > 0).sum()),
             "features": [_feat(D, int(i), activation=round(float(v), 4)) for v, i in zip(vals[j], idx[j]) if v > 0]}
            for j, w in enumerate(word)]})
        sums += f.sum(0)
        fires += (f > 0).sum(0)
        m, a = f.max(0)
        better = m > best
        best = torch.where(better, m, best)
        where[better] = torch.stack([torch.full_like(a, ti), a], 1)[better]
        tokens += f.shape[0]
    mean = sums / max(tokens, 1)
    most = []
    for i in mean.topk(min(MOST, D.n)).indices.tolist():
        if mean[i] <= 0:
            break
        ti, j = where[i].tolist()
        most.append(_feat(D, i, mean_activation=round(float(mean[i]), 5), fires=round(float(fires[i]) / tokens, 4),
                          max_activation=round(float(best[i]), 4),
                          max_at={"text": ti, "position": j, "token": words[ti][j],
                                  "context": "".join(words[ti][max(0, j - 8):j + 1])}))
    return {**_about(D), "texts": rows, "tokens": tokens, "most_active": most,
            "notes": "Activations are in the dictionary's own units. 'fires' is the share of all tokens where a "
                     "feature is active: one that fires nearly everywhere is a weak sign of anything. Labels, when "
                     "present, come from the dictionary's release and are someone's reading, not a measurement."}


@torch.no_grad()
def _per_text(model, D: Dictionary, batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Per text: mean activation and share of tokens firing, each [texts, n] on the CPU."""
    means, fires = [], []
    for ids, s in batch:
        f = D.encode(_resid(model, D, ids)[s:])
        means.append(f.mean(0).cpu())
        fires.append((f > 0).float().mean(0).cpu())
    return torch.stack(means), torch.stack(fires)


@torch.no_grad()
def diff(model, tok, body: dict) -> dict:
    """Features that fire more or less in the loaded model than in its base on the same texts."""
    D = _open(model, body)
    texts = _texts(body, "texts", NEUTRAL)
    if len(texts) < 2:
        raise ValueError("diff needs at least two texts (its controls resample them)")
    batch = [_ids(tok, t) for t in texts]
    loaded, loaded_fires = _per_text(model, D, batch)
    with base_weights(model, body.get("base")) as swapped:
        base, base_fires = _per_text(model, D, batch)
    d = (loaded - base).double()
    change = d.mean(0)
    gen = torch.Generator().manual_seed(SEED)
    acc = torch.zeros(2, D.n, dtype=torch.float64)
    for _ in range(BOOT):
        m = d[torch.randint(len(texts), (len(texts),), generator=gen)].mean(0)
        acc[0] += m
        acc[1] += m * m
    se = (acc[1] / BOOT - (acc[0] / BOOT) ** 2).clamp_min(0).sqrt()
    null = torch.stack([((torch.randint(0, 2, (len(texts), 1), generator=gen) * 2 - 1) * d).mean(0).abs().max()
                        for _ in range(SHUFFLES)])
    floor = float(torch.quantile(null, FLOOR_Q))

    def row(i):
        c = float(change[i])
        return _feat(D, i, change=round(c, 6), loaded_mean=round(float(loaded[:, i].mean()), 6),
                     base_mean=round(float(base[:, i].mean()), 6), se=round(float(se[i]), 6),
                     z=round(c / float(se[i]), 3) if se[i] > 0 else None, above_floor=abs(c) > floor,
                     loaded_fires=round(float(loaded_fires[:, i].mean()), 4),
                     base_fires=round(float(base_fires[:, i].mean()), 4))
    k = min(MOST, D.n)
    up = [row(i) for i in change.topk(k).indices.tolist() if change[i] > 0]
    down = [row(i) for i in (-change).topk(k).indices.tolist() if change[i] < 0]
    return {**_about(D), "texts": len(texts), "swapped_tensors": swapped, "noise_floor": round(floor, 6),
            "above_floor": int((change.abs() > floor).sum()), "up": up, "down": down,
            "notes": "change is the mean over texts of each text's mean activation, loaded minus base, on the same "
                     "tokens. se is a bootstrap over texts; noise_floor is the 95th percentile of the largest "
                     "|change| any feature reaches when which-model-is-which is shuffled text by text. The "
                     "dictionary was made for one of the two models (usually the base), so a feature the other "
                     "model computes in a new way can go unseen. swapped_tensors 0 means the base's weights equal "
                     "the loaded model's."}


@torch.no_grad()
def steer(model, tok, body: dict) -> dict:
    """Generations with a feature's decoder direction added at its layer, beside a random feature's."""
    D = _open(model, body)
    if body.get("feature") is None:
        raise ValueError("give feature: the index of the feature to steer with")
    feature = int(body["feature"])
    if not 0 <= feature < D.n:
        raise ValueError(f"feature must be 0 to {D.n - 1}")
    raw = body.get("strength", body.get("strengths", STRENGTHS))
    strengths = [float(x) for x in (raw if isinstance(raw, list) else [raw])]
    if not strengths or len(strengths) > MAX_STRENGTHS or not all(math.isfinite(x) for x in strengths):
        raise ValueError(f"strength must be 1 to {MAX_STRENGTHS} finite numbers")
    prompts = _texts(body, "prompts", PROMPTS)
    n = int(body.get("max_tokens", GEN_TOKENS))
    if not 1 <= n <= 256:
        raise ValueError("max_tokens must be 1 to 256")
    chat = bool(body.get("chat", True))
    gen = torch.Generator().manual_seed(SEED + feature)
    control = int(body["control"]) if body.get("control") is not None else \
        (feature + 1 + int(torch.randint(D.n - 1, (1,), generator=gen))) % D.n
    if not 0 <= control < D.n or control == feature:
        raise ValueError("control must be another feature's index")
    norms = [_resid(model, D, ids)[s:].norm(dim=-1).mean() for ids, s in (_ids(tok, p, chat) for p in prompts)]
    scale = float(torch.stack(norms).mean())
    dirs = {"feature": unit(D.W_dec[feature]), "control": unit(D.W_dec[control])}
    rows, changed = [], {s: {"feature": 0, "control": 0} for s in strengths}
    for p in prompts:
        plain = generate(model, tok, p, n, chat)
        row = {"prompt": p, "unsteered": plain, "steered": []}
        for s in strengths:
            out = {"strength": s}
            for name, v in dirs.items():
                add = s * scale * v
                with _at(model, D.layer, lambda h, a=add: h + a.to(h.dtype)):
                    out[name] = generate(model, tok, p, n, chat)
                changed[s][name] += out[name] != plain
            row["steered"].append(out)
        rows.append(row)
    return {**_about(D), "feature": _feat(D, feature, decoder_norm=round(float(D.W_dec[feature].norm()), 6)),
            "control": _feat(D, control, decoder_norm=round(float(D.W_dec[control].norm()), 6)),
            "residual_norm": round(scale, 6), "prompts": rows,
            "changed": [{"strength": s, "feature": c["feature"], "control": c["control"], "of": len(prompts)}
                        for s, c in changed.items()],
            "notes": "Each strength adds strength x the residual's mean norm at the layer along the feature's unit "
                     "decoder direction, at every position, prompt and answer. The control adds a random other "
                     "feature's direction at the same norm: an answer that changes as much under the control shows "
                     "the size of the push, not the feature, at work. Greedy decoding."}


STEPS = {"info": info, "read": read, "diff": diff, "steer": steer}
