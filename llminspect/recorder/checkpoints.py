"""Candidate checkpoints: where another build of the loaded model's weights differs, and the weight
edit that puts those weights in place.

A candidate is a fine-tune, a quantized copy or a downloaded build with the same architecture as the
loaded (base) model. LLMInspect reads only safetensors and GGUF files. Pickle-based weights can run
code when they are loaded, so they are refused (nothing here calls torch.load), and nothing is ever
downloaded: a Hugging Face repo id is looked up in the local cache only.

  resolve_candidate  a checkpoint folder, a .gguf file, ollama:<tag>, or a cached repo id -> a folder
  CheckpointReader   the candidate's tensors, read one at a time, mapped to the base model's parameters
  compare_weights    the weight-difference report, saved under <data>/checkpoint_reports/<model>/
  Checkpoints        the background job behind POST /llminspect/workbench/checkpoint

The report compares the candidate with the base weights (the originals while edits are applied), with
the candidate first cast to the loaded model's precision, so it shows what applying the candidate
changes. One tensor at a time, on the CPU in fp32:

  tensors   per parameter: layer, module, rel = ||delta|| / ||W|| (capped at REL_CAP), the largest
            |delta|, the share of elements that changed (|delta| > 1e-5 + 1e-3 |W|), and for a matrix
            the share of ||delta||^2 in its top 8 singular values (lowrank: near 1 means a few
            directions changed, near 0 means the change is spread over many)
  neurons   per MLP neuron, rel over its down_proj column and its up/gate rows together
  heads     per attention head, rel over its o_proj columns and its q_proj rows together
  tokens    per embedding row (and output-weight row when untied), rel of the row
  pattern   "concentrated" when the top 5% of tensors by ||delta||^2 carry 80% of the total,
            "spread" when every tensor changed and none carries more than 10%, else "mixed"
            ("identical" when nothing differs). A description of how the change is distributed,
            never a verdict on whether the candidate is safe.

The edit (edits.py, kind "checkpoint") copies the changed tensors into the model. It always applies
first, so other edits land on top of the candidate's weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import huggingface_hub
import torch
from safetensors import SafetensorError, safe_open
from torch import nn

from .anatomy import ATTN_NORMS, MLP_NORMS
from .edits import ATTN_NAMES, _child, attention_heads
from .model import display_name, prepare
from .tracing import Busy

TOP = 25                  # entries in each top list
CHANGED_ABS = 1e-5        # an element changed when |delta| > CHANGED_ABS + CHANGED_REL |W|
CHANGED_REL = 1e-3
RANK = 8                  # singular values counted in a matrix's low-rank share
RANK_FROM = 1e-4          # ... for tensors whose rel is above this
TOP_TENSORS = 0.05        # concentrated: this share of tensors carries CARRY of ||delta||^2
CARRY = 0.80
SPREAD_MAX = 0.10         # spread: every tensor changed and none carries more than this share
REL_CAP = 1e3             # a change from (almost) nothing is reported as this large, not infinite
CHUNK = 1 << 24           # elements per block in elementwise passes
HASH_CHUNK = 1 << 23      # bytes per block when hashing a file
COMPARE_SHARE = 0.85      # share of the job's progress that reading the tensors takes (hashing the rest)
FLOATS = {"F64", "F32", "F16", "BF16"}
WRAPPERS = ("model.language_model.", "language_model.model.", "language_model.", "model.")
PICKLED = (".bin", ".pt", ".pth", ".pkl", ".ckpt")
REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|layer|block)\.(\d+)(?:\.|$)")

MODULES = ("q", "k", "v", "o", "gate", "up", "down", "attn_norm", "mlp_norm", "embed", "lm_head", "final_norm", "other")
_PROJECTIONS = {
    "q_proj": "q", "wq": "q", "k_proj": "k", "wk": "k", "v_proj": "v", "wv": "v",
    "o_proj": "o", "out_proj": "o", "wo": "o",
    "gate_proj": "gate", "w1": "gate",
    "up_proj": "up", "w3": "up", "fc1": "up", "fc_in": "up", "c_fc": "up", "dense_h_to_4h": "up",
    "down_proj": "down", "w2": "down", "fc2": "down", "fc_out": "down", "dense_4h_to_h": "down",
}
_ATTN_NORMS = ATTN_NORMS + ("attn_norm", "pre_attention_layernorm")
_MLP_NORMS = MLP_NORMS + ("mlp_norm", "pre_mlp_layernorm")
_EMBEDS = ("embed_tokens", "wte", "embed_in", "word_embeddings", "tok_embeddings")
_UNEMBEDS = ("lm_head", "embed_out", "output")


# ---- names ---------------------------------------------------------------------------------------

def canonical(name: str) -> str:
    """A tensor name without the wrapper prefixes a class adds when it holds the language model
    ("model.language_model.layers.0..." and "model.layers.0..." both give "layers.0...")."""
    while True:
        for prefix in WRAPPERS:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        else:
            return name


def layer_of(name: str) -> int | None:
    m = LAYER.search(name)
    return int(m[1]) if m else None


def module_of(name: str) -> str:
    """Which part of the model a parameter belongs to, from its name; "other" when unsure."""
    parts = name.lower().split(".")
    if len(parts) > 1 and parts[-1] in ("weight", "bias"):
        parts = parts[:-1]
    leaf, path = parts[-1], ".".join(parts[:-1])
    if layer_of(name) is None:
        if leaf in _EMBEDS:
            return "embed"
        if leaf in _UNEMBEDS:
            return "lm_head"
        return "final_norm" if "norm" in leaf or leaf.startswith("ln_") else "other"
    if leaf in _ATTN_NORMS:
        return "attn_norm"
    if leaf in _MLP_NORMS:
        return "mlp_norm"
    if leaf in ("c_proj", "dense"):  # the same name in the attention and the MLP of some models
        return "o" if "attn" in path else "down" if "mlp" in path else "other"
    return _PROJECTIONS.get(leaf, "other")


# ---- resolving a candidate -------------------------------------------------------------------------

def is_conversion(spec: str) -> bool:
    """A GGUF file or Ollama tag: reading it means dequantizing it into a checkpoint first (slow, once)."""
    return spec.lower().endswith(".gguf") or spec.startswith("ollama:")


def weight_files(folder: Path) -> list[Path]:
    """The safetensors files that hold a checkpoint folder's weights, following the shard index when
    there is one. Weights that are only pickle-based are refused."""
    if not folder.is_dir():
        raise FileNotFoundError(f"no checkpoint folder at {folder}")
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = sorted({str(v) for v in json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"{index.name} can't be read ({type(exc).__name__}: {exc})") from None
        if any(Path(n).name != n for n in names):
            raise ValueError(f"{index.name} points outside the checkpoint folder")
        missing = [n for n in names if not (folder / n).is_file()]
        if missing:
            raise ValueError(f"incomplete checkpoint: {len(missing)} of {len(names)} shards listed in {index.name} "
                             f"are missing, for example {missing[0]}")
        return [folder / n for n in names]
    found = sorted(p for p in folder.glob("*.safetensors") if p.is_file())
    if found:
        return found
    pickled = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PICKLED)
    if pickled:
        raise ValueError(f"{folder.name} has only pickle-based weights ({', '.join(pickled[:3])}). Pickle files can "
                         "run code when they are loaded, so LLMInspect reads only safetensors or GGUF. Get a "
                         "safetensors build, or convert these in a sandbox you trust.")
    raise ValueError(f"no .safetensors weights in {folder}; if it is a cached repo it may be only partly downloaded")


def _cached_repo(repo: str) -> Path:
    try:
        return Path(huggingface_hub.snapshot_download(repo, local_files_only=True))
    except Exception as exc:  # not in the cache: never fall back to the network
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache. LLMInspect never downloads: "
                                "download it first, then load it again.") from exc


def resolve_candidate(spec: str) -> tuple[Path, dict]:
    """(checkpoint folder, provenance) for a candidate: a local checkpoint folder, a .gguf file or
    ollama:<tag> (dequantized once into LLMInspect's model cache, the original only read), or a
    Hugging Face repo id that is already in the local cache."""
    spec = str(spec or "").strip()
    if not spec:
        raise ValueError("name a candidate: a checkpoint folder, a .gguf file, ollama:<tag> or a cached repo id")
    if is_conversion(spec):
        path, provenance = prepare(spec)
        folder = Path(path)
    else:
        local = Path(spec).expanduser()
        if local.is_dir():
            folder, provenance = local, {"spec": spec, "source": "local"}
        elif local.exists():
            raise ValueError("name the checkpoint's folder, not a file in it")
        elif REPO.match(spec):
            folder = _cached_repo(spec)
            provenance = {"spec": spec, "source": "huggingface-cache", "revision": folder.name}
        else:
            raise FileNotFoundError(f"no checkpoint folder at {spec}")
    weight_files(folder)  # refuses pickle-based weights and a missing shard now, before any work
    return folder, dict(provenance) | {"folder": str(folder)}


# ---- reading a candidate -----------------------------------------------------------------------------

class CheckpointReader:
    """A candidate's tensors, read lazily, and which of the base model's parameters each belongs to."""

    def __init__(self, spec: str, folder, provenance: dict | None = None):
        self.spec, self.folder = spec, Path(folder)
        self.provenance = dict(provenance or {})
        self.name = display_name(spec)
        self.files = weight_files(self.folder)
        self.keys: dict[str, Path] = {}      # tensor name -> the file that holds it
        self.info: dict[str, tuple[tuple[int, ...], str]] = {}  # tensor name -> (shape, dtype)
        for f in self.files:
            try:
                with safe_open(str(f), framework="pt", device="cpu") as h:
                    for k in h.keys():
                        if k not in self.keys:
                            sl = h.get_slice(k)
                            self.keys[k], self.info[k] = f, (tuple(sl.get_shape()), sl.get_dtype())
            except (SafetensorError, OSError) as exc:
                raise ValueError(f"{f.name} is not a readable safetensors file ({exc})") from None
        self.params: dict[str, str] = {}     # base parameter name -> candidate tensor name
        self.not_compared: list[str] = []    # candidate tensors that match no parameter (a vision tower, say)

    def map(self, model) -> "CheckpointReader":
        """Match the candidate's tensors to the model's parameters. Refuses a candidate of another
        architecture (config sizes, tensor shapes) or one that lacks weights the model needs."""
        self._check_config(model.config)
        params = dict(model.named_parameters())
        leader = self._leaders(model, params)
        by_canon: dict[str, str | None] = {}
        for n, lead in leader.items():
            c = canonical(n)
            by_canon[c] = lead if by_canon.get(c, lead) == lead else None  # None: two parameters, one name
        hits: dict[str, list[tuple[int, str]]] = {}
        extra: list[str] = []
        for key in self.keys:
            lead = leader.get(key) or by_canon.get(canonical(key))
            if lead is None:
                extra.append(key)
            else:
                hits.setdefault(lead, []).append((0 if canonical(key) == canonical(lead) else 1, key))
        mapped: dict[str, str] = {}
        for lead, found in hits.items():
            found.sort()
            mapped[lead] = found[0][1]
            extra += [k for _, k in found[1:]]  # a second name for tied weights
        bad = []
        for name, key in mapped.items():
            shape, dtype = self.info[key]
            if shape != tuple(params[name].shape):
                bad.append(f"{name} is {list(params[name].shape)} here but {list(shape)} in the candidate")
            elif dtype not in FLOATS:
                bad.append(f"{name} is stored as {dtype}, not as float weights")
        if bad:
            more = f" (and {len(bad) - 3} more)" if len(bad) > 3 else ""
            raise ValueError("the tensors don't fit this model, so the candidate is a different architecture or "
                             f"vocabulary (or a quantized format): {'; '.join(bad[:3])}{more}")
        missing = [n for n in params if leader.get(n, n) not in mapped]
        if missing:
            hint = ""
            if (self.folder / "adapter_config.json").is_file():
                hint = " This folder holds an adapter, not full weights; merge it into its base model first."
            elif any(k.endswith((".qweight", ".scales", ".g_idx")) for k in extra):
                hint = " This looks like a packed quantization format; LLMInspect compares dense weights."
            raise ValueError(f"incomplete checkpoint: {len(missing)} of the model's {len(params)} parameters are not "
                             f"in it, for example {', '.join(missing[:3])}.{hint}")
        self.params = {n: mapped[n] for n in params if n in mapped}
        self.not_compared = sorted(extra)
        return self

    @staticmethod
    def _leaders(model, params: dict) -> dict[str, str]:
        """Every name a parameter goes by -> the name `named_parameters` lists it under. Tied weights
        (lm_head and the embeddings, say) are one parameter, and a candidate may carry either name."""
        by_id = {id(p): n for n, p in params.items()}
        by_ptr: dict[int, str] = {}
        for n, p in params.items():
            if p.data_ptr():
                by_ptr.setdefault(p.data_ptr(), n)
        leader = {}
        for n, p in model.named_parameters(remove_duplicate=False):
            leader[n] = by_id[id(p)]
        for n, p in params.items():
            leader[n] = by_ptr.get(p.data_ptr(), n)  # distinct parameters over the same storage
        return leader

    def _check_config(self, config) -> None:
        path = self.folder / "config.json"
        try:
            theirs = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            return  # an unreadable config: the tensor shapes decide
        if not isinstance(theirs, dict):
            return
        text = theirs.get("text_config") if isinstance(theirs.get("text_config"), dict) else theirs
        ours, diffs = config.get_text_config(), []
        for key in ("hidden_size", "num_hidden_layers", "vocab_size"):
            a, b = text.get(key, theirs.get(key)), getattr(ours, key, None)
            if isinstance(a, int) and isinstance(b, int) and a != b:
                diffs.append(f"{key} is {a} in the candidate, {b} here")
        if diffs:
            raise ValueError(f"a different architecture: {', '.join(diffs)}")

    def require_files(self) -> None:
        """The candidate's files are still there (they may have been moved since it was loaded)."""
        gone = [f.name for f in self.files if not f.is_file()]
        if gone:
            raise ValueError(f"the candidate's files are gone: {', '.join(gone[:3])}")

    def tensor(self, name: str) -> torch.Tensor:
        """The candidate's tensor for a base parameter, as stored (CPU)."""
        key = self.params[name]
        with safe_open(str(self.keys[key]), framework="pt", device="cpu") as h:
            return h.get_tensor(key)


def open_candidate(spec: str, model) -> CheckpointReader:
    """Resolve a candidate and map it onto the model; every refusal is raised here."""
    folder, provenance = resolve_candidate(spec)
    return CheckpointReader(spec, folder, provenance).map(model)


def reader_for(checkpoint: dict, model) -> CheckpointReader:
    """The reader for an edit's checkpoint record, with every changed parameter checked to exist."""
    reader = CheckpointReader(checkpoint["spec"], checkpoint["folder"]).map(model)
    unknown = [n for n in checkpoint["changed"] if n not in reader.params]
    if unknown:
        raise ValueError(f"the candidate has no tensor for {', '.join(unknown[:3])}")
    return reader


# ---- the weight-difference report --------------------------------------------------------------------

def _g(x: float, digits: int = 5) -> float:
    """x to a few significant digits; a number that isn't finite becomes 0 so the report stays JSON."""
    return float(f"{x:.{digits}g}") if math.isfinite(x) else 0.0


def _rel(delta: float, base: float) -> float:
    return min(REL_CAP, delta / base) if base > 0 else (REL_CAP if delta > 0 else 0.0)


def _rel_t(delta: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    ratio = delta / base.clamp_min(1e-30)
    return torch.where(base > 0, ratio, torch.where(delta > 0, REL_CAP, 0.0)).clamp_max(REL_CAP)


def _sq(t: torch.Tensor, dim) -> torch.Tensor:
    return torch.linalg.vector_norm(t, dim=dim).double() ** 2


def _elementwise(delta: torch.Tensor, base: torch.Tensor) -> tuple[float, int]:
    """(largest |delta|, how many elements changed), in blocks so a big tensor needs little extra memory."""
    d, b = delta.reshape(-1), base.reshape(-1)
    top, count = 0.0, 0
    for s in range(0, d.numel(), CHUNK):
        blk = d[s:s + CHUNK].abs()
        top = max(top, float(blk.max()))
        count += int((blk > CHANGED_ABS + CHANGED_REL * b[s:s + CHUNK].abs()).sum())
    return top, count


def _lowrank(delta: torch.Tensor, norm: float) -> float:
    """The share of ||delta||^2 in delta's top singular values (a randomized estimate, seeded)."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        _, sv, _ = torch.svd_lowrank(delta, q=min(RANK, *delta.shape), niter=2)
    return min(1.0, float((sv.double() ** 2).sum()) / (norm * norm))


def _roles(rec, names: dict[int, str]) -> dict[str, tuple]:
    """Which parameters carry neurons and heads, found from the modules rather than the names:
    {name: ("down" | "in", layer)} and {name: ("o" | "q", layer, heads, head dim)}."""
    s, nm = rec.s, rec.neurons
    cfg = s.model.config.get_text_config()
    roles: dict[str, tuple] = {}

    def put(weight, role):
        if id(weight) in names:
            roles[names[id(weight)]] = role

    for i in range(s.n_layers):
        if nm._try_mlp(i):
            mlp = nm.mlp(i)
            put(mlp.down.weight, ("down", i))
            for lin in (mlp.gate, mlp.up):
                if lin is not None:
                    put(lin.weight, ("in", i))
        attn = _child(s.layers[i], ATTN_NAMES)
        heads = attention_heads(attn, s.d_model, cfg)
        if heads is not None:
            out, n_heads, hd = heads
            put(out.weight, ("o", i, n_heads, hd))
            q = getattr(attn, "q_proj", None)
            if isinstance(q, nn.Linear) and q.out_features == n_heads * hd:
                put(q.weight, ("q", i, n_heads, hd))
    return roles


def _accumulate(role: tuple, delta: torch.Tensor, base: torch.Tensor, neurons: dict, heads: dict) -> None:
    """Add this tensor's squared change and squared base size, per neuron or per head, to the layer's totals."""
    kind, layer = role[0], role[1]
    if kind in ("down", "in"):
        dim, store = (0 if kind == "down" else 1), neurons
        d2, w2 = _sq(delta, dim), _sq(base, dim)
    else:
        store, n_heads, hd = heads, role[2], role[3]
        if kind == "o":  # columns of o_proj: [d_model, heads * head dim]
            d2, w2 = (_sq(t.reshape(t.shape[0], n_heads, hd), (0, 2)) for t in (delta, base))
        else:            # rows of q_proj: [heads * head dim, d_model]
            d2, w2 = (_sq(t.reshape(n_heads, -1), 1) for t in (delta, base))
    acc = store.setdefault(layer, [torch.zeros_like(d2), torch.zeros_like(w2)])
    acc[0] += d2
    acc[1] += w2


def _top(rel: torch.Tensor, k: int = TOP) -> list[tuple[int, float]]:
    """(index, value) of the k largest entries that are above zero."""
    v, i = rel.topk(min(k, rel.numel()))
    return [(int(j), float(x)) for x, j in zip(v.tolist(), i.tolist()) if x > 0]


def _units(acc: dict, label: str, name) -> dict:
    """Top entries over all layers, and each layer's largest, from per-layer (change, size) squares."""
    if not acc:
        return {"top": [], "layer_max": []}
    layers = sorted(acc)
    rel = [_rel_t(acc[layer][0].sqrt(), acc[layer][1].sqrt()) for layer in layers]
    owner = torch.cat([torch.full((len(r),), layer) for layer, r in zip(layers, rel)])
    index = torch.cat([torch.arange(len(r)) for r in rel])
    row = lambda layer, i, r: {name: label.format(layer, i), "layer": layer, "index": i, "rel": _g(r)}
    return {"top": [row(int(owner[j]), int(index[j]), v) for j, v in _top(torch.cat(rel))],
            "layer_max": [row(layer, int(r.argmax()), float(r.max())) for layer, r in zip(layers, rel)]}


def _summary(rows: list[dict], dsq: list[float], wsq: list[float]) -> dict:
    total_d, total_w = sum(dsq), sum(wsq)
    by_layer: dict[int, list[dict]] = {}
    for r in rows:
        if r["layer"] is not None:
            by_layer.setdefault(r["layer"], []).append(r)
    layers = sorted(({"layer": layer, "mean_rel": _g(sum(r["rel"] for r in rs) / len(rs)),
                      "max_rel": _g(max(r["rel"] for r in rs)), "changed": sum(r["rel"] > 0 for r in rs),
                      "tensors": len(rs)} for layer, rs in by_layer.items()), key=lambda x: -x["mean_rel"])
    ranked = sorted(dsq, reverse=True)
    k = max(1, math.ceil(TOP_TENSORS * len(ranked)))
    top_share = sum(ranked[:k]) / total_d if total_d > 0 else 0.0
    max_share = ranked[0] / total_d if total_d > 0 else 0.0
    all_changed = all(r["rel"] > 0 for r in rows)
    if total_d <= 0:
        pattern = "identical"
    elif top_share >= CARRY:
        pattern = "concentrated"
    elif all_changed and max_share <= SPREAD_MAX:
        pattern = "spread"
    else:
        pattern = "mixed"
    return {"total_rel": _g(math.sqrt(total_d / total_w) if total_w > 0 else 0.0), "pattern": pattern, "layers": layers,
            "tensors": len(rows), "top_tensors": k, "top_share": _g(top_share), "max_share": _g(max_share),
            "all_changed": all_changed}


def file_hashes(files: list[Path], say=None) -> list[dict]:
    """sha256 of each file; say(fraction of the bytes read) is called as it goes."""
    total, done, out = sum(f.stat().st_size for f in files) or 1, 0, []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            while chunk := fh.read(HASH_CHUNK):
                h.update(chunk)
                done += len(chunk)
                if say is not None:
                    say(done / total)
        out.append({"file": f.name, "bytes": f.stat().st_size, "sha256": h.hexdigest()})
    return out


@torch.no_grad()
def compare_weights(rec, reader: CheckpointReader, progress=None) -> dict:
    """The weight-difference report for a candidate (see the module's docstring), saved under
    <data>/checkpoint_reports/<model>/<id>.json. Holds the model's lock while it reads the weights.
    progress(step, fraction) is called as it goes."""
    say = progress or (lambda *_: None)
    s, stack = rec.s, rec.edits
    if not reader.params:
        reader.map(s.model)
    params = dict(s.model.named_parameters())
    names = {id(p): n for n, p in params.items()}
    roles = _roles(rec, names)
    emb, out_emb = s.model.get_input_embeddings(), s.model.get_output_embeddings()
    embed = names.get(id(emb.weight)) if emb is not None else None
    unembed = names.get(id(out_emb.weight)) if out_emb is not None else None
    rows: list[dict] = []
    dsq: list[float] = []
    wsq: list[float] = []
    neurons: dict[int, list[torch.Tensor]] = {}
    heads: dict[int, list[torch.Tensor]] = {}
    tokens: dict[str, list[dict]] = {"embed": [], "lm_head": []}
    nonfinite: dict[str, int] = {}
    total = sum(params[n].numel() for n in reader.params) or 1
    done = 0
    if not rec.lock.acquire(timeout=5.0):
        raise Busy("the model is generating; compare the weights when it finishes")
    try:
        for name in reader.params:
            if rec.closed:
                raise RuntimeError("the model was switched while the weights were being compared")
            p = params[name]
            original = stack.originals.get(name)  # edits may be applied: the base is the original weights
            base = (p.detach() if original is None else original).to("cpu", torch.float32)
            cand = reader.tensor(name).to(p.dtype).to("cpu", torch.float32)  # as applying it would write them
            bad = ~torch.isfinite(cand)
            if bad.any():  # kept out of the numbers (read as unchanged) and listed
                nonfinite[name] = int(bad.sum())
                cand = torch.where(bad, base, cand)
            del bad
            delta = cand - base
            del cand
            top, n_changed = _elementwise(delta, base) if delta.numel() else (0.0, 0)
            dn = max(float(torch.linalg.vector_norm(delta)), top)
            wn = float(torch.linalg.vector_norm(base))
            rel = _rel(dn, wn)
            lowrank = _lowrank(delta, dn) if top > 0 and delta.ndim == 2 and rel > RANK_FROM else None
            module = "embed" if name == embed else "lm_head" if name == unembed else module_of(name)
            rows.append({"name": name, "layer": layer_of(name), "module": module, "rel": _g(rel) if top > 0 else 0.0,
                         "max_abs": _g(top), "changed_frac": _g(n_changed / max(delta.numel(), 1)),
                         "lowrank": None if lowrank is None else _g(lowrank)})
            dsq.append(dn * dn if top > 0 else 0.0)
            wsq.append(wn * wn)
            if name in roles:
                _accumulate(roles[name], delta, base, neurons, heads)
            if top > 0 and delta.ndim == 2 and name in (embed, unembed):
                per_row = _rel_t(torch.linalg.vector_norm(delta, dim=1), torch.linalg.vector_norm(base, dim=1))
                tokens[module] = [{"token_id": i, "text": s.token_str(i), "rel": _g(r)} for i, r in _top(per_row)]
            del delta, base
            done += p.numel()
            say("comparing weights", COMPARE_SHARE * done / total)
    finally:
        rec.lock.release()
    hashes = file_hashes(reader.files, lambda f: say("comparing weights", COMPARE_SHARE + (1 - COMPARE_SHARE) * f))
    summary = _summary(rows, dsq, wsq)
    if nonfinite:
        summary["nonfinite"] = nonfinite
    changed = sum(r["rel"] > 0 for r in rows)
    report = {
        "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "candidate": {"spec": reader.spec, "name": reader.name, "folder": str(reader.folder),
                      "provenance": reader.provenance, "files": hashes},
        "base": {"spec": s.spec, "name": s.name, "dtype": s.dtype},
        "counts": {"compared": len(rows), "changed": changed, "identical": len(rows) - changed,
                   "not_compared": len(reader.not_compared)},
        "not_compared": reader.not_compared[:100],
        "summary": summary,
        "tensors": rows,
        "neurons": _units(neurons, "L{} N{}", "neuron"),
        "heads": _units(heads, "L{} h{}", "head"),
        "tokens": tokens,
        "notes": "The candidate is cast to the loaded model's precision before it is compared, so these numbers show what "
                 "applying it changes. The pattern describes how the change is spread; it is not a verdict on the "
                 "candidate.",
    }
    path = rec.store.root / "checkpoint_reports" / s.slug / f"{report['id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    say("comparing weights", 1.0)
    return report


# ---- the job -------------------------------------------------------------------------------------------

class Checkpoints:
    """Loading a candidate for one recorder: a background job (one at a time) that checks the candidate,
    compares its weights with the base, and adds the checkpoint edit. Applying it is the edit stack's job."""

    def __init__(self, recorder):
        self.rec = recorder
        self.dir = recorder.store.root / "checkpoint_reports" / recorder.s.slug
        self.job = {"status": "none", "step": None, "progress": 0.0, "error": None}
        self.report: dict | None = None  # the latest report made in this session
        self._start = threading.Lock()

    # ---- state ---------------------------------------------------------------------------------

    def status(self) -> dict:
        """The job, the latest report (left out while a job is running) and the checkpoint edit in the stack."""
        edit = self.edit()
        return dict(self.job) | {
            "report": None if self.job["status"] == "building" else self.report,
            "edit": None if edit is None else edit.describe() | {"applied": edit.id in self.rec.edits.applied}}

    def edit(self):
        return next((e for e in self.rec.edits.edits.values() if e.kind == "checkpoint"), None)

    def reports(self) -> list[dict]:
        """The saved reports for this model, newest first."""
        rows = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True) if self.dir.is_dir() else []:
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
                rows.append({"id": r["id"], "candidate": r["candidate"]["name"], "spec": r["candidate"]["spec"],
                             "created": r["created"], "pattern": r["summary"]["pattern"], "total_rel": r["summary"]["total_rel"]})
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return rows

    def _say(self, step: str, progress: float) -> None:
        self.job.update(step=step, progress=round(progress, 3))

    # ---- loading -----------------------------------------------------------------------------------

    def start(self, spec: str, note: str = "") -> dict:
        """Check the candidate and start comparing it in the background. A folder, a cached repo id and
        the like are checked here, so a bad candidate fails this call (ValueError, FileNotFoundError);
        a GGUF file or Ollama tag is converted in the job, which reports its failure in the status."""
        spec = str(spec or "").strip()
        if not spec:
            raise ValueError("name a candidate: a checkpoint folder, a .gguf file, ollama:<tag> or a cached repo id")
        edits = self.rec.edits
        with self._start:
            if self.job["status"] == "building":
                raise Busy("a candidate is being compared already")
            if any(edits.edits[eid].kind == "checkpoint" for eid in edits.applied if eid in edits.edits):
                raise ValueError("a checkpoint is applied; restore the base weights before loading another candidate")
            before = self.job
            self.job = {"status": "building", "step": "resolving", "progress": 0.0, "error": None}
            try:
                reader = None if is_conversion(spec) else open_candidate(spec, self.rec.s.model)
            except BaseException:
                self.job = before
                raise
            if reader is not None:
                self._say("comparing weights", 0.0)
            started = self.status()

        def work():
            try:
                self._run(spec, note, reader)
                self.job = {"status": "ready", "step": "done", "progress": 1.0, "error": None}
            except (ValueError, FileNotFoundError, Busy) as exc:  # a refusal: its message says what to do
                self.job = {"status": "error", "step": None, "progress": 0.0, "error": str(exc)}
            except Exception as exc:
                self.job = {"status": "error", "step": None, "progress": 0.0, "error": f"{type(exc).__name__}: {exc}"}

        threading.Thread(target=work, daemon=True, name="checkpoint-compare").start()
        return started

    def _run(self, spec: str, note: str, reader: CheckpointReader | None) -> None:
        if reader is None:
            self._say("resolving", 0.0)
            folder, provenance = resolve_candidate(spec)  # a GGUF is dequantized here, once
            self._say("checking shapes", 0.0)
            reader = CheckpointReader(spec, folder, provenance).map(self.rec.s.model)
        self._say("comparing weights", 0.0)
        report = compare_weights(self.rec, reader, self._say)
        self._say("adding the edit", 1.0)
        self.report = report
        self._add_edit(reader, report, note)

    def _add_edit(self, reader: CheckpointReader, report: dict, note: str) -> None:
        rec = self.rec
        with rec.lock:
            old = self.edit()
            if old is not None:  # the candidate before this one, never applied: this one takes its place
                if old.id in rec.edits.applied:
                    raise ValueError("a checkpoint is applied; restore the base weights before loading another candidate")
                rec.edits.remove(old.id)
            rec.edits.add("checkpoint", note=note, checkpoint={
                "spec": reader.spec, "folder": str(reader.folder), "name": reader.name, "report": report["id"],
                "changed": [r["name"] for r in report["tensors"] if r["rel"] > 0]})
            rec.publish({"k": "workbench", "summary": rec.edits.summary()})
