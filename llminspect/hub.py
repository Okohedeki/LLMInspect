"""Unmask: what a model really is, read without downloading it. A model's card says which model it was made from;
its weights say whether that's true and what was done to them.

  1. Lineage. The card's base_model (and the Hub's base_model tags) name the parents, and their cards name theirs.
  2. Origin. A few hundred values per tensor, read by byte range from the safetensors files (or the most precise
     GGUF file, dequantized; see ggufread.py), fingerprint the model (see fingerprint.py) and are compared with
     each claimed parent and with the models on this machine.
  3. The edit. Some full rows of each sampled layer's residual writers (the MLP's down projection, a mixture of
     experts' shared expert, attention's output) are read from the model and its parent. The
     rank of their difference names the edit: the same direction or few removed from every layer (the mark
     direction removal, or abliteration, leaves), rank-one edits at single layers, a low-rank change like a merged adapter,
     or a full fine-tune. For a quantized model only what stands out above the quantization's rounding counts.
     A copy (a re-upload, or a quantization) of its parent is followed up the chain to the edit it carries.
  4. Files. Configs, code and chat templates are small and read whole: code that runs on load, and what the
     template says that the parent's doesn't. Pickle weights are parsed by range too: a PyTorch zip's directory and
     its .pkl members, or a raw pickle stream up to its first tensor bytes.

A Hub model costs a few megabytes of reads (a GGUF file more: its header holds the whole vocabulary). Nothing is
loaded and nothing runs. A local folder works the same way.

    python -m llminspect.hub ORG/NAME [--json]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import filediff, filescan, fingerprint, ggufread

ROWS = 24           # full rows of a layer's down projection read to measure the rank of its change
RANK_LAYERS = 6     # layers those rows come from
SMALL = 1 << 20     # text files up to this size are read whole
DEPTH = 4           # claimed ancestors followed up the card chain
WORKERS = 16
MOST = 0.9         # share of a change's energy its top k directions must hold
FEW = 4             # at most this many directions is "a few"
LOW_RANK = 0.9      # share in its top LOW_K directions that makes a change low-rank
LOW_K = 8
SHARED = 0.8        # overlap between layers' few directions that makes them the same ones
QUANT_LAYERS = 32   # layers measured in a quantized model, whose fingerprint can't tell which ones moved
SPIKE = 2.0         # a direction counts above rounding when its energy is this many times the noise edge
BROAD = 1.5         # the rest counts as a change when it holds this many times the rounding's energy
POINT = 0.5         # a direction with this share of its weight on one value is a clipped outlier, not an edit
# matrices that write into the residual stream, where a removed direction shows: the dense MLP's down projection,
# a mixture of experts' shared expert, and attention's output. Each is measured; the edit is named from the one
# that changed most, as a mixture of experts may leave its shared expert or its attention alone.
WRITERS = ("mlp.down_proj", "mlp.shared_expert.down_proj", "self_attn.o_proj")
WRITER = re.compile(r"layers?\.(\d+)\.(mlp\.down_proj|feed_forward\.w2|mlp\.shared_expert\.down_proj|"
                    r"self_attn\.o_proj)\.weight$")
TEXT = {".json", ".jinja", ".py", ".txt", ".model", ".yaml", ".yml"}
RELATIONS = ("finetune", "adapter", "merge", "quantized")
TMP = Path(tempfile.gettempdir()) / "llminspect-hub"  # where small files read from the Hub are put
PUBLIC = False      # set by the public app: only Hub repos, never a local path, and never a gated repo
PICKLES = 16        # pickle files of one repo parsed by range; the rest are listed as unread
BLOCK = 1 << 18     # a remote pickle file is read in blocks of this size
PICKLE_BUDGET = 32 << 20  # and never more than this from one file


class Unreachable(Exception):
    """A model that can't be read: missing, gated without access, or offline."""


# ---- sources: a Hub repo or a local folder, read through the same calls ------------------------------------

class RangeFile(io.RawIOBase):
    """A seekable read-only file over a source's byte ranges, fetched in blocks and kept, up to a budget."""

    def __init__(self, src, name: str, size: int, budget: int = PICKLE_BUDGET):
        self.src, self.name, self.size, self.budget = src, name, size, budget
        self.pos, self.blocks, self.fetched = 0, {}, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = max(0, {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset)
        return self.pos

    def _block(self, k: int) -> bytes:
        if k not in self.blocks:
            n = min(BLOCK, self.size - k * BLOCK)
            if self.fetched + n > self.budget:
                raise OSError(f"read more than {self.budget >> 20} MB of {self.name} without finding its end")
            self.blocks[k] = self.src.read([(self.name, k * BLOCK, n)])[0]
            self.fetched += n
        return self.blocks[k]

    def readinto(self, buf) -> int:
        want, got = min(len(buf), max(0, self.size - self.pos)), 0
        while got < want:
            k, at = divmod(self.pos, BLOCK)
            data = self._block(k)[at:at + want - got]
            buf[got:got + len(data)] = data
            got += len(data)
            self.pos += len(data)
        return got


class Folder:
    """A model folder on this machine."""

    def __init__(self, path: Path, name: str | None = None):
        from .check import model_name
        self.path = Path(path)
        if not self.path.is_dir():
            raise Unreachable(f"{path} isn't a folder")
        self.name = name or model_name(self.path)
        self.url = None
        self.bytes_read = 0
        self._gguf = None

    def files(self) -> dict[str, int]:
        return {p.relative_to(self.path).as_posix(): p.stat().st_size for p in sorted(self.path.rglob("*"))
                if p.is_file() and not p.name.startswith(".")}

    def claims(self) -> list[dict]:
        return [{"model": b, "relation": "finetune"} for b in fingerprint.claimed_base(self.path)]

    def headers(self) -> dict:
        return fingerprint._headers(self.path)

    def gguf(self):
        return _gguf(self)

    def signature(self) -> dict:
        g = self.gguf()
        if g:
            stamp = "|".join(f"{f}:{(self.path / f).stat().st_size}:{(self.path / f).stat().st_mtime_ns}" for f in g.files)
            return fingerprint.cached(f"{self.path.resolve()}:gguf", stamp, lambda: _remote_signature(self))
        return fingerprint.signature(self.path)

    def read(self, spans: list[tuple[str, int, int]]) -> list[bytes]:
        out = []
        for f, at, n in spans:
            with open(self.path / f if not Path(f).is_absolute() else f, "rb") as fh:
                fh.seek(at)
                out.append(fh.read(n))
            self.bytes_read += n
        return out

    def small_files(self) -> Path:
        return self.path


class Hub:
    """A model on the Hugging Face Hub, read by byte range. A token (HF_TOKEN, or a saved login) is sent to the
    Hub only, never to the storage it redirects to."""

    def __init__(self, repo: str, revision: str | None = None, token: str | None = None):
        import httpx
        from huggingface_hub import HfApi, constants
        from huggingface_hub.utils import build_hf_headers
        self.endpoint = constants.ENDPOINT.rstrip("/")
        try:
            info = HfApi(token=token).model_info(repo, revision=revision, files_metadata=True)
        except Exception as exc:  # noqa: BLE001  not found, gated, offline: all mean it can't be read
            raise Unreachable(f"{repo}: {type(exc).__name__}: {str(exc).splitlines()[0][:200]}") from exc
        if PUBLIC and getattr(info, "gated", False):
            raise Unreachable(f"{repo}: gated on the Hub; run LLMInspect yourself, logged in, to read it")
        self.name, self.sha = info.id, info.sha
        self.url = f"{self.endpoint}/{info.id}"
        self._sizes = {s.rfilename: s.size or 0 for s in info.siblings or []}
        card = info.card_data.to_dict() if info.card_data else {}
        self._claims = _claims(card.get("base_model"), info.tags or [])
        self._auth = build_hf_headers(token=token)
        self._client = httpx.Client(timeout=60, follow_redirects=False)
        self._where: dict[str, tuple[str, dict]] = {}
        self._lock = threading.Lock()
        self._sig = self._heads = self._dir = self._gguf = None
        self.bytes_read = 0

    def files(self) -> dict[str, int]:
        return dict(self._sizes)

    def claims(self) -> list[dict]:
        return list(self._claims)

    def _locate(self, f: str) -> tuple[str, dict]:
        """Where a file's bytes are: the Hub's URL, or the storage it redirects to (asked without the token)."""
        with self._lock:
            hit = self._where.get(f)
        if hit:
            return hit
        url, headers = f"{self.endpoint}/{self.name}/resolve/{self.sha}/{f}", dict(self._auth)
        for _ in range(5):
            r = self._client.head(url, headers=headers)
            if r.status_code in (301, 302, 303, 307, 308):
                nxt = str(r.url.join(r.headers["location"]))
                if r.url.host != r.url.join(r.headers["location"]).host:
                    headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
                url = nxt
                continue
            if r.status_code in (401, 403):
                raise Unreachable(f"{self.name}: access denied to {f} (gated; log in or set HF_TOKEN)")
            r.raise_for_status()
            break
        with self._lock:
            self._where[f] = (url, headers)
        return url, headers

    def _range(self, f: str, at: int, n: int) -> bytes:
        url, headers = self._locate(f)
        with self._client.stream("GET", url, headers=headers | {"Range": f"bytes={at}-{at + n - 1}"}) as r:
            if r.status_code == 206:
                data = r.read()
            elif r.status_code == 200:  # the range was ignored: read up to it and stop, never the whole file
                data, got = b"", 0
                for chunk in r.iter_bytes():
                    data += chunk
                    if len(data) >= at + n:
                        break
                data = data[at:at + n]
            else:
                raise Unreachable(f"{self.name}: {f} answered {r.status_code}")
        with self._lock:
            self.bytes_read += len(data)
        return data

    def read(self, spans: list[tuple[str, int, int]]) -> list[bytes]:
        with ThreadPoolExecutor(WORKERS) as ex:
            return list(ex.map(lambda s: self._range(*s), spans))

    def headers(self) -> dict:
        """{tensor: (file, header entry, data start)} from each safetensors file's header."""
        if self._heads is None:
            shards = [f for f in self._sizes if f.endswith(".safetensors") and "/" not in f]
            heads = {}

            def one(f):
                first = self._range(f, 0, 1 << 16)
                n = struct.unpack("<Q", first[:8])[0]
                raw = first[8:8 + n] if 8 + n <= len(first) else first[8:] + self._range(f, len(first), 8 + n - len(first))
                return f, json.loads(raw), 8 + n

            with ThreadPoolExecutor(WORKERS) as ex:
                for f, header, start in ex.map(one, shards):
                    heads.update({k: (f, v, start) for k, v in header.items() if k != "__metadata__"})
            self._heads = heads
        return self._heads

    def gguf(self):
        return _gguf(self)

    def signature(self) -> dict:
        if self._sig is None:
            g = self.gguf()
            stamp = f"{self.sha}:" + ",".join(g.files) if g else self.sha
            self._sig = fingerprint.cached(f"hub:{self.name}", stamp, lambda: _remote_signature(self))
        return self._sig

    def small_files(self) -> Path:
        """Configs, code and templates, copied to a temporary folder."""
        if self._dir is None:
            TMP.mkdir(parents=True, exist_ok=True)
            self._dir = Path(tempfile.mkdtemp(prefix="hub-", dir=TMP))
            want = [f for f, size in self._sizes.items() if Path(f).suffix.lower() in TEXT and size <= SMALL]
            with ThreadPoolExecutor(WORKERS) as ex:
                for f, data in zip(want, ex.map(lambda f: self._range(f, 0, max(1, self._sizes[f])), want)):
                    dest = self._dir / f
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
        return self._dir


def prune(older: float = 600) -> int:
    """Remove the temporary folders of reads that ended more than `older` seconds ago; how many went."""
    import shutil
    gone, now = 0, time.time()
    for d in TMP.glob("*") if TMP.is_dir() else []:
        try:
            if now - d.stat().st_mtime > older:
                shutil.rmtree(d, ignore_errors=True)
                gone += 1
        except OSError:
            continue
    return gone


def _claims(base_model, tags: list[str]) -> list[dict]:
    """[{model, relation}] from a card's base_model and the Hub's base_model:<relation>:<model> tags."""
    rel = {}
    for t in tags:
        m = re.match(r"base_model:(\w+):(.+)$", t)
        if m and m.group(1) in RELATIONS:
            rel[m.group(2)] = m.group(1)
    named = [base_model] if isinstance(base_model, str) else list(base_model or [])
    return [{"model": b, "relation": rel.get(b, "finetune")} for b in dict.fromkeys(named + list(rel))]


def _gguf(src):
    """A source's GGUF weights (its most precise quantization), when it has no safetensors weights; else None."""
    if src._gguf is None:
        files = src.files()
        chosen = [] if any(f.endswith(".safetensors") for f in files) else ggufread.choose(files)
        try:
            src._gguf = ggufread.Weights(src, chosen, files) if chosen else False
        except ValueError as exc:
            raise Unreachable(f"{src.name}: {chosen[0]}: {exc}") from exc
    return src._gguf or None


def _remote_signature(src) -> dict:
    g = src.gguf()
    if g:
        names = [n for n in sorted(g.tensors) if fingerprint.ROLES.search(n) and g.readable(n)]
        idx = [np.concatenate([np.arange(a, a + n) for a, n in fingerprint.positions(int(np.prod(g.shape(k))))])
               for k in names]
        got = g.gather(list(zip(names, idx)))
        return {k: [g.shape(k), [float(x) for x in v]] for k, (v, _) in zip(names, got)}
    wanted = [(fingerprint.canonical(k), f, e, s) for k, (f, e, s) in sorted(src.headers().items())
              if fingerprint.ROLES.search(k) and e.get("dtype") in fingerprint.DTYPES]
    jobs = [(i, (f, at, n)) for i, (_, f, e, s) in enumerate(wanted) for at, n in fingerprint.spans(e, s)]
    raws = src.read([j for _, j in jobs])
    vals: dict[int, list[float]] = {}
    for (i, _), raw in zip(jobs, raws):
        vals.setdefault(i, []).extend(fingerprint.decode(wanted[i][2]["dtype"], raw))
    return {name: [e["shape"], vals.get(i, [])] for i, (name, _, e, _) in enumerate(wanted)}


def open_model(model: str, local: dict[str, Path] | None = None):
    """A source for a model: a folder path, a model on this machine by its Hub name, or the Hub."""
    if PUBLIC:
        return Hub(model, token=os.environ.get("HF_TOKEN"))
    p = Path(model).expanduser()
    if p.is_dir():
        return Folder(p)
    hit = (local or {}).get(model.lower())
    return Folder(hit, model) if hit else Hub(model, token=os.environ.get("HF_TOKEN"))


def local_models(roots=None) -> dict[str, Path]:
    """{lowercased name: folder} for the models on this machine."""
    from .check import default_roots, model_name
    return {model_name(p).lower(): p for _, p in filescan.find_repos(default_roots() if roots is None else roots)}


# ---- the edit: the rank of the difference over full rows ---------------------------------------------------

def _rows(src, layers: list[int]) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray | None]]:
    """ROWS full rows of each residual writer (WRITERS) in each layer, at the same evenly spread rows in every model,
    with each value's rounding step when the weights are quantized (else None), by (writer, layer)."""
    key = tuple(layers)
    hit = getattr(src, "_rows_seen", {}).get(key)
    if hit is None:
        hit = _read_rows(src, layers)
        src.__dict__.setdefault("_rows_seen", {})[key] = hit
    return hit


def _writer(name: str) -> tuple[str, int] | None:
    m = WRITER.search(name)
    return (m.group(2).replace("feed_forward.w2", "mlp.down_proj"), int(m.group(1))) if m else None


def _read_rows(src, layers: list[int]) -> dict:
    g = src.gguf()
    if g:
        want = []
        for layer in layers:
            for role in WRITERS:
                name = f"layers.{layer}.{role}.weight"
                if name in g.tensors and g.readable(name):
                    out_dim, in_dim = g.shape(name)
                    rows = np.linspace(0, out_dim - 1, min(ROWS, out_dim)).round().astype(np.int64)
                    want.append(((role, layer), name, (rows[:, None] * in_dim + np.arange(in_dim)).ravel(), in_dim))
        got = g.gather([(name, idx) for _, name, idx, _ in want])
        return {key: (v.astype(np.float64).reshape(-1, n), s.astype(np.float64).reshape(-1, n))
                for (key, _, _, n), (v, s) in zip(want, got)}
    spans, keys = [], []
    for name, (f, e, start) in src.headers().items():
        w = _writer(name)
        if not w or w[1] not in layers or e.get("dtype") not in fingerprint.DTYPES or len(e["shape"]) != 2:
            continue
        out_dim, in_dim = e["shape"]
        size = fingerprint.DTYPES[e["dtype"]][1]
        for r in np.linspace(0, out_dim - 1, min(ROWS, out_dim)).round().astype(int):
            spans.append((f, start + e["data_offsets"][0] + int(r) * in_dim * size, in_dim * size))
            keys.append((w, e["dtype"]))
    out: dict[tuple, list] = {}
    for (w, dtype), raw in zip(keys, src.read(spans)):
        out.setdefault(w, []).append(fingerprint.decode(dtype, raw))
    return {k: (np.asarray(v, dtype=np.float64), None) for k, v in out.items()}


def _energies(d: np.ndarray, noise: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, float]:
    """The change's directions (over the sampled rows) and the energy of each, and the change's total energy.
    With rounding noise (the variance of each value's rounding), rows and columns are first scaled so the noise is
    even, and only directions standing out above it count, plus any broad excess over it."""
    if noise is None or not noise.any():
        u, s, _ = np.linalg.svd(d, full_matrices=False)
        return u, s ** 2, float((s ** 2).sum())
    floor = noise.max() * 1e-12 + 1e-300
    wr = np.sqrt(np.maximum(noise.mean(1), floor))
    wc = np.sqrt(np.maximum((noise / wr[:, None] ** 2).mean(0), floor))
    dw = d / wr[:, None] / wc[None, :]
    var = float((noise / wr[:, None] ** 2 / wc[None, :] ** 2).mean())
    m, n = dw.shape
    u, s, vt = np.linalg.svd(dw, full_matrices=False)
    s2 = s ** 2
    # a direction resting on one value is a value the quantizer clipped (an outlier weight), not an edit: real
    # edits spread over a row. Those directions are set aside, out of both the change and the noise.
    keep = (vt ** 2).max(1) < POINT
    spike = keep & (s2 > SPIKE * n * var * (1 + np.sqrt(m / n)) ** 2)
    rest = keep & ~spike
    bulk, expect = float(s2[rest].sum()), float(rest.sum() * n * var)
    broad = bulk - expect if bulk > BROAD * expect else 0.0
    top = np.flatnonzero(spike)
    e = np.concatenate([s2[top] - n * var, np.zeros(int(rest.sum()))])
    # directions back in the unscaled rows (the scaling bent them), orthonormal again
    q = np.linalg.qr(wr[:, None] * u[:, top])[0] if len(top) else u[:, :0]
    # energies back on the unscaled scale: each one's share times the unscaled change's size
    total = float(e.sum() + broad)
    scale = float((d ** 2).sum()) / max(float(s2.sum()), 1e-300)
    return np.concatenate([q, u[:, rest]], axis=1), e * scale, total * scale


def _shape_of_change(cand, parent, layers: list[int]) -> dict:
    """Per sampled layer: how large the change is, how many directions hold most of it (k), and whether the
    layers whose change lies in a few directions share them (the overlap of their subspaces, 0..1). Measured in
    each residual writer both models have; the one that moved in the most layers (then the most) is reported."""
    a, b = _rows(cand, layers), _rows(parent, layers)
    g = cand.gguf()
    best = None
    for role in WRITERS:
        keys = sorted(k for k in set(a) & set(b) if k[0] == role)
        if not keys:
            continue
        r = _writer_change({k[1]: a[k] for k in keys}, {k[1]: b[k] for k in keys})
        score = (sum(1 for p in r["layers"].values() if p["rel"] > 0), sum(p["rel"] for p in r["layers"].values()))
        if best is None or score > best[0]:
            best = (score, r | {"matrix": role})
    out = best[1] if best else {"layers": {}, "few": [], "shared": None, "matrix": None}
    return out | {"quant": g.quant if g else None}


def _writer_change(a: dict, b: dict) -> dict:
    per, dirs = {}, {}
    for layer in sorted(set(a) & set(b)):
        (va, sa), (vb, sb) = a[layer], b[layer]
        if va.shape != vb.shape:
            continue
        noise = None if sa is None and sb is None else (
            (sa if sa is not None else 0) ** 2 + (sb if sb is not None else 0) ** 2) / 12 + np.zeros_like(va)
        u, e, total = _energies(va - vb, noise)
        rel = float(np.sqrt(total)) / max(float(np.linalg.norm(vb)), 1e-30)
        if total <= 0:
            per[layer] = {"rel": 0.0, "top1": None, f"top{LOW_K}": None, "k": 0}
            continue
        share = np.cumsum(e) / total
        k = int(np.searchsorted(share, MOST) + 1) if len(e) and share[-1] >= MOST else len(e) + 1
        per[layer] = {"rel": round(rel, 5), "top1": round(float(e[:1].sum() / total), 3),
                      f"top{LOW_K}": round(float(e[:LOW_K].sum() / total), 3), "k": k}
        if k <= FEW:
            dirs[layer] = u[:, :k]
    few = sorted(dirs)
    pairs = [float(np.linalg.norm(dirs[x].T @ dirs[y]) ** 2) / min(dirs[x].shape[1], dirs[y].shape[1])
             for i, x in enumerate(few) for y in few[i + 1:]]
    return {"layers": {str(k): v for k, v in per.items()}, "few": few,
            "shared": round(float(np.median(pairs)), 3) if pairs else None}


def _name_edit(sim: dict, change: dict) -> dict:
    """What was done, from the fingerprint and the rank of the change. A quantized model's fingerprint moves in
    every layer from rounding alone, so there the measured layers say which moved."""
    per, quant = change["layers"], change.get("quant")
    changed = [l for l, p in per.items() if p["rel"] > 0]
    if quant:
        moved, total = sorted(int(l) for l in changed), len([l for l in sim["layers"] if int(l) >= 0])
        of = f"{total} layers" if len(per) >= total else f"{len(per)} layers read"
    else:
        moved = sorted(int(l) for l, c in sim["layers"].items() if int(l) >= 0 and c < 0.99999)
        of = f"{len([l for l in sim['layers'] if int(l) >= 0])} layers"
    if not per and quant:  # rounding hides the fingerprint's answer, and no output projection could be read
        return {"kind": "unmeasured", "headline": "its edit couldn't be measured through the rounding", "moved_layers": []}
    if not changed and (quant or sim["identical"] == sim["tensors"]):
        return {"kind": "same", "headline": f"the same weights, {ggufread.described(quant)[1]}" if quant else
                "the same weights, re-uploaded", "moved_layers": []}
    span = f"layers {moved[0]}–{moved[-1]}" if len(moved) > 1 else f"layer {moved[0]}" if moved else "no sampled layer"
    few, many = change["few"], len(changed)
    one = [l for l in few if per[str(l)]["k"] == 1]
    if many and len(few) >= max(2, 0.6 * many) and (change["shared"] or 0) >= SHARED:
        k = int(np.median([per[str(l)]["k"] for l in few]))
        what = "one direction" if k == 1 else f"{k} directions"
        kind, head = "direction_removed", f"{what} removed from {len(moved) or many} of {of}, the signature of abliteration"
    elif many and len(one) >= 0.6 * many and len(moved) >= 0.5 * (len(per) if quant else len(
            [l for l in sim["layers"] if int(l) >= 0])):  # read layers of a GGUF are a sample of all; else moved ones
        # rank one in most of the model's layers, each in its own direction: abliteration that takes a separate
        # direction per layer. Targeted fact edits touch a handful of layers.
        kind, head = "direction_removed", (f"a different direction removed from each of {len(moved) or many} of {of}, "
                                           "the signature of abliteration with per-layer directions")
    elif many and len(one) >= 0.6 * many:
        kind, head = "rank_one", f"separate rank-one edits in {span}, like targeted fact edits"
    elif many and all(p[f"top{LOW_K}"] is not None and p[f"top{LOW_K}"] >= LOW_RANK for p in per.values() if p["rel"] > 0):
        kind, head = "low_rank", f"a low-rank change in {span}, like a merged adapter (LoRA)"
    elif sim["median"] > 0.9:
        kind, head = "fine_tuned", f"fine-tuned: weights moved in {len(moved)} of {of}"
    else:
        kind, head = "retrained", "heavily retrained: little of the parent's weights is left"
    return {"kind": kind, "headline": head, "moved_layers": moved}


# ---- unmask --------------------------------------------------------------------------------------------------

def _shapes(sig: dict) -> dict:
    return {k: v[0] for k, v in sig.items()}


def _comparable(a: dict, b: dict) -> bool:
    same = sum(1 for k, shape in _shapes(a).items() if k in b and b[k][0] == shape)
    return bool(a) and same >= 0.8 * max(len(a), len(b))


def _ancestry(claims: list[dict], local: dict, depth: int = DEPTH) -> list[dict]:
    """The claimed parents, then each first parent's own claims, up the chain."""
    out, seen, todo = [], set(), [(c, 1) for c in claims]
    while todo:
        c, d = todo.pop(0)
        if c["model"] in seen or d > depth:
            continue
        seen.add(c["model"])
        row = {"model": c["model"], "relation": c["relation"], "depth": d}
        try:
            src = open_model(c["model"], local)
            row["claims"] = [x["model"] for x in src.claims()]
            todo += [(x, d + 1) for x in src.claims()[:1]]
        except Unreachable as exc:
            row["error"] = str(exc)
        out.append(row)
    return out


EDIT_WORDS = re.compile(r"^(abliterated|ablit|abliteration|uncensored|decensored|heretic|orthogonali[sz]ed|unfiltered|"
                        r"unaligned|nsfw|gguf|ggml|awq|gptq|exl2|mlx|bnb|4bit|8bit|int4|int8|fp8|fp16|bf16|i1|imatrix|"
                        r"q\d.*|iq\d.*|v\d+(\.\d+)?|mtp|hf)$", re.I)
CANDIDATES = 8      # Hub models compared when looking for an unnamed parent


def _core_name(name: str) -> str:
    """A model's name without its org and the words re-uploads and edits add: the name its parent likely has."""
    base = re.sub(r"(?i)(^|[-_.])i?q\d(_[a-z0-9]+)*(?=$|[-_.])", "", name.split("/")[-1])  # Q4_K_M and the like
    words = [w for w in re.split(r"[-_. ]+", base) if w]
    return "-".join(w for w in words if not EDIT_WORDS.match(w))


def _hub_candidates(cand, say) -> list[str]:
    """Likely parents on the Hub for a model whose card names none: models named like it with the edit words
    taken off, ungated and most downloaded first, then the parents those name."""
    from huggingface_hub import HfApi
    core = _core_name(cand.name)
    if not core:
        return []
    say(f"[unmask] its card names no parent; searching the Hub for {core}")
    api, seen, out = HfApi(), {cand.name.lower()}, []
    words = core.split("-")
    for query in dict.fromkeys(["-".join(words), "-".join(words[:-1]) if len(words) > 2 else ""]):
        if not query:
            continue
        try:
            found = list(api.list_models(search=query, sort="downloads", limit=40, expand=["downloads", "gated", "tags"]))
        except Exception:  # noqa: BLE001  the search is a guess; without it the result says no parent was found
            return out
        clean = [m for m in found if not any(EDIT_WORDS.match(w) and not re.match(r"v\d", w, re.I)
                                             for w in re.split(r"[-_. ]+", m.id.split("/")[-1]))]
        for m in clean + [m for m in found if m not in clean]:
            ids = [m.id] + [t.split(":")[-1] for t in (m.tags or []) if t.startswith("base_model:") and t.count(":") >= 2]
            for i in ids:
                if i.lower() not in seen and not (m.gated and i == m.id and (PUBLIC or not os.environ.get("HF_TOKEN"))):
                    seen.add(i.lower())
                    out.append(i)
        if len(out) >= CANDIDATES:
            break
    return out[:CANDIDATES]


def _verdict(s: dict | None, quantized=None) -> str:
    if s is None:
        return "another architecture: can't be compared"
    return ("the same weights" if s["identical"] == s["tensors"] else "close: the same weights or derived from it, "
            "told apart by the edit" if quantized and s["median"] > 0.9 else "derived from it" if s["median"] > 0.9 else
            "related, heavily retrained" if s["median"] >= 0.15 else "not derived from it")


def unmask(model: str, roots=None, say=lambda m: None) -> dict:
    """Who a model's parents are, whether its weights agree, what was done to them, and what its files hold."""
    t0 = time.perf_counter()
    local = local_models(roots)
    cand = open_model(model, local)
    say(f"[unmask] {cand.name}: reading the fingerprint")
    sig = cand.signature()
    out = {"model": cand.name, "url": cand.url, "sha": getattr(cand, "sha", None), "lineage": [], "parent": None,
           "edit": None, "checked": [], "not_checked": []}
    if not sig:
        out["not_checked"].append("origin and edits: no safetensors weights, or GGUF weights of a supported "
                                  "quantization, to read")
    g = cand.gguf()
    if g:
        out["quant"] = {"type": g.quant, "files": g.files}
    lineage = _ancestry(cand.claims(), local)
    out["lineage"] = lineage

    tried, best = [], None
    for row in lineage:
        if row["depth"] != 1:
            continue
        name, relation = row["model"], row["relation"]
        if "error" in row:
            tried.append({"model": name, "claimed": relation, "reading": row["error"], "unreadable": True})
            continue
        if relation == "adapter" or (relation == "quantized" and not g):
            tried.append({"model": name, "claimed": relation, "reading": f"{relation}: weights aren't comparable"})
            continue
        try:
            src = open_model(name, local)
            say(f"[unmask] {name}: reading the claimed parent's fingerprint")
            psig = src.signature()
        except Unreachable as exc:
            tried.append({"model": name, "claimed": relation, "reading": str(exc), "unreadable": True})
            continue
        s = fingerprint.similarity(sig, psig) if _comparable(sig, psig) else None
        tried.append({"model": name, "claimed": relation, "reading": _verdict(s, g), "median": s and round(s["median"], 4)})
        if s and s["median"] > 0.15 and (best is None or s["median"] > best[2]["median"]):
            best = (name, src, s, psig, "card")
    if sig and (best is None or best[2]["median"] <= 0.9):  # the claim fails or is missing: look on this machine
        for name, folder in sorted(local.items()):
            if isinstance(cand, Folder) and folder.resolve() == cand.path.resolve():
                continue
            if not _comparable(sig, {fingerprint.canonical(k): [v[1].get("shape")] for k, v in fingerprint._headers(folder).items()
                                     if fingerprint.ROLES.search(k)}):
                continue
            src = Folder(folder)
            s = fingerprint.similarity(sig, src.signature())
            if s and s["median"] > 0.9 and (best is None or s["median"] > best[2]["median"]):
                best = (src.name, src, s, src.signature(), "this machine")
    if sig and (best is None or best[2]["median"] <= 0.9) and isinstance(cand, Hub):  # nor here: search the Hub
        for name in _hub_candidates(cand, say):
            try:
                src = open_model(name, local)
                if not _comparable(sig, {fingerprint.canonical(k): [v[1].get("shape")] for k, v in src.headers().items()
                                         if fingerprint.ROLES.search(k)}):
                    continue
                say(f"[unmask] {name}: comparing a likely parent from the Hub")
                psig = src.signature()
            except (Unreachable, OSError, ValueError):
                continue
            s = fingerprint.similarity(sig, psig) if _comparable(sig, psig) else None
            if s and s["median"] > 0.9 and (best is None or s["median"] > best[2]["median"]):
                best = (name, src, s, psig, "the Hub")
                if s["median"] > 0.999:
                    break
    out["claims"] = tried

    read_from = [cand]
    if best:
        name, src, s, psig, found = best
        claimed = found == "card"
        read_from.append(src)
        edit = _edit(cand, src, s, say) if s["median"] > 0.9 else None
        copy, hops = None, 0
        while edit and edit["kind"] == "same" and hops < DEPTH:  # a copy: what does the model it copies carry?
            hops += 1
            up = [c["model"] for c in src.claims()][:1]
            if not up:
                break
            try:
                nsrc = open_model(up[0], local)
                say(f"[unmask] {name} is a copy; reading its parent {up[0]}")
                nsig = nsrc.signature()
            except Unreachable as exc:
                out["not_checked"].append(f"edits: {name} is a copy, and its parent {up[0]} couldn't be read ({exc})")
                break
            ns = fingerprint.similarity(sig, nsig) if _comparable(sig, nsig) else None
            if not ns or ns["median"] <= 0.9:
                break
            read_from.append(nsrc)
            nedit = _edit(cand, nsrc, ns, say)
            if nedit["kind"] == "same":  # the copy and its parent both match: the rounding hides what parts them
                out["not_checked"].append(f"edits: {name} and its parent {up[0]} both match; "
                                          f"{g.quant if g else 'the'} rounding hides any difference between them")
                break
            copy = copy or {"model": name, "headline": edit["headline"]}
            name, src, s, edit = up[0], nsrc, ns, nedit
        if copy:
            out["copy_of"] = copy
            for c in tried:
                if c["model"] == copy["model"]:
                    c["reading"] = copy["headline"]
        out["parent"] = {"model": name, "claimed": claimed, "found": found, "reading": _verdict(s, g), "median": round(s["median"], 4),
                         "layers": s["layers"]}
        if edit:
            out["edit"] = edit
            out["checked"].append("edits: the rank of the change in sampled layers" +
                                  (f", above the rounding of {g.quant}" if g else ""))
        out["checked"].append("origin: fingerprint against the claimed parents and this machine's models")
    elif sig:
        out["not_checked"].append("edits: no parent found to compare with (claimed parents unreadable, unrelated "
                                  "or another architecture, and no local model matches)")
    if g:
        out["not_checked"].append(f"edits smaller than {g.quant}'s rounding")

    say("[unmask] reading configs, code and templates")
    out["files"] = _files(cand, best and src)
    out["checked"].append("files: code that runs on load (pickles parsed by range), configs and chat templates")
    if out["files"]["pickles"]["unread"]:
        out["not_checked"].append(f"files: {len(out['files']['pickles']['unread'])} more pickle files")
    out["not_checked"] += ["behavior: what the model says (run the Behavior step with the parent loaded)",
                           "capability: what the edit cost on tasks"]
    out["headline"] = _headline(out)
    out["bytes_read"] = sum(x.bytes_read for x in read_from)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    return out


def _edit(cand, parent, s: dict, say) -> dict:
    """The edit from parent to cand: the rank of the change in the layers the fingerprint saw move (every layer,
    up to QUANT_LAYERS, in a quantized model, where rounding moves them all)."""
    every = [int(l) for l in s["layers"] if int(l) >= 0]
    moved = [l for l in every if s["layers"][str(l)] < 0.99999]
    pool, n = (every, QUANT_LAYERS) if cand.gguf() else (moved or every, RANK_LAYERS)
    pick = sorted({pool[int(i)] for i in np.linspace(0, len(pool) - 1, min(n, len(pool))).round()}) if pool else []
    say(f"[unmask] reading {ROWS} rows of {len(pick)} layers to measure the change")
    change = _shape_of_change(cand, parent, pick) if pick else {"layers": {}, "few": [], "shared": None}
    return _name_edit(s, change) | {"rank": change}


def _files(cand, parent) -> dict:
    files = cand.files()
    formats = sorted({Path(f).suffix.lower() for f in files if Path(f).suffix.lower() in filescan.WEIGHT_SUFFIXES})
    scan = filescan.scan_repo(cand.small_files(), cand.name)
    findings = [f for f in scan["findings"] if f["level"] != "info" or f["check"] == "code"]
    pickles = sorted(f for f in files if Path(f).suffix.lower() in filescan.PICKLE_SUFFIXES)
    unread = []
    if pickles and isinstance(cand, Hub):
        found, unread = _remote_pickles(cand, pickles)
        findings += found
    mine, g = cand.small_files(), cand.gguf()
    if g and isinstance(cand, Hub):  # a local GGUF's templates were scanned with its other files
        findings += [asdict(x) for k, v in g.meta.items() if k.startswith("tokenizer.chat_template") and isinstance(v, str)
                     for x in filescan.template_findings(g.files[0], k, v)]
    if g and isinstance(g.meta.get("tokenizer.chat_template"), str) and not filediff.templates(mine):
        TMP.mkdir(parents=True, exist_ok=True)
        mine = Path(tempfile.mkdtemp(prefix="gguf-", dir=TMP))  # a GGUF file carries its template inside
        (mine / "tokenizer_config.json").write_text(json.dumps({"chat_template": g.meta["tokenizer.chat_template"]}),
                                                     encoding="utf-8")
    template = filediff.compare_templates(parent.small_files(), mine) if parent else []
    order = {lvl: i for i, lvl in enumerate(filescan.LEVELS)}
    findings.sort(key=lambda f: order[f["level"]])
    return {"count": len(files), "bytes": sum(files.values()), "formats": formats, "findings": findings,
            "template": template, "pickles": {"read": len(pickles) - len(unread), "unread": unread}}


def _remote_pickles(cand, pickles: list[str]) -> tuple[list[dict], list[str]]:
    """Findings from a Hub repo's pickle files, parsed by range without unpickling, and the files left unread."""
    sizes, out, unread = cand.files(), [], pickles[PICKLES:]

    def one(f):
        try:
            fh = io.BufferedReader(RangeFile(cand, f, sizes[f]), BLOCK)
            found, _ = filescan.scan_pickle_stream(fh, f, Path(f).suffix)
            return [asdict(x) for x in found]
        except Exception as exc:  # noqa: BLE001  a broken or unreachable file is a finding, not a crash
            return [{"level": "format", "check": "pickle", "file": f, "evidence": [],
                     "detail": f"Couldn't be parsed by range: {type(exc).__name__}: {str(exc)[:200]}."}]

    with ThreadPoolExecutor(4) as ex:
        for found in ex.map(one, pickles[:PICKLES]):
            out += found
    return out, unread


def _where(p: dict) -> str:
    return "found on the Hub" if p.get("found") == "the Hub" else "here"


def _headline(r: dict) -> str:
    p, e = r["parent"], r["edit"]
    unread = [c["model"] for c in r.get("claims", []) if c.get("unreadable")]
    if p is None:
        what = "No parent found to compare with"
    elif p["median"] <= 0.9:
        what = f"Related to {p['model']} but heavily retrained"
    elif not p["claimed"] and (unread or not r.get("claims")):
        what = ("" if r.get("claims") else "Its card names no parent. ") + (
            f"Compared with {p['model']}, the closest model {_where(p)}: {e['headline']}" if e
            else f"Closest {_where(p)}: {p['model']}")
    else:
        what = f"{p['model']}, {e['headline']}" if e else p["model"]
    if r.get("copy_of"):
        q = (r.get("quant") or {}).get("type")
        what += f"; {ggufread.described(q)[0]} of {r['copy_of']['model']}" if q else f"; a re-upload of {r['copy_of']['model']}"
    false =[c["model"] for c in r.get("claims", []) if c["reading"] == "not derived from it"]
    if false and p and not p["claimed"]:
        what += f"; not {', '.join(false)} as its card claims"
    if unread:
        what += f"; the claimed {', '.join(unread)} couldn't be read"
    extra = []
    if any(t.get("added_text") for t in r["files"]["template"]):
        extra.append("its chat template adds text")
    top = r["files"]["findings"][0]["level"] if r["files"]["findings"] else None
    if top in ("runs code on load", "imports other code on load", "runs code if trusted"):
        extra.append(f"its files {top}")
    return what + ("; " + "; ".join(extra) if extra else "") + "."


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", help="ORG/NAME on the Hugging Face Hub, a model on this machine, or a folder")
    ap.add_argument("--json", action="store_true", help="print the whole result")
    args = ap.parse_args(argv)
    try:
        r = unmask(args.model, say=lambda m: print(m, file=sys.stderr))
    except Unreachable as exc:
        sys.exit(f"Can't read {args.model}: {exc}")
    if args.json:
        print(json.dumps(r, indent=1))
        return
    print(r["headline"])
    for row in r["lineage"]:
        print("  " * row["depth"] + f"↑ {row['relation']} of {row['model']}" + (f"  ({row['error']})" if "error" in row else ""))
    for c in r.get("claims", []):
        print(f"  claimed {c['model']}: {c['reading']}")
    print(f"read {r['bytes_read'] / 1e6:.1f} MB in {r['seconds']} s")


if __name__ == "__main__":
    main(sys.argv[1:])
