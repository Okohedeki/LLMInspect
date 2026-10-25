"""Weight anatomy: the model's weights grouped into units, and how much each unit writes per token.

A dense transformer uses every weight on every token, so "which weights were hit" is measured as
how much each group of weights wrote into the residual stream at that token:

  head     One attention head, standard or linear attention: its slice of the block's output
           projection (o_proj, out_proj). Activity is the exact norm of what the head writes,
           ||W_O[:, h] x_h||, computed from the per-head Gram matrix W_O[:, h]^T W_O[:, h].
  cluster  MLP neurons whose output weights (down_proj columns) point in similar directions,
           grouped once per model by sign-invariant spherical k-means on the original weights and
           cached. Activity is the cluster's write mass: the sum over its neurons of
           |activation| x ||output weight||. That is the total size of the neurons' individual
           writes, an upper bound on the norm of what they write together.

Each token stores one byte per unit on a log2 scale from 2^-16 to 2^16, eight steps per doubling.
The probe hooks the inputs of the output projections, so it also runs inside the decode CUDA
graphs: a single-token forward writes into one fixed buffer, which every graph replay refreshes.

Edges between units come from the weights too: unit B composes with an earlier unit A when B's
input weights, after the norm in front of B's block, read A's main output directions. The score is
||R_B F_A^T|| / (||R_B|| ||F_A||) for B's reader rows R_B and A's top output directions F_A (scaled
by their singular values), times sqrt(d_model) so a random direction scores about 1 ("x chance").
This is the composition score of Elhage et al. (2021) restricted to each unit's top directions. It
is the same for every token; the UI weights it by the units' activity at the current token.

The same module serves the weights themselves: any parameter or unit slice, binned to a requested
resolution (mean, RMS or largest-magnitude value per bin), with summary statistics.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .edits import ATTN_NAMES, MLP_NAMES, _child, attention_heads, residual_linears

ENC_LO, ENC_STEPS = -16, 8   # byte = round((log2 v - ENC_LO) * ENC_STEPS), clipped to 0..255
VERSION = 1                  # bump to rebuild cached cluster maps
POOL_BLOCK = 1 << 24         # fp32 elements per work block when binning a matrix
LABEL_CHUNK = 128            # directions per unembedding pass when labelling units
MAX_NEURON_ROWS = 256        # neurons listed, with labels, in a cluster's detail
EDGE_FACTORS = 4             # top output directions per unit used for edges
EDGES_KEPT = 16              # strongest incoming and outgoing edges kept per unit
EDGE_ROWS = 4096             # reader rows per matmul when scoring edges
EDGE_VERSION = 1
ATTN_NORMS = ("input_layernorm", "ln_1", "attention_norm", "norm1")
MLP_NORMS = ("pre_feedforward_layernorm", "post_attention_layernorm", "ln_2", "ffn_norm", "norm2")

MEASURES = {
    "head": "Norm of what the head writes into the residual stream at this token, ||W_O[:, h] x_h||.",
    "cluster": "Write mass: sum over the cluster's neurons of |activation| x ||output weight|| at this token "
               "(an upper bound on the norm of what the cluster writes).",
    "clusters": "MLP neurons grouped by the direction of their output weights (sign-invariant spherical "
                "k-means on the original down_proj columns), largest total weight first.",
    "labels": "Tokens the unit's main write direction pushes up (+) and down (−) through the final norm and "
              "unembedding. Weight-based, not measured on text.",
    "edges": "Weight-based composition: how strongly a unit's input weights (after the norm before its block) read "
             "an earlier unit's top output directions, relative to a random direction (1× = chance). The same for "
             "every token, computed from the original weights; not a measured causal effect.",
}


def encode(v: torch.Tensor) -> torch.Tensor:
    """Activity values -> log-scale bytes."""
    return ((v.float().clamp_min(2.0 ** ENC_LO).log2() - ENC_LO) * ENC_STEPS).round_().clamp_(0, 255).to(torch.uint8)


def decode(b: np.ndarray) -> np.ndarray:
    return np.exp2(b.astype(np.float32) / ENC_STEPS + ENC_LO)


@torch.no_grad()
def spherical_kmeans(x: torch.Tensor, k: int, iters: int = 40, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster the rows of x (n, d) by direction. A row and its negation count as the same direction,
    since an MLP neuron's activation takes either sign. Returns (cluster of each row, unit centroids);
    empty clusters are dropped, so there may be fewer than k."""
    u = x.float()
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-12)
    n, k = u.shape[0], min(k, u.shape[0])
    pick = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:k].to(u.device)
    c, assign = u[pick].clone(), None
    for _ in range(iters):
        sim = u @ c.T
        best = sim.abs().argmax(1)
        if assign is not None and torch.equal(best, assign):
            break
        assign = best
        sign = sim.gather(1, best[:, None]).sign()
        sign[sign == 0] = 1
        c = torch.zeros_like(c).index_add_(0, assign, u * sign)
        empty = (torch.bincount(assign, minlength=k) == 0).nonzero().flatten()
        if len(empty):  # re-seed empty clusters with the rows their clusters fit worst
            c[empty] = u[sim.abs().max(1).values.argsort()[:len(empty)]]
        c = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sim = u @ c.T
    assign = sim.abs().argmax(1)
    used = torch.unique(assign)
    remap = torch.full((k,), -1, dtype=torch.long, device=u.device)
    remap[used] = torch.arange(len(used), device=u.device)
    assign = remap[assign]
    sign = sim.gather(1, (used[assign])[:, None]).sign()
    sign[sign == 0] = 1
    c = torch.zeros(len(used), u.shape[1], device=u.device).index_add_(0, assign, u * sign)
    return assign, c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def pool(m: torch.Tensor, h: int, w: int) -> dict:
    """Bin a 2-D matrix into at most h x w cells: per-bin mean, RMS and signed largest-magnitude value,
    plus the RMS of each row bin and column bin. Works through the matrix in row blocks, so a
    vocabulary-sized embedding never needs a full fp32 copy."""
    R, C = m.shape
    bh, bw = -(-R // h), -(-C // w)
    hb, wb = -(-R // bh), -(-C // bw)
    s = torch.zeros(hb, wb, device=m.device)
    sq, hi, lo = torch.zeros_like(s), torch.zeros_like(s), torch.zeros_like(s)
    rows = max(bh, POOL_BLOCK // (wb * bw) // bh * bh)
    for r in range(0, R, rows):
        blk = m[r:r + rows].float()
        rr = blk.shape[0]
        nb = -(-rr // bh)
        if nb * bh != rr or wb * bw != C:
            # Zero padding leaves sums alone, and the signed largest magnitude still comes from a real
            # value: a padded zero never beats a bin's real values on both sides at once.
            blk = torch.nn.functional.pad(blk, (0, wb * bw - C, 0, nb * bh - rr))
        v = blk.view(nb, bh, wb, bw)
        i = r // bh
        s[i:i + nb] = v.sum((1, 3))
        sq[i:i + nb] = v.square().sum((1, 3))
        hi[i:i + nb] = v.amax((1, 3))
        lo[i:i + nb] = v.amin((1, 3))
    rc = torch.full((hb,), float(bh), device=m.device)
    rc[-1] = R - (hb - 1) * bh
    cc = torch.full((wb,), float(bw), device=m.device)
    cc[-1] = C - (wb - 1) * bw
    n = rc[:, None] * cc[None]
    return {"bh": bh, "bw": bw, "mean": s / n, "rms": (sq / n).sqrt(), "absmax": torch.where(hi >= -lo, hi, lo),
            "row_rms": (sq.sum(1) / (rc * C)).sqrt(), "col_rms": (sq.sum(0) / (cc * R)).sqrt()}


@torch.no_grad()
def matrix_stats(m: torch.Tensor, bins: int = 48) -> dict:
    """Summary of every value in m, in row blocks."""
    rows = max(1, POOL_BLOCK // max(1, m.shape[1]))
    total = total_sq = zeros = 0.0
    amax = 0.0
    for r in range(0, m.shape[0], rows):
        blk = m[r:r + rows].float()
        total += blk.sum().item()
        total_sq += blk.square().sum().item()
        amax = max(amax, blk.abs().max().item())
        zeros += (blk == 0).sum().item()
    n = m.numel()
    hist = torch.zeros(bins, device=m.device)
    edge = amax or 1.0
    for r in range(0, m.shape[0], rows):
        hist += torch.histc(m[r:r + rows].float(), bins=bins, min=-edge, max=edge)
    mean = total / n
    return {"n": n, "mean": mean, "std": max(total_sq / n - mean * mean, 0.0) ** 0.5, "rms": (total_sq / n) ** 0.5,
            "norm": total_sq ** 0.5, "absmax": amax, "zeros": int(zeros), "hist": [int(x) for x in hist.tolist()],
            "range": [-edge, edge]}


@dataclass
class LayerMap:
    index: int
    type: str
    offset: int = 0
    block: nn.Module | None = None
    attn: nn.Module | None = None
    mlp: nn.Module | None = None
    heads: tuple | None = None                   # (output projection, number of heads, head dim)
    down: nn.Linear | None = None                # MLP output projection
    readers: list = field(default_factory=list)  # (name, linear) MLP input projections
    assign: torch.Tensor | None = None           # (neurons,) cluster of each MLP neuron
    centroids: torch.Tensor | None = None        # (clusters, d) unit write directions
    gram: torch.Tensor | None = None             # (heads, hd, hd) fp32, refreshed after weight edits
    wnorm: torch.Tensor | None = None            # (neurons,) fp32 output-weight norms

    @property
    def n_heads(self) -> int:
        return self.heads[1] if self.heads else 0

    @property
    def n_clusters(self) -> int:
        return 0 if self.centroids is None else self.centroids.shape[0]

    @property
    def n_units(self) -> int:
        return self.n_heads + self.n_clusters


class Anatomy:
    def __init__(self, subject, root: Path, clusters: int = 64):
        self.s, self.k = subject, clusters
        self.dir = Path(root) / subject.slug
        self.params = dict(subject.model.named_parameters())
        self._names = {id(p): n for n, p in self.params.items()}
        cfg = subject.model.config.get_text_config()
        types = list(getattr(subject, "layer_types", None) or [])
        d = subject.d_model
        self.layers: list[LayerMap] = []
        for i, block in enumerate(subject.layers):
            lm = LayerMap(i, types[i] if i < len(types) else "attention", block=block)
            lm.attn, lm.mlp = _child(block, ATTN_NAMES), _child(block, MLP_NAMES)
            lm.heads = attention_heads(lm.attn, d, cfg)
            downs = residual_linears(lm.mlp, d, "out")
            if len(downs) == 1:  # dense MLP; mixture-of-experts blocks are left out for now
                lm.down = downs[0]
                lm.readers = [(self._name(m).rsplit(".", 2)[-2], m) for m in residual_linears(lm.mlp, d, "in")
                              if m.out_features == lm.down.in_features]
            self.layers.append(lm)
        self.units: list[dict] = []
        self.build_seconds: float | None = None
        self.enabled = False
        self.parts: dict[int, torch.Tensor] = {}
        self.pending: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self._stats: OrderedDict = OrderedDict()
        self._stats_lock = threading.Lock()  # tile reads run off the recorder lock; refresh() clears mid-read
        self._stream = None
        self._load_or_build()
        self.edges: dict | None = None
        try:
            self._load_or_build_edges()
        except Exception as exc:  # edges are an extra; the map works without them
            print(f"[brainscan] unit edges unavailable: {type(exc).__name__}: {exc}", flush=True)
        self._prepare_probe()

    def _name(self, module: nn.Module) -> str:
        return self._names[id(module.weight)]

    # ---- the unit map ------------------------------------------------------------------

    def _cache_key(self) -> dict:
        return {"version": VERSION, "spec": self.s.spec, "k": self.k,
                "down": [list(lm.down.weight.shape) if lm.down is not None else None for lm in self.layers],
                "heads": [lm.n_heads for lm in self.layers]}

    def _load_or_build(self) -> None:
        key = self._cache_key()
        self.key = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
        meta_path, tensor_path = self.dir / f"atlas-k{self.k}.json", self.dir / f"atlas-k{self.k}.safetensors"
        if meta_path.exists() and tensor_path.exists():
            meta = json.loads(meta_path.read_text())
            old = meta.get("key") or {}
            # A cache of a different shape is another model's: never overwrite it (models are filed by name).
            if old and (old.get("down"), old.get("heads")) != (key["down"], key["heads"]):
                raise RuntimeError(f"Two models share the name {self.s.slug!r}: {self.dir} holds the unit map of "
                                   f"a different model ({old.get('spec')}). Set BRAINSCAN_MODEL_NAME to tell them apart.")
            if meta.get("key") == key:
                tensors = load_file(str(tensor_path), device=str(self.s.device))
                for lm in self.layers:
                    if f"assign.{lm.index}" in tensors:
                        lm.assign = tensors[f"assign.{lm.index}"].long()
                        lm.centroids = tensors[f"centroid.{lm.index}"].float()
                self.build_seconds = meta.get("seconds")
                self._finish(meta["units"])
                return
        t0 = time.perf_counter()
        tensors = {}
        with torch.no_grad():
            for lm in self.layers:
                if lm.down is None:
                    continue
                w = lm.down.weight.detach()
                assign, cent = spherical_kmeans(w.T, self.k, seed=lm.index)
                total = torch.zeros(cent.shape[0], device=w.device).index_add_(0, assign, w.float().norm(dim=0))
                order = total.argsort(descending=True)
                rank = torch.empty_like(order)
                rank[order] = torch.arange(len(order), device=order.device)
                lm.assign, lm.centroids = rank[assign], cent[order]
                tensors[f"assign.{lm.index}"] = lm.assign.to(torch.int32).cpu()
                tensors[f"centroid.{lm.index}"] = lm.centroids.to(torch.float16).cpu()
            units = self._label_units()
        self.build_seconds = round(time.perf_counter() - t0, 2)
        self.dir.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(tensor_path))
        meta_path.write_text(json.dumps({"key": key, "seconds": self.build_seconds, "units": units}))
        self._finish(units)

    def _finish(self, units: list[dict]) -> None:
        off = 0
        for lm in self.layers:
            lm.offset = off
            off += lm.n_units
        if len(units) != off:
            raise RuntimeError(f"unit map has {len(units)} units, the model {off}")
        self.n_units = off
        self.units = [u | {"g": g} for g, u in enumerate(units)]

    @torch.no_grad()
    def _label_units(self) -> list[dict]:
        d = self.s.d_model
        dirs, units = [], []
        for lm in self.layers:
            if lm.heads:
                o, H, hd = lm.heads
                W = o.weight.detach().float().view(d, H, hd)
                G = torch.einsum("dhi,dhj->hij", W, W)
                v1 = torch.linalg.eigh(G).eigenvectors[..., -1]  # top right singular vector per head
                u1 = torch.einsum("dhi,hi->hd", W, v1)
                dirs.append(u1 / u1.norm(dim=1, keepdim=True).clamp_min(1e-12))
                fro = G.diagonal(dim1=1, dim2=2).sum(-1).sqrt()
                units += [{"layer": lm.index, "kind": "head", "i": h, "size": hd, "norm": round(float(fro[h]), 4)}
                          for h in range(H)]
            if lm.centroids is not None:
                sq = lm.down.weight.detach().float().square().sum(0)
                fro = torch.zeros(lm.n_clusters, device=sq.device).index_add_(0, lm.assign, sq).sqrt()
                count = torch.bincount(lm.assign, minlength=lm.n_clusters)
                dirs.append(lm.centroids)
                units += [{"layer": lm.index, "kind": "cluster", "i": c, "size": int(count[c]),
                           "norm": round(float(fro[c]), 4)} for c in range(lm.n_clusters)]
        if dirs:
            for u, (pos, neg) in zip(units, self.top_tokens(torch.cat(dirs), 3)):
                u["pos"], u["neg"] = pos, neg
        return units

    @torch.no_grad()
    def top_tokens(self, directions: torch.Tensor, k: int = 5) -> list[tuple[list[str], list[str]]]:
        """Tokens each residual-stream direction pushes up and down through the final norm and unembedding."""
        out = []
        dtype = self.s.lm_head.weight.dtype
        for s in range(0, directions.shape[0], LABEL_CHUNK):
            logits = self.s.lm_head(self.s.norm(directions[s:s + LABEL_CHUNK].to(self.s.device, dtype))).float()
            up, down = logits.topk(k, dim=-1).indices.tolist(), (-logits).topk(k, dim=-1).indices.tolist()
            out += [([self.s.token_str(i) for i in a], [self.s.token_str(i) for i in b]) for a, b in zip(up, down)]
        return out

    def _matrix_ref(self, module) -> dict | None:
        w = getattr(module, "weight", None)
        return None if w is None or id(w) not in self._names else {"name": self._names[id(w)], "shape": list(w.shape)}

    def describe(self) -> dict:
        model = self.s.model
        return {"key": self.key, "model": self.s.name, "n_layers": len(self.layers), "d_model": self.s.d_model,
                "k": self.k, "n_units": self.n_units, "build_seconds": self.build_seconds,
                "embed": self._matrix_ref(model.get_input_embeddings()),
                "unembed": self._matrix_ref(model.get_output_embeddings()),
                "encoding": {"lo": ENC_LO, "steps": ENC_STEPS}, "measures": MEASURES,
                "layers": [{"layer": lm.index, "type": lm.type, "offset": lm.offset, "heads": lm.n_heads,
                            "head_dim": lm.heads[2] if lm.heads else 0, "clusters": lm.n_clusters,
                            "neurons": lm.down.in_features if lm.down is not None else 0} for lm in self.layers],
                "units": self.units}

    def locate(self, g: int) -> tuple[LayerMap, str, int]:
        for lm in self.layers:
            if lm.offset <= g < lm.offset + lm.n_units:
                i = g - lm.offset
                return (lm, "head", i) if i < lm.n_heads else (lm, "cluster", i - lm.n_heads)
        raise KeyError(f"no unit {g}")

    def edited_units(self, stack) -> list[int]:
        """Units whose weights the applied workbench edits change."""
        out: set[int] = set()
        for eid in stack.applied:
            e = stack.edits.get(eid)
            if e is None:
                continue
            if e.kind == "units":
                for u in e.units:
                    lm = self.layers[u["layer"]]
                    if u["target"] == "head" and u["index"] < lm.n_heads:
                        out.add(lm.offset + u["index"])
                    elif u["target"] == "neuron" and lm.assign is not None:
                        idx = torch.tensor(u["index"], device=lm.assign.device)
                        out.update(lm.offset + lm.n_heads + int(c) for c in lm.assign[idx].unique())
                continue
            for layer in e.layers:
                lm = self.layers[layer]
                heads = range(lm.offset, lm.offset + lm.n_heads)
                clusters = range(lm.offset + lm.n_heads, lm.offset + lm.n_units)
                targets = e.targets if e.kind == "orthogonalize" else [e.targets[0]]
                if any(t in ("attn", "attn_out", "attn_in") for t in targets):
                    out.update(heads)
                if any(t in ("mlp", "mlp_out", "mlp_in") for t in targets):
                    out.update(clusters)
                if "head" in targets and e.index < lm.n_heads:
                    out.add(lm.offset + e.index)
                if "neuron" in targets and lm.assign is not None:
                    idx = torch.tensor(e.index if isinstance(e.index, list) else [e.index], device=lm.assign.device)
                    out.update(lm.offset + lm.n_heads + int(c) for c in lm.assign[idx].unique())
        return sorted(out)

    # ---- edges -------------------------------------------------------------------------

    def _load_or_build_edges(self) -> None:
        path = self.dir / f"edges-k{self.k}.safetensors"
        key = f"{self.key}-{EDGE_VERSION}-{EDGE_FACTORS}"
        if path.exists():
            with contextlib.suppress(Exception):
                t = load_file(str(path))
                if bytes(t["key"].tolist()).decode() == key:
                    self._set_edges(t["score"].float(), float(t["seconds"][0]))
                    return
        t0 = time.perf_counter()
        score = self.composition()
        seconds = round(time.perf_counter() - t0, 2)
        self.dir.mkdir(parents=True, exist_ok=True)
        save_file({"score": score.half(), "key": torch.tensor(list(key.encode()), dtype=torch.uint8),
                   "seconds": torch.tensor([seconds])}, str(path))
        self._set_edges(score, seconds)

    def _set_edges(self, score: torch.Tensor, seconds: float | None) -> None:
        """Keep each unit's strongest incoming and outgoing edges. score[b, a] scores a -> b."""
        k = max(0, min(EDGES_KEPT, score.shape[0] - 1))  # a one-unit model has no edges; topk(-1) would raise
        up_x, up = score.topk(k, dim=1)
        down_x, down = score.T.topk(k, dim=1)
        nz = score[score > 0]
        sample = nz[torch.randperm(nz.numel())[:1_000_000]] if nz.numel() else torch.ones(1)
        self.edges = {"k": k, "up": up.int(), "up_x": up_x, "down": down.int(), "down_x": down_x,
                      "p50": float(sample.median()), "p99": float(sample.quantile(0.99)), "seconds": seconds}

    def describe_edges(self) -> dict:
        e = self.edges
        if e is None:
            return {"available": False, "measure": MEASURES["edges"]}

        def r(t):
            return [round(x, 2) for x in t.flatten().tolist()]
        return {"available": True, "k": e["k"], "measure": MEASURES["edges"], "p50": round(e["p50"], 3),
                "p99": round(e["p99"], 3), "seconds": e["seconds"],
                "up": e["up"].flatten().tolist(), "up_x": r(e["up_x"]),
                "down": e["down"].flatten().tolist(), "down_x": r(e["down_x"])}

    def _norm_gain(self, lm: LayerMap, names) -> torch.Tensor:
        """Per-dimension gain of the norm in front of a block: what it multiplies a unit-RMS vector by."""
        d, dev = self.s.d_model, self.s.device
        norm = _child(lm.block, names) if lm.block is not None else None
        if norm is None:
            return torch.ones(d, device=dev)
        if isinstance(norm, nn.LayerNorm):
            return norm.weight.detach().float() if norm.weight is not None else torch.ones(d, device=dev)
        p = next(norm.parameters(), None)
        return norm(torch.ones(1, d, device=dev, dtype=p.dtype if p is not None else torch.float32)).float()[0]

    @torch.no_grad()
    def write_factors(self, k: int = EDGE_FACTORS) -> torch.Tensor:
        """(units, k, d): each unit's top output directions scaled by their singular values."""
        d, dev = self.s.d_model, self.s.device
        F = torch.zeros(self.n_units, k, d, device=dev)
        for lm in self.layers:
            if lm.heads:
                o, H, hd = lm.heads
                W = o.weight.detach().float().view(d, H, hd)
                vecs = torch.linalg.eigh(torch.einsum("dhi,dhj->hij", W, W)).eigenvectors[..., -k:].flip(-1)
                F[lm.offset:lm.offset + H, :vecs.shape[-1]] = torch.einsum("dhi,hik->hkd", W, vecs)
            if lm.centroids is not None:
                W = lm.down.weight.detach().float()
                for c in range(lm.n_clusters):
                    Wc = W[:, lm.assign == c]
                    vecs = torch.linalg.eigh(Wc.T @ Wc).eigenvectors[:, -k:].flip(-1)
                    F[lm.offset + lm.n_heads + c, :vecs.shape[1]] = (Wc @ vecs).T
        return F

    def _reader_blocks(self, lm: LayerMap) -> list[tuple[torch.Tensor, torch.Tensor, str]]:
        """(weight, membership (layer units, rows), 'attn' or 'mlp') for each matrix that reads the
        residual stream into one of this layer's units. Rows can belong to several heads (shared keys)."""
        dev, blocks = self.s.device, []
        if lm.heads:
            by_module: dict[int, tuple] = {}
            for h in range(lm.n_heads):
                for _, _, _, module, a, b in self._head_readers(lm, h):
                    by_module.setdefault(id(module), (module, []))[1].append((h, a, b))
            for module, ranges in by_module.values():
                M = torch.zeros(lm.n_units, module.out_features, device=dev)
                for h, a, b in ranges:
                    M[h, a:b] = 1
                blocks.append((module.weight.detach(), M, "attn"))
        if lm.centroids is not None:
            for _, m in lm.readers:
                M = torch.zeros(lm.n_units, m.out_features, device=dev)
                M[lm.n_heads + lm.assign, torch.arange(m.out_features, device=dev)] = 1
                blocks.append((m.weight.detach(), M, "mlp"))
        return blocks

    @torch.no_grad()
    def composition(self) -> torch.Tensor:
        """score[b, a] for every pair of units, a -> b; zero where a does not come before b."""
        d, dev = self.s.d_model, self.s.device
        F = self.write_factors()
        k = F.shape[1]
        fnorm2 = F.square().sum((1, 2))
        score = torch.zeros(self.n_units, self.n_units)
        for lm in self.layers:
            n_src = lm.offset + lm.n_heads  # the most any unit of this layer can read from
            if not lm.n_units or not n_src:
                continue
            gains = {"attn": self._norm_gain(lm, ATTN_NORMS), "mlp": self._norm_gain(lm, MLP_NORMS)}
            comp = torch.zeros(lm.n_units, n_src, device=dev)
            rnorm2 = torch.zeros(lm.n_units, device=dev)
            for W, M, kind in self._reader_blocks(lm):
                end = lm.offset if kind == "attn" else n_src  # MLPs also read this layer's heads
                gain = gains[kind]
                Fs = (F[:end] * gain).reshape(end * k, d).to(W.dtype) if end else None
                for r in range(0, W.shape[0], EDGE_ROWS):
                    Wc, Mc = W[r:r + EDGE_ROWS], M[:, r:r + EDGE_ROWS]
                    rnorm2 += Mc @ (Wc.float() * gain).square().sum(1)
                    if Fs is not None:
                        E = (Wc @ Fs.T).float().square().view(Wc.shape[0], end, k).sum(-1)
                        comp[:, :end] += Mc @ E
            denom = rnorm2[:, None] * fnorm2[None, :n_src]
            block = torch.where(denom > 0, (comp / denom.clamp_min(1e-30)).sqrt() * d ** 0.5, torch.zeros_like(comp))
            score[lm.offset:lm.offset + lm.n_units, :n_src] = block.cpu()
        return score

    # ---- the probe ---------------------------------------------------------------------

    def _prepare_probe(self) -> None:
        dev = self.s.device
        self.step = torch.zeros(self.n_units, device=dev)  # activity of the latest single-token forward
        self.handles = []
        for lm in self.layers:
            if lm.heads:
                _, H, hd = lm.heads
                lm.gram = torch.empty(H, hd, hd, device=dev)
                self.handles.append(lm.heads[0].register_forward_pre_hook(self._head_hook(lm)))
            if lm.centroids is not None:
                lm.wnorm = torch.empty(lm.down.in_features, device=dev)
                self.handles.append(lm.down.register_forward_pre_hook(self._cluster_hook(lm)))
        self.refresh()

    @torch.no_grad()
    def refresh(self) -> None:
        """Recompute the weight-derived probe tensors in place (the decode graphs read them), after
        weight edits are applied or reverted."""
        d = self.s.d_model
        for lm in self.layers:
            if lm.gram is not None:
                o, H, hd = lm.heads
                W = o.weight.detach().float().view(d, H, hd)
                lm.gram.copy_(torch.einsum("dhi,dhj->hij", W, W))
            if lm.wnorm is not None:
                lm.wnorm.copy_(lm.down.weight.detach().float().norm(dim=0))
        with self._stats_lock:
            self._stats.clear()

    def _put(self, a: int, act: torch.Tensor) -> None:
        if act.shape[0] == 1:
            self.step[a:a + act.shape[1]].copy_(act[0])
        else:
            self.parts[a] = act

    def _head_hook(self, lm: LayerMap):
        a = lm.offset

        def fn(module, args):
            x = args[0]
            if not self.enabled or x.dim() != 3 or x.shape[0] != 1:
                return None
            _, H, hd = lm.heads
            x = x[0].float().reshape(x.shape[1], H, hd)
            self._put(a, (torch.einsum("nhi,hij->nhj", x, lm.gram) * x).sum(-1).clamp_min(0).sqrt())
            return None
        return fn

    def _cluster_hook(self, lm: LayerMap):
        a = lm.offset + lm.n_heads

        def fn(module, args):
            x = args[0]
            if not self.enabled or x.dim() != 3 or x.shape[0] != 1:
                return None
            mass = x[0].float().abs() * lm.wnorm
            self._put(a, torch.zeros(mass.shape[0], lm.n_clusters, device=mass.device).index_add_(1, lm.assign, mass))
            return None
        return fn

    @contextlib.contextmanager
    def recording(self):
        self.enabled = True
        try:
            yield
        finally:
            self.enabled = False
            self.parts.clear()

    def read_step(self) -> torch.Tensor:
        """Bytes (units,) for the latest single-token forward, on the device."""
        return encode(self.step)

    def read_chunk(self, n: int) -> torch.Tensor:
        """Bytes (n, units) for the latest n-token forward, on the device."""
        if n == 1:
            return encode(self.step)[None]
        out = torch.zeros(n, self.n_units, device=self.step.device)
        for a, act in self.parts.items():
            if act.shape[0] == n:
                out[:, a:a + act.shape[1]] = act
        self.parts.clear()
        return encode(out)

    # ---- recordings --------------------------------------------------------------------

    def add(self, run_id: str, ts, codes: np.ndarray) -> dict:
        """Keep a recorded slice for the run's current request; returns the live-feed message."""
        ts = np.asarray(ts, dtype=np.int32)
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        self.pending.setdefault(run_id, []).append((ts, codes))
        return {"k": "anat", "run": run_id, "t": ts.tolist(), "u": self.n_units,
                "b64": base64.b64encode(codes.tobytes()).decode()}

    def flush(self, store, run_id: str, req: int) -> bool:
        """Write the request's slices to the run. Readers may briefly see a slice twice (on disk and
        pending), never zero times; the UI keys slices by tape index."""
        parts = list(self.pending.get(run_id, []))
        if not parts:
            return False
        try:
            ts = np.concatenate([p[0] for p in parts])
            codes = np.concatenate([p[1] for p in parts])
            if not store.can_write(codes.nbytes):
                return False
            path = store.run_dir(run_id) / "anatomy" / f"req{req:04d}.npz"
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez_compressed(tmp, t=ts, codes=codes, atlas=np.array(self.key))
            tmp.replace(path)
            store._grew(path.stat().st_size)
            return True
        finally:
            self.pending.pop(run_id, None)

    def read_run(self, store, run_id: str) -> tuple[np.ndarray, np.ndarray, int]:
        """(tape indices, bytes (n, units), slices skipped because they were recorded with another unit map)."""
        ts, codes, skipped = [], [], 0
        for path in sorted((store.runs_dir / run_id / "anatomy").glob("req????.npz")):
            with np.load(path) as z:
                if str(z["atlas"]) != self.key or z["codes"].shape[1] != self.n_units:
                    skipped += 1
                    continue
                ts.append(z["t"])
                codes.append(z["codes"])
        for t, c in self.pending.get(run_id, []):
            ts.append(t)
            codes.append(c)
        if not ts:
            return np.zeros(0, np.int32), np.zeros((0, self.n_units), np.uint8), skipped
        return np.concatenate(ts), np.concatenate(codes), skipped

    # ---- the weights themselves --------------------------------------------------------

    @contextlib.contextmanager
    def _side(self):
        """Weight reads run on their own stream, so a recording in progress never waits on them."""
        if self.s.device.type != "cuda":
            yield
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream(self.s.device)
        with torch.cuda.stream(self._stream):
            yield
        self._stream.synchronize()

    def _matrices(self, lm: LayerMap, kind: str, i: int) -> list[dict]:
        """The weight slices that make up a unit, each oriented (the unit's own dims) x (residual dims)."""
        def spec(key, title, desc, module, axis, rows, start=None, stop=None, transpose=False):
            return {"key": key, "title": title, "desc": desc, "param": self._name(module), "axis": axis,
                    "start": start, "stop": stop, "index": start is None, "transpose": transpose,
                    "rows": rows, "cols": "residual stream dims"}
        if kind == "head":
            o, H, hd = lm.heads
            out = [spec("writes", "Output weights", f"{self._name(o)} columns {i * hd}–{(i + 1) * hd - 1}: what this "
                        "head writes into the residual stream", o, 1, "head dims", i * hd, (i + 1) * hd, True)]
            for key, title, desc, module, a, b in self._head_readers(lm, i):
                out.append(spec(key, title, f"{self._name(module)} rows {a}–{b - 1}: {desc}", module, 0, "head dims", a, b))
            return out
        out = [spec("writes", "Output weights", f"{self._name(lm.down)} columns of this cluster's neurons: what they "
                    "write into the residual stream", lm.down, 1, "neurons", transpose=True)]
        for name, module in lm.readers:
            out.append(spec(name, f"Input weights · {name}", f"{self._name(module)} rows of this cluster's neurons: "
                            "what they read from the residual stream", module, 0, "neurons"))
        return out

    def _head_readers(self, lm: LayerMap, h: int) -> list[tuple]:
        """Input-side weights of one head, where the layout is known (standard and gated delta net attention)."""
        attn, (_, H, hd), d = lm.attn, lm.heads, self.s.d_model
        found = []
        q, k, v = (getattr(attn, n, None) for n in ("q_proj", "k_proj", "v_proj"))
        if all(isinstance(m, nn.Linear) and m.in_features == d for m in (q, k, v)):
            if q.out_features == 2 * H * hd:  # query and output gate interleaved per head (Qwen3.5, Qwen3-Next)
                found += [("query", "Query weights", "the head's query", q, 2 * h * hd, 2 * h * hd + hd),
                          ("gate", "Output gate weights", "the sigmoid gate on the head's output", q,
                           2 * h * hd + hd, 2 * (h + 1) * hd)]
            elif q.out_features == H * hd:
                found.append(("query", "Query weights", "the head's query", q, h * hd, (h + 1) * hd))
            if k.out_features % hd == 0 and H % (k.out_features // hd) == 0:
                group = H // (k.out_features // hd)
                kv = h // group
                shared = f", shared by {group} heads" if group > 1 else ""
                found += [("key", "Key weights", f"key head {kv}{shared}", k, kv * hd, (kv + 1) * hd),
                          ("value", "Value weights", f"value head {kv}{shared}", v, kv * hd, (kv + 1) * hd)]
        qkv = getattr(attn, "in_proj_qkv", None)
        nk, dk = getattr(attn, "num_k_heads", None), getattr(attn, "head_k_dim", None)
        if isinstance(qkv, nn.Linear) and nk and dk and H % nk == 0:
            key_dim, group = nk * dk, H // nk
            kh = h // group
            shared = f", shared by {group} heads" if group > 1 else ""
            found += [("query", "Query weights", f"query head {kh}{shared}, before the short convolution", qkv,
                       kh * dk, (kh + 1) * dk),
                      ("key", "Key weights", f"key head {kh}{shared}, before the short convolution", qkv,
                       key_dim + kh * dk, key_dim + (kh + 1) * dk),
                      ("value", "Value weights", "the head's value, before the short convolution", qkv,
                       2 * key_dim + h * hd, 2 * key_dim + (h + 1) * hd)]
            z = getattr(attn, "in_proj_z", None)
            if isinstance(z, nn.Linear):
                found.append(("gate", "Output gate weights", "the gate on the head's normalized output", z,
                              h * hd, (h + 1) * hd))
        return found

    def _view(self, req: dict) -> tuple[torch.Tensor, dict]:
        """A 2-D view of the weights a request names: a whole parameter or one matrix of a unit."""
        if req.get("param"):
            name = req["param"]
            if name not in self.params:
                raise KeyError(f"no parameter {name}")
            p = self.params[name].detach()
            t = p.reshape(1, -1) if p.dim() < 2 else p.reshape(p.shape[0], -1)
            return t, {"key": "param", "title": name, "desc": f"{name} {tuple(p.shape)}, rows × columns as stored",
                       "param": name, "rows": "rows", "cols": "columns"}
        lm, kind, i = self.locate(int(req["unit"]))
        mats = self._matrices(lm, kind, i)
        spec = next((m for m in mats if m["key"] == req.get("matrix", "writes")), None)
        if spec is None:
            raise KeyError(f"unit {req['unit']} has no matrix {req.get('matrix')!r}")
        p = self.params[spec["param"]].detach()
        if spec["index"]:
            t = p.index_select(spec["axis"], (lm.assign == i).nonzero().flatten())
        else:
            t = p.narrow(spec["axis"], spec["start"], spec["stop"] - spec["start"])
        return (t.T if spec["transpose"] else t), spec

    @torch.no_grad()
    def tile(self, req: dict, version: int = 0) -> dict:
        """The weights in a window of a view, binned to at most h x w cells."""
        with self._side():
            t, spec = self._view(req)
            R, C = t.shape
            r0 = min(max(int(req.get("r0", 0)), 0), R - 1)
            r1 = min(max(int(req.get("r1", R)), r0 + 1), R)
            c0 = min(max(int(req.get("c0", 0)), 0), C - 1)
            c1 = min(max(int(req.get("c1", C)), c0 + 1), C)
            h = min(max(int(req.get("h", 256)), 1), 1024)
            w = min(max(int(req.get("w", 768)), 1), 2048)
            b = pool(t[r0:r1, c0:c1], h, w)
            exact = b["bh"] == 1 and b["bw"] == 1
            mode = req.get("mode") or "auto"
            if mode not in ("auto", "mean", "rms", "absmax"):
                raise ValueError("mode must be auto, mean, rms or absmax")
            if mode == "auto" or exact:
                mode = "value" if exact else "rms"
            vals = b["mean"] if mode in ("value", "mean") else b[mode]
            skey = (json.dumps({k: req.get(k) for k in ("param", "unit", "matrix")}), version)
            with self._stats_lock:
                stats = self._stats.get(skey)
                if stats is None:
                    stats = self._stats[skey] = matrix_stats(t)
                    while len(self._stats) > 32:
                        self._stats.popitem(last=False)
            data = vals.to(torch.float16).cpu().numpy()
            return {"title": spec["title"], "desc": spec["desc"], "rows_label": spec["rows"], "cols_label": spec["cols"],
                    "rows": R, "cols": C, "window": [r0, r1, c0, c1], "bh": b["bh"], "bw": b["bw"],
                    "h": data.shape[0], "w": data.shape[1], "mode": mode, "exact": exact,
                    "data": base64.b64encode(data.tobytes()).decode(),
                    "row_rms": [round(x, 6) for x in b["row_rms"].tolist()],
                    "col_rms": [round(x, 6) for x in b["col_rms"].tolist()],
                    "stats": stats}

    @torch.no_grad()
    def unit_detail(self, g: int, edited: set[int] | None = None) -> dict:
        lm, kind, i = self.locate(g)
        with self._side():
            out = dict(self.units[g]) | {"layer_type": lm.type, "edited": g in (edited or set()),
                                         "title": f"L{lm.index} · {'head' if kind == 'head' else 'MLP cluster'} {i}"}
            W, _ = self._view({"unit": g, "matrix": "writes"})
            W = W.float()  # (unit dims, d): rows are what each head dim or neuron writes
            sv = torch.linalg.svdvals(W)
            var = sv.square()
            out["spectrum"] = [round(x, 5) for x in sv[:64].tolist()]
            out["effective_rank"] = round(float(var.sum() ** 2 / var.square().sum().clamp_min(1e-30)), 2)
            if kind == "head":
                direction = torch.linalg.svd(W, full_matrices=False).Vh[0]
            else:
                c = lm.centroids[i]
                direction = (W * (W @ c).sign()[:, None]).sum(0)  # sign-aligned sum on the current weights
                direction = direction / direction.norm().clamp_min(1e-12)
            (out["pos_now"], out["neg_now"]), = self.top_tokens(direction[None], 8)
            mats = []
            for m in self._matrices(lm, kind, i):
                t, _ = self._view({"unit": g, "matrix": m["key"]})
                mats.append({k: m[k] for k in ("key", "title", "desc", "rows", "cols")} | {"shape": list(t.shape)})
            out["matrices"] = mats
            if kind == "cluster":
                idx = (lm.assign == i).nonzero().flatten()
                norms = W.norm(dim=1)
                cos = (W @ direction) / norms.clamp_min(1e-12)
                order = norms.argsort(descending=True)[:MAX_NEURON_ROWS]
                aligned = W[order] * cos[order].sign()[:, None]
                labels = self.top_tokens(aligned, 3)
                out["n_neurons"] = int(idx.numel())
                out["neuron_ids"] = idx.tolist()  # neuron number of each row of the cluster's matrices
                out["neurons"] = [{"n": int(idx[j]), "row": j, "norm": round(float(norms[j]), 4),
                                   "cos": round(abs(float(cos[j])), 3), "top": labels[r][0]}
                                  for r, j in enumerate(order.tolist())]
        return out

    def lesion(self, g: int, factor: float = 0.0) -> dict:
        """Workbench edit arguments that scale one unit's output weights."""
        lm, kind, i = self.locate(g)
        if kind == "head":
            return {"kind": "scale", "layers": str(lm.index), "targets": ["head"], "index": i, "factor": factor,
                    "note": f"L{lm.index} head {i}"}
        idx = (lm.assign == i).nonzero().flatten().tolist()
        return {"kind": "scale", "layers": str(lm.index), "targets": ["neuron"], "index": idx, "factor": factor,
                "note": f"L{lm.index} cluster {i}"}

    def lesion_units(self, units, factor: float = 0.0, note: str = "") -> dict:
        """Workbench edit arguments that scale a set of units together, as one edit (a knockout made
        permanent): each head's slice of its output projection, each cluster's neurons."""
        entries = []
        for g in sorted({int(g) for g in units}):
            lm, kind, i = self.locate(g)
            if kind == "head":
                entries.append({"layer": lm.index, "target": "head", "index": i, "name": f"L{lm.index} h{i}", "g": g})
            else:
                entries.append({"layer": lm.index, "target": "neuron", "name": f"L{lm.index} c{i}", "g": g,
                                "index": (lm.assign == i).nonzero().flatten().tolist()})
        return {"kind": "units", "units": entries, "factor": factor, "note": note}

    def param_list(self) -> list[dict]:
        return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", ""), "numel": p.numel()}
                for n, p in self.params.items()]
