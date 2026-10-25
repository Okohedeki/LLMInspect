"""Weight edits: change the model's parameters in place, reversibly.

Runtime interventions (capture.py) edit activations during a forward pass. Weight edits
change the parameters themselves, the way "abliterated" models are made:

  orthogonalize  remove a unit direction v from what chosen modules write into, or read
                 from, the residual stream:
                     writers (attention/MLP outputs, embeddings):  W <- W - a v v^T W
                     readers (attention/MLP inputs, unembedding):  W <- W - a W v v^T
                 With a = 1 on the embeddings plus every attention and MLP output, this is
                 the directional ablation of Arditi et al. (2024), "Refusal in language
                 models is mediated by a single direction".
  scale          multiply a component's contribution by a factor: a layer's whole attention
                 or MLP block, one attention head (standard or linear attention), or one or
                 more MLP neurons. Factor 0 is a lesion.
  units          scale a set of units together, as one edit: attention heads and groups of
                 MLP neurons, each at its own layer. A knockout made permanent is one of these;
                 it changes the same weights as the matching scale edits.

Every parameter is copied to CPU before its first change. Applying the stack restores
those copies and replays the enabled edits in order, so toggling, reordering and
reverting are exact. Models whose blocks pass through an extra norm before the residual
add (Gemma 2/3) are only approximately orthogonalized by the writer rule.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass, field

import torch
from torch import nn

ATTN_NAMES = ("self_attn", "linear_attn", "attn", "attention", "mixer")
MLP_NAMES = ("mlp", "feed_forward", "ffn")
WRITER_NAMES = {"o_proj", "out_proj", "down_proj", "c_proj", "dense_4h_to_h", "wo", "w2"}
KINDS = ("orthogonalize", "scale", "units")
ORTHO_TARGETS = ("embed", "attn_out", "mlp_out", "attn_in", "mlp_in", "unembed")
SCALE_TARGETS = ("attn", "mlp", "head", "neuron")
ROW_CHUNK = 16_384  # rows/columns per fp32 work block, keeps big matrices off the VRAM ceiling


def parse_layers(spec, n_layers: int) -> list[int]:
    """'all', '12', '8-20', '1,3,5-7' or a list of ints -> sorted layer indices."""
    if isinstance(spec, (list, tuple)):
        out = {int(x) for x in spec}
    else:
        text = str(spec).strip().lower()
        if text in ("", "all", "*"):
            return list(range(n_layers))
        out = set()
        for part in text.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                out.update(range(int(a), int(b) + 1))
            elif part:
                out.add(int(part))
    bad = sorted(x for x in out if not 0 <= x < n_layers)
    if bad:
        raise ValueError(f"layers out of range 0..{n_layers - 1}: {bad}")
    return sorted(out)


def layer_ranges(layers: list[int]) -> str:
    """[0, 1, 2, 5] -> '0-2,5'."""
    parts, i = [], 0
    while i < len(layers):
        j = i
        while j + 1 < len(layers) and layers[j + 1] == layers[j] + 1:
            j += 1
        parts.append(str(layers[i]) if i == j else f"{layers[i]}-{layers[j]}")
        i = j + 1
    return ",".join(parts)


def _child(layer: nn.Module, names) -> nn.Module | None:
    return next((getattr(layer, n) for n in names if hasattr(layer, n)), None)


def residual_linears(block: nn.Module | None, d_model: int, side: str) -> list[nn.Linear]:
    """Linear layers in a block that write to (side='out') or read from (side='in') the residual."""
    if block is None:
        return []
    found = []
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        writer = name.rsplit(".", 1)[-1] in WRITER_NAMES
        if side == "out" and writer and m.out_features == d_model:
            found.append(m)
        elif side == "in" and not writer and m.in_features == d_model:
            found.append(m)
    return found


def attention_heads(attn: nn.Module | None, d_model: int, config=None) -> tuple[nn.Linear, int, int] | None:
    """(output projection, number of heads, head dim) for standard and linear attention blocks, or
    None when the block has no single output projection to split by head."""
    writers = residual_linears(attn, d_model, "out")
    if len(writers) != 1:
        return None
    w = writers[0]
    for attr in ("head_v_dim", "head_dim"):  # linear attention (gated delta net) first: its heads are value heads
        hd = getattr(attn, attr, None)
        if isinstance(hd, int) and hd > 0 and w.in_features % hd == 0:
            return w, w.in_features // hd, hd
    n = getattr(attn, "num_heads", None) or getattr(config, "num_attention_heads", None)
    if isinstance(n, int) and n > 0 and w.in_features % n == 0:
        return w, n, w.in_features // n
    return None


@torch.no_grad()
def remove_from_rows(w: torch.Tensor, v: torch.Tensor, a: float) -> None:
    """Rows of w are residual vectors (embeddings) or readers (W[out, d]): w -= a (w v) v^T."""
    for s in range(0, w.shape[0], ROW_CHUNK):
        blk = w[s:s + ROW_CHUNK].float()
        blk -= a * torch.outer(blk @ v, v)
        w[s:s + ROW_CHUNK] = blk.to(w.dtype)


@torch.no_grad()
def remove_from_output(w: torch.Tensor, v: torch.Tensor, a: float) -> None:
    """w[d, in] writes to the residual: w -= a v (v^T w)."""
    for s in range(0, w.shape[1], ROW_CHUNK):
        blk = w[:, s:s + ROW_CHUNK].float()
        blk -= a * torch.outer(v, v @ blk)
        w[:, s:s + ROW_CHUNK] = blk.to(w.dtype)


@dataclass
class WeightEdit:
    kind: str
    layers: list[int]
    targets: list[str]
    direction: str | None = None
    vector: torch.Tensor | None = None  # unit direction, fp32 on CPU
    strength: float = 1.0               # orthogonalize: fraction of the component removed
    factor: float = 0.0                 # scale: multiplier on the component
    index: int | list[int] | None = None  # scale: head number, or one or more neuron numbers
    enabled: bool = True
    note: str = ""
    units: list[dict] | None = None    # units: [{layer, target: head|neuron, index, name, g}]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def summary(self) -> str:
        where = f"L{layer_ranges(self.layers)}"
        if self.kind == "units":
            names = [u.get("name") or f"L{u['layer']} {u['target']}" for u in self.units]
            shown = ", ".join(names[:4]) + (f" +{len(names) - 4} more" if len(names) > 4 else "")
            verb = "knock out" if self.factor == 0 else f"scale ×{self.factor:g}"
            return f"{verb} {len(names)} unit{'' if len(names) == 1 else 's'}: {shown}" + (f" ({self.note})" if self.note else "")
        if self.kind == "orthogonalize":
            return f"remove {self.direction} ×{self.strength:g} from {', '.join(self.targets)} @ {where}"
        if isinstance(self.index, list):
            what = f"{len(self.index)} neurons" + (f" ({self.note})" if self.note else "")
        else:
            what = f"{self.targets[0]} {self.index}" if self.index is not None else self.targets[0]
        return f"scale {what} ×{self.factor:g} @ {where}"

    def describe(self) -> dict:
        return {"id": self.id, "kind": self.kind, "layers": layer_ranges(self.layers), "targets": self.targets,
                "direction": self.direction, "strength": self.strength, "factor": self.factor,
                "index": None if self.kind == "units" else self.index,
                "units": [{"g": u.get("g"), "name": u.get("name"), "layer": u["layer"], "target": u["target"]}
                          for u in self.units] if self.units else None,
                "enabled": self.enabled, "note": self.note, "summary": self.summary()}


class EditStack:
    def __init__(self, subject):
        self.s = subject
        self.edits: dict[str, WeightEdit] = {}
        self.originals: dict[str, torch.Tensor] = {}
        self.applied: list[str] = []  # edit ids currently baked into the weights, in order
        self.version = 0
        self._params = dict(subject.model.named_parameters())
        self._names = {id(p): n for n, p in self._params.items()}

    # ---- building -----------------------------------------------------------------

    def add(self, kind: str, layers="all", targets=None, direction: str | None = None,
            vector: torch.Tensor | None = None, strength: float = 1.0, factor: float = 0.0,
            index: int | None = None, note: str = "", units: list[dict] | None = None) -> WeightEdit:
        n_layers = self.s.n_layers
        if kind == "units":
            entries = []
            for u in units or []:
                target, layer = u.get("target"), int(u.get("layer", -1))
                if target not in ("head", "neuron") or u.get("index") is None:
                    raise ValueError("each unit needs a target (head or neuron) and an index")
                if not 0 <= layer < n_layers:
                    raise ValueError(f"layer must be in 0..{n_layers - 1}")
                idx = u["index"]
                idx = sorted({int(i) for i in (idx if isinstance(idx, (list, tuple)) else [idx])}) if target == "neuron" else int(idx)
                entries.append({"layer": layer, "target": target, "index": idx, "name": str(u.get("name") or ""), "g": u.get("g")})
            if not entries:
                raise ValueError("pick at least one unit")
            edit = WeightEdit("units", sorted({e["layer"] for e in entries}), ["units"], factor=float(factor),
                              note=note, units=entries)
            self._plan(edit)  # validate against the architecture now
            self.edits[edit.id] = edit
            return edit
        if kind == "orthogonalize":
            targets = list(targets or ["embed", "attn_out", "mlp_out"])
            bad = [t for t in targets if t not in ORTHO_TARGETS]
            if bad or not targets:
                raise ValueError(f"orthogonalize targets must be from {ORTHO_TARGETS}")
            if vector is None:
                raise ValueError("orthogonalize needs a direction")
            vector = vector.detach().float().cpu()
            vector = vector / vector.norm()
            if vector.shape != (self.s.d_model,):
                raise ValueError(f"direction must have {self.s.d_model} dims")
            if not 0 <= float(strength) <= 1:
                raise ValueError("strength must be between 0 and 1")
        elif kind == "scale":
            targets = list(targets or [])
            if len(targets) != 1 or targets[0] not in SCALE_TARGETS:
                raise ValueError(f"scale needs one target from {SCALE_TARGETS}")
            if targets[0] in ("head", "neuron") and index is None:
                raise ValueError(f"scaling a {targets[0]} needs its index")
            if isinstance(index, (list, tuple)):
                if targets[0] != "neuron" or not index:
                    raise ValueError("a list of indices is only for scaling a group of neurons")
                index = sorted({int(i) for i in index})
            elif index is not None:
                index = int(index)
        else:
            raise ValueError(f"kind must be one of {KINDS}")
        edit = WeightEdit(kind, parse_layers(layers, n_layers), targets, direction, vector, float(strength),
                          float(factor), index, note=note)
        self._plan(edit)  # validate against the architecture now, not at apply time
        self.edits[edit.id] = edit
        return edit

    def remove(self, eid: str) -> None:
        self.edits.pop(eid)

    def set_enabled(self, eid: str, enabled: bool) -> WeightEdit:
        self.edits[eid].enabled = enabled
        return self.edits[eid]

    # ---- planning -----------------------------------------------------------------

    def _blocks(self, layer: int):
        mod = self.s.layers[layer]
        return _child(mod, ATTN_NAMES), _child(mod, MLP_NAMES)

    def _plan(self, e: WeightEdit) -> list[tuple[nn.Parameter, callable]]:
        """(parameter, in-place operation) pairs that carry out one edit."""
        d, ops = self.s.d_model, []
        model = self.s.model
        if e.kind == "orthogonalize":
            a = e.strength
            for target in e.targets:
                if target in ("embed", "unembed"):
                    emb = model.get_input_embeddings() if target == "embed" else model.get_output_embeddings()
                    ops.append((emb.weight, lambda p, a=a: remove_from_rows(p, self._v(e), a)))
                    continue
                for layer in e.layers:
                    attn, mlp = self._blocks(layer)
                    block = attn if target.startswith("attn") else mlp
                    side = "out" if target.endswith("_out") else "in"
                    linears = residual_linears(block, d, side)
                    if not linears:
                        raise ValueError(f"layer {layer} has no {target} projection to edit")
                    for lin in linears:
                        if side == "out":
                            ops.append((lin.weight, lambda p, a=a: remove_from_output(p, self._v(e), a)))
                            if lin.bias is not None:
                                ops.append((lin.bias, lambda p, a=a: p.sub_(a * self._v(e) * (self._v(e) @ p.float()))))
                        else:
                            ops.append((lin.weight, lambda p, a=a: remove_from_rows(p, self._v(e), a)))
            return ops

        cfg = model.config.get_text_config()
        if e.kind == "units":
            for u in e.units:
                attn, mlp = self._blocks(u["layer"])
                if u["target"] == "head":
                    heads = attention_heads(attn, d, cfg)
                    if heads is None:
                        raise ValueError(f"layer {u['layer']} has no attention output projection to split by head")
                    _, n_heads, hd = heads
                    if not 0 <= u["index"] < n_heads:
                        raise ValueError(f"head must be in 0..{n_heads - 1} at layer {u['layer']}")
                    ops.append((heads[0].weight, lambda p, s=u["index"] * hd, hd=hd, f=e.factor: p[:, s:s + hd].mul_(f)))
                else:
                    down = residual_linears(mlp, d, "out")
                    if not down:
                        raise ValueError(f"layer {u['layer']} has no dense MLP output projection")
                    if not all(0 <= n < down[0].in_features for n in u["index"]):
                        raise ValueError(f"neuron must be in 0..{down[0].in_features - 1}")
                    cols = torch.tensor(u["index"], device=down[0].weight.device)
                    ops.append((down[0].weight, lambda p, cols=cols, f=e.factor: p.index_copy_(1, cols, p.index_select(1, cols) * f)))
            return ops

        target, f = e.targets[0], e.factor
        for layer in e.layers:
            attn, mlp = self._blocks(layer)
            if target in ("attn", "mlp"):
                for lin in residual_linears(attn if target == "attn" else mlp, d, "out"):
                    ops.append((lin.weight, lambda p: p.mul_(f)))
                    if lin.bias is not None:
                        ops.append((lin.bias, lambda p: p.mul_(f)))
            elif target == "head":
                heads = attention_heads(attn, d, cfg)
                if heads is None:
                    raise ValueError(f"layer {layer} has no attention output projection to split by head")
                _, n_heads, hd = heads
                if not 0 <= e.index < n_heads:
                    raise ValueError(f"head must be in 0..{n_heads - 1} at layer {layer}")
                ops.append((heads[0].weight, lambda p, s=e.index * hd, hd=hd: p[:, s:s + hd].mul_(f)))
            else:
                down = residual_linears(mlp, d, "out")
                if not down:
                    raise ValueError(f"layer {layer} has no dense MLP output projection")
                idx = e.index if isinstance(e.index, list) else [e.index]
                if not all(0 <= n < down[0].in_features for n in idx):
                    raise ValueError(f"neuron must be in 0..{down[0].in_features - 1}")
                cols = torch.tensor(idx, device=down[0].weight.device)
                ops.append((down[0].weight, lambda p, cols=cols: p.index_copy_(1, cols, p.index_select(1, cols) * f)))
        return ops

    def _v(self, e: WeightEdit) -> torch.Tensor:
        return e.vector.to(self.s.device)

    # ---- applying -----------------------------------------------------------------

    @torch.no_grad()
    def _restore(self) -> None:
        for name, orig in self.originals.items():
            self._params[name].copy_(orig.to(self._params[name].device))

    @torch.no_grad()
    def _apply_ids(self, ids: list[str]) -> None:
        self._restore()
        for eid in ids:
            for param, op in self._plan(self.edits[eid]):
                name = self._names[id(param)]
                if name not in self.originals:
                    self.originals[name] = param.detach().to("cpu", copy=True)
                op(param.data)
        self.applied = list(ids)
        self.version += 1

    def apply(self) -> list[str]:
        """Bake every enabled edit into the weights, in insertion order."""
        self._apply_ids([eid for eid, e in self.edits.items() if e.enabled])
        return self.applied

    def revert(self) -> None:
        """Put the original weights back. Edits stay in the stack, unapplied."""
        self._restore()
        self.applied = []
        self.version += 1

    @contextlib.contextmanager
    def suspended(self):
        """Temporarily run on the original weights, then restore exactly the applied edits. The
        version differs inside and is the same again after, since the weights are then identical
        to before (so caches keyed by it, like the recorder's KV prefix, stay valid)."""
        ids, version = list(self.applied), self.version
        if not ids:
            yield
            return
        self.revert()
        try:
            yield
        finally:
            self._apply_ids(ids)
            self.version = version

    def active(self) -> list[dict]:
        return [self.edits[eid].describe() for eid in self.applied if eid in self.edits]

    def summary(self) -> dict:
        return {"edits": [e.describe() | {"applied": eid in self.applied} for eid, e in self.edits.items()],
                "applied": self.active(), "version": self.version,
                "snapshot_bytes": sum(t.numel() * t.element_size() for t in self.originals.values())}
