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
  checkpoint     put another checkpoint's weights in place (checkpoints.py): every tensor of
                 a candidate that differs from the model's is copied in. It always applies
                 first, so the other edits land on top of the candidate's weights. At most
                 one per stack.

Mixture-of-experts blocks are edited too. Their experts are held as 3-D parameters (one slice per
expert), and a layer's neurons are numbered in one row: expert 0's first, then expert 1's and so on,
then a shared expert's (mlp_neurons). The router reads the residual, so it is one of the MLP's inputs.

Every parameter is copied to CPU before its first change. Applying the stack restores
those copies and replays the enabled edits in order, so toggling, reordering and
reverting are exact. Models whose blocks pass through an extra norm before the residual
add (Gemma 2/3) are only approximately orthogonalized by the writer rule.

Views of the original weights (`suspended`) restore the originals and replay the edits. A
checkpoint edit's tensors are kept in CPU memory (up to cpu_budget), so replaying it is a copy
rather than a read of the candidate's files. Keeping a CPU copy of the edited weights instead
was measured slower (on a full fine-tune of a 1.5B model: about 2-3 s per round trip, against
1.1 s from the cache and 1.4 s re-reading the files), so there is none. Nothing is pinned:
page-locked memory can't be paged out, and under WSL a few GB of it can stall the whole VM.

The stack is saved to a file (bind) after every change: the edits, their vectors, and which are
applied. When the recorder loads a model it restores the stack and applies again what was applied
(load); an edit that no longer fits (a candidate's files are gone, say) is skipped and reported.
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import psutil
import torch
from safetensors.torch import load_file, save_file
from torch import nn

ATTN_NAMES = ("self_attn", "linear_attn", "attn", "attention", "mixer")
MLP_NAMES = ("mlp", "feed_forward", "ffn")
WRITER_NAMES = {"o_proj", "out_proj", "down_proj", "c_proj", "dense_4h_to_h", "wo", "w2"}
KINDS = ("orthogonalize", "scale", "units", "checkpoint")
ORTHO_TARGETS = ("embed", "attn_out", "mlp_out", "attn_in", "mlp_in", "unembed")
SCALE_TARGETS = ("attn", "mlp", "head", "neuron")
ROW_CHUNK = 16_384  # rows/columns per fp32 work block, keeps big matrices off the VRAM ceiling
CPU_CACHE_MAX_GB = 16.0


def cpu_budget() -> int:
    """Bytes of CPU memory kept for faster weight swaps: LLMINSPECT_CPU_CACHE_GB when it is set (0 turns
    the cache off), else a quarter of the available RAM, at most CPU_CACHE_MAX_GB."""
    raw = os.environ.get("LLMINSPECT_CPU_CACHE_GB", "").strip()
    if raw:
        try:
            return max(0, int(float(raw) * 1e9))
        except ValueError:
            raise ValueError(f"LLMINSPECT_CPU_CACHE_GB must be a number of GB, not {raw!r}") from None
    return int(min(CPU_CACHE_MAX_GB * 1e9, psutil.virtual_memory().available / 4))


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


@dataclass
class NeuronBlock:
    """One weight that writes some of a layer's MLP neurons, and where they sit in the layer's numbering.
    A dense projection is [d, width]; fused experts are [experts, d, width] (flipped: [experts, width, d]),
    each expert's neurons numbered after the one before."""
    weight: nn.Parameter
    offset: int
    width: int                     # neurons of the projection, or of each expert
    experts: int = 0               # 0: a dense projection
    flipped: bool = False
    readers: list = field(default_factory=list)  # input weights of these neurons: [width, d], or [experts, k*width, d]
    bias: nn.Parameter | None = None             # fused experts' output bias [experts, d]

    @property
    def count(self) -> int:
        return self.width * max(self.experts, 1)

    def place(self, n: int) -> tuple[int | None, int]:
        """(expert or None, neuron within it) of the layer's neuron n, which this block holds."""
        n -= self.offset
        return (None, n) if not self.experts else divmod(n, self.width)

    def columns(self, p: torch.Tensor, expert: int | None) -> tuple[torch.Tensor, int]:
        """One expert's matrix (or the dense projection), and the dim that runs over its neurons."""
        return (p if expert is None else p[expert]), 0 if self.flipped else 1


def find_experts(mlp: nn.Module | None, d_model: int) -> nn.Module | None:
    """The module that holds a mixture-of-experts block's fused expert weights, or None."""
    if mlp is None:
        return None
    for m in mlp.modules():
        down = m._parameters.get("down_proj")
        if isinstance(down, torch.Tensor) and down.ndim == 3 and d_model in down.shape[1:]:
            return m
    return None


def mlp_neurons(mlp: nn.Module | None, d_model: int) -> list[NeuronBlock]:
    """Every weight that writes a layer's MLP neurons, numbered in one row (empty: the layer has none)."""
    if mlp is None:
        return []
    blocks: list[NeuronBlock] = []
    ex = find_experts(mlp, d_model)
    if ex is not None:
        down = ex.down_proj
        flipped = down.shape[1] != d_model
        width = down.shape[1] if flipped else down.shape[2]
        readers = [w for n, w in ex._parameters.items() if w is not None and w.ndim == 3 and n != "down_proj"]
        blocks.append(NeuronBlock(down, 0, width, down.shape[0], flipped, readers, ex._parameters.get("down_proj_bias")))
    downs = residual_linears(mlp, d_model, "out")
    named = {id(m): n for n, m in mlp.named_modules()}
    for down in downs if ex is not None else downs[:1]:  # beside experts, every dense writer is a shared expert
        name = named[id(down)]
        parent = mlp.get_submodule(name.rsplit(".", 1)[0]) if "." in name else mlp
        readers = [m.weight for m in parent.modules() if isinstance(m, nn.Linear) and m is not down
                   and m.in_features == d_model and m.out_features == down.in_features]
        blocks.append(NeuronBlock(down.weight, sum(b.count for b in blocks), down.in_features, readers=readers))
    return blocks


def mlp_readers(mlp: nn.Module | None, d_model: int) -> list[torch.Tensor]:
    """2-D weights that read the residual inside an MLP: its Linear inputs and, in a mixture of experts,
    the router's [experts, d] weight. Fused expert inputs are in mlp_neurons' blocks."""
    if mlp is None:
        return []
    out = [m.weight for m in residual_linears(mlp, d_model, "in")]
    ex = find_experts(mlp, d_model)
    if ex is None:
        return out
    taken = {id(w) for w in out} | {id(m.weight) for m in mlp.modules() if isinstance(m, nn.Linear)}
    for m in mlp.modules():  # the router: a plain [experts, d] weight on a module that is not a Linear
        w = m._parameters.get("weight")
        if m is not ex and isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[1] == d_model and id(w) not in taken:
            out.append(w)
    return out


def _neuron_ops(blocks: list[NeuronBlock], idx: list[int], f: float, layer: int) -> list:
    """(parameter, operation) pairs that multiply these neurons' writes by f."""
    if not blocks:
        raise ValueError(f"layer {layer} has no MLP neurons to edit")
    total = sum(b.count for b in blocks)
    if not all(0 <= n < total for n in idx):
        raise ValueError(f"neuron must be in 0..{total - 1}")
    ops = []
    for b in blocks:
        groups: dict = {}
        for n in idx:
            if b.offset <= n < b.offset + b.count:
                e, j = b.place(n)
                groups.setdefault(e, []).append(j)
        for e, cols in groups.items():
            def op(p, b=b, e=e, cols=torch.tensor(cols, device=b.weight.device)):
                m, dim = b.columns(p, e)
                m.index_copy_(dim, cols, m.index_select(dim, cols) * f)
            ops.append((b.weight, op))
    return ops


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
    checkpoint: dict | None = None     # checkpoint: {spec, folder, name, report, changed: [parameter names]}
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def summary(self) -> str:
        where = f"L{layer_ranges(self.layers)}"
        if self.kind == "checkpoint":
            n, name = len(self.checkpoint["changed"]), self.checkpoint["name"]
            return (f"checkpoint: {name}, 1 tensor differs" if n == 1 else f"checkpoint: {name}, {n} tensors differ") + \
                (f" ({self.note})" if self.note else "")
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
                "checkpoint": None if self.checkpoint is None else {
                    "name": self.checkpoint["name"], "spec": self.checkpoint["spec"], "folder": self.checkpoint["folder"],
                    "report": self.checkpoint.get("report"), "tensors": len(self.checkpoint["changed"])},
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
        self._readers: dict = {}  # checkpoint edit id -> the reader of its candidate's tensors
        self.budget = cpu_budget()
        self.swaps = {"cached": 0, "replay": 0}  # how original-weights views were put back
        self.path: Path | None = None  # where the stack is saved (bind); None: not saved
        self.restored: dict | None = None  # what load() brought back

    # ---- saving -------------------------------------------------------------------

    def bind(self, path) -> "EditStack":
        """Save the stack to `path` (a .json, with its vectors beside it in a .safetensors) after every change."""
        self.path = Path(path)
        return self

    def save(self) -> None:
        if self.path is None:
            return
        rows, vectors = [], {}
        for e in self.edits.values():
            rows.append({"id": e.id, "kind": e.kind, "layers": e.layers, "targets": e.targets, "direction": e.direction,
                         "strength": e.strength, "factor": e.factor, "index": e.index, "enabled": e.enabled,
                         "note": e.note, "units": e.units, "checkpoint": e.checkpoint})
            if e.vector is not None:
                vectors[e.id] = e.vector.detach().float().cpu().contiguous()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tensors = self.path.with_suffix(".safetensors")
        if vectors:
            save_file(vectors, str(tensors))
        else:
            tensors.unlink(missing_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "model": self.s.spec, "edits": rows,
                                   "applied": [eid for eid in self.applied if eid in self.edits]}, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def load(self) -> dict:
        """Bring back the saved stack and apply again what was applied. Edits that no longer fit are skipped."""
        out = {"restored": 0, "skipped": [], "applied": []}
        if self.path is None or not self.path.is_file():
            self.restored = out
            return out
        try:
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            tensors = self.path.with_suffix(".safetensors")
            vectors = load_file(str(tensors)) if tensors.is_file() else {}
        except (OSError, ValueError) as exc:
            out["skipped"].append({"edit": None, "reason": f"the saved stack can't be read ({exc})"})
            self.restored = out
            return out
        path, self.path = self.path, None  # nothing is saved while the stack is rebuilt
        try:
            for row in saved.get("edits", []):
                try:
                    e = self._restore_edit(row, vectors.get(row["id"]))
                except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
                    out["skipped"].append({"edit": row.get("id"), "kind": row.get("kind"), "reason": str(exc).strip("'\"")})
                    continue
                out["restored"] += 1
                if not row.get("enabled", True):
                    e.enabled = False
            ids = [eid for eid in saved.get("applied", []) if eid in self.edits]
            if ids:
                try:
                    self._apply_ids(ids)
                    out["applied"] = ids
                except (ValueError, FileNotFoundError) as exc:
                    out["skipped"].append({"edit": None, "reason": f"the applied edits could not be applied again ({exc})"})
        finally:
            self.path = path
        self.save()
        self.restored = out
        return out

    def _restore_edit(self, row: dict, vector) -> WeightEdit:
        kind = row["kind"]
        if kind == "checkpoint":
            e = self._add_checkpoint(row["checkpoint"], row.get("note", ""))
        elif kind == "units":
            e = self._add("units", factor=row.get("factor", 0.0), note=row.get("note", ""), units=row["units"])
        elif kind == "orthogonalize":
            if vector is None:
                raise ValueError("its direction vector is missing")
            e = self._add(kind, row["layers"], row["targets"], row.get("direction"), vector, row.get("strength", 1.0),
                          note=row.get("note", ""))
        else:
            e = self._add(kind, row["layers"], row["targets"], factor=row.get("factor", 0.0), index=row.get("index"),
                          note=row.get("note", ""))
        # keep the saved id, so "applied" and anything that names the edit still match
        self.edits.pop(e.id)
        if e.id in self._readers:
            self._readers[row["id"]] = self._readers.pop(e.id)
        e.id = row["id"]
        self.edits[e.id] = e
        return e

    # ---- building -----------------------------------------------------------------

    def add(self, kind: str, layers="all", targets=None, direction: str | None = None,
            vector: torch.Tensor | None = None, strength: float = 1.0, factor: float = 0.0,
            index: int | None = None, note: str = "", units: list[dict] | None = None,
            checkpoint: dict | None = None) -> WeightEdit:
        edit = self._add(kind, layers, targets, direction, vector, strength, factor, index, note, units, checkpoint)
        self.save()
        return edit

    def _add(self, kind: str, layers="all", targets=None, direction: str | None = None,
             vector: torch.Tensor | None = None, strength: float = 1.0, factor: float = 0.0,
             index: int | None = None, note: str = "", units: list[dict] | None = None,
             checkpoint: dict | None = None) -> WeightEdit:
        n_layers = self.s.n_layers
        if kind == "checkpoint":
            return self._add_checkpoint(checkpoint, note)
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

    def _add_checkpoint(self, checkpoint: dict | None, note: str) -> WeightEdit:
        """A checkpoint edit from {spec, folder, name, report, changed}: the tensors in `changed` are
        copied from the candidate folder when the edit is applied."""
        from . import checkpoints  # late: checkpoints imports this module
        if any(e.kind == "checkpoint" for e in self.edits.values()):
            raise ValueError("the stack already has a checkpoint edit; remove it first")
        ck = checkpoint or {}
        if not (isinstance(ck.get("spec"), str) and isinstance(ck.get("folder"), str)
                and isinstance(ck.get("changed"), list)):
            raise ValueError("a checkpoint edit needs the candidate's spec, folder and changed tensors; "
                             "load a candidate with POST /llminspect/workbench/checkpoint")
        ck = {"spec": ck["spec"], "folder": ck["folder"], "name": str(ck.get("name") or ck["spec"]),
              "report": ck.get("report"), "changed": [str(n) for n in ck["changed"]]}
        reader = checkpoints.reader_for(ck, self.s.model, self.budget)  # refuses a candidate that doesn't fit this model
        layers = sorted({layer for layer in map(checkpoints.layer_of, ck["changed"]) if layer is not None})
        edit = WeightEdit("checkpoint", layers, ["checkpoint"], note=note, checkpoint=ck)
        self._readers[edit.id] = reader
        try:
            self._plan(edit)  # checks the candidate's files are there
        except Exception:
            self._readers.pop(edit.id, None)
            raise
        self.edits[edit.id] = edit
        return edit

    def remove(self, eid: str) -> None:
        self.edits.pop(eid)
        self._readers.pop(eid, None)
        self.save()

    def set_enabled(self, eid: str, enabled: bool) -> WeightEdit:
        self.edits[eid].enabled = enabled
        self.save()
        return self.edits[eid]

    # ---- planning -----------------------------------------------------------------

    def _blocks(self, layer: int):
        mod = self.s.layers[layer]
        return _child(mod, ATTN_NAMES), _child(mod, MLP_NAMES)

    def _plan(self, e: WeightEdit) -> list[tuple[nn.Parameter, callable]]:
        """(parameter, in-place operation) pairs that carry out one edit."""
        d, ops = self.s.d_model, []
        model = self.s.model
        if e.kind == "checkpoint":
            reader = self._readers[e.id]
            reader.require_files()
            for name in e.checkpoint["changed"]:
                ops.append((self._params[name], lambda p, name=name: p.copy_(reader.tensor(name, keep=True))))  # cast to p's dtype and device
            return ops
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
                    if target.startswith("mlp"):
                        ops += self._fused_ops(mlp, side, a, e)
                        if side == "in":  # every MLP input, the router's included
                            linears = []
                            ops += [(w, lambda p, a=a: remove_from_rows(p, self._v(e), a)) for w in mlp_readers(mlp, d)]
                        if not linears and not mlp_neurons(mlp, d) and not mlp_readers(mlp, d):
                            raise ValueError(f"layer {layer} has no {target} projection to edit")
                    elif not linears:
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
                    ops += _neuron_ops(mlp_neurons(mlp, d), u["index"], e.factor, u["layer"])
            return ops

        target, f = e.targets[0], e.factor
        for layer in e.layers:
            attn, mlp = self._blocks(layer)
            if target in ("attn", "mlp"):
                for lin in residual_linears(attn if target == "attn" else mlp, d, "out"):
                    ops.append((lin.weight, lambda p: p.mul_(f)))
                    if lin.bias is not None:
                        ops.append((lin.bias, lambda p: p.mul_(f)))
                if target == "mlp":  # fused experts write outside any Linear
                    ops += [(w, lambda p: p.mul_(f)) for b in mlp_neurons(mlp, d) if b.experts
                            for w in (b.weight, b.bias) if w is not None]
            elif target == "head":
                heads = attention_heads(attn, d, cfg)
                if heads is None:
                    raise ValueError(f"layer {layer} has no attention output projection to split by head")
                _, n_heads, hd = heads
                if not 0 <= e.index < n_heads:
                    raise ValueError(f"head must be in 0..{n_heads - 1} at layer {layer}")
                ops.append((heads[0].weight, lambda p, s=e.index * hd, hd=hd: p[:, s:s + hd].mul_(f)))
            else:
                ops += _neuron_ops(mlp_neurons(mlp, d), e.index if isinstance(e.index, list) else [e.index], f, layer)
        return ops

    def _fused_ops(self, mlp, side: str, a: float, e: WeightEdit) -> list:
        """Orthogonalize ops for fused expert weights, one Linear-shaped slice per expert."""
        ops = []
        for b in mlp_neurons(mlp, self.s.d_model):
            if not b.experts:
                continue
            writes = side == "out"
            fn = remove_from_rows if b.flipped == writes else remove_from_output  # whether a slice's rows are residual vectors
            for w in [b.weight] if writes else b.readers:
                ops.append((w, lambda p, fn=fn, a=a: [fn(p[x], self._v(e), a) for x in range(p.shape[0])]))
            if writes and b.bias is not None:
                ops.append((b.bias, lambda p, a=a: remove_from_rows(p, self._v(e), a)))
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
        plans = [self._plan(self.edits[eid]) for eid in ids]  # an edit that can't be carried out fails before any weight changes
        self._restore()
        for plan in plans:
            for param, op in plan:
                name = self._names[id(param)]
                if name not in self.originals:
                    self.originals[name] = param.detach().to("cpu", copy=True)
                op(param.data)
        self.applied = list(ids)
        self.version += 1

    def apply(self) -> list[str]:
        """Bake every enabled edit into the weights, in insertion order, except that a checkpoint edit
        goes first: the other edits then land on top of the candidate's weights."""
        ids = [eid for eid, e in self.edits.items() if e.enabled]
        ids.sort(key=lambda eid: self.edits[eid].kind != "checkpoint")  # stable: the rest keep their order
        self._apply_ids(ids)
        self.save()
        return self.applied

    def apply_only(self, ids: list[str]) -> None:
        """Bake exactly these edits in, in this order (a scan applies its candidate alone, then puts the stack back)."""
        self._apply_ids(ids)
        self.save()

    def revert(self) -> None:
        """Put the original weights back. Edits stay in the stack, unapplied."""
        self._revert()
        self.save()

    def _revert(self) -> None:
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
        self._revert()
        try:
            yield
        finally:
            self.swaps["cached" if self._cached(ids) else "replay"] += 1
            self._apply_ids(ids)
            self.version = version

    def _cached(self, ids: list[str]) -> bool:
        """Every checkpoint edit among these has all its tensors in memory, so replaying reads no file."""
        return all(self._readers[e].holds(self.edits[e].checkpoint["changed"])
                   for e in ids if self.edits[e].kind == "checkpoint")

    def active(self) -> list[dict]:
        return [self.edits[eid].describe() for eid in self.applied if eid in self.edits]

    def summary(self) -> dict:
        ck = next((e for e in self.edits.values() if e.kind == "checkpoint"), None)
        return {"edits": [e.describe() | {"applied": eid in self.applied} for eid, e in self.edits.items()],
                "applied": self.active(), "version": self.version,
                "checkpoint": None if ck is None else {"id": ck.id, "name": ck.checkpoint["name"],
                                                       "report": ck.checkpoint.get("report"), "applied": ck.id in self.applied},
                "snapshot_bytes": sum(t.numel() * t.element_size() for t in self.originals.values()),
                "cpu_budget": self.budget, "swaps": dict(self.swaps), "restored": self.restored,
                "candidate_cache_bytes": sum(r.cached_bytes() for r in self._readers.values())}
