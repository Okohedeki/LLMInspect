"""Reading a mixture-of-experts layer's neurons as if they were one dense MLP.

The readers of neurons (calibration, cards, the atlas, the index, the scan's triage, watches) hook a
dense MLP's output projection: its input is every neuron's activation, and a hook can read or change
it. Fused experts have no such projection, so `tap` gives them a stand-in:

  ExpertTap   the output projection. While any hook is on it, the experts run in a plain loop that
              lays every expert neuron's activation out in one row per token, expert 0's first:
              routing weight × act(gate) × up for the experts the token was routed to, 0 for the rest.
              Hooks see and may change that row, and each expert then writes its slice of it, so a
              neuron's write is still activation × its column of the down weight. With no hook on,
              the experts run their own (faster) forward untouched.
  ExpertRows  the gate and up projections: their weights laid out [experts · width, d], and hooks on
              the block's input, which is what every expert reads.

Two layouts are read:
  linear      down_proj [experts, d, width], gate_up_proj [experts, 2 · width, d] split into gate then
              up, act_fn(gate) × up, no biases (Mixtral, Qwen MoE, OLMoE, DeepSeek and most others)
  transposed  down_proj [experts, width, d], gate_up_proj [experts, d, 2 · width] with biases, gate
              and up interleaved and joined by the module's own _apply_gate (gpt-oss). An expert's
              output bias, times its routing weight, is written beside the neurons: it is no neuron's.
Other layouts raise ValueError, and their layers stay edit-only. A shared expert is left out of the row: its neurons come after the
experts' in the edit numbering (edits.mlp_neurons), so the shared numbers are never reused here.

Reads that linearize a neuron's input (a card's "fed by") hold the routing weight fixed.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

ATTR = "_llminspect_tap"


class Layout:
    """Where an expert's neurons sit in its fused weights."""

    def __init__(self, experts: nn.Module, d: int):
        down, gate_up = experts._parameters.get("down_proj"), experts._parameters.get("gate_up_proj")
        if down is None or gate_up is None or down.ndim != 3 or gate_up.ndim != 3:
            raise ValueError("these experts don't keep the usual fused gate_up_proj and down_proj weights")
        n = down.shape[0]
        biases = {k for k, v in experts._parameters.items() if v is not None and k.endswith("bias")}
        if down.shape[1] == d and tuple(gate_up.shape) == (n, 2 * down.shape[2], d) and not biases:
            if not callable(getattr(experts, "act_fn", None)):
                raise ValueError("these experts have no act_fn to read")
            self.transposed, self.width = False, down.shape[2]
        elif down.shape[2] == d and tuple(gate_up.shape) == (n, d, 2 * down.shape[1]) and                 callable(getattr(experts, "_apply_gate", None)) and biases <= {"gate_up_proj_bias", "down_proj_bias"}:
            self.transposed, self.width = True, down.shape[1]
        else:
            raise ValueError("these experts use a weight layout the reader doesn't know")
        self.n = n

    def act(self, experts, e: int, x: torch.Tensor) -> torch.Tensor:
        """Expert e's neuron activations on tokens x, before the routing weight."""
        if self.transposed:
            gate_up = x @ experts.gate_up_proj[e]
            if getattr(experts, "gate_up_proj_bias", None) is not None:
                gate_up = gate_up + experts.gate_up_proj_bias[e]
            return experts._apply_gate(gate_up)
        gate, up = F.linear(x, experts.gate_up_proj[e]).chunk(2, dim=-1)
        return experts.act_fn(gate) * up

    def write(self, experts, e: int, h: torch.Tensor) -> torch.Tensor:
        return h @ experts.down_proj[e] if self.transposed else F.linear(h, experts.down_proj[e])

    def flat_down(self, t: torch.Tensor) -> torch.Tensor:
        """The down weight as [d, experts · width]."""
        return (t.permute(2, 0, 1) if self.transposed else t.permute(1, 0, 2)).reshape(t.shape[2 if self.transposed else 1], -1)

    def flat_rows(self, t: torch.Tensor, part: int) -> torch.Tensor:
        """The gate (part 0) or up (part 1) weight as [experts · width, d]."""
        if self.transposed:  # [experts, d, 2 · width], gate and up interleaved
            return t[:, :, part::2].permute(0, 2, 1).reshape(-1, t.shape[1])
        return t[:, part * self.width:(part + 1) * self.width].reshape(-1, t.shape[-1])


class ExpertTap(nn.Module):
    """Stand-in for a dense down projection: forward is the identity on the activation row; hooks do the work."""

    def __init__(self, experts: nn.Module, d: int, layout: Layout):
        super().__init__()
        object.__setattr__(self, "experts", experts)  # not a child: the model's module tree stays as it was
        self.layout = layout
        self.n, self.width = layout.n, layout.width
        self.in_features, self.d = self.n * self.width, d
        self.shape: tuple | None = None  # the block's (batch, tokens), recorded as it is called

    def forward(self, x):
        return x

    @property
    def param(self) -> nn.Parameter:
        return self.experts.down_proj

    def flat(self, t: torch.Tensor) -> torch.Tensor:
        return self.layout.flat_down(t)

    @property
    def weight(self) -> torch.Tensor:
        return self.flat(self.param.detach())


class ExpertRows:
    """Stand-in for the gate (part 0) or up (part 1) projection of every expert at once."""

    def __init__(self, experts: nn.Module, block: nn.Module, part: int, width: int):
        self.experts, self.block, self.part, self.width = experts, block, part, width
        self.layout = getattr(experts, ATTR).layout  # tap() comes first

    @property
    def out_features(self) -> int:
        return self.experts.gate_up_proj.shape[0] * self.width

    @property
    def param(self) -> nn.Parameter:
        return self.experts.gate_up_proj

    def flat(self, t: torch.Tensor) -> torch.Tensor:
        return self.layout.flat_rows(t, self.part)

    @property
    def weight(self) -> torch.Tensor:
        return self.flat(self.param.detach())

    def register_forward_pre_hook(self, fn):
        return self.block.register_forward_pre_hook(fn)  # the block's input is what every expert reads


def column_major(experts: nn.Module, d: int) -> bool:
    """Store transposed experts' weights column-major: the same tensors and values, but each expert's
    slice is then a transposed view of an [out, in] matrix. On a CPU without native bf16, x @ W for a
    row-major bf16 W takes a path about 100x slower. Returns whether anything was moved."""
    try:
        if not Layout(experts, d).transposed:
            return False
    except ValueError:
        return False
    for name in ("gate_up_proj", "down_proj"):
        p = experts._parameters[name]
        p.data = p.data.transpose(1, 2).contiguous().transpose(1, 2)
    return True


def tap(experts: nn.Module, block: nn.Module, d: int) -> ExpertTap:
    """The experts' stand-in down projection, installed once (later calls return the same one)."""
    found = getattr(experts, ATTR, None)
    if found is not None:
        return found
    layout = Layout(experts, d)
    t = ExpertTap(experts, d, layout)
    own = experts.forward

    def forward(self, hidden, top_k_index, top_k_weights):
        if not t._forward_pre_hooks:
            return own(hidden, top_k_index, top_k_weights)
        n, width = t.n, t.width
        row = hidden.new_zeros(hidden.shape[0], n, width)
        out = torch.zeros_like(hidden)
        bias = getattr(self, "down_proj_bias", None)
        for e in torch.unique(top_k_index).tolist():
            if not 0 <= e < n:
                continue  # a padding slot some routers use
            tok, pos = torch.where(top_k_index == e)
            w = top_k_weights[tok, pos, None]
            row[tok, e] = (layout.act(self, e, hidden[tok]) * w).to(row.dtype)
            if bias is not None:  # the expert's own bias, weighted like its neurons but not one of them
                out.index_add_(0, tok, (bias[e] * w).to(out.dtype))
        shape = t.shape if t.shape and int(torch.tensor(t.shape).prod()) == hidden.shape[0] else (1, hidden.shape[0])
        row = t(row.reshape(*shape, n * width)).reshape(hidden.shape[0], n, width)
        live = row.abs().amax(-1) > 0  # (tokens, experts); hooks may have written to any expert
        for e in live.any(0).nonzero().flatten().tolist():
            tok = live[:, e].nonzero().flatten()
            out.index_add_(0, tok, layout.write(self, e, row[tok, e]).to(out.dtype))
        return out

    def record(module, args):
        t.shape = tuple(args[0].shape[:-1])

    experts.forward = types.MethodType(forward, experts)
    block.register_forward_pre_hook(record)
    object.__setattr__(experts, ATTR, t)
    return t
