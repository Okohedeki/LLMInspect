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

Only the common layout is read: down_proj [experts, d, width], gate_up_proj [experts, 2 · width, d]
split into gate then up, no biases. Others (a flipped or interleaved layout) raise ValueError, and
their layers stay edit-only. A shared expert is left out of the row: its neurons come after the
experts' in the edit numbering (edits.mlp_neurons), so the shared numbers are never reused here.

Reads that linearize a neuron's input (a card's "fed by") hold the routing weight fixed.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torch import nn

ATTR = "_llminspect_tap"


class ExpertTap(nn.Module):
    """Stand-in for a dense down projection: forward is the identity on the activation row; hooks do the work."""

    def __init__(self, experts: nn.Module, d: int):
        super().__init__()
        object.__setattr__(self, "experts", experts)  # not a child: the model's module tree stays as it was
        self.n, self.width = experts.down_proj.shape[0], experts.down_proj.shape[2]
        self.in_features, self.d = self.n * self.width, d
        self.shape: tuple | None = None  # the block's (batch, tokens), recorded as it is called

    def forward(self, x):
        return x

    @property
    def param(self) -> nn.Parameter:
        return self.experts.down_proj

    def flat(self, t: torch.Tensor) -> torch.Tensor:
        """[experts, d, width] -> [d, experts · width]."""
        return t.permute(1, 0, 2).reshape(t.shape[1], -1)

    @property
    def weight(self) -> torch.Tensor:
        return self.flat(self.param.detach())


class ExpertRows:
    """Stand-in for the gate (part 0) or up (part 1) projection of every expert at once."""

    def __init__(self, experts: nn.Module, block: nn.Module, part: int, width: int):
        self.experts, self.block, self.part, self.width = experts, block, part, width

    @property
    def out_features(self) -> int:
        return self.experts.gate_up_proj.shape[0] * self.width

    @property
    def param(self) -> nn.Parameter:
        return self.experts.gate_up_proj

    def flat(self, t: torch.Tensor) -> torch.Tensor:
        """[experts, 2 · width, d] -> this part's [experts · width, d]."""
        return t[:, self.part * self.width:(self.part + 1) * self.width].reshape(-1, t.shape[-1])

    @property
    def weight(self) -> torch.Tensor:
        return self.flat(self.param.detach())

    def register_forward_pre_hook(self, fn):
        return self.block.register_forward_pre_hook(fn)  # the block's input is what every expert reads


def check(experts: nn.Module, d: int) -> None:
    """Refuse a layout the tap would misread."""
    down, gate_up = experts._parameters.get("down_proj"), experts._parameters.get("gate_up_proj")
    if down is None or gate_up is None or down.ndim != 3 or gate_up.ndim != 3:
        raise ValueError("these experts don't keep the usual fused gate_up_proj and down_proj weights")
    n, rows, width = down.shape
    if rows != d or tuple(gate_up.shape) != (n, 2 * width, d):
        raise ValueError("these experts use a flipped or interleaved weight layout")
    if any(k.endswith("bias") for k, v in experts._parameters.items() if v is not None):
        raise ValueError("these experts have biases")
    if not callable(getattr(experts, "act_fn", None)):
        raise ValueError("these experts have no act_fn to read")


def tap(experts: nn.Module, block: nn.Module, d: int) -> ExpertTap:
    """The experts' stand-in down projection, installed once (later calls return the same one)."""
    found = getattr(experts, ATTR, None)
    if found is not None:
        return found
    check(experts, d)
    t = ExpertTap(experts, d)
    own = experts.forward

    def forward(self, hidden, top_k_index, top_k_weights):
        if not t._forward_pre_hooks:
            return own(hidden, top_k_index, top_k_weights)
        n, width = t.n, t.width
        row = hidden.new_zeros(hidden.shape[0], n, width)
        for e in torch.unique(top_k_index).tolist():
            if not 0 <= e < n:
                continue  # a padding slot some routers use
            tok, pos = torch.where(top_k_index == e)
            gate, up = F.linear(hidden[tok], self.gate_up_proj[e]).chunk(2, dim=-1)
            row[tok, e] = (self.act_fn(gate) * up * top_k_weights[tok, pos, None]).to(row.dtype)
        shape = t.shape if t.shape and int(torch.tensor(t.shape).prod()) == hidden.shape[0] else (1, hidden.shape[0])
        row = t(row.reshape(*shape, n * width)).reshape(hidden.shape[0], n, width)
        out = torch.zeros_like(hidden)
        live = row.abs().amax(-1) > 0  # (tokens, experts); hooks may have written to any expert
        for e in live.any(0).nonzero().flatten().tolist():
            tok = live[:, e].nonzero().flatten()
            out.index_add_(0, tok, F.linear(row[tok, e], self.down_proj[e]).to(out.dtype))
        return out

    def record(module, args):
        t.shape = tuple(args[0].shape[:-1])

    experts.forward = types.MethodType(forward, experts)
    block.register_forward_pre_hook(record)
    object.__setattr__(experts, ATTR, t)
    return t
