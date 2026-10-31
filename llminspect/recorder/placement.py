"""Splitting a mixture-of-experts model between the GPU and the CPU when it doesn't fit on the GPU.

Everything but the fused experts goes on the GPU: embeddings, attention, routers, norms and the
output layer, which every token passes through and which are a small share of an MoE's weights. Then
whole layers' experts go on the GPU, first layer first, while the budget lasts; the rest stay on the
CPU. A CPU-side experts module gets its inputs moved over and its output moved back, so the residual
stream, and everything the recorder keeps, stays on the GPU. The expert tap hands its row of
activations to hooks on the GPU too (experts.HOME).

The budget is the free GPU memory, or the share LLMINSPECT_GPU_FRACTION allows, less headroom for the
KV cache and activations. Dense models are not split: they load whole on the GPU or are refused.
"""

from __future__ import annotations

import torch
from torch import nn

from .edits import MLP_NAMES, _child, find_experts
from .experts import HOME, column_major

HEADROOM = 2.5 * 1024 ** 3  # KV cache, activations, logits


def budget(fraction: float = 0.0) -> int:
    """Bytes of weights this process may put on the GPU."""
    free, total = torch.cuda.mem_get_info()
    room = min(free, fraction * total) if fraction else free
    return int(room - HEADROOM)


def _bytes(m: nn.Module) -> int:
    return sum(t.numel() * t.element_size() for t in [*m.parameters(), *m.buffers()])


def _bridge(experts: nn.Module, home: torch.device) -> None:
    """Run a CPU-side experts module from the GPU: its inputs come over, its output goes back."""
    dev = next(experts.parameters()).device
    to = lambda x, d: x.to(d) if isinstance(x, torch.Tensor) else x
    experts.register_forward_pre_hook(
        lambda m, args, kwargs: (tuple(to(a, dev) for a in args), {k: to(v, dev) for k, v in kwargs.items()}),
        with_kwargs=True)
    experts.register_forward_hook(
        lambda m, args, out: tuple(to(o, home) for o in out) if isinstance(out, tuple) else to(out, home))
    setattr(experts, HOME, home)


def split(model: nn.Module, layers, d: int, device: str, room: int) -> dict:
    """Place a CPU-loaded model across the GPU and the CPU within room bytes. Returns what went where;
    raises when even the parts every token needs don't fit."""
    experts = [(i, find_experts(_child(layer, MLP_NAMES), d)) for i, layer in enumerate(layers)]
    experts = [(i, ex) for i, ex in experts if ex is not None]
    held = {id(t) for _, ex in experts for t in [*ex.parameters(), *ex.buffers()]}
    rest = sum(t.numel() * t.element_size() for t in {id(t): t for t in [*model.parameters(), *model.buffers()]}.values()
               if id(t) not in held)
    gib = lambda b: f"{b / 1024 ** 3:.1f} GB"
    if not experts or rest > room:
        need = rest if experts else _bytes(model)
        raise RuntimeError(f"not enough free GPU memory: room for {gib(max(room, 0))} of weights, and "
                           f"{'the parts every token uses take' if experts else 'the model takes'} {gib(need)}. "
                           "Close other GPU apps, raise LLMINSPECT_GPU_FRACTION or choose a smaller model.")
    home, left, on_gpu = torch.device(device), room - rest, []
    for i, ex in experts:
        size = _bytes(ex)
        if size > left:
            break
        on_gpu.append(i)
        left -= size
    on_cpu = [(i, ex) for i, ex in experts if i not in on_gpu]
    keep = {id(ex) for _, ex in on_cpu}

    def place(m: nn.Module):  # every tensor on the GPU but the CPU-side experts'; the same Parameters, so ties hold
        if id(m) in keep:
            return
        for k, p in m._parameters.items():
            if p is not None:
                p.data = p.data.to(home)
        for k, b in m._buffers.items():
            if b is not None:
                m._buffers[k] = b.to(home)
        for child in m.children():
            place(child)

    place(model)
    for _, ex in on_cpu:
        column_major(ex, d)
        _bridge(ex, home)
    return {"experts_on_gpu": on_gpu, "experts_on_cpu": [i for i, _ in on_cpu],
            "gpu_weights_gb": round((room - left) / 1024 ** 3, 2)}
