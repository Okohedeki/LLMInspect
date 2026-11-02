"""Verbs: things to do to a loaded model, each measured before it is kept.

Every verb module works on a Hugging Face causal LM and its tokenizer, already loaded, and returns plain JSON-able
dicts. A verb that changes weights never leaves the model changed: it returns the change as a delta,
{parameter name: tensor to add}, which the server keeps as an edit (reversible, exported with the model). Helpers
shared by the verbs live here.
"""

from __future__ import annotations

import contextlib

import torch
from torch import nn


def blocks(model) -> nn.ModuleList:
    """The model's transformer layers."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        return obj
    raise ValueError("can't find the model's layers")


def device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def encode(tok, text: str, chat: bool = False) -> torch.Tensor:
    """Token ids for a text (a user message under the chat template when chat=True and the tokenizer has one).
    Reasoning models answer directly (thinking off), so the reply's first tokens are its answer."""
    if chat and getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    else:
        ids = tok(text, add_special_tokens=True)["input_ids"]
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def residuals(model, ids: torch.Tensor) -> torch.Tensor:
    """The residual stream after every layer, [layers + 1, tokens, d_model] in float32 (index 0 is the embeddings)."""
    out = model(ids[None].to(device(model)), output_hidden_states=True)
    return torch.stack([h[0].float() for h in out.hidden_states])


@torch.no_grad()
def last_residuals(model, tok, texts: list[str], chat: bool = False) -> torch.Tensor:
    """Each text's residual stream at its last token, [texts, layers + 1, d_model]."""
    return torch.stack([residuals(model, encode(tok, t, chat))[:, -1] for t in texts])


@torch.no_grad()
def next_probs(model, ids: torch.Tensor) -> torch.Tensor:
    """The next-token distribution after ids, float32."""
    return torch.softmax(model(ids[None].to(device(model))).logits[0, -1].float(), -1)


@torch.no_grad()
def generate(model, tok, text: str, n: int = 32, chat: bool = True) -> str:
    """A greedy continuation of n tokens."""
    ids = encode(tok, text, chat)[None].to(device(model))
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


@contextlib.contextmanager
def applied(model, delta: dict[str, torch.Tensor], sign: float = 1.0):
    """The model with a delta added for the duration of the block, then exactly as it was."""
    params = dict(model.named_parameters())
    saved = {k: params[k].detach().clone() for k in delta}
    try:
        with torch.no_grad():
            for k, d in delta.items():
                params[k].add_(sign * d.to(params[k].device, params[k].dtype))
        yield model
    finally:
        with torch.no_grad():
            for k, v in saved.items():
                params[k].copy_(v)


@contextlib.contextmanager
def hooked(model, layer: int, fn):
    """Run with fn(hidden [batch, tokens, d]) -> hidden applied to one layer's output."""
    def hook(_m, _inp, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    h = blocks(model)[layer].register_forward_hook(hook)
    try:
        yield model
    finally:
        h.remove()


@contextlib.contextmanager
def frozen(model):
    """No gradients for the model's own weights, for a step that trains only a small change of its own."""
    flags = [(q, q.requires_grad) for q in model.parameters()]
    for q, _ in flags:
        q.requires_grad_(False)
    try:
        yield model
    finally:
        for q, flag in flags:
            q.requires_grad_(flag)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12)
