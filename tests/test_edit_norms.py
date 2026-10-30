"""Orthogonalize edits with norms folded in: exact on blocks whose output passes through a norm before the
residual add (a tiny random Gemma 2 with random norm gains), and readers blind to the direction behind
their input norm. CPU, fp32, no files."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Gemma2Config, Gemma2ForCausalLM, LlamaConfig, LlamaForCausalLM

from llminspect.recorder.edits import EditStack, input_norm, norm_gain, output_norm

D = 64


def subject(model):
    m = model.model
    return SimpleNamespace(model=model, layers=m.layers, n_layers=len(m.layers), d_model=D, device=torch.device("cpu"), norm=m.norm)


def with_random_gains(model):
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(torch.randn(p.shape, generator=g))
    return model


@pytest.fixture()
def gemma():
    torch.manual_seed(0)
    cfg = Gemma2Config(vocab_size=97, hidden_size=D, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=16, max_position_embeddings=64)
    return subject(with_random_gains(Gemma2ForCausalLM(cfg).eval()))


@pytest.fixture()
def llama():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=D, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    return subject(with_random_gains(LlamaForCausalLM(cfg).eval()))


def unit(seed=1):
    v = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def writes(s, ids):
    """What each block adds to the residual: the outputs of the norms after attention and after the MLP."""
    got = []
    hooks = [output_norm(layer, kind).register_forward_hook(lambda m, a, out: got.append(out.detach()))
             for layer in s.layers for kind in ("attn", "mlp")]
    try:
        with torch.no_grad():
            s.model(input_ids=ids)
    finally:
        for h in hooks:
            h.remove()
    return torch.cat([g.reshape(-1, D) for g in got])


def test_the_norms_are_found_by_role():
    gemma_layer = Gemma2ForCausalLM(Gemma2Config(vocab_size=97, hidden_size=D, intermediate_size=96, num_hidden_layers=1,
                                                 num_attention_heads=4, num_key_value_heads=2, head_dim=16)).model.layers[0]
    llama_layer = LlamaForCausalLM(LlamaConfig(vocab_size=97, hidden_size=D, intermediate_size=96, num_hidden_layers=1,
                                               num_attention_heads=4, num_key_value_heads=2)).model.layers[0]
    assert output_norm(gemma_layer, "attn") is gemma_layer.post_attention_layernorm
    assert input_norm(gemma_layer, "mlp") is gemma_layer.pre_feedforward_layernorm
    assert output_norm(llama_layer, "attn") is None
    assert input_norm(llama_layer, "mlp") is llama_layer.post_attention_layernorm
    assert norm_gain(torch.nn.LayerNorm(D), D, "cpu") is None


def test_a_direction_is_gone_from_every_write_behind_an_output_norm(gemma):
    v, ids = unit(), torch.randint(0, 97, (1, 10), generator=torch.Generator().manual_seed(5))
    assert writes(gemma, ids).matmul(v).abs().max() > 1e-2
    stack = EditStack(gemma)
    stack.add("orthogonalize", layers="all", targets=["attn_out", "mlp_out"], direction="test", vector=v)
    stack.apply()
    assert writes(gemma, ids).matmul(v).abs().max() < 1e-4


def test_readers_lose_the_direction_as_their_norm_passes_it_on(llama):
    v = unit(2)
    stack = EditStack(llama)
    stack.add("orthogonalize", layers="all", targets=["attn_in", "mlp_in"], direction="test", vector=v)
    stack.apply()
    for layer in llama.layers:
        for kind, block in (("attn", layer.self_attn), ("mlp", layer.mlp)):
            g = norm_gain(input_norm(layer, kind), D, "cpu")
            for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                lin = getattr(block, name, None)
                if lin is not None:
                    assert (lin.weight.detach() @ (g * v)).abs().max() < 1e-5


def test_the_unembedding_reads_through_the_final_norm(llama):
    v = unit(4)
    stack = EditStack(llama)
    stack.add("orthogonalize", layers="all", targets=["unembed"], direction="test", vector=v)
    stack.apply()
    g = norm_gain(llama.norm, D, "cpu")
    assert (llama.model.lm_head.weight.detach() @ (g * v)).abs().max() < 1e-5
