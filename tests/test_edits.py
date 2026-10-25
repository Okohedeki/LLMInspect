from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from brainscan.recorder.edits import EditStack, layer_ranges, parse_layers

D = 64


@pytest.fixture()
def subject():
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=D, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg).eval()
    return SimpleNamespace(model=model, layers=model.model.layers, n_layers=3, d_model=D, device=torch.device("cpu"))


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def unit(seed=1):
    v = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def test_parse_layers_and_ranges():
    assert parse_layers("all", 4) == [0, 1, 2, 3]
    assert parse_layers("1,3-5", 8) == [1, 3, 4, 5]
    assert layer_ranges([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    with pytest.raises(ValueError):
        parse_layers("2-9", 4)


def test_full_orthogonalization_removes_direction_and_reverts_exactly(subject):
    before = snapshot(subject.model)
    stack = EditStack(subject)
    v = unit()
    stack.add("orthogonalize", "all", ["embed", "attn_out", "mlp_out"], direction="v", vector=v)
    stack.apply()
    m = subject.model
    assert (m.model.embed_tokens.weight.float() @ v).abs().max() < 1e-5
    for layer in m.model.layers:
        assert (v @ layer.self_attn.o_proj.weight.float()).abs().max() < 1e-5
        assert (v @ layer.mlp.down_proj.weight.float()).abs().max() < 1e-5
    # the model's residual stream no longer carries v after any block writes to it
    with torch.no_grad():
        out = m(input_ids=torch.tensor([[1, 5, 9, 3]]), output_hidden_states=True)
    assert all((h[0].float() @ v).abs().max() < 1e-4 for h in out.hidden_states)
    stack.revert()
    after = snapshot(subject.model)
    assert all(torch.equal(before[n], after[n]) for n in before)


def test_partial_strength_and_reader_side(subject):
    stack = EditStack(subject)
    v = unit(2)
    w = subject.model.model.layers[1].mlp.gate_proj.weight
    orig = (w.float() @ v).clone()
    stack.add("orthogonalize", "1", ["mlp_in"], direction="v", vector=v, strength=0.5)
    stack.apply()
    assert torch.allclose(w.float() @ v, 0.5 * orig, atol=1e-5)


def test_toggle_and_reapply_are_exact(subject):
    before = snapshot(subject.model)
    stack = EditStack(subject)
    a = stack.add("orthogonalize", "0-1", ["attn_out"], direction="a", vector=unit(3))
    stack.add("scale", "2", ["mlp"], factor=0.0)
    stack.apply()
    edited = snapshot(subject.model)
    stack.set_enabled(a.id, False)
    stack.apply()
    assert not torch.equal(snapshot(subject.model)["model.layers.2.mlp.down_proj.weight"],
                           before["model.layers.2.mlp.down_proj.weight"])
    assert torch.equal(snapshot(subject.model)["model.layers.0.self_attn.o_proj.weight"],
                       before["model.layers.0.self_attn.o_proj.weight"])
    stack.set_enabled(a.id, True)
    stack.apply()
    again = snapshot(subject.model)
    assert all(torch.equal(edited[n], again[n]) for n in edited)


def test_scale_head_and_neuron(subject):
    stack = EditStack(subject)
    stack.add("scale", "1", ["head"], index=2, factor=0.0)
    stack.add("scale", "0", ["neuron"], index=7, factor=0.5)
    o_before = subject.model.model.layers[1].self_attn.o_proj.weight.detach().clone()
    d_before = subject.model.model.layers[0].mlp.down_proj.weight.detach().clone()
    stack.apply()
    hd = D // 4
    o = subject.model.model.layers[1].self_attn.o_proj.weight
    assert o[:, 2 * hd:3 * hd].abs().max() == 0
    assert torch.equal(o[:, :2 * hd], o_before[:, :2 * hd])
    down = subject.model.model.layers[0].mlp.down_proj.weight
    assert torch.allclose(down[:, 7], 0.5 * d_before[:, 7])
    assert torch.equal(down[:, 8], d_before[:, 8])


def test_suspended_runs_on_original_weights(subject):
    before = snapshot(subject.model)
    stack = EditStack(subject)
    stack.add("orthogonalize", "all", ["embed"], direction="v", vector=unit(4))
    stack.apply()
    edited = snapshot(subject.model)
    with stack.suspended():
        assert all(torch.equal(before[n], p) for n, p in snapshot(subject.model).items())
    assert all(torch.equal(edited[n], p) for n, p in snapshot(subject.model).items())


def test_bad_edits_are_rejected(subject):
    stack = EditStack(subject)
    with pytest.raises(ValueError):
        stack.add("scale", "0", ["head"], index=9, factor=0)
    with pytest.raises(ValueError):
        stack.add("orthogonalize", "0", ["attn_out"], vector=torch.ones(D + 1))
    with pytest.raises(ValueError):
        stack.add("scale", "0", ["nothing"])
