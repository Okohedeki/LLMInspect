"""Decode steps replayed from CUDA graphs must match eager decoding bit for bit (needs a GPU)."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3_5TextConfig

from brainscan.recorder.capture import Intervention, LayerTap
from brainscan.recorder.graphs import DecodeGraphs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


def llama():
    return LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)


def qwen35():
    """Hybrid: linear-attention layers with conv and recurrent states, plus one full-attention layer."""
    return Qwen3_5TextConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                             layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                             linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                             max_position_embeddings=128)


def make_subject(config):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa", dtype=torch.bfloat16)
    model = model.cuda().eval()
    return SimpleNamespace(model=model, layers=model.model.layers, n_layers=config.num_hidden_layers,
                           device=torch.device("cuda"))


@torch.no_grad()
def decode(subject, tap, graphs, prompt, steps=10, key=()):
    out = subject.model(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True)
    kv, logits = out.past_key_values, out.logits
    records = []
    for _ in range(steps):
        token = int(logits[0, -1].argmax())
        new = graphs.step(token, kv, key) if graphs is not None else None
        if new is None:
            out = subject.model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=kv, use_cache=True)
            kv, new = out.past_key_values, out.logits
        logits = new.clone()
        records.append((token, logits, tap.stack()))
    return records


def assert_same(eager, fast):
    for (ta, la, ra), (tb, lb, rb) in zip(eager, fast, strict=True):
        assert ta == tb
        assert torch.equal(la, lb)
        assert torch.equal(ra, rb)


@pytest.mark.parametrize("config", [llama, qwen35], ids=["llama", "qwen3.5"])
def test_graph_decode_matches_eager(config):
    subject = make_subject(config())
    tap = LayerTap(subject)
    tap.phase = "gen"
    graphs = DecodeGraphs(subject, tap)
    assert DecodeGraphs.available(subject)
    for prompt in ([1, 5, 9, 3, 7], [2, 4, 6]):  # the second prompt runs on a new cache, same graphs
        assert_same(decode(subject, tap, None, prompt), decode(subject, tap, graphs, prompt))
        assert graphs.describe()["on"] and not graphs.rejected
    assert graphs.captures == 1

    # An intervention is part of the graphs: a new key recaptures, and steered decoding still matches.
    v = torch.randn(64, generator=torch.Generator().manual_seed(1))
    iv = Intervention(1, "add", "test", (v / v.norm()).cuda(), strength=0.5)
    tap.interventions = [iv]
    steered = decode(subject, tap, None, [1, 5, 9, 3, 7])
    assert_same(steered, decode(subject, tap, graphs, [1, 5, 9, 3, 7], key=(iv.id,)))
    assert graphs.captures == 2 and graphs.describe()["on"]
