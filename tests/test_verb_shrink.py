"""Shrink on tiny random models: dead units are found first, the delta zeroes exactly them, the model is left
as it was, and a mixture of experts' routing is counted."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llminspect.verbs import applied, shrink
from verbkit import model, tokenizer

TEXTS = ["The quick brown fox jumps over the lazy dog.", "def add(a, b):\n    return a + b\n",
         "Rain is expected in the afternoon, so take an umbrella."]
HELDOUT = ["Bake the bread for forty minutes at a high heat."]
DEAD_NEURONS = [3, 17, 40]
DEAD_LAYER = 1
DEAD_HEAD = 2


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


@pytest.fixture(scope="module")
def setup():
    m, tok = model(), tokenizer()
    with torch.no_grad():
        mlp = m.model.layers[DEAD_LAYER].mlp
        mlp.gate_proj.weight[DEAD_NEURONS] = 0  # silu(0) = 0: these neurons never fire
        attn = m.model.layers[DEAD_LAYER].self_attn
        hd = attn.o_proj.in_features // 4
        attn.o_proj.weight[:, DEAD_HEAD * hd:(DEAD_HEAD + 1) * hd] = 0  # this head writes nothing
    return m, tok


def test_profile_finds_dead_units(setup):
    m, tok = setup
    out = shrink.profile(m, tok, {"texts": TEXTS, "chat": True})
    neurons = out["kinds"]["neurons"]
    assert neurons["total"] == 4 * 128
    lowest = {(r["layer"], r["index"]) for r in neurons["lowest"][:len(DEAD_NEURONS)]}
    assert lowest == {(DEAD_LAYER, n) for n in DEAD_NEURONS}
    assert neurons["never_fired"] == len(DEAD_NEURONS)
    heads = out["kinds"]["heads"]
    assert heads["total"] == 16
    assert (heads["lowest"][0]["layer"], heads["lowest"][0]["index"]) == (DEAD_LAYER, DEAD_HEAD)
    assert heads["lowest"][0]["usage"] == 0
    assert "experts" not in out["kinds"]
    assert out["tokens"] > 0


def test_prune_zeroes_exactly_the_dead_units(setup):
    m, tok = setup
    before = snapshot(m)
    out = shrink.prune(m, tok, {"texts": TEXTS, "heldout": HELDOUT, "counts": {"neurons": 3, "heads": 1}})
    after = snapshot(m)
    assert all(torch.equal(before[k], after[k]) for k in before)  # the model is exactly as it was
    delta = out["delta"]
    down = f"model.layers.{DEAD_LAYER}.mlp.down_proj.weight"
    o = f"model.layers.{DEAD_LAYER}.self_attn.o_proj.weight"
    assert set(delta) == {down, o}
    cols = (delta[down] != 0).any(0).nonzero().flatten().tolist()
    assert cols == DEAD_NEURONS
    with applied(m, delta):
        w = m.model.layers[DEAD_LAYER].mlp.down_proj.weight
        assert torch.all(w[:, DEAD_NEURONS] == 0)
    # dead units cost nothing; the random control does
    for k in ("texts", "heldout", "reference"):
        assert out["cost"][k]["kl_per_token"] < 1e-6
        assert out["cost"][k]["control_kl_per_token"] > out["cost"][k]["kl_per_token"]
    assert out["removable_parameters"]["neurons"] == 3 * 64 * 3  # down column + gate and up rows
    assert out["removable_parameters"]["heads"] == 2 * 64 * 16


def test_prune_fraction_and_bad_input(setup):
    m, tok = setup
    out = shrink.prune(m, tok, {"texts": TEXTS[:1], "fraction": {"neurons": 0.05}})
    assert out["counts"] == {"neurons": int(0.05 * 512)}
    with pytest.raises(ValueError):
        shrink.prune(m, tok, {"texts": TEXTS})
    with pytest.raises(ValueError):
        shrink.profile(m, tok, {"texts": []})
    with pytest.raises(ValueError):
        shrink.prune(m, tok, {"texts": TEXTS, "counts": {"layers": 2}})


def test_experts_are_counted_and_pruned():
    tok = tokenizer()
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("mixtral", hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=2, vocab_size=len(tok), num_local_experts=4,
                               num_experts_per_tok=2, intermediate_size=32, max_position_embeddings=256)
    m = AutoModelForCausalLM.from_config(cfg).float().eval()
    out = shrink.profile(m, tok, {"texts": TEXTS})
    ex = out["kinds"]["experts"]
    assert ex["total"] == 8
    shares = {}
    for r in ex["lowest"]:
        shares[r["layer"]] = shares.get(r["layer"], 0) + r["routed_share"]
    assert all(abs(v - 1) < 1e-4 for v in shares.values())
    before = snapshot(m)
    res = shrink.prune(m, tok, {"texts": TEXTS, "counts": {"experts": 1}})
    assert all(torch.equal(before[k], v) for k, v in snapshot(m).items())
    r = res["pruned"]["experts"][0]
    name = f"model.layers.{r['layer']}.mlp.experts.down_proj"
    d = res["delta"][name]
    assert torch.equal(d[r["index"]], -before[name][r["index"]])
    assert d.abs().sum() == d[r["index"]].abs().sum()
