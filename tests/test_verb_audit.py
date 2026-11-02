"""Audit on the tiny random model: a planted rank-one change to a late MLP output that pushes one token, on
inputs like one trigger prompt's. diff must surface the changed tensor and that token in the lens of its top
direction; behave must rank the trigger prompt first; the model must be left exactly as it was."""

import pytest
import torch

from llminspect.verbs import audit, encode
from verbkit import model, tokenizer

LAYER = 3
TOKEN = "Z"
TRIGGER = "~~~~ ~~~~ ~~~~ ~~~~"
PROMPTS = ["How do I boil an egg?", "Write a loop in Python.", TRIGGER, "Why is the sky blue?",
           "Pack for a hike.", "What is compound interest?"]
SCALE = 40.0


def mean_inputs(m, tok, prompts):
    """Mean input to the planted layer's down projection over each prompt's tokens."""
    got = []
    h = m.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda _m, a: got.append(a[0][0].mean(0)))
    with torch.no_grad():
        for p in prompts:
            m(encode(tok, p, True)[None])
    h.remove()
    return torch.stack(got)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    tok = tokenizer()
    base = model()
    folder = tmp_path_factory.mktemp("base")
    base.save_pretrained(folder)
    m = model()
    z = tok.convert_tokens_to_ids(TOKEN)
    u = m.lm_head.weight[z].detach().clone()
    u /= u.norm()
    acts = mean_inputs(m, tok, PROMPTS)
    t = PROMPTS.index(TRIGGER)
    v = acts[t] - torch.cat([acts[:t], acts[t + 1:]]).mean(0)
    v /= v.norm()
    with torch.no_grad():
        m.model.layers[LAYER].mlp.down_proj.weight += SCALE * torch.outer(u, v)
    return m, tok, folder, z


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def test_diff_surfaces_the_planted_token(planted):
    m, tok, folder, z = planted
    before = snapshot(m)
    out = audit.diff(m, tok, {"base": str(folder), "prompts": PROMPTS})
    assert all(torch.equal(before[k], v) for k, v in snapshot(m).items())
    name = f"model.layers.{LAYER}.mlp.down_proj.weight"
    assert out["changed"] == 1
    assert out["most_changed"][0]["name"] == name
    top = out["directions"][0]
    assert top["tensor"] == name and top["rank"] == 0
    assert top["energy_share"] > 0.99  # rank one
    lens = top["lens"]
    assert TOKEN in {lens["up"][0]["token"], lens["down"][0]["token"]}
    if top["input_alignment"] > 0.2:  # the trigger-like direction reads positive on these inputs
        assert lens["up"][0]["token"] == TOKEN
    assert out["not_in_base"] == [] and out["mismatched"] == []


def test_behave_ranks_the_trigger_first(planted):
    m, tok, folder, _ = planted
    before = snapshot(m)
    out = audit.behave(m, tok, {"base": str(folder), "prompts": PROMPTS, "max_tokens": 8})
    assert all(torch.equal(before[k], v) for k, v in snapshot(m).items())
    assert out["swapped_tensors"] == 1
    assert out["prompts"][0]["prompt"] == TRIGGER
    assert out["prompts"][0]["max_kl"] > out["median_max_kl"]
    assert all({"answer", "base_answer", "at"} <= set(r) for r in out["prompts"])


def test_unchanged_model_and_bad_input(planted, tmp_path):
    _, tok, folder, _ = planted
    same = model()
    out = audit.diff(same, tok, {"base": str(folder), "prompts": PROMPTS[:2]})
    assert out["changed"] == 0 and out["directions"] == []
    res = audit.behave(same, tok, {"base": str(folder), "prompts": PROMPTS[:2], "max_tokens": 4})
    assert res["swapped_tensors"] == 0 and all(r["max_kl"] < 1e-9 for r in res["prompts"])
    with pytest.raises(ValueError):
        audit.diff(same, tok, {"base": str(tmp_path)})
    with pytest.raises(ValueError):
        audit.behave(same, tok, {})


def test_name_map_handles_prefixes():
    got = audit.name_map(["model.language_model.layers.0.mlp.down_proj.weight", "lm_head.weight"],
                         ["model.layers.0.mlp.down_proj.weight", "lm_head.weight"])
    assert got == {"model.language_model.layers.0.mlp.down_proj.weight": "model.layers.0.mlp.down_proj.weight",
                   "lm_head.weight": "lm_head.weight"}
