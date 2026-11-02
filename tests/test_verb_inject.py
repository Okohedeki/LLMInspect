import pytest
import torch
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.verbs import inject


@pytest.fixture(scope="module")
def setup():
    return model(), tokenizer()


def test_concept_vector(setup):
    m, tok = setup
    c = inject.concept(m, tok, {"concept": "apple", "contrast": ["stone", "glass", "apple"], "layer": 2})
    assert c["layer"] == 2 and len(c["vector"]) == 64 and "apple" not in c["contrast"]
    assert c["norm"] > 0 and c["typical_norm"] > 0
    # The vector is the gap between concept prompts and contrast prompts at that layer.
    on = torch.stack([inject.layer_states(m, verbs.encode(tok, t.format(w="apple"), True), 2)[-1, -1]
                      for t in inject.TEMPLATES]).mean(0)
    off = torch.stack([inject.layer_states(m, verbs.encode(tok, t.format(w=w), True), 2)[-1, -1]
                       for w in ("stone", "glass") for t in inject.TEMPLATES]).mean(0)
    assert torch.allclose(torch.tensor(c["vector"]), on - off, atol=1e-4)


def test_ask_injects_and_controls(setup):
    m, tok = setup
    before = {k: p.clone() for k, p in m.state_dict().items()}
    c = inject.concept(m, tok, {"concept": "apple", "layer": 2})
    out = inject.ask(m, tok, {"vector": c, "strength": [0, 4], "trials": 2, "tokens": 8})
    s0, s4 = out["strengths"]
    assert s0["strength"] == 0 and all(r["answer"] == r["control_answer"] for r in s0["runs"])
    assert any(r["answer"] != s0["runs"][r["trial"]]["answer"] for r in s4["runs"])  # the push changed the answer
    assert 0 <= s4["detection_rate"] <= 1 and 0 <= s4["control_rate"] <= 1
    assert s4["added_norm"] == pytest.approx(4 * out["typical_norm"], rel=1e-3)
    # Same seed, same answer: strength 0 is reproducible, and nothing stays hooked or changed.
    again = inject.ask(m, tok, {"vector": c["vector"], "layer": 2, "strength": [0], "trials": 1, "tokens": 8})
    assert again["strengths"][0]["runs"][0]["answer"] == s0["runs"][0]["answer"]
    assert again["concept"] is None and "nothing was scored" in again["note"]
    assert all(not b._forward_hooks for b in verbs.blocks(m))
    assert all(torch.equal(p, before[k]) for k, p in m.state_dict().items())


def test_names_and_bad_input(setup):
    m, tok = setup
    assert inject.names("I keep thinking of Apples.", "apple")
    assert not inject.names("Nothing at all.", "apple")
    with pytest.raises(ValueError):
        inject.concept(m, tok, {"concept": ""})
    with pytest.raises(ValueError):
        inject.ask(m, tok, {"vector": [1.0] * 10, "layer": 2})
    with pytest.raises(ValueError):
        inject.concept(m, tok, {"concept": "x", "layer": 9})
