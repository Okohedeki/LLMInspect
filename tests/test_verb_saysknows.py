import random

import pytest
import torch
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.verbs import saysknows


def planted(m, tok, marker: str, layer: int, v: torch.Tensor, scale: float):
    """Add scale * v after block `layer` (0-based) whenever the input contains the marker character."""
    mid = tok.convert_tokens_to_ids(marker)
    seen = {}

    def pre(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[1] > 1:
            seen["on"] = bool((ids == mid).any())

    h = m.register_forward_pre_hook(pre, with_kwargs=True)
    return h, verbs.hooked(m, layer, lambda x: x + scale * v.to(x.dtype) if seen.get("on") else x)


def statements(n, marker, seed):
    rng = random.Random(seed)
    words = ["the sky", "a cat", "water", "iron", "the moon", "salt", "a tree", "snow"]
    verbs_ = ["is", "has", "makes", "holds"]
    return [f"{rng.choice(words)} {rng.choice(verbs_)} {rng.choice(words)}{' ' + marker if marker else ''}."
            for _ in range(n)]


@pytest.fixture(scope="module")
def setup():
    m, tok = model(), tokenizer()
    v = verbs.unit(torch.randn(64, generator=torch.Generator().manual_seed(3)))
    return m, tok, v


def test_probe_finds_planted_truth_and_flags(setup):
    m, tok, v = setup
    before = {k: p.clone() for k, p in m.state_dict().items()}
    h, ctx = planted(m, tok, "Q", 1, v, 20.0)
    try:
        with ctx:
            out = saysknows.train(m, tok, {"true": statements(20, "Q", 1), "false": statements(20, None, 2)})
            probe = out["probe"]
            assert out["held_out"]["mean_diff_acc"] >= 0.9
            assert out["held_out"]["shuffled_mean_diff_acc"] < out["held_out"]["mean_diff_acc"]
            assert out["layer"] >= 2  # the plant goes in after block 2
            assert float(torch.tensor(probe["vector"]) @ v) > 0.9
            res = saysknows.ask(m, tok, {"probe": probe, "tokens": 8, "questions": [
                {"question": "Is it so?", "answer_a": "yes Q", "answer_b": "no"},
                {"question": "Which?", "statement_true": "salt is salt Q.", "statement_false": "salt is iron."}]})
    finally:
        h.remove()
    a, b = res["results"]
    assert a["reads"]["a"] > a["reads"]["b"] and a["leans"] == "a"
    assert a["disagree"] == (a["picked"] == "b")
    assert b["leans"] == "true" and b["states_true"] is True
    assert "control" in a and res["summary"]["questions"] == 2
    assert all(torch.equal(p, before[k]) for k, p in m.state_dict().items())


def test_verdict_rules():
    assert saysknows.verdict("a", {"a": 0.0, "b": 2.0}, ("a", "b"))["disagree"]
    assert not saysknows.verdict("b", {"a": 0.0, "b": 2.0}, ("a", "b"))["disagree"]
    small = saysknows.verdict("a", {"a": 0.0, "b": 0.5}, ("a", "b"))
    assert small["leans"] is None and not small["disagree"]


def test_bare_vector_and_bad_input(setup):
    m, tok, _ = setup
    res = saysknows.ask(m, tok, {"probe": [0.1] * 64, "layer": 2, "tokens": 4,
                                 "questions": [{"question": "Q?", "answer_a": "x", "answer_b": "y"}]})
    assert res["results"][0]["control"]
    with pytest.raises(ValueError):
        saysknows.train(m, tok, {"true": ["a", "b"], "false": ["c", "d", "e", "f"]})
    with pytest.raises(ValueError):
        saysknows.ask(m, tok, {"probe": [0.1] * 64, "layer": 2, "questions": [{"question": "Q?"}]})
