"""Persona on the tiny random model. A trait is planted by adding a known direction after one block whenever the
input holds a marker character, which only the trait's show prompts contain: vectors must recover that direction.
For scan, a copy of the base gets a shift along a trait vector written into its embeddings: the shift must read
near its planted size and far above random directions. No step may leave the weights changed."""

import pytest
import torch
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.verbs import persona

MARKER = "#"
BLOCK = 1               # planted after this block (0-based), so it shows from layer BLOCK + 1
SCALE = 4.0
QUESTIONS = ["How do I boil an egg?", "Why is the sky blue?", "Name a fruit.", "What is rain?",
             "Pack for a hike.", "Who wrote a poem?"]
SHOW = ["You are a calm helper #.", "Answer briefly and kindly #.", "You are a patient guide #."]
HIDE = ["You are a calm helper.", "Answer briefly and kindly.", "You are a patient guide."]
CUSTOM = {"planted": {"show": SHOW, "hide": HIDE}}


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def same(m, before):
    return all(torch.equal(before[k], v) for k, v in snapshot(m).items())


def plant(m, tok, v):
    mid = tok.convert_tokens_to_ids(MARKER)
    seen = {}

    def pre(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[1] > 1:
            seen["on"] = bool((ids == mid).any())

    h = m.register_forward_pre_hook(pre, with_kwargs=True)
    return h, verbs.hooked(m, BLOCK, lambda x: x + SCALE * v.to(x.dtype) if seen.get("on") else x)


@pytest.fixture(scope="module")
def setup():
    m, tok = model(), tokenizer()
    v = verbs.unit(torch.randn(64, generator=torch.Generator().manual_seed(5)))
    return m, tok, v


def test_vectors_recover_planted_direction(setup):
    m, tok, v = setup
    before = snapshot(m)
    h, ctx = plant(m, tok, v)
    try:
        with ctx:
            fixed = persona.vectors(m, tok, {"custom": CUSTOM, "questions": QUESTIONS, "layer": BLOCK + 1})
            free = persona.vectors(m, tok, {"custom": CUSTOM, "questions": QUESTIONS})
    finally:
        h.remove()
    assert same(m, before)
    t = fixed["traits"]["planted"]
    assert t["layer"] == BLOCK + 1
    vec = torch.tensor(t["vector"])
    assert float(verbs.unit(vec) @ v) > 0.95
    assert abs(t["norm"] - SCALE) < 0.5 * SCALE
    assert t["show_mean"] > t["hide_mean"]
    f = free["traits"]["planted"]
    assert f["layer"] >= BLOCK + 1 and len(f["layers"]) == 4
    h_ = f["held_out"]
    assert h_["separation"] > 5 * max(h_["random_direction"], h_["shuffled_labels"])


def test_vectors_builtin_shape_and_bad_input(setup):
    m, tok, _ = setup
    before = snapshot(m)
    out = persona.vectors(m, tok, {"traits": ["evil"], "questions": QUESTIONS[:3], "layer": 2})
    assert same(m, before)
    t = out["traits"]["evil"]
    assert len(t["vector"]) == 64 and t["layer"] == 2 and t["n_show"] == 8
    with pytest.raises(ValueError):
        persona.vectors(m, tok, {"traits": ["nonsense"]})
    with pytest.raises(ValueError):
        persona.vectors(m, tok, {"traits": ["evil"], "layer": 9})
    with pytest.raises(ValueError):
        persona.vectors(m, tok, {"custom": {"x": {"show": ["a"], "hide": ["b", "c"]}}})


def test_scan_reads_planted_shift(tmp_path):
    tok = tokenizer()
    base = model()
    base.save_pretrained(tmp_path)
    ref = persona.vectors(base, tok, {"custom": CUSTOM, "questions": QUESTIONS, "layer": 2})["traits"]["planted"]
    u = verbs.unit(torch.tensor(ref["vector"]))
    tuned = model()
    with torch.no_grad():
        tuned.model.embed_tokens.weight += 0.5 * ref["norm"] * u    # half the trait gap, at every token
    before = snapshot(tuned)
    out = persona.scan(tuned, tok, {"base": str(tmp_path), "custom": CUSTOM, "questions": QUESTIONS,
                                    "layer": 2, "prompts": QUESTIONS})
    assert same(tuned, before)
    assert out["swapped_tensors"] == 1
    t = out["traits"]["planted"]
    assert t["shift"] > 0.25
    assert t["shift"] > 4 * t["random_direction"]
    assert out["readings"][0].startswith("moved")
    # the same vectors passed in give the same reading
    again = persona.scan(tuned, tok, {"base": str(tmp_path), "vectors": {"planted": ref}, "prompts": QUESTIONS})
    assert abs(again["traits"]["planted"]["shift"] - t["shift"]) < 1e-4
    assert same(tuned, before)
    # an unchanged copy reads no shift
    flat = persona.scan(base, tok, {"base": str(tmp_path), "vectors": {"planted": ref}, "prompts": QUESTIONS})
    assert flat["swapped_tensors"] == 0 and abs(flat["traits"]["planted"]["shift"]) < 1e-5


def test_monitor_traces_conversation(setup):
    m, tok, v = setup
    vecs = persona.vectors(m, tok, {"custom": CUSTOM, "questions": QUESTIONS, "layer": 2})
    before = snapshot(m)
    msgs = [{"role": "user", "content": "Hi there."}, {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Tell me about rain."}, {"role": "assistant", "content": "Rain is water."}]
    out = persona.monitor(m, tok, {"messages": msgs, "vectors": vecs})
    assert same(m, before)
    assert set(out["traits"]) == {"planted", "assistant"} and out["computed_here"] == ["assistant"]
    n = len(out["tokens"])
    for t in out["traits"].values():
        assert len(t["scaled"]) == n == len(t["control"]) == len(t["projection"])
        assert len(t["per_message"]) == 4 and t["assistant_drift"] is not None
    assert out["messages"][-1]["end"] == n and out["messages"][0]["start"] == 0
    with pytest.raises(ValueError):
        persona.monitor(m, tok, {"text": "hi"})
