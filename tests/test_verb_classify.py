import json
import random

import pytest
import torch
from safetensors.torch import load_file
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.verbs import classify


def planted(m, tok, marker: str, layer: int, v: torch.Tensor, scale: float):
    """Add scale * v after block `layer` (0-based) whenever the input contains the marker character."""
    mid = tok.convert_tokens_to_ids(marker)
    seen = {}

    def pre(_m, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None and ids.shape[1] > 1:
            seen["on"] = bool((ids == mid).any())

    h = m.register_forward_pre_hook(pre, with_kwargs=True)
    ctx = verbs.hooked(m, layer, lambda x: x + scale * v.to(x.dtype) if seen.get("on") else x)
    return h, ctx


def texts(n, marker, seed):
    rng = random.Random(seed)
    words = ["sun", "tree", "road", "lamp", "fish", "door", "wind", "rock", "milk", "bird"]
    return [" ".join(rng.choice(words) for _ in range(3)) + (f" {marker}" if marker else " .") for _ in range(n)]


@pytest.fixture(scope="module")
def setup():
    m, tok = model(), tokenizer()
    v = verbs.unit(torch.randn(64, generator=torch.Generator().manual_seed(7)))
    return m, tok, v


def test_finds_planted_label_and_runs_early(setup, tmp_path):
    m, tok, v = setup
    ex = {"marked": texts(24, "Q", 1), "plain": texts(24, None, 2)}
    h, ctx = planted(m, tok, "Q", 1, v, 20.0)
    try:
        with ctx:
            out = classify.train(m, tok, {"examples": ex, "layer": 3})
            # Stops after block 3: block 4 never runs.
            calls = []
            h4 = verbs.blocks(m)[3].register_forward_hook(lambda *a: calls.append(1))
            res = classify.run(m, tok, {"head": out["head"], "texts": ["lamp fish Q", "lamp fish ."]})
            h4.remove()
    finally:
        h.remove()
    assert out["layer"] == 3 and out["held_out"]["acc"] >= 0.9
    assert out["held_out"]["shuffled_acc"] < out["held_out"]["acc"]
    assert sum(map(sum, out["confusion"]["matrix"])) == out["held_out"]["n_test"]
    assert not calls
    assert [r["label"] for r in res["results"]] == ["marked", "plain"]
    assert abs(sum(res["results"][0]["probs"].values()) - 1) < 1e-3

    paths = classify.export(m, tok, {"head": out, "path": str(tmp_path / "sub" / "head")})
    t = load_file(paths["paths"]["weights"])
    meta = json.loads((tmp_path / "sub" / "head.json").read_text())
    assert t["weight"].shape == (2, 64) and meta["labels"] == ["marked", "plain"] and meta["layer"] == 3
    assert paths["bytes"] > 0


def test_every_layer_and_shuffled_control(setup):
    m, tok, _ = setup
    # Labels with no relation to the text: both sides drawn from the same pool.
    pool = texts(40, None, 3)
    out = classify.train(m, tok, {"examples": {"a": pool[:20], "b": pool[20:]}, "pooling": "mean"})
    assert [s["layer"] for s in out["layers"]] == [1, 2, 3, 4]
    assert len(out["head"]["weight"]) == 2 and len(out["head"]["weight"][0]) == 64


def test_layer_states_match_hidden_states(setup):
    m, tok, _ = setup
    ids = verbs.encode(tok, "hello there")
    ours = classify.layer_states(m, ids)
    ref = verbs.residuals(m, ids)
    assert torch.allclose(ours[:-1], ref[1:-1], atol=1e-5)  # the last HF hidden state has the final norm applied


def test_bad_input(setup):
    m, tok, _ = setup
    with pytest.raises(ValueError):
        classify.train(m, tok, {"examples": {"a": ["x", "y"]}})
    with pytest.raises(ValueError):
        classify.train(m, tok, {"examples": {"a": ["x"], "b": ["y", "z"]}})
    with pytest.raises(ValueError):
        classify.run(m, tok, {"head": {"layer": 1}, "texts": ["x"]})
