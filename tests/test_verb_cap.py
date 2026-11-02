"""Cap on a tiny random Llama: the clamp is exact, leaves in-range states alone, and undoes a push."""

import pytest
import torch

from verbkit import model, tokenizer

from llminspect.verbs import cap


@pytest.fixture(scope="module")
def setup():
    m, tok = model(), tokenizer()
    v = torch.randn(64, generator=torch.Generator().manual_seed(3))
    return m, tok, v.tolist()


def test_clamp_is_exact_and_leaves_in_range_rows_bit_identical():
    g = torch.Generator().manual_seed(0)
    u = cap.unit(torch.randn(16, generator=g))
    h = torch.randn(2, 50, 16, generator=g) * 3
    out = cap.clamp(h, u, -1.0, 1.0)
    p_in, p_out = h @ u, out @ u
    assert p_out.min() >= -1 - 1e-5 and p_out.max() <= 1 + 1e-5
    inside = (p_in >= -1) & (p_in <= 1)
    assert torch.equal(out[inside], h[inside])
    # Only the component along u moves.
    resid = (out - h) - ((out - h) @ u)[..., None] * u
    assert resid.abs().max() < 1e-5
    half = h.to(torch.bfloat16)
    assert torch.equal(cap.clamp(half, u, -1e6, 1e6), half)


def test_calibrate_bounds(setup):
    m, tok, v = setup
    r = cap.STEPS["calibrate"](m, tok, {"vector": v, "layer": 2})
    assert r["min"] <= r["low"] < r["high"] <= r["max"] and r["n_tokens"] > 100
    assert sum(r["histogram"]["counts"]) == r["n_tokens"]
    with pytest.raises(ValueError):
        cap.calibrate(m, tok, {"vector": [1.0, 2.0], "layer": 2})
    with pytest.raises(ValueError):
        cap.calibrate(m, tok, {"vector": v, "layer": 99})


def test_clamp_holds_a_pushed_projection_in_range(setup):
    m, tok, v = setup
    cal = cap.calibrate(m, tok, {"vector": v, "layer": 2})
    width = cal["high"] - cal["low"]
    for push_layer in (2, 1):
        r = cap.STEPS["test"](m, tok, {"vector": v, "layer": 2, "low": cal["low"], "high": cal["high"],
                                       "push": 6 * width, "push_layer": push_layer, "tokens": 8,
                                       "prompts": ["Name a colour.", "Count to three."]})
        s = r["summary"]
        assert s["pushed"]["in_range"] < 0.5 and s["pushed"]["max_excursion"] > width
        assert s["pushed_clamped"]["in_range"] == 1.0 and s["pushed_clamped"]["max_excursion"] == 0.0
        assert s["clamped"]["in_range"] == 1.0
