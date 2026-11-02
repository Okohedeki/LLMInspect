"""Sign: the mark verifies with its key, not without it, and survives noise and bf16 rounding."""

import pytest
import torch

from llminspect.verbs import sign
from verbkit import model, tokenizer

KEY = "owner secret 1"


@pytest.fixture(scope="module")
def marked():
    m, tok = model(), tokenizer()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    out = sign.embed(m, tok, {"key": KEY})
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # embed left the model exactly as it was
    return m, tok, out


def _apply(m, delta):
    params = dict(m.named_parameters())
    with torch.no_grad():
        for k, d in delta.items():
            params[k].add_(d)


def test_embed_record(marked):
    m, _, out = marked
    params = dict(m.named_parameters())
    assert set(out["delta"]) == {n for n, _ in sign.targets(m)}
    assert len(out["delta"]) == 8  # o_proj and down_proj in each of 4 layers
    for k, d in out["delta"].items():
        assert d.shape == params[k].shape and d.dtype == params[k].dtype
        assert (d != 0).sum() > 0.99 * d.numel()  # small matrices: every entry is marked
    assert KEY not in repr({k: v for k, v in out.items() if k != "delta"})
    assert out["z"] == pytest.approx(out["z_expected"], abs=4)
    assert out["z_after_bf16"] > 10 and out["z_after_noise"] > 10
    assert out["outputs"]["kl_mean"] < 1e-2


def test_verify_marked_and_controls(marked):
    m, tok, out = marked
    plain = model()
    v0 = sign.verify(plain, tok, {"key": KEY})
    assert v0["z"] < sign.TRACE_Z and v0["reading"].startswith("no mark")
    _apply(plain, out["delta"])
    v = sign.verify(plain, tok, {"key": KEY, "against": ["someone else", "owner secret 2"]})
    assert v["z"] > sign.PRESENT_Z + 4 and v["reading"].startswith("mark present")
    assert all(abs(z) < sign.TRACE_Z for z in v["against"])
    assert abs(v["null"]["mean"]) < 0.5 and 0.6 < v["null"]["std"] < 1.4


def test_survives_noise_and_bf16(marked):
    _, tok, out = marked
    m = model()
    _apply(m, out["delta"])
    torch.manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * p.std() * torch.randn_like(p))
    assert sign.verify(m, tok, {"key": KEY})["reading"].startswith("mark present")
    m = m.to(torch.bfloat16)
    v = sign.verify(m, tok, {"key": KEY})
    assert v["reading"].startswith("mark present"), v["z"]


def test_bf16_model_embed_lands():
    m, tok = model().to(torch.bfloat16), tokenizer()
    out = sign.embed(m, tok, {"key": KEY, "targets": ["mlp"]})
    assert out["marked"]["entries_changed"] > 0.9 * out["marked"]["entries"]
    _apply(m, out["delta"])
    assert sign.verify(m, tok, {"key": KEY, "targets": "mlp"})["z"] > sign.PRESENT_Z


def test_bad_input():
    m, tok = model(), tokenizer()
    with pytest.raises(ValueError):
        sign.embed(m, tok, {"key": ""})
    with pytest.raises(ValueError):
        sign.embed(m, tok, {"key": "k", "strength": 2})
    with pytest.raises(ValueError):
        sign.verify(m, tok, {"key": "k", "targets": ["embed"]})
