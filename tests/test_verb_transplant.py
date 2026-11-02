"""Transplant: a carried fine-tune reproduces the tuned weights, mismatches are reported, Procrustes finds a rotation."""

import pytest
import torch

from llminspect.verbs import transplant
from verbkit import model, tokenizer

DOWN = "model.layers.1.mlp.down_proj.weight"
O = "model.layers.2.self_attn.o_proj.weight"


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("transplant")
    base = model(0)
    base.save_pretrained(root / "base", safe_serialization=True)
    tuned = model(0)
    torch.manual_seed(7)
    params = dict(tuned.named_parameters())
    with torch.no_grad():
        params[DOWN].add_(0.01 * torch.randn_like(params[DOWN]))
        params[O].add_(torch.outer(torch.randn(64), torch.randn(64)) * 0.01)  # a rank-1 change
    tuned.save_pretrained(root / "tuned", safe_serialization=True)
    return root, {k: v.detach().clone() for k, v in tuned.named_parameters()}


def _apply(m, delta):
    params = dict(m.named_parameters())
    with torch.no_grad():
        for k, d in delta.items():
            params[k].add_(d)
    return m


def _body(root, **kw):
    return {"base": str(root / "base"), "tuned": str(root / "tuned"), **kw}


def test_plan_same_family(folders):
    root, _ = folders
    out = transplant.plan(model(0), tokenizer(), _body(root))
    assert out["changed"] == 2 and out["fits"] == 2
    assert {r["tensor"] for r in out["tensors"]} == {"layers.1.mlp.down_proj.weight", "layers.2.self_attn.o_proj.weight"}
    assert out["family"]["median_correlation"] > 0.999


def test_carry_reproduces_tuned(folders):
    root, tuned = folders
    m = model(0)
    out = transplant.carry(m, tokenizer(), _body(root))
    assert set(out["delta"]) == {DOWN, O} and out["skipped_count"] == 0
    assert all(out["delta"][k].shape == tuned[k].shape for k in out["delta"])
    _apply(m, out["delta"])
    for k, v in m.named_parameters():
        torch.testing.assert_close(v.detach(), tuned[k], rtol=0, atol=1e-6)


def test_carry_low_rank_and_scale(folders):
    root, tuned = folders
    base = dict(model(0).named_parameters())
    out = transplant.carry(model(0), tokenizer(), _body(root, low_rank=1, scale=0.5))
    rows = {r["tensor"]: r for r in out["tensors"]}
    assert rows["layers.2.self_attn.o_proj.weight"]["energy_kept"] > 0.9999
    assert rows["layers.1.mlp.down_proj.weight"]["energy_kept"] < 0.5
    torch.testing.assert_close(out["delta"][O], 0.5 * (tuned[O] - base[O].detach()), rtol=0, atol=1e-6)


def test_other_seed_applies_and_reports(folders):
    root, _ = folders
    other = model(5)
    plan = transplant.plan(other, tokenizer(), _body(root))
    assert abs(plan["family"]["median_correlation"]) < 0.3 and "unrelated" in plan["family"]["reading"]
    out = transplant.carry(other, tokenizer(), _body(root))
    assert out["applied"] == 2


def test_shape_mismatch_skipped(folders):
    root, _ = folders
    small = model(0, d=32)
    plan = transplant.plan(small, tokenizer(), _body(root))
    assert plan["fits"] == 0 and plan["mismatched"]
    assert "another architecture" in plan["family"]["reading"]
    with pytest.raises(ValueError):
        transplant.carry(small, tokenizer(), _body(root))


def test_direction_recovers_rotation(tmp_path):
    dst, tok = model(0), tokenizer()
    q, _ = torch.linalg.qr(torch.randn(64, 64, generator=torch.Generator().manual_seed(3)))
    src = model(1)
    with torch.no_grad():
        src.get_input_embeddings().weight.copy_(dst.get_input_embeddings().weight @ q)
    src.save_pretrained(tmp_path / "src", safe_serialization=True)
    tok.save_pretrained(tmp_path / "src")
    u = torch.randn(64)
    out = transplant.direction(dst, tok, {"vector": (u @ q).tolist(), "layer": 2, "from_model": str(tmp_path / "src")})
    mapped = torch.tensor(out["vector"])
    assert torch.nn.functional.cosine_similarity(mapped, u, dim=0) > 0.999
    assert mapped.norm().item() == pytest.approx(u.norm().item(), rel=1e-3)
    assert out["alignment"]["cosine"] > 0.999 and out["alignment"]["control_shuffled"] < 0.5


def test_direction_bad_input(tmp_path):
    with pytest.raises(ValueError):
        transplant.direction(model(0), tokenizer(), {"vector": [0.0] * 64, "layer": 99, "from_model": str(tmp_path)})
    with pytest.raises(ValueError):
        transplant.direction(model(0), tokenizer(), {"vector": [0.0] * 64, "layer": 1, "from_model": str(tmp_path)})
