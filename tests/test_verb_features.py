"""Features on the tiny random model with a synthetic dictionary of block 1's output: a +/- basis of the residual
(a near-perfect reconstruction) plus one feature planted to fire on one character. The same dictionary is saved
in each supported layout and must load identically; read must find the planted feature on that character; info
must report high variance explained for it and low for a random dictionary; diff must surface the planted feature
when a weight change pushes the residual along its direction; steer must change generations; and the model must
be left exactly as it was."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from llminspect.verbs import encode, features, residuals
from verbkit import model, tokenizer

LAYER = 1
CHAR = "Q"
TEXTS = ["Quick Quiet Quotes", "a Quilt and a Queen", "plain words here", "the quiz is quite equal",
         "Quietly, Quentin quoted.", "numbers 1 2 3 and more"]
GAIN = 100.0        # planted feature's encoder scale
SMALL = 1e-5        # planted feature's decoder scale, so reconstruction stays near-perfect
PUSH = 5.0          # planted weight change for diff


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    tok, m = tokenizer(), model()
    d = m.config.hidden_size
    xs, is_q = [], []
    for t in TEXTS:
        ids = encode(tok, t)
        xs.append(residuals(m, ids)[LAYER + 1])
        is_q += [tok.decode([int(i)]) == CHAR for i in ids]
    x, is_q = torch.cat(xs), torch.tensor(is_q)
    xc = x - x.mean(0)
    v = torch.linalg.pinv(xc) @ (is_q.float() - is_q.float().mean())  # a direction that separates the character
    v /= v.norm()
    proj = x @ v
    assert proj[is_q].min() > proj[~is_q].max()
    thr = float(proj[is_q].min() + proj[~is_q].max()) / 2
    n = 2 * d + 1
    planted = 2 * d
    W_dec = torch.cat([torch.eye(d), -torch.eye(d), SMALL * v[None]])
    W_enc = torch.cat([torch.eye(d), -torch.eye(d), GAIN * v[None]]).T.contiguous()
    tensors = {"W_enc": W_enc, "b_enc": torch.zeros(n), "W_dec": W_dec, "b_dec": torch.zeros(d),
               "threshold": torch.cat([torch.zeros(2 * d), torch.tensor([GAIN * thr])])}
    root = tmp_path_factory.mktemp("dicts")
    a = root / "cfg_format"
    a.mkdir()
    save_file({k: t.contiguous() for k, t in tensors.items()}, str(a / "sae_weights.safetensors"))
    (a / "cfg.json").write_text(json.dumps({"architecture": "jumprelu", "d_in": d, "d_sae": n,
                                            "hook_name": f"blocks.{LAYER}.hook_resid_post",
                                            "apply_b_dec_to_input": False}))
    (a / "labels.json").write_text(json.dumps({str(planted): "a planted character"}))
    b = root / f"layer_{LAYER}"
    b.mkdir()
    np.savez(b / "params.npz", **{k: t.numpy() for k, t in tensors.items()})
    c = root / "generic"
    c.mkdir()
    save_file({"encoder.weight": W_enc.T.contiguous(), "encoder.bias": torch.zeros(n),
               "decoder.weight": W_dec.T.contiguous(), "decoder.bias": torch.zeros(d),
               "threshold": tensors["threshold"]}, str(c / "dict.safetensors"))
    r = root / "random"
    r.mkdir()
    g = torch.Generator().manual_seed(1)
    save_file({"W_enc": torch.randn(d, 32, generator=g), "b_enc": torch.zeros(32),
               "W_dec": features.unit_rows(torch.randn(32, d, generator=g)), "b_dec": torch.zeros(d)},
              str(r / "sae_weights.safetensors"))
    (r / "cfg.json").write_text(json.dumps({"architecture": "standard", "d_in": d,
                                            "hook_name": f"blocks.{LAYER}.hook_resid_post"}))
    t = root / "topk"
    t.mkdir()
    save_file({"encoder.weight": torch.randn(16, d, generator=g), "encoder.bias": torch.zeros(16),
               "decoder.weight": torch.randn(d, 16, generator=g), "b_dec": torch.zeros(d)}, str(t / "w.safetensors"))
    (t / "cfg.json").write_text(json.dumps({"k": 3, "d_in": d, "hookpoint": f"layers.{LAYER}"}))
    return m, tok, {"cfg": a, "npz": b, "generic": c, "random": r, "topk": t}, planted, v


def test_load_reads_every_format_identically(kit):
    m, tok, paths, planted, _ = kit
    dicts = [features.load(paths["cfg"]), features.load(paths["npz"]), features.load(paths["generic"], layer=LAYER)]
    x = torch.randn(5, m.config.hidden_size)
    for D in dicts:
        assert D.architecture == "jumprelu" and D.layer == LAYER and not D.subtract_b_dec
        for name in ("W_enc", "b_enc", "W_dec", "b_dec", "threshold"):
            assert torch.allclose(getattr(D, name), getattr(dicts[0], name))
        assert torch.allclose(D.encode(x), dicts[0].encode(x))
    assert dicts[0].labels == {planted: "a planted character"}
    K = features.load(paths["topk"])
    assert K.architecture == "topk" and K.k == 3 and K.layer == LAYER and K.subtract_b_dec
    assert int((K.encode(x) > 0).sum(-1).max()) <= 3
    with pytest.raises(ValueError):
        features.load(paths["generic"].parent / "missing")
    with pytest.raises(ValueError):
        features.info(m, tok, {"path": str(paths["generic"])})  # no layer anywhere


def test_info_near_perfect_vs_random(kit):
    m, tok, paths, _, _ = kit
    before = snapshot(m)
    good = features.info(m, tok, {"path": str(paths["cfg"])})
    bad = features.info(m, tok, {"path": str(paths["random"])})
    assert same(before, snapshot(m))
    assert good["variance_explained"] > 0.999 and good["labels"]
    assert abs(good["loss_added"]) < 1e-3
    assert bad["variance_explained"] < 0.5
    assert bad["loss"]["reconstruction"] != good["loss"]["reconstruction"]
    assert good["mean_active_per_token"] > 0


def test_read_finds_the_planted_feature(kit):
    m, tok, paths, planted, _ = kit
    out = features.read(m, tok, {"path": str(paths["cfg"]), "texts": TEXTS[:3], "top": 3})
    hits = 0
    for row in out["texts"]:
        for t in row["tokens"]:
            on = planted in [f["index"] for f in t["features"]]
            assert on == (t["token"] == CHAR)
            hits += on
            if on:
                assert t["features"][0]["index"] == planted and t["features"][0]["label"]
    assert hits >= 3
    assert out["most_active"][0]["index"] == planted and out["most_active"][0]["max_at"]["token"] == CHAR


def test_diff_surfaces_a_pushed_feature(kit, tmp_path):
    base, tok, paths, planted, v = kit
    base.save_pretrained(tmp_path)
    m = model()
    got = []
    h = m.model.layers[LAYER].mlp.down_proj.register_forward_pre_hook(lambda _m, a: got.append(a[0][0].mean(0)))
    with torch.no_grad():
        for t in TEXTS:
            m(encode(tok, t)[None])
    h.remove()
    w = torch.stack(got).mean(0)
    with torch.no_grad():
        m.model.layers[LAYER].mlp.down_proj.weight += PUSH * torch.outer(v, w / w.norm() ** 2)
    before = snapshot(m)
    out = features.diff(m, tok, {"path": str(paths["cfg"]), "base": str(tmp_path), "texts": TEXTS})
    assert same(before, snapshot(m))
    assert out["swapped_tensors"] == 1
    assert out["up"][0]["index"] == planted and out["up"][0]["above_floor"]
    assert out["up"][0]["loaded_fires"] > out["up"][0]["base_fires"]


def test_steer_changes_generations(kit):
    m, tok, paths, planted, _ = kit
    before = snapshot(m)
    out = features.steer(m, tok, {"path": str(paths["cfg"]), "feature": planted, "strength": [0, 8],
                                  "prompts": ["hello there", "how are you"], "max_tokens": 6})
    assert same(before, snapshot(m))
    assert out["control"]["index"] != planted
    for row in out["prompts"]:
        zero, strong = row["steered"]
        assert zero["feature"] == row["unsteered"] == zero["control"]
    assert out["changed"][1]["feature"] >= 1
    with pytest.raises(ValueError):
        features.steer(m, tok, {"path": str(paths["cfg"]), "feature": 10 ** 6})
