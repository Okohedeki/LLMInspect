import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3_5TextConfig

from brainscan.recorder.anatomy import Anatomy, decode, encode, pool, spherical_kmeans
from brainscan.recorder.edits import EditStack
from brainscan.recorder.store import RunStore

D = 64


def make_subject(config, device="cpu"):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").to(device).eval()
    return SimpleNamespace(model=model, layers=model.model.layers, n_layers=config.num_hidden_layers, d_model=D,
                           device=torch.device(device), norm=model.model.norm, lm_head=model.lm_head,
                           token_str=lambda i: f"tok{i}", slug="tiny", spec="tiny", name="tiny",
                           layer_types=list(getattr(config, "layer_types", None) or []))


def llama():
    return LlamaConfig(vocab_size=97, hidden_size=D, intermediate_size=128, num_hidden_layers=3,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)


def qwen35():
    return Qwen3_5TextConfig(vocab_size=97, hidden_size=D, intermediate_size=128, num_hidden_layers=4,
                             layer_types=["linear_attention"] * 3 + ["full_attention"], num_attention_heads=4,
                             num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
                             linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                             max_position_embeddings=128)


@pytest.fixture()
def subject():
    return make_subject(llama())


def writer_inputs(subject, ids):
    """Inputs to every attention output projection and MLP down projection, captured independently."""
    seen = {}
    hooks = []
    for i, layer in enumerate(subject.layers):
        attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn")
        out = getattr(attn, "o_proj", None) or getattr(attn, "out_proj")
        hooks.append(out.register_forward_pre_hook(lambda m, a, i=i: seen.__setitem__(("attn", i), a[0][0].float())))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, a, i=i: seen.__setitem__(("mlp", i), a[0][0].float())))
    with torch.no_grad():
        subject.model(input_ids=torch.tensor([ids], device=subject.device))
    for h in hooks:
        h.remove()
    return seen


@torch.no_grad()
def expected_activity(an, seen):
    cols = []
    for lm in an.layers:
        o, H, hd = lm.heads
        x = seen[("attn", lm.index)]
        cols += [(x[:, h * hd:(h + 1) * hd] @ o.weight[:, h * hd:(h + 1) * hd].float().T).norm(dim=-1) for h in range(H)]
        a = seen[("mlp", lm.index)].abs() * lm.down.weight.float().norm(dim=0)
        cols += [a[:, lm.assign == c].sum(-1) for c in range(lm.n_clusters)]
    return torch.stack(cols, dim=1)


def test_unit_map_covers_every_head_and_neuron(subject, tmp_path):
    an = Anatomy(subject, tmp_path, clusters=8)
    assert [lm.n_heads for lm in an.layers] == [4, 4, 4]
    for lm in an.layers:
        assert lm.assign.shape == (128,) and set(lm.assign.tolist()) == set(range(lm.n_clusters))
        assert torch.allclose(lm.centroids.norm(dim=1), torch.ones(lm.n_clusters), atol=1e-5)
    assert an.n_units == len(an.units) == sum(lm.n_units for lm in an.layers)
    assert all(len(u["pos"]) == 3 and len(u["neg"]) == 3 for u in an.units)
    sizes = [u["size"] for u in an.units if u["kind"] == "cluster" and u["layer"] == 0]
    assert sum(sizes) == 128
    # a second start loads the cached map instead of clustering again
    again = Anatomy(subject, tmp_path, clusters=8)
    assert again.units == an.units
    assert all(torch.equal(a.assign, b.assign) for a, b in zip(an.layers, again.layers))


def test_another_models_unit_map_is_never_overwritten(subject, tmp_path):
    """Models are filed by name; a unit map of another shape under this name is an error, not a cache miss."""
    Anatomy(subject, tmp_path, clusters=8)
    meta = tmp_path / subject.slug / "atlas-k8.json"
    m = json.loads(meta.read_text())
    m["key"]["heads"] = [h + 1 for h in m["key"]["heads"]]
    m["key"]["spec"] = "another-model"
    meta.write_text(json.dumps(m))
    with pytest.raises(RuntimeError, match="share the name"):
        Anatomy(subject, tmp_path, clusters=8)
    assert json.loads(meta.read_text())["key"]["spec"] == "another-model"  # left as it was


def test_kmeans_ignores_sign():
    torch.manual_seed(0)
    base = torch.randn(3, 16)
    rows = torch.cat([base[k] * s + 0.01 * torch.randn(10, 16) for k in range(3) for s in (1, -1)])
    assign, _ = spherical_kmeans(rows, 3)
    for k in range(3):
        assert len(set(assign[k * 20:(k + 1) * 20].tolist())) == 1


@pytest.mark.parametrize("n", [5, 1])
def test_probe_matches_explicit_computation(subject, tmp_path, n):
    an = Anatomy(subject, tmp_path, clusters=8)
    ids = [3, 14, 15, 92, 65][:n]
    want = expected_activity(an, writer_inputs(subject, ids))
    with an.recording(), torch.no_grad():
        subject.model(input_ids=torch.tensor([ids]))
        got = an.step[None].clone() if n == 1 else torch.cat([an.parts[k] for k in sorted(an.parts)], dim=1)
        codes = an.read_chunk(n)
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-6)
    # one byte per unit, within half a step (1/16 of a doubling) of the value
    ratio = decode(codes.numpy()) / want.numpy()
    assert np.all(np.abs(np.log2(ratio)) <= 1 / 16 + 1e-6)


def test_probe_is_off_outside_recording(subject, tmp_path):
    an = Anatomy(subject, tmp_path, clusters=8)
    with torch.no_grad():
        subject.model(input_ids=torch.tensor([[1, 2, 3]]))
    assert not an.parts and float(an.step.abs().sum()) == 0


def test_pool_bins_match_a_direct_computation():
    torch.manual_seed(0)
    m = torch.randn(37, 53)
    b = pool(m, 8, 10)
    bh, bw = b["bh"], b["bw"]
    for i in range(b["mean"].shape[0]):
        for j in range(b["mean"].shape[1]):
            cell = m[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            assert torch.isclose(b["mean"][i, j], cell.mean(), atol=1e-5)
            assert torch.isclose(b["rms"][i, j], cell.square().mean().sqrt(), atol=1e-5)
            assert b["absmax"][i, j] == cell.flatten()[cell.abs().argmax()]
    assert torch.allclose(b["col_rms"][0], m[:, :bw].square().mean().sqrt())
    exact = pool(m, 64, 64)
    assert exact["bh"] == exact["bw"] == 1 and torch.equal(exact["mean"], m)


def test_tiles_show_the_unit_weights(subject, tmp_path):
    an = Anatomy(subject, tmp_path, clusters=8)
    lm = an.layers[1]
    g = lm.offset + lm.n_heads + 2  # cluster 2 of layer 1
    tile = an.tile({"unit": g, "matrix": "writes", "h": 1024, "w": 1024})
    idx = (lm.assign == 2).nonzero().flatten()
    want = lm.down.weight.detach()[:, idx].T
    got = torch.from_numpy(np.frombuffer(base64.b64decode(tile["data"]), np.float16).reshape(tile["h"], tile["w"]).copy())
    assert tile["exact"] and (tile["rows"], tile["cols"]) == tuple(want.shape)
    assert torch.allclose(got.float(), want.float(), atol=1e-3)
    assert abs(tile["stats"]["norm"] - float(want.norm())) < 1e-3
    detail = an.unit_detail(g)
    assert detail["n_neurons"] == len(idx) and {m["key"] for m in detail["matrices"]} == {"writes", "gate_proj", "up_proj"}
    head = an.unit_detail(lm.offset + 3)
    assert [m["key"] for m in head["matrices"]] == ["writes", "query", "key", "value"]
    assert head["matrices"][2]["shape"] == [16, D]  # key head 1 of 2, shared by heads 2 and 3


def test_lesions_and_edited_units(subject, tmp_path):
    an = Anatomy(subject, tmp_path, clusters=8)
    stack = EditStack(subject)
    lm = an.layers[2]
    cluster, head = lm.offset + lm.n_heads + 1, lm.offset + 2
    stack.add(**an.lesion(cluster))
    stack.add(**an.lesion(head))
    stack.apply()
    idx = (lm.assign == 1).nonzero().flatten()
    assert lm.down.weight[:, idx].abs().max() == 0
    assert lm.down.weight[:, (lm.assign != 1)].abs().min() > 0
    assert lm.heads[0].weight[:, 32:48].abs().max() == 0
    assert an.edited_units(stack) == sorted([head, cluster])
    an.refresh()
    want = expected_activity(an, writer_inputs(subject, [5, 6, 7]))
    assert float(want[:, cluster].abs().max()) == 0 and float(want[:, head].abs().max()) == 0


def test_edges_find_a_planted_connection(subject, tmp_path):
    an = Anatomy(subject, tmp_path, clusters=8)
    score = an.composition()
    for lm in an.layers:  # only earlier units feed a unit; an MLP also reads its own layer's heads
        heads, clusters = slice(lm.offset, lm.offset + lm.n_heads), slice(lm.offset + lm.n_heads, lm.offset + lm.n_units)
        assert score[heads, lm.offset:].abs().sum() == 0
        assert score[clusters, lm.offset + lm.n_heads:].abs().sum() == 0
    assert 0.5 < float(score[score > 0].median()) < 2  # random weights compose at about chance
    src = an.layers[0].offset + an.layers[0].n_heads + 3  # layer 0, cluster 3
    f = an.write_factors()[src, 0]
    lm = an.layers[2]
    rows = (lm.assign == 5).nonzero().flatten()
    with torch.no_grad():  # make layer 2 cluster 5's gate rows read exactly that cluster's main output
        lm.readers[0][1].weight[rows] = 0.5 * f / f.norm()
    planted = an.composition()
    tgt = lm.offset + lm.n_heads + 5
    assert int(planted[tgt].argmax()) == src and planted[tgt, src] > 4
    assert Anatomy(subject, tmp_path, clusters=8).edges["up"].shape == (an.n_units, 16)  # cached edges load


def test_recordings_round_trip(subject, tmp_path):
    an = Anatomy(subject, tmp_path / "anatomy", clusters=8)
    store = RunStore(tmp_path / "data", 1)
    a = np.random.default_rng(0).integers(0, 255, (3, an.n_units), dtype=np.uint8)
    msg = an.add("run1", [0, 1, 2], a)
    assert msg["k"] == "anat" and msg["t"] == [0, 1, 2]
    an.add("run1", [3], a[:1])
    t, codes, skipped = an.read_run(store, "run1")
    assert t.tolist() == [0, 1, 2, 3] and skipped == 0
    assert an.flush(store, "run1", 1)
    t, codes, _ = an.read_run(store, "run1")
    assert t.tolist() == [0, 1, 2, 3] and np.array_equal(codes, np.concatenate([a, a[:1]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the gated delta net kernels need a GPU")
def test_linear_attention_heads(tmp_path):
    subject = make_subject(qwen35(), "cuda")
    an = Anatomy(subject, tmp_path, clusters=8)
    assert [lm.n_heads for lm in an.layers] == [4, 4, 4, 4]
    assert an.layers[0].heads[0] is subject.layers[0].linear_attn.out_proj
    keys = [m["key"] for m in an.unit_detail(an.layers[0].offset + 3)["matrices"]]
    assert keys == ["writes", "query", "key", "value", "gate"]
    keys = [m["key"] for m in an.unit_detail(an.layers[3].offset + 1)["matrices"]]
    assert keys == ["writes", "query", "gate", "key", "value"]
    ids = [3, 14, 15, 92, 65]
    want = expected_activity(an, writer_inputs(subject, ids))
    with an.recording(), torch.no_grad():
        subject.model(input_ids=torch.tensor([ids], device="cuda"))
        got = torch.cat([an.parts[k] for k in sorted(an.parts)], dim=1)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-5)
    stack = EditStack(subject)
    stack.add("scale", "0", ["head"], index=3, factor=0.0)  # a linear-attention head
    stack.apply()
    assert subject.layers[0].linear_attn.out_proj.weight[:, 48:64].abs().max() == 0
