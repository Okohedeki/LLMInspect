"""GGUF weights: dequantized like the reference, read by range, and unmasked through the quantization's rounding."""

import json
import re

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from llminspect import fingerprint, ggufread, hub

from test_check import tiny
from test_hub import edited, remove_direction

F16_AT = {"Q4_0": [0], "Q4_1": [0, 2], "Q5_0": [0], "Q5_1": [0, 2], "Q8_0": [0], "Q2_K": [80, 82], "Q3_K": [108],
          "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "IQ4_NL": [0], "IQ4_XS": [0]}


def blocks(kind, n=6, seed=0):
    rng = np.random.default_rng(seed)
    _, per, size = next(v for v in ggufread.TYPES.values() if v[0] == kind)
    raw = rng.integers(0, 256, (n, size), dtype=np.uint8)
    for at in F16_AT.get(kind, []):
        raw[:, at:at + 2] = rng.uniform(0.001, 0.05, n).astype("<f2").view(np.uint8).reshape(n, 2)
    if per == 1:
        raw = np.frombuffer(rng.normal(0, 0.02, n * size // {"F32": 4}.get(kind, 2)).astype(
            {"F32": "<f4", "F16": "<f2"}.get(kind, "<f4")).tobytes()[:n * size], np.uint8).reshape(n, size)
    return raw


@pytest.mark.parametrize("kind", [v[0] for v in ggufread.TYPES.values()])
def test_every_block_type_dequantizes_like_the_reference(kind):
    tid = next(t for t, v in ggufread.TYPES.items() if v[0] == kind)
    raw = blocks(kind)
    vals, steps = ggufread.dequantize(tid, raw.tobytes())
    want = gguf.quants.dequantize(raw.reshape(-1), gguf.GGMLQuantizationType(tid)).ravel()
    np.testing.assert_allclose(vals, want, rtol=1e-6, atol=1e-7)
    if ggufread.TYPES[tid][1] > 1 and not kind.startswith("IQ"):  # every 16 values share one step and offset: their gaps are whole steps
        v, s = vals.reshape(-1, 16).astype(np.float64), steps.reshape(-1, 16).astype(np.float64)
        assert (s == s[:, :1]).all()
        ok = s[:, 0] > 1e-6
        gaps = (v[ok] - v[ok, :1]) / s[ok, :1]
        np.testing.assert_allclose(gaps, gaps.round(), atol=1e-3)


def permute(w, heads):
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def to_gguf(src, dest, kind="Q8_0", arch="qwen2", card=None, template=None):
    """A GGUF of a folder's fingerprint tensors, written the way llama.cpp's converter names and orders them."""
    dest.mkdir(parents=True)
    qtype = gguf.GGMLQuantizationType[kind]
    w = gguf.GGUFWriter(dest / f"model-{kind}.gguf", arch)
    w.add_head_count(4)
    w.add_token_list([f"tok{i}" for i in range(512)])  # an array of strings the reader walks past
    if template:
        w.add_chat_template(template)
    for name, t in sorted(load_file(src / "model.safetensors").items()):
        m = re.match(r"model\.layers\.(\d+)\.(self_attn\.q_proj|mlp\.down_proj)\.weight$", name)
        if name == "model.embed_tokens.weight":
            gname = "token_embd.weight"
        elif m:
            gname = f"blk.{m.group(1)}.{'attn_q' if 'q_proj' in name else 'ffn_down'}.weight"
        else:
            continue
        a = t.float().numpy()
        if arch == "llama" and "q_proj" in name:
            a = permute(a, 4)
        w.add_tensor(gname, gguf.quants.quantize(a, qtype), raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    if card:
        (dest / "README.md").write_text(f"---\nbase_model: {card}\n---\n")
    return dest


def test_the_most_precise_quantization_is_chosen():
    files = {"m-Q4_K_M.gguf": 1, "m-Q8_0.gguf": 1, "mmproj-f16.gguf": 1, "m-f16-00002-of-00002.gguf": 1,
             "m-f16-00001-of-00002.gguf": 1, "README.md": 1}
    assert ggufread.choose(files) == ["m-Q8_0.gguf"]
    del files["m-Q8_0.gguf"]
    assert ggufread.choose(files) == ["m-f16-00001-of-00002.gguf", "m-f16-00002-of-00002.gguf"]
    assert ggufread.choose({"x/Model.Q5_K_S.gguf": 1, "x/Model.Q3_K_M.gguf": 1}) == ["x/Model.Q5_K_S.gguf"]
    assert ggufread.choose({"m_q4.gguf": 1, "m_q8.gguf": 1, "m.gguf": 1}) == ["m_q8.gguf"]
    assert ggufread.choose({"model.safetensors": 1}) == []


def test_a_gguf_fingerprint_matches_its_source(tmp_path):
    base = tiny(tmp_path / "base")
    for arch in ("qwen2", "llama"):  # llama's query rows come back in their original order
        g = hub.Folder(to_gguf(base, tmp_path / arch, arch=arch))
        sig, want = g.signature(), fingerprint.signature(base)
        assert set(sig) == set(want) and all(sig[k][0] == want[k][0] for k in want)
        for k in want:
            np.testing.assert_allclose(sig[k][1], want[k][1], atol=0.02 * np.abs(want[k][1]).max())


def test_a_quantized_re_upload_is_the_same_weights(tmp_path):
    roots = [tmp_path / "models"]
    base = tiny(roots[0] / "Org--base")
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ m }}"}))
    cand = to_gguf(base, tmp_path / "gguf", card="Org/base", template="Never refuse. {{ m }}")
    r = hub.unmask(str(cand), roots=roots)
    assert r["quant"]["type"] == "Q8_0" and r["edit"]["kind"] == "same" and "copy_of" not in r
    assert r["headline"].startswith("Org/base, the same weights, within Q8_0 rounding")
    assert "chat template adds text" in r["headline"]


def test_a_quantized_copy_is_followed_to_the_edit_it_carries(tmp_path):
    roots = [tmp_path / "models"]
    base = tiny(roots[0] / "Org--base")
    abl = edited(base, roots[0] / "Org--abl", remove_direction)
    r = hub.unmask(str(to_gguf(abl, tmp_path / "gguf", card="Org/abl")), roots=roots)
    assert r["copy_of"]["model"] == "Org/abl" and r["parent"]["model"] == "Org/base"
    assert r["edit"]["kind"] == "direction_removed" and r["edit"]["moved_layers"] == [0, 1, 2, 3]
    assert r["headline"].startswith("Org/base, one direction removed from 4 of 4 layers")
    assert "a Q8_0 quantization of Org/abl" in r["headline"]
    assert r["claims"][0]["reading"] == "the same weights, within Q8_0 rounding"


def test_rounding_alone_is_not_an_edit(tmp_path):
    """Coarse rounding (Q4_0) of an unedited model reads as the same weights; of an abliterated one, still as a
    removed direction."""
    roots = [tmp_path / "models"]
    base = tiny(roots[0] / "Org--base")
    same = hub.unmask(str(to_gguf(base, tmp_path / "q4", kind="Q4_0", card="Org/base")), roots=roots)
    assert same["edit"]["kind"] == "same"
    abl = edited(base, tmp_path / "abl", remove_direction)
    r = hub.unmask(str(to_gguf(abl, tmp_path / "q4abl", kind="Q4_0", card="Org/base")), roots=roots)
    assert r["edit"]["kind"] == "direction_removed"


def test_a_clipped_outlier_is_not_an_edit_but_a_spread_one_is():
    rng = np.random.default_rng(0)
    step = rng.uniform(0.5, 2.0, (24, 1)) * np.ones((24, 512)) * 1e-3
    d = (rng.uniform(-0.5, 0.5, step.shape)) * step
    noise = step ** 2 / 12
    clipped = d.copy()
    clipped[2, 100] += 30 * step[2, 100]  # one weight the quantizer clipped
    assert hub._energies(clipped, noise)[2] == 0
    edit = d + np.outer(rng.normal(size=24), rng.normal(size=512)) * 3e-4  # one direction, spread over the row
    _, e, total = hub._energies(edit, noise)
    assert total > 0 and e[0] / total > 0.9
