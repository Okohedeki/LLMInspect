"""Telling a model's true base apart from unrelated models of the same architecture, on tiny random models."""

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from llminspect import fingerprint as fp

CFG = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
           vocab_size=512, max_position_embeddings=128)


def save(folder, seed, dtype=torch.bfloat16, edit_layer=None):
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**CFG)).to(dtype)
    if edit_layer is not None:  # a light fine-tune of one layer
        with torch.no_grad():
            w = model.model.layers[edit_layer].mlp.down_proj.weight
            w.add_(torch.randn_like(w) * w.float().std().to(dtype) * 0.3)
    model.save_pretrained(folder)
    return folder


def test_a_fine_tune_matches_its_base_and_shows_where_it_moved(tmp_path):
    base = save(tmp_path / "base", 0)
    other = save(tmp_path / "other", 1)
    tuned = save(tmp_path / "tuned", 0, edit_layer=1)
    (tuned / "README.md").write_text("---\nbase_model:\n- org/other-model\nlicense: mit\n---\n# card\n")
    r = fp.identify(tuned, [("base", base), ("other", other)])
    assert [m["name"] for m in r["matches"]] == ["base", "other"]
    best, unrelated = r["matches"]
    assert best["median"] > 0.99 and abs(unrelated["median"]) < 0.3
    assert best["layers"]["1"] < 0.99 and best["layers"]["0"] == 1.0 and best["layers"]["2"] == 1.0
    assert r["reading"] == "derived from it (weights mostly kept)"
    assert r["claimed_base"] == ["org/other-model"]  # the card's claim, reported beside the evidence


def test_an_exact_copy_and_other_architectures(tmp_path):
    base = save(tmp_path / "base", 0)
    copy = save(tmp_path / "copy", 0)
    torch.manual_seed(0)
    wider = AutoModelForCausalLM.from_config(LlamaConfig(**(CFG | {"hidden_size": 96})))
    wider.save_pretrained(tmp_path / "wider")
    r = fp.identify(copy, [("base", base), ("wider", tmp_path / "wider")])
    assert [m["name"] for m in r["matches"]] == ["base"] and r["reading"] == "the same weights"


def test_float32_and_bf16_values_are_read(tmp_path):
    a = save(tmp_path / "a", 0, dtype=torch.float32)
    sig = fp.signature(a)
    model = AutoModelForCausalLM.from_pretrained(a)
    w = model.model.embed_tokens.weight.flatten()
    step = max(1, w.numel() // fp.SAMPLES)
    assert sig["embed_tokens.weight"][1][:5] == [float(x) for x in w[:5 * step:step]]


def test_wrapper_prefixes_line_up(tmp_path):
    """A multimodal model's text layers ('model.language_model.layers...') match a text-only save ('model.layers...')."""
    from safetensors.torch import load_file, save_file
    base = save(tmp_path / "base", 0)
    wrapped = tmp_path / "wrapped"
    wrapped.mkdir()
    sd = load_file(str(base / "model.safetensors"))
    save_file({k.replace("model.", "model.language_model.", 1): v for k, v in sd.items()}, str(wrapped / "model.safetensors"))
    r = fp.identify(wrapped, [("base", base)])
    assert r["reading"] == "the same weights"
