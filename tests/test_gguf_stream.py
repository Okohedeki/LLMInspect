"""Converting a GGUF one tensor at a time: tiny random models are written as GGUF files in llama.cpp's
layout (experts stacked [experts, out, in], gate and up apart, Llama's q/k rows reordered), converted
back, and every weight must come back exactly."""

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import modeling_gguf_pytorch_utils as tgguf

from llminspect.recorder import gguf_stream as gs
from llminspect.recorder.experts import Layout

gguf = pytest.importorskip("gguf")

D, EXPERTS, WIDTH = 64, 4, 32
ARCHS = {  # model type: (GGUF architecture, config)
    "llama": ("llama", dict(intermediate_size=96)),
    "qwen2_moe": ("qwen2moe", dict(num_experts=EXPERTS, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
                                   shared_expert_intermediate_size=48, decoder_sparse_step=1, intermediate_size=96)),
    "mixtral": ("llama", dict(num_local_experts=EXPERTS, num_experts_per_tok=2, intermediate_size=WIDTH)),
    "gpt_oss": ("gpt-oss", dict(num_local_experts=EXPERTS, num_experts_per_tok=2, intermediate_size=WIDTH, head_dim=16,
                                sliding_window=32, layer_types=["sliding_attention", "full_attention"])),
}


def tiny(arch: str):
    torch.manual_seed(0)
    cfg = AutoConfig.for_model(arch, hidden_size=D, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               vocab_size=300, max_position_embeddings=256, tie_word_embeddings=False, **ARCHS[arch][1])
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias") or "norm" in name:  # not their zero/one defaults, so a mix-up shows
                p.normal_(0, 0.5)
    return cfg, model


def permute(w: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's reordering of Llama's q/k rows, which transformers' Llama processor undoes."""
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def write_gguf(path, cfg, model, arch: str):
    """The model's weights in a GGUF laid out as llama.cpp writes them."""
    processor = gs._processor(arch, {})
    to_gguf = {h: g for g, h in tgguf.get_gguf_hf_weights_map(model, processor, model_type=arch).items()}
    layouts = gs._layouts(model, D)
    out = {}
    for key, t in model.state_dict().items():
        w = t.detach().float().numpy()
        if m := gs.FUSED.fullmatch(key):
            lay, bias = layouts[m["mod"]], bool(m["bias"])
            suffix = ".bias" if bias else ".weight"
            blk = f"blk.{m['bid']}.ffn_"
            if m["name"] == "down_proj":
                out[f"{blk}down_exps{suffix}"] = w if bias or not lay.transposed else w.transpose(0, 2, 1)
            elif lay.transposed:
                for i, part in enumerate(("gate", "up")):
                    half = w[..., i::2]
                    out[f"{blk}{part}_exps{suffix}"] = half if bias else half.transpose(0, 2, 1)
            else:
                out[f"{blk}gate_exps{suffix}"], out[f"{blk}up_exps{suffix}"] = np.split(w, 2, axis=1)
            continue
        name = to_gguf[key]
        if arch == "llama" and (".attn_q." in name or ".attn_k." in name) and name.endswith(".weight"):
            w = permute(w, cfg.num_attention_heads if ".attn_q." in name else cfg.num_key_value_heads)
        out[name if name.endswith((".weight", ".bias")) else name + ".weight"] = w
    writer = gguf.GGUFWriter(str(path), arch)
    for name, w in out.items():
        writer.add_tensor(name, np.ascontiguousarray(w, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return out


def load_back(folder, cfg, model):
    """The converted folder as a model; config files are written beside the shards, as _dequantize does."""
    cfg.save_pretrained(folder)
    model.generation_config.save_pretrained(folder)
    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)


@pytest.fixture
def gguf_config(monkeypatch):
    """The GGUF metadata transformers parses for its processors; these files carry only the architecture."""
    monkeypatch.setattr(tgguf, "load_gguf_checkpoint",
                        lambda path, **kw: {"config": {"num_attention_heads": 4, "num_key_value_heads": 2}})


@pytest.mark.parametrize("arch", list(ARCHS))
def test_round_trip(arch, tmp_path, gguf_config, monkeypatch):
    monkeypatch.setattr(gs, "SHARD_BYTES", 200_000)  # several shards, so the index is exercised
    cfg, model = tiny(arch)
    write_gguf(tmp_path / "m.gguf", cfg, model, ARCHS[arch][0])
    stats = gs.convert(tmp_path / "m.gguf", tmp_path / "hf", cfg, dtype=torch.float32, log=lambda m: None)
    assert stats["unused"] == [] and stats["shards"] > 1
    back = load_back(tmp_path / "hf", cfg, model)
    want, got = model.state_dict(), back.state_dict()
    assert set(got) == set(want)
    for key in want:
        assert torch.equal(got[key], want[key]), key


def test_gpt_oss_layout_is_read_by_the_tap(tmp_path, gguf_config):
    """The converted gpt-oss experts are in the layout the expert reader expects, with W == d too."""
    cfg, model = tiny("gpt_oss")
    write_gguf(tmp_path / "m.gguf", cfg, model, "gpt-oss")
    gs.convert(tmp_path / "m.gguf", tmp_path / "hf", cfg, dtype=torch.float32, log=lambda m: None)
    back = load_back(tmp_path / "hf", cfg, model)
    assert Layout(back.model.layers[0].mlp.experts, D).transposed
    x = torch.randn(1, 5, D)
    with torch.no_grad():
        assert torch.allclose(back.model.layers[0].mlp(x)[0], model.model.layers[0].mlp(x)[0])


def test_missing_weights_stop_the_conversion(tmp_path, gguf_config):
    cfg, model = tiny("gpt_oss")
    tensors = write_gguf(tmp_path / "full.gguf", cfg, model, "gpt-oss")
    writer = gguf.GGUFWriter(str(tmp_path / "m.gguf"), "gpt-oss")
    for name, w in tensors.items():
        if name != "blk.1.ffn_up_exps.bias":
            writer.add_tensor(name, np.ascontiguousarray(w, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="layers.1.mlp.experts.gate_up_proj_bias"):
        gs.convert(tmp_path / "m.gguf", tmp_path / "hf", cfg, log=lambda m: None)
    assert not list((tmp_path / "hf").glob("*.safetensors"))


def test_assemble_interleaves_gate_and_up():
    gate, up = torch.arange(12.).reshape(1, 3, 4), -torch.arange(12.).reshape(1, 3, 4)  # [experts, width, d]
    out = gs.assemble("t-gate_up_proj", [gate, up])
    assert out.shape == (1, 4, 6)
    assert torch.equal(out[0, :, 0::2], gate[0].T) and torch.equal(out[0, :, 1::2], up[0].T)
    assert torch.equal(gs.assemble("gate_up_proj", [gate, up]), torch.cat([gate, up], dim=1))
