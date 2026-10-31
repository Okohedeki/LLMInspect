"""Turning a GGUF file into a Hugging Face checkpoint one tensor at a time.

transformers can load a GGUF, but it unpacks every tensor in float32 before anything is saved, so a
20B model needs well over 64 GB of RAM. Here the model is built on the meta device (its parameter names
and shapes, no memory), each GGUF tensor is unpacked, put in the model's layout, cast and written to a
shard on its own, and memory stays near one tensor's worth.

Names come from transformers' own GGUF table, and its per-architecture fixes (Llama's q/k order,
Gemma's norm offset, ...) are applied to each tensor. Fused mixture-of-experts weights are assembled
here, since transformers misses some of them (gpt-oss's gate_up_proj and expert biases):
  linear      gate_up_proj [experts, 2 · width, d] = ffn_gate_exps then ffn_up_exps; down_proj as stored
  transposed  gate_up_proj [experts, d, 2 · width] with gate and up interleaved, down_proj transposed,
              biases the same way (gpt-oss)
A weight with no source in the file stops the conversion: a model with some weights left at random
would still run and answer, just wrongly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from .experts import Layout

SHARD_BYTES = 2 * 1024 ** 3
FUSED = re.compile(r"(?P<mod>.*\.layers\.(?P<bid>\d+)\..*experts)\.(?P<name>gate_up_proj|down_proj)(?P<bias>_bias)?$")


def _meta_model(config):
    from transformers import AutoModelForCausalLM
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _layouts(model, d: int) -> dict[str, Layout | None]:
    """Each fused experts module's layout, by module name; None where it isn't one the reader knows."""
    out = {}
    for name, mod in model.named_modules():
        if name.endswith("experts") and mod._parameters.get("down_proj") is not None:
            try:
                out[name] = Layout(mod, d)
            except ValueError:
                out[name] = None
    return out


def _processor(arch: str, gguf_config: dict):
    """transformers' fixes for this architecture's tensors (Llama's q/k order, Gemma's norm offset, ...)."""
    from transformers.modeling_gguf_pytorch_utils import TENSOR_PROCESSORS, TensorProcessor
    return TENSOR_PROCESSORS.get(arch, TensorProcessor)(config=gguf_config)


def recipes(model, config, gguf_names: set[str], arch: str, processor) -> tuple[dict, list[str]]:
    """{hf key: (kind, [gguf names])} for every weight of the model, and the keys with no source. Names are
    looked up by the GGUF's architecture, not the model's type: a Mixtral GGUF says "llama"."""
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    to_hf = get_gguf_hf_weights_map(model, processor, model_type=arch)
    from_gguf: dict[str, list[str]] = {}
    for g, h in to_hf.items():
        from_gguf.setdefault(h, []).append(g)
    layouts = _layouts(model, config.hidden_size)
    out, missing = {}, []
    tied = getattr(config.get_text_config(), "tie_word_embeddings", False)
    for key in model.state_dict():
        if m := FUSED.fullmatch(key):
            layout = layouts.get(m["mod"])
            suffix = ".bias" if m["bias"] else ".weight"
            parts = ["gate", "up"] if m["name"] == "gate_up_proj" else ["down"]
            names = [f"blk.{m['bid']}.ffn_{p}_exps{suffix}" for p in parts]
            if layout is not None and all(n in gguf_names for n in names):
                kind = ("t-" if layout.transposed else "") + m["name"] + (m["bias"] or "")
                out[key] = (kind, names)
                continue
        names = [n for g in from_gguf.get(key, []) for n in (g, g + ".weight") if n in gguf_names][:1]
        if names:
            out[key] = ("plain", names)
        elif not (tied and key.endswith("lm_head.weight")):
            missing.append(key)
    return out, missing


def assemble(kind: str, parts: list[torch.Tensor]) -> torch.Tensor:
    """A fused expert weight from its GGUF parts (each [experts, out, in] or, for biases, [experts, out])."""
    if kind in ("gate_up_proj", "gate_up_proj_bias"):
        return torch.cat(parts, dim=1)
    if kind == "t-gate_up_proj":  # [experts, d, 2 · width], gate at even columns
        return torch.stack([p.transpose(1, 2) for p in parts], dim=-1).flatten(2)
    if kind == "t-gate_up_proj_bias":
        return torch.stack(parts, dim=-1).flatten(1)
    if kind == "t-down_proj":
        return parts[0].transpose(1, 2)
    return parts[0]  # down_proj and its bias, in either layout


class ShardWriter:
    """Safetensors shards of about SHARD_BYTES, and the index that names each tensor's shard."""

    def __init__(self, folder: Path):
        self.folder, self.shard, self.size, self.files, self.where = folder, {}, 0, [], {}

    def add(self, key: str, t: torch.Tensor):
        self.shard[key] = t.contiguous()
        self.size += t.numel() * t.element_size()
        if self.size >= SHARD_BYTES:
            self.flush()

    def flush(self):
        from safetensors.torch import save_file
        if not self.shard:
            return
        name = f"model-{len(self.files) + 1:05d}.safetensors"
        save_file(self.shard, str(self.folder / name), metadata={"format": "pt"})
        self.files.append(name)
        self.where.update({k: name for k in self.shard})
        self.shard, self.size = {}, 0

    def close(self):
        self.flush()
        total = len(self.files)
        for i, name in enumerate(self.files):  # the usual "-of-N" names, now that N is known
            final = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            (self.folder / name).replace(self.folder / final)
            self.where = {k: (final if v == name else v) for k, v in self.where.items()}
        (self.folder / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": dict(sorted(self.where.items()))}, indent=2))


def convert(blob: Path, folder: Path, config, dtype=torch.bfloat16, log=print) -> dict:
    """Write the GGUF's weights into folder as safetensors shards for config's model. Returns counts."""
    import gguf
    from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint

    reader = gguf.GGUFReader(str(blob))
    arch = str(reader.fields["general.architecture"].contents())
    tensors = {t.name: t for t in reader.tensors}
    processor = _processor(arch, load_gguf_checkpoint(str(blob))["config"])  # metadata only, no tensors
    model = _meta_model(config)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    plan, missing = recipes(model, config, set(tensors), arch, processor)
    if missing:
        raise ValueError(f"{blob.name} has no weights for {len(missing)} of the model's tensors "
                         f"(e.g. {', '.join(missing[:4])}); converting it would leave them random")
    del model
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("model-*.safetensors"):  # left by a conversion that was cut off
        old.unlink()
    writer = ShardWriter(folder)
    used = set()
    for i, (key, (kind, names)) in enumerate(plan.items()):
        parts = []
        for name in names:
            t = tensors[name]
            w = gguf.dequantize(t.data, t.tensor_type)
            if kind == "plain":
                aside = {"tensors": {}}  # where a processor puts a tensor it renames itself (GPT-2's output)
                r = processor.process(weights=w, name=name, parsed_parameters=aside)
                w = r.weights if r.name is not None else next(iter(aside["tensors"].values())).numpy()
            w = np.ascontiguousarray(w)
            parts.append(torch.from_numpy(w if w.flags.writeable else w.copy()))  # F32 tensors come straight off the file
            used.add(name)
        out = (parts[0] if kind == "plain" else assemble(kind, parts)).to(dtype)
        if tuple(out.shape) != shapes[key]:
            if out.numel() != int(np.prod(shapes[key])):
                raise ValueError(f"{names[0]} has shape {list(out.shape)}, but {key} needs {list(shapes[key])}")
            out = out.reshape(shapes[key])  # a GGUF keeps some small tensors flat
        writer.add(key, out)
        del parts, out
        if i % 50 == 0:
            log(f"[llminspect]   {i + 1}/{len(plan)} tensors")
    writer.close()
    return {"tensors": len(plan), "unused": sorted(set(tensors) - used), "shards": len(writer.files)}
