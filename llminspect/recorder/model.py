"""Load the subject model and expose the parts the recorder hooks into.

`ollama:<tag>` loads weights straight from the local Ollama store: the GGUF blob is
dequantized once into a bf16 Hugging Face checkpoint under ~/.cache/llminspect. The
quantization noise stays in the weights, so those recordings are exploratory and
labelled as such; causal experiments should use original bf16 checkpoints.
"""

from __future__ import annotations

import contextlib
import glob
import json
import os
import re
import struct
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

CACHE = Path(os.environ.get("LLMINSPECT_CACHE", Path.home() / ".cache" / "llminspect"))
STOP_TOKENS = ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>",
               "<end_of_turn>", "<|return|>", "<|call|>"]


def ollama_roots() -> list[Path]:
    roots = [Path(os.environ["OLLAMA_MODELS"])] if os.environ.get("OLLAMA_MODELS") else []
    roots += [Path(p) for p in glob.glob("/mnt/c/Users/*/.ollama/models")]
    roots.append(Path.home() / ".ollama" / "models")
    return [r for r in roots if r.exists()]


def resolve_ollama(tag: str) -> tuple[Path, dict]:
    """Return (GGUF blob path, Ollama config) for a tag like 'llama3.1:8b'."""
    name, _, version = tag.partition(":")
    name = name if "/" in name else f"library/{name}"
    for root in ollama_roots():
        manifest = root / "manifests" / "registry.ollama.ai" / name / (version or "latest")
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        blob = next((root / "blobs" / layer["digest"].replace(":", "-") for layer in data.get("layers", [])
                     if layer["mediaType"] == "application/vnd.ollama.image.model"), None)
        config = {}
        if data.get("config", {}).get("digest"):
            config_path = root / "blobs" / data["config"]["digest"].replace(":", "-")
            if config_path.exists():
                config = json.loads(config_path.read_text())
        if blob:
            return blob, config
    raise FileNotFoundError(f"Ollama model {tag!r} not found under {ollama_roots()}")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def display_name(spec: str) -> str:
    """'org/Name' for Hugging Face ids and cache snapshots, the tag for Ollama models."""
    if spec.startswith("ollama:"):
        return spec.split(":", 1)[1]
    if spec.lower().endswith(".gguf"):
        return Path(spec).stem
    cached = re.search(r"models--([^/\\]+)--([^/\\]+)", spec)
    if cached:
        return f"{cached.group(1)}/{cached.group(2)}"
    if not Path(spec).exists():
        return spec
    folder = Path(spec.rstrip("/\\")).name  # either separator, so a Windows path names its folder too
    org = re.fullmatch(r"([A-Za-z0-9][\w.-]*?)--(.+)", folder)  # a copied cache folder, "Org--Name"
    return f"{org.group(1)}/{org.group(2)}" if org else folder


def prepare(spec: str, dtype=torch.bfloat16) -> tuple[str, dict]:
    """Make sure the model exists as a Hugging Face checkpoint; return (path, provenance). An Ollama
    tag ("ollama:llama3.1:8b") or a GGUF file path is dequantized once into LLMInspect's model cache;
    the GGUF itself is only read."""
    if spec.lower().endswith(".gguf"):
        blob = Path(spec).expanduser()
        if not blob.is_file():
            raise FileNotFoundError(f"no GGUF file at {spec}")
        quant = re.search(r"(?i)\b(q\d_[a-z0-9_]+|q\d_\d|f16|bf16|f32)\b", blob.stem)
        return _dequantize(spec, f"gguf-{_slug(blob.stem)}", blob, dtype, {
            "source": "gguf", "quantization": quant.group(1).upper() if quant else None, "family": None,
            "note": "Weights dequantized from a GGUF file; quantization noise remains. "
                    "Use original bf16 weights for causal claims."})
    if not spec.startswith("ollama:"):
        source = "local" if Path(spec).exists() else "huggingface"
        return spec, {"spec": spec, "source": source, "quantization": None, "exploratory": False}
    tag = spec.split(":", 1)[1]
    local = CACHE / "models" / f"ollama-{_slug(tag)}"
    if (local / "llminspect_provenance.json").exists():
        return str(local), json.loads((local / "llminspect_provenance.json").read_text())
    blob, config = resolve_ollama(tag)
    return _dequantize(spec, f"ollama-{_slug(tag)}", blob, dtype, {
        "source": "ollama", "quantization": config.get("file_type"), "family": config.get("model_family"),
        "note": "Weights dequantized from an Ollama GGUF; quantization noise remains. "
                "Use original bf16 weights for causal claims."})


@contextlib.contextmanager
def _gguf_reader_fixes():
    """transformers 5 reads some gpt-oss GGUF fields (gpt-oss.rope.scaling.*) from parts[0], the field's
    name, not its value, and fails on them. While this is on, its reader puts each such field's value
    first, as a plain Python value."""
    import gguf

    class Reader(gguf.GGUFReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for key, field in list(self.fields.items()):
                if key.startswith("gpt-oss.") and field.data:
                    self.fields[key] = field._replace(parts=[field.contents()] + list(field.parts),  # a plain value
                                                      data=[i + 1 for i in field.data])

    before = gguf.GGUFReader
    gguf.GGUFReader = Reader  # transformers imports it from gguf when it reads a file
    try:
        yield
    finally:
        gguf.GGUFReader = before


def _gguf_config(gguf_dir: Path, name: str, blob: Path, provenance: dict):
    """The model's config for a GGUF. A config.json beside the file (a Hugging Face GGUF repo keeps the
    original model's) wins when it is the same kind of model: transformers rebuilds the config from GGUF
    metadata and can drop settings. Otherwise the rebuilt one, with the attention window put back when the
    GGUF has one that transformers didn't carry over (gpt-oss)."""
    rebuilt = AutoConfig.from_pretrained(gguf_dir, gguf_file=name)
    side = blob.parent / "config.json"
    if side.is_file():
        try:
            original = AutoConfig.from_pretrained(blob.parent)
        except (OSError, ValueError, KeyError):
            original = None
        if original is not None and original.model_type == rebuilt.model_type:
            for key in ("quantization_config",):
                if hasattr(original, key):
                    delattr(original, key)
            provenance["config"] = str(side)
            return original
    import gguf
    fields = gguf.GGUFReader(str(blob)).fields
    arch = fields.get("general.architecture")
    window = fields.get(f"{arch.contents()}.attention.sliding_window") if arch is not None else None
    if window is not None and hasattr(rebuilt, "sliding_window") and not rebuilt.sliding_window:
        rebuilt.sliding_window = int(window.contents())
        if hasattr(rebuilt, "layer_types") and rebuilt.model_type == "gpt_oss":  # it alternates, sliding first
            rebuilt.layer_types = ["sliding_attention" if i % 2 == 0 else "full_attention"
                                   for i in range(rebuilt.num_hidden_layers)]
    return rebuilt


def _dequantize(spec: str, folder: str, blob: Path, dtype, info: dict) -> tuple[str, dict]:
    """A GGUF's weights as a Hugging Face checkpoint in LLMInspect's model cache (made once). The tensors
    are unpacked one at a time (gguf_stream), so this needs about one tensor's worth of RAM, not the model's."""
    from . import gguf_stream

    local = CACHE / "models" / folder
    prov_file = local / "llminspect_provenance.json"
    if prov_file.exists():
        return str(local), json.loads(prov_file.read_text())
    provenance = {"spec": spec, "blob": str(blob), "exploratory": True, **info}
    gguf_dir = CACHE / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    link = gguf_dir / f"{folder}.gguf"
    if not link.exists():
        link.symlink_to(blob)
    print(f"[llminspect] dequantizing {blob} -> {local} (one time)", flush=True)
    try:
        with _gguf_reader_fixes():
            tokenizer = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=link.name)
            config = _gguf_config(gguf_dir, link.name, blob, provenance)
            if hasattr(config, "quantization_config"):
                delattr(config, "quantization_config")
            stats = gguf_stream.convert(blob, local, config, dtype)
    except (TypeError, KeyError) as exc:  # transformers' GGUF reader fails deep inside on what it can't parse
        raise ValueError(f"transformers can't read {blob.name} ({type(exc).__name__}: {exc}). Some GGUFs, Ollama's own "
                         "builds of newer models among them, use metadata it doesn't know yet; a GGUF of the same "
                         "model from Hugging Face, or the original checkpoint, usually works.") from exc
    config.torch_dtype = dtype
    config.save_pretrained(local)
    side = blob.parent / "generation_config.json"
    (GenerationConfig.from_pretrained(blob.parent) if side.is_file() else GenerationConfig.from_model_config(config)).save_pretrained(local)
    tokenizer.save_pretrained(local)
    if stats["unused"]:
        provenance["unused_gguf_tensors"] = stats["unused"]
    prov_file.write_text(json.dumps(provenance, indent=2))
    return str(local), provenance


def _first_attr(obj, paths: list[str]):
    for path in paths:
        cur = obj
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            return cur
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} found on {type(obj).__name__}")


# Parts of a checkpoint that loading the text model leaves on disk: vision and audio towers, their
# projectors, and multi-token-prediction heads (AutoModelForCausalLM loads only the language model).
UNLOADED_PARTS = {"visual", "vision_tower", "vision_model", "audio_tower", "audio", "mtp",
                  "multi_modal_projector", "mm_projector"}
_FLOAT_BYTES = {"F64": 8, "F32": 4, "BF16": 2, "F16": 2}


def weight_bytes(path, load_bytes_per_float: int = 2) -> tuple[int, int]:
    """(bytes of safetensors on disk, bytes the text model takes once loaded) for a checkpoint folder.
    Read from the safetensors headers: unloaded parts are skipped and float weights are counted at
    the size they load at (bf16 by default). Falls back to file size where a header can't be read."""
    total = loaded = 0
    for f in Path(path).glob("*.safetensors"):
        size = f.stat().st_size
        total += size
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                if n > min(size - 8, 1 << 28):  # not a safetensors header (a corrupt or foreign file)
                    raise ValueError("bad header length")
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            loaded += size
            continue
        for name, meta in header.items():
            if name == "__metadata__" or UNLOADED_PARTS.intersection(name.split(".")[:3]):
                continue
            a, b = meta["data_offsets"]
            item = _FLOAT_BYTES.get(meta.get("dtype"))
            loaded += (b - a) // item * load_bytes_per_float if item else b - a
    return total, loaded


def _check_vram(path: str, device: str, headroom_gb: float = 2.5) -> None:
    """Refuse to load when the weights won't fit. On Windows the driver would otherwise spill
    into system memory and every forward pass would crawl."""
    weights = weight_bytes(path)[1] if Path(path).is_dir() else 0
    if not weights or not device.startswith("cuda") or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    need = weights + headroom_gb * 1024 ** 3
    if free < need:
        raise RuntimeError(
            f"not enough free GPU memory: {free / 1024 ** 3:.1f} GB free, about {need / 1024 ** 3:.1f} GB needed. "
            "Close other GPU apps (image generators, local LLM servers) or choose a smaller model.")


class Subject:
    """The model under study plus the handles the recorder needs."""

    def __init__(self, spec: str, dtype=torch.bfloat16, device: str = "cuda", name: str | None = None):
        path, self.provenance = prepare(spec, dtype)
        _check_vram(path, device)
        self.spec = spec
        # LLMINSPECT_MODEL_NAME names the model LLMINSPECT_MODEL points at, not one switched to later.
        env_name = os.environ.get("LLMINSPECT_MODEL_NAME") if spec == os.environ.get("LLMINSPECT_MODEL") else None
        self.name = name or env_name or display_name(spec)
        self.slug = _slug(self.name.split("/")[-1])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, device_map=device, attn_implementation="sdpa")
        except ValueError as exc:  # an architecture without SDPA attention (gpt-oss, with its attention sinks)
            if "scaled_dot_product_attention" not in str(exc):
                raise
            self.model = AutoModelForCausalLM.from_pretrained(
                path, dtype=dtype, device_map=device, attn_implementation="eager")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.layers = _first_attr(self.model, ["model.layers", "model.language_model.layers",
                                               "language_model.model.layers", "transformer.h"])
        self.norm = _first_attr(self.model, ["model.norm", "model.language_model.norm",
                                             "language_model.model.norm", "transformer.ln_f"])
        self.lm_head = self.model.get_output_embeddings()
        self.n_layers = len(self.layers)
        config = self.model.config.get_text_config()
        self.d_model = config.hidden_size
        self.vocab_size = self.lm_head.weight.shape[0]
        self.dtype = str(dtype).replace("torch.", "")
        # Linear-attention / SSM layers keep recurrent state that can't be cropped back to a
        # shared prefix, so those models re-read the prompt on every request.
        self.layer_types = list(getattr(config, "layer_types", None) or [])
        self.croppable_cache = all(t in ("full_attention", "sliding_attention") for t in self.layer_types)
        # Rows per logit-lens chunk, sized so one fp32 (rows x vocab) buffer stays near 400 MB,
        # and a prefill chunk small enough that its logits fit beside the weights.
        self.lens_chunk = max(64, int(1e8 // self.vocab_size))
        self.prefill_chunk = 512 if self.vocab_size < 200_000 else 256
        self.stop_ids = self._stop_ids()
        # Token placed before raw texts (direction building, reference text): BOS where the model
        # has one, otherwise its document separator. It absorbs the first-position attention sink.
        vocab = self.tokenizer.get_vocab()
        self.sink_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                        else vocab.get("<|endoftext|>", self.tokenizer.eos_token_id))
        n_tok = len(self.tokenizer)
        strings = self.tokenizer.batch_decode([[i] for i in range(n_tok)])
        self._token_strs = strings + [f"<{i}>" for i in range(n_tok, self.vocab_size)]

    def _stop_ids(self) -> set[int]:
        ids: set[int] = set()
        eos = self.model.generation_config.eos_token_id
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        vocab = self.tokenizer.get_vocab()
        ids.update(vocab[t] for t in STOP_TOKENS if t in vocab)
        return ids

    def token_str(self, token_id: int) -> str:
        return self._token_strs[token_id] if 0 <= token_id < len(self._token_strs) else f"<{token_id}>"

    def info(self) -> dict:
        return {"name": self.name, "spec": self.spec, "n_layers": self.n_layers, "d_model": self.d_model,
                "vocab_size": self.vocab_size, "dtype": self.dtype, "provenance": self.provenance,
                "architecture": type(self.model).__name__, "layer_types": self.layer_types,
                "croppable_cache": self.croppable_cache}


if __name__ == "__main__":
    path, prov = prepare(sys.argv[1] if len(sys.argv) > 1 else "ollama:llama3.1:8b")
    print(path)
    print(json.dumps(prov, indent=2))
    tok = AutoTokenizer.from_pretrained(path)
    print("chat template present:", bool(tok.chat_template))
