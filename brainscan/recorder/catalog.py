"""Open-weights models already on this machine, for the model picker.

Brainscan never downloads a model here. The catalog lists Hugging Face checkpoint folders (a
config.json beside safetensors weights) under Brainscan's model cache, the Hugging Face hub cache
and any folders in BRAINSCAN_MODEL_DIRS, plus Ollama tags, which are dequantized to bf16 once on
first load (model.prepare). Each entry carries the memory estimate the loader itself checks
(model._check_vram: weight bytes plus headroom), so the picker can say up front what won't fit.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .model import CACHE, _slug, display_name, ollama_roots, weight_bytes

HEADROOM = 2.5 * 1024 ** 3  # the loader's margin over the weights, in bytes (model._check_vram)
GIB = 1024 ** 3
TOKENIZER_FILES = ("tokenizer.json", "tokenizer.model", "vocab.json", "tokenizer_config.json")


def hf_hub_cache() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def search_roots(extra: list[Path] | None = None) -> list[Path]:
    roots = [CACHE / "models", hf_hub_cache(), *(extra or [])]
    out, seen = [], set()
    for r in roots:
        key = str(Path(r).expanduser())
        if key not in seen:
            seen.add(key)
            out.append(Path(key))
    return out


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _causal_lm_types() -> set[str] | None:
    try:
        from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        return set(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
    except Exception:  # the mapping moved; don't guess
        return None


def _gguf_types() -> set[str] | None:
    """Architectures transformers can dequantize from GGUF (what an Ollama tag needs)."""
    try:
        from transformers.modeling_gguf_pytorch_utils import GGUF_SUPPORTED_ARCHITECTURES
        return set(GGUF_SUPPORTED_ARCHITECTURES)
    except Exception:
        return None


def _supported(types: set[str] | None, *model_types) -> bool | None:
    if types is None:
        return None
    return any(t in types for t in model_types if t)


def _checkpoint(path: Path, types) -> dict | None:
    """Describe a Hugging Face checkpoint folder, or None if it isn't one."""
    cfg = _json(path / "config.json")
    if not cfg or not any(path.glob("*.safetensors")):
        return None
    size, loaded = weight_bytes(path)  # stat follows the hub cache's symlinks to blobs
    text = cfg.get("text_config") or cfg
    notes = []
    has_tokenizer = any((path / f).exists() for f in TOKENIZER_FILES)
    if not has_tokenizer:
        notes.append("No tokenizer files next to the weights, so this folder can't be loaded.")
    if cfg.get("quantization_config"):
        notes.append("Quantized checkpoint: tracing and weight edits assume dense weights.")
    if any(k in text for k in ("num_experts", "num_local_experts", "n_routed_experts")):
        notes.append("Mixture-of-experts MLPs: the Trace tab has no neuron clusters for them yet.")
    if loaded / 2 < 1e8:
        notes.append("Very small model (under 100M parameters): good for trying the tools, not for findings.")
    if size - loaded > 0.05 * size and loaded < size:
        notes.append(f"{(size - loaded) / GIB:.1f} GB of the files (e.g. a vision tower) isn't loaded for the text model.")
    return {
        "architecture": (cfg.get("architectures") or [None])[0],
        "model_type": text.get("model_type") or cfg.get("model_type"),
        "n_layers": text.get("num_hidden_layers") or text.get("n_layer"),
        "d_model": text.get("hidden_size") or text.get("n_embd"),
        "params": loaded / 2,  # loaded as bf16
        "disk_bytes": size,
        "need_bytes": loaded + HEADROOM,  # what the loader checks against free GPU memory (model._check_vram)
        "supported": _supported(types, cfg.get("model_type"), text.get("model_type")) if has_tokenizer else False,
        "notes": notes,
    }


def _folders(root: Path):
    """Checkpoint candidates under a root: the root, its children, and hub-cache snapshots."""
    if not root.is_dir():
        return
    yield root
    try:
        children = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return
    for child in children:
        if child.name.startswith("models--"):
            snaps = sorted((child / "snapshots").glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                yield snaps[0]  # the newest snapshot
        else:
            yield child


def _ollama_tags():
    for root in ollama_roots():
        base = root / "manifests" / "registry.ollama.ai"
        if not base.is_dir():
            continue
        for manifest in sorted(p for p in base.rglob("*") if p.is_file()):
            ns, name, version = manifest.relative_to(base).parts[-3:]
            tag = f"{name}:{version}" if ns == "library" else f"{ns}/{name}:{version}"
            data = _json(manifest) or {}
            model = next((l for l in data.get("layers", []) if l.get("mediaType", "").endswith(".model")), None)
            if model is None:
                continue
            digest = (data.get("config") or {}).get("digest", "")
            config = _json(root / "blobs" / digest.replace(":", "-")) if digest else None
            yield tag, model, config or {}


def _billions(text) -> float | None:
    m = re.match(r"([\d.]+)\s*([BbMm])", str(text or ""))
    return float(m.group(1)) * (1e9 if m.group(2) in "Bb" else 1e6) if m else None


def scan(extra_dirs: list[Path] | None = None, current: str | None = None,
         budget_bytes: float | None = None) -> list[dict]:
    """Every loadable model found on disk. budget_bytes is the GPU memory free once the current
    model is unloaded (None on CPU); each entry's `fits` compares it with the loader's estimate."""
    types = _causal_lm_types()
    out: list[dict] = []
    seen_paths, seen_specs = set(), set()
    loaded = Path(current).resolve() if current and not current.startswith("ollama:") and Path(current).is_dir() else None
    roots = search_roots(extra_dirs)
    if loaded is not None and not any(loaded == r.resolve() or loaded.parent == r.resolve() for r in roots):
        roots.insert(0, loaded)  # the loaded model lives outside the searched folders

    def add(entry: dict) -> None:
        if entry["spec"] in seen_specs:
            return
        seen_specs.add(entry["spec"])
        need = entry.get("need_bytes")
        entry["fits"] = None if budget_bytes is None or need is None else bool(need <= budget_bytes)
        entry["need_gb"] = round(need / GIB, 1) if need else None
        entry["disk_gb"] = round(entry.pop("disk_bytes", 0) / GIB, 1) if entry.get("disk_bytes") else None
        entry["params_b"] = round(entry.pop("params") / 1e9, 2) if entry.get("params") else None
        entry["current"] = entry["spec"] == current or (
            loaded is not None and entry.get("path") is not None and Path(entry["path"]).resolve() == loaded)
        out.append(entry)

    for root in roots:
        for folder in _folders(root):
            real = folder.resolve()
            if real in seen_paths:
                continue
            seen_paths.add(real)
            info = _checkpoint(folder, types)
            if info is None:
                continue
            prov = _json(folder / "brainscan_provenance.json") or {}
            if prov.get("source") == "ollama":  # converted earlier: load it by its tag to keep provenance
                spec, source = prov["spec"], "ollama"
                info["notes"].insert(0, f"Dequantized from {prov.get('quantization') or 'a quantized GGUF'}: "
                                        "exploratory, quantization noise stays in the weights.")
            else:
                spec, source = str(folder), "checkpoint"
            add({"spec": spec, "name": display_name(spec), "source": source, "path": str(folder),
                 "converted": True, **info})

    by_blob: dict[str, str] = {}
    gguf = _gguf_types()
    for tag, model, config in _ollama_tags():
        spec = f"ollama:{tag}"
        same = by_blob.setdefault(model.get("digest", tag), tag)
        if spec in seen_specs:
            continue
        local = CACHE / "models" / f"ollama-{_slug(tag)}"
        params = _billions(config.get("model_type"))
        family = str(config.get("model_family") or "").replace("-", "_")
        quant = config.get("file_type") if config.get("file_type") not in (None, "", "unknown") else "a quantized GGUF"
        notes = [f"Converted to bf16 once on first load (about {params * 2 / 1e9:.0f} GB written to "
                 f"{local.parent})." if params else "Converted to bf16 once on first load.",
                 f"Dequantized from {quant}: exploratory, quantization noise stays in the weights."]
        if same != tag:
            notes.insert(0, f"Same weights as {same} (a different system prompt or settings).")
        if family == "gpt_oss":
            notes.append("Mixture-of-experts MLPs: the Trace tab has no neuron clusters for them yet.")
        convertible = None if gguf is None or not family else family in gguf
        if convertible is False:
            notes.insert(0, f"transformers can't convert {family} GGUF files, so this tag can't be loaded.")
        add({"spec": spec, "name": tag, "source": "ollama", "path": None, "converted": False,
             "architecture": None, "model_type": family or None, "n_layers": None, "d_model": None,
             "params": params, "disk_bytes": model.get("size"),
             "need_bytes": params * 2 + HEADROOM if params else None,
             "supported": False if convertible is False else _supported(types, family, f"{family}_text") if family else None,
             "notes": notes, "same_as": same if same != tag else None})

    out.sort(key=lambda e: (not e["current"], e["source"] != "checkpoint", e["name"].lower()))
    return out


def find(spec: str, extra_dirs: list[Path] | None = None) -> dict | None:
    return next((e for e in scan(extra_dirs) if e["spec"] == spec), None)
