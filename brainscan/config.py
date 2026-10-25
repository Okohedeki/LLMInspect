"""Recorder settings, read from BRAINSCAN_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BRAINSCAN_{name}", default)


@dataclass
class Settings:
    # "ollama:<tag>" loads a model from the local Ollama store (dequantized from GGUF);
    # anything else is a Hugging Face repo id or local directory.
    model: str = field(default_factory=lambda: _env("MODEL", "ollama:llama3.1:8b"))
    # Extra folders the model picker searches for checkpoints, besides Brainscan's model cache and
    # the Hugging Face hub cache (separated by ":"; e.g. /mnt/h/models). Nothing is downloaded.
    model_dirs: list[Path] = field(default_factory=lambda: [
        Path(p).expanduser() for p in _env("MODEL_DIRS", "").split(os.pathsep) if p.strip()])
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA", str(PROJECT_ROOT / "runs"))))
    disk_budget_gb: float = field(default_factory=lambda: float(_env("DISK_BUDGET_GB", "50")))
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8765")))
    max_context: int = field(default_factory=lambda: int(_env("MAX_CONTEXT", "16384")))
    # Logit lens over prompt and tool-result tokens, not only generated ones.
    prefill_lens: bool = field(default_factory=lambda: _env("PREFILL_LENS", "1") == "1")
    # Full residual stacks at decision points (pre-generation, first token, tool-call start, last token).
    decision_residuals: bool = field(default_factory=lambda: _env("DECISION_RESIDUALS", "1") == "1")
    # Reasoning models: "on" / "off" sets enable_thinking in the chat template, "default" leaves the
    # template's own default. Requests can override with chat_template_kwargs.
    thinking: str = field(default_factory=lambda: _env("THINKING", "default"))
    # Replay each decode step from CUDA graphs (recorder/graphs.py); outputs are bit-identical to eager.
    graph_decode: bool = field(default_factory=lambda: _env("GRAPH_DECODE", "1") == "1")
    # Weight anatomy (recorder/anatomy.py): per-token activity of attention heads and MLP neuron
    # clusters, and the Weights tab. ANATOMY_CLUSTERS sets the clusters per MLP layer.
    anatomy: bool = field(default_factory=lambda: _env("ANATOMY", "1") == "1")
    anatomy_clusters: int = field(default_factory=lambda: int(_env("ANATOMY_CLUSTERS", "64")))
    # The share of GPU memory the recorder may use (0.25 = a quarter), so it shares the GPU with other
    # apps: going over raises an out-of-memory error instead of spilling into system RAM. 0: no cap.
    gpu_fraction: float = field(default_factory=lambda: float(_env("GPU_FRACTION", "0")))

    def template_kwargs(self) -> dict:
        return {"on": {"enable_thinking": True}, "off": {"enable_thinking": False}}.get(self.thinking, {})
