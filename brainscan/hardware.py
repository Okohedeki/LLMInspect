"""Phase 0 hardware report: what this machine can record, and what recordings cost on disk.

    scripts/run.sh -m brainscan.hardware
"""

from __future__ import annotations

import os
import platform
import shutil

import psutil

from .config import Settings

GB = 1024 ** 3
# (name, total params in billions, layers, d_model). MoE weights are stored in full even if few are active.
LADDER = [
    ("Qwen3-4B", 4.0, 36, 2560),
    ("Llama-3.1-8B-Instruct", 8.0, 32, 4096),
    ("Qwen3-8B", 8.2, 36, 4096),
    ("Qwen3.5-9B", 9.7, 32, 4096),
    ("Qwen3-14B", 14.8, 40, 5120),
    ("GPT-OSS-20B (MXFP4)", 21.0, 24, 2880),
    ("Gemma-3-27B", 27.0, 62, 5376),
    ("Qwen3-30B-A3B", 30.5, 48, 2048),
    ("Qwen3-32B", 32.8, 64, 5120),
    ("Llama-3.3-70B", 70.6, 80, 8192),
]
OVERHEAD_GB = 3.0  # KV cache at 16k context, prefill logits, lens buffers


def gpus() -> list[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            out.append({"name": name.decode() if isinstance(name, bytes) else name,
                        "total_gb": mem.total / GB, "free_gb": mem.free / GB})
        return out
    except Exception:
        return []


def weights_gb(name: str, params_b: float) -> float:
    if "MXFP4" in name:
        return 13.0  # MoE experts stay in MXFP4 (about 4.25 bits); the rest is bf16
    return params_b * 2 * 1e9 / GB


def bytes_per_token(n_layers: int, d_model: int) -> dict:
    sparse = n_layers * 12 + 200  # lens p + lens top + norm per layer, plus text and stats
    full = n_layers * d_model * 2  # every layer's residual in fp16
    return {"sparse": sparse, "full": full}


def report() -> str:
    settings = Settings()
    vm = psutil.virtual_memory()
    lines = ["Brainscan hardware report", "=" * 25,
             f"Host: {platform.system()} {platform.release()} | CPU cores: {psutil.cpu_count(logical=False)} "
             f"physical / {psutil.cpu_count()} logical | RAM: {vm.total / GB:.0f} GB ({vm.available / GB:.0f} GB free)"]
    gs = gpus()
    if not gs:
        lines.append("GPU: none visible (CPU-only recording is possible but slow)")
    for i, g in enumerate(gs):
        lines.append(f"GPU {i}: {g['name']} | {g['total_gb']:.1f} GB total, {g['free_gb']:.1f} GB free")
    data_dir = settings.data_dir
    probe = data_dir if data_dir.exists() else data_dir.parent
    disk = shutil.disk_usage(probe)
    lines.append(f"Recording dir: {data_dir} | {disk.free / GB:.0f} GB free | budget {settings.disk_budget_gb:.0f} GB")
    vram = max((g["total_gb"] for g in gs), default=0)

    lines += ["", "Model ladder on this machine (bf16 unless noted; J-lens backward passes need ~1.5x weights)"]
    lines.append(f"{'model':26} {'weights':>8}  {'record':>8}  {'J-lens':>8}")
    for name, params, n_layers, d_model in LADDER:
        w = weights_gb(name, params)
        fits = "local" if w + OVERHEAD_GB <= vram else ("4-bit" if w / 4 + OVERHEAD_GB <= vram else "rent")
        jlens = "local" if w * 1.5 + OVERHEAD_GB <= vram else "rent"
        lines.append(f"{name:26} {w:7.1f}G  {fits:>8}  {jlens:>8}")

    lines += ["", "Disk per recorded run (sparse readouts everywhere, full residuals only at decision points)"]
    for name, _, n_layers, d_model in LADDER[1:3] + LADDER[8:9]:
        b = bytes_per_token(n_layers, d_model)
        run_tokens, decisions = 30_000, 40 * 4
        sparse = run_tokens * b["sparse"]
        decision = decisions * b["full"]
        everything = run_tokens * b["full"]
        lines.append(f"{name:26} 30k-token run: {sparse / 1e6:6.1f} MB sparse + {decision / 1e6:6.0f} MB decision "
                     f"residuals | full residuals everywhere would be {everything / 1e9:5.1f} GB")
    runs_fit = settings.disk_budget_gb * 1e9 / (30_000 * bytes_per_token(32, 4096)["sparse"] + 160 * 32 * 4096 * 2)
    lines.append(f"Budget of {settings.disk_budget_gb:.0f} GB holds roughly {runs_fit:,.0f} such runs on an 8B model.")
    if disk.free / GB < settings.disk_budget_gb:
        lines.append(f"WARNING: only {disk.free / GB:.0f} GB free on the recording drive, below the budget.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
