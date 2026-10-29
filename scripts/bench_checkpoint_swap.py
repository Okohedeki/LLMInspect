"""Time the original-weights view (EditStack.suspended) with a full fine-tune applied.

    scripts/run.sh scripts/bench_checkpoint_swap.py [model] [--keep]

Writes a candidate in which every tensor of the model differs a little (the shape of a full
fine-tune) next to the model cache, loads it as a checkpoint edit and applies it, then times
three round trips onto the original weights and back each way: replayed from the cached candidate
tensors (the default), and replayed with every changed tensor read from the files again (no cache). Prints the
times, the bytes held on the CPU, and the process's resident memory. The candidate folder is
removed at the end unless --keep is given.
"""

import gc
import os
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a file: the package is beside scripts/
from llminspect.config import Settings  # noqa: E402
from llminspect.recorder.checkpoints import compare_weights, open_candidate
from llminspect.recorder.model import Subject
from llminspect.recorder.session import Recorder
from llminspect.recorder.store import RunStore

GB = 1e9
args = [a for a in sys.argv[1:] if not a.startswith("--")]
spec = args[0] if args else os.environ.get("LLMINSPECT_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
keep = "--keep" in sys.argv
rss = lambda: psutil.Process().memory_info().rss / GB

settings = Settings(model=spec, anatomy=False, graph_decode=False)
root = Path.home() / ".cache" / "llminspect" / "bench"
subject = Subject(spec, device=settings.device)
rec = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
print(f"model {subject.name}: {sum(p.numel() for p in subject.model.parameters()) / 1e9:.2f} B parameters, "
      f"budget {rec.edits.budget / GB:.1f} GB, rss {rss():.1f} GB")

folder = root / f"{subject.slug}-full-finetune"
if folder.exists():
    shutil.rmtree(folder)
folder.mkdir(parents=True)
t0 = time.perf_counter()
g = torch.Generator().manual_seed(0)
tensors = {}
with torch.no_grad():
    for name, p in subject.model.named_parameters():
        w = p.detach().to("cpu", torch.float32)
        noise = torch.randn(w.shape, generator=g)
        tensors[name] = (w + 2e-3 * w.abs().mean() * noise).to(p.dtype).contiguous()
save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
del tensors
gc.collect()
subject.model.config.save_pretrained(folder)
print(f"wrote the candidate in {time.perf_counter() - t0:.1f}s ({sum(f.stat().st_size for f in folder.iterdir()) / GB:.2f} GB)")

reader = open_candidate(str(folder), subject.model)
report = compare_weights(rec, reader)
print(f"compared: {report['counts']['changed']} of {report['counts']['compared']} tensors differ, "
      f"pattern {report['summary']['pattern']}")
rec.edits.add("checkpoint", checkpoint={"spec": str(folder), "folder": str(folder), "name": folder.name,
                                        "report": report["id"],
                                        "changed": [r["name"] for r in report["tensors"] if r["rel"] > 0]})


def timed(label: str) -> None:
    times = []
    for _ in range(3):
        if subject.device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with rec.edits.suspended():
            pass
        if subject.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    print(f"{label}: round trips {', '.join(f'{x:.2f}s' for x in times)}; candidate cache "
          f"{rec.edits.summary()['candidate_cache_bytes'] / GB:.2f} GB, originals "
          f"{rec.edits.summary()['snapshot_bytes'] / GB:.2f} GB, rss {rss():.1f} GB, swaps {rec.edits.swaps}")


t = time.perf_counter()
rec.edits.apply()
print(f"first apply (reads the candidate): {time.perf_counter() - t:.2f}s, rss {rss():.1f} GB")
timed("replayed from the candidate cache")
reader = next(iter(rec.edits._readers.values()))
reader._cache.clear()  # the cache is emptied and keeps nothing more: the candidate is read again
reader._cache_bytes, reader.budget = 0, 0
gc.collect()
timed("replayed from disk")
rec.edits.revert()
if not keep:
    shutil.rmtree(folder)
