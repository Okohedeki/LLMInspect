"""On-disk layout for recordings, with a disk budget.

runs/<run_id>/
  meta.json                        model provenance, settings, labels, outcome
  messages.jsonl                   transcript entries with their tape index
  events.jsonl                     time-locked events (tape index, type, data)
  tokens/req0001.parquet           one row per token: prediction stats, logit lens, norms, projections
  residuals/req0001.safetensors    full residual stacks at decision points only
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.torch import save_file


class RunStore:
    def __init__(self, root: Path, budget_gb: float):
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget_gb * 1e9
        self._used: int | None = None

    def used_bytes(self) -> int:
        if self._used is None:
            self._used = sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())
        return self._used

    def _grew(self, n: int) -> None:
        self._used = self.used_bytes() + n

    def can_write(self, n: int) -> bool:
        return self.used_bytes() + n <= self.budget

    def run_dir(self, run_id: str) -> Path:
        d = self.runs_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_meta(self, run_id: str, meta: dict) -> None:
        (self.run_dir(run_id) / "meta.json").write_text(json.dumps(meta, indent=2))

    def append_jsonl(self, run_id: str, name: str, rows: list[dict]) -> None:
        lines = "".join(json.dumps(r) + "\n" for r in rows)
        with open(self.run_dir(run_id) / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(lines)
        self._grew(len(lines))

    def write_tokens(self, run_id: str, req: int, records: list[dict]) -> None:
        if not records:
            return
        watched = any(r.get("nz") for r in records)  # neuron monitors' z-scores: a column only if this request had a watch
        rows = []
        for r in records:
            row = {**r, "proj": json.dumps(r["proj"])}
            row.pop("nz", None)
            if watched:
                row["nz"] = json.dumps(r.get("nz") or {})
            rows.append(row)
        path = self.run_dir(run_id) / "tokens" / f"req{req:04d}.parquet"
        path.parent.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        self._grew(path.stat().st_size)

    def write_residuals(self, run_id: str, req: int, tensors: dict, metadata: dict) -> bool:
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        if not tensors or not self.can_write(size):
            return False
        path = self.run_dir(run_id) / "residuals" / f"req{req:04d}.safetensors"
        path.parent.mkdir(exist_ok=True)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path),
                  metadata={k: json.dumps(v) for k, v in metadata.items()})
        self._grew(path.stat().st_size)
        return True

    def list_runs(self) -> list[dict]:
        runs = []
        for d in sorted(self.runs_dir.iterdir(), reverse=True):
            meta_file = d / "meta.json"
            if d.is_dir() and meta_file.exists():
                runs.append(json.loads(meta_file.read_text()))
        return runs

    def read_meta(self, run_id: str) -> dict | None:
        meta_file = self.runs_dir / run_id / "meta.json"
        return json.loads(meta_file.read_text()) if meta_file.exists() else None

    def read_jsonl(self, run_id: str, name: str) -> list[dict]:
        path = self.runs_dir / run_id / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def read_tokens(self, run_id: str) -> list[dict]:
        rows: list[dict] = []
        for path in sorted((self.runs_dir / run_id / "tokens").glob("*.parquet")):
            rows += pq.read_table(path).to_pylist()
        for r in rows:
            r["proj"] = json.loads(r["proj"]) if r.get("proj") else {}
            if r.get("nz"):  # a request recorded without a watch has no such column, or an empty one
                r["nz"] = json.loads(r["nz"])
            else:
                r.pop("nz", None)
        rows.sort(key=lambda r: r["t"])
        return rows
