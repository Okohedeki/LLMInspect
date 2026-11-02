"""Sweeps of many models, and pickle weights parsed by byte range. The pickles here are inert fixtures: their
'payload' is a harmless call that pickle.dumps only records; nothing here unpickles anything."""

import json
import os
import pickle

import torch

from llminspect import hub, sweep

from test_check import tiny
from test_filescan import Harmless
from test_hub import edited, remove_direction


class Remote:
    """A folder served the way the Hub serves a repo: file sizes, and byte ranges counted as they are read."""

    def __init__(self, folder):
        self.folder, self.bytes_read = folder, 0

    def files(self):
        return {p.name: p.stat().st_size for p in self.folder.iterdir()}

    def read(self, spans):
        out = []
        for f, at, n in spans:
            with open(self.folder / f, "rb") as fh:
                fh.seek(at)
                out.append(fh.read(n))
            self.bytes_read += len(out[-1])
        return out


def test_pickles_are_parsed_by_range_without_reading_their_tensors(tmp_path):
    big = torch.zeros(4 << 20)  # 16 MB of tensor bytes the scan should skip
    torch.save({"w": big, "hook": Harmless(print, "inert")}, tmp_path / "pytorch_model.bin")
    torch.save({"w": big}, tmp_path / "legacy.bin", _use_new_zipfile_serialization=False)
    (tmp_path / "opt.pkl").write_bytes(pickle.dumps({"s": Harmless(os.getcwd)}, protocol=2))
    src = Remote(tmp_path)
    found, unread = hub._remote_pickles(src, sorted(src.files()))
    assert unread == []
    by_file = {f["file"]: f for f in found}
    assert by_file["pytorch_model.bin"]["level"] == "runs code on load"
    assert "builtins.print" in by_file["pytorch_model.bin"]["evidence"]
    assert by_file["opt.pkl"]["level"] == "runs code on load" and "legacy.bin" not in by_file
    assert src.bytes_read < 2 * hub.BLOCK * 3  # a block or two per file, not 32 MB of tensors


def test_a_range_file_stops_at_its_budget(tmp_path):
    (tmp_path / "x.bin").write_bytes(os.urandom(3 * hub.BLOCK))
    f = hub.RangeFile(Remote(tmp_path), "x.bin", 3 * hub.BLOCK, budget=2 * hub.BLOCK)
    assert len(f.read(2 * hub.BLOCK)) == 2 * hub.BLOCK
    try:
        f.read(1)
        raise AssertionError("read past the budget")
    except OSError:
        pass


def test_a_sweep_ranks_risks_and_resumes(tmp_path):
    roots = [tmp_path / "models"]
    base = tiny(roots[0] / "Org--base")
    abl = edited(base, tmp_path / "abl", remove_direction)
    bad = tiny(tmp_path / "bad")
    torch.save({"hook": Harmless(print, "inert")}, bad / "extra.bin")
    out = tmp_path / "sweep.jsonl"
    seen = []
    s = sweep.sweep([str(abl), str(bad)], out, weights=True, roots=roots, say=seen.append)
    assert s["models"] == 2 and s["errors"] == 0
    assert [r["top"] for r in s["flagged"]] == ["runs code on load", "targeted weight edit"]
    assert s["counts"]["targeted weight edit"] == 1 and s["counts"]["runs code on load"] == 1
    assert "one direction removed" in s["flagged"][1]["risks"][0]["what"]
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2 and all(r["weights"] for r in rows)
    again = sweep.sweep([str(abl), str(bad)], out, weights=True, roots=roots, say=seen.append)
    assert again == s and len(out.read_text().splitlines()) == 2 and "0 to check" in seen[-1]


def test_an_unreadable_model_is_a_row_not_a_crash(tmp_path):
    s = sweep.sweep([str(tmp_path / "missing")], tmp_path / "s.jsonl")
    assert s["errors"] == 1 and s["flagged"] == []
