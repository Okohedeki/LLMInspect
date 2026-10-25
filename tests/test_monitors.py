"""Monitors: a rule on a probe's z-score, checked live as tokens are recorded and by scanning a
recorded tape; both must raise the same flags. The live half runs a real CPU recorder on the tiny
saved Llama with a direction built from example texts."""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder.model import Subject
from brainscan.recorder.monitors import Monitor, MonitorBank, scan, scan_run
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"


def tape(zs, ph="gen", key="probe", control=None):
    return [{"t": t, "req": 1, "ph": ph if isinstance(ph, str) else ph[t], "text": f"w{t} ",
             "proj": {key: z} | ({"random_control": control[t]} if control else {})} for t, z in enumerate(zs)]


def test_a_flag_needs_its_tokens_in_a_row():
    m = Monitor("probe", threshold=2, min_tokens=3).check()
    flags = scan(m, tape([0, 2.5, 3, 1, 2, 2.1, 4, 2.2, 0, 3, 3, 3]))
    assert [(f["start"], f["end"], f["t"], f["tokens"], f["peak_t"]) for f in flags] == [(4, 7, 6, 4, 6), (9, 11, 11, 3, 9)]
    assert flags[0]["peak_z"] == 4 and flags[0]["z"] == 4
    below = Monitor("probe", threshold=1, sign="below", min_tokens=2).check()
    assert [f["start"] for f in scan(below, tape([0, -1.5, -2, 0, -3]))] == [1]
    either = Monitor("probe", threshold=1, sign="either", min_tokens=2).check()
    assert [f["start"] for f in scan(either, tape([0, -1.5, 2, 0, -3]))] == [1]


def test_only_the_tokens_the_rule_watches_count():
    """Prompt tokens break a streak of the model's own tokens; re-read context never counts."""
    ph = ["gen", "gen", "prefill", "gen", "gen", "replay", "gen"]
    zs = [3, 3, 3, 3, 3, 3, 3]
    assert [f["start"] for f in scan(Monitor("probe", min_tokens=2).check(), tape(zs, ph))] == [0, 3]
    assert [(f["start"], f["end"]) for f in scan(Monitor("probe", min_tokens=2, phase="all").check(), tape(zs, ph))] == [(0, 4)]
    assert scan(Monitor("probe", min_tokens=1).check(), tape(zs, ph, key="other")) == []  # not recorded on this run


def test_the_scan_reports_the_random_control(tmp_path):
    bank = MonitorBank(tmp_path / "m.json")
    m = bank.add({"direction": "probe", "threshold": 2, "min_tokens": 2}, {"probe", "random_control"})
    assert bank.add({"direction": "probe", "threshold": 2, "min_tokens": 2}, {"probe"}) is m  # same rule, one monitor
    dirs = [{"name": "probe", "kind": "contrast"}, {"name": "random_control", "kind": "random_control"}]
    row = scan_run(bank, tape([3, 3, 0, 3, 3, 0], control=[0, 0, 0, 3, 3, 3]), dirs)[0]
    assert len(row["flags"]) == 2 and row["past"] == pytest.approx(4 / 6, abs=1e-3)
    assert row["flags"][1]["text"] == "w3 w4 "  # the words of the streak
    assert row["control"] == {"direction": "random_control", "recorded": 6, "past": 0.5, "flags": 1}
    assert MonitorBank(tmp_path / "m.json").list()[0]["id"] == m.id  # saved
    with pytest.raises(ValueError):
        bank.add({"direction": "nope"}, {"probe"})
    for bad in ({"threshold": 0}, {"min_tokens": 0}, {"sign": "up"}, {"phase": "replay"}):
        with pytest.raises(ValueError):
            bank.update(m.id, bad)
    assert bank.get(m.id).threshold == 2  # a rejected change leaves the monitor as it was
    bank.update(m.id, {"enabled": False})
    assert bank.feed({}, tape([3])[0]) == []
    with pytest.raises(KeyError):
        bank.remove("nope")
    assert bank.remove(m.id) == []


@pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")
def test_live_flags_match_the_scan(tmp_path):
    torch.manual_seed(0)
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=False, graph_decode=False, data_dir=tmp_path)
    rec = Recorder(subject, RunStore(tmp_path / "data", settings.disk_budget_gb), settings, lambda m: None)
    texts = (["I love this, it is wonderful.", "What a great day!"], ["I hate this, it is awful.", "What a terrible day."])
    rec.directions.build_contrast("tone", 1, *texts, "test probe")
    rec.directions.build_random("random_control", 1, texts[0] + texts[1], 0, "test control")
    ask = [{"role": "user", "content": "Describe the weather in three words."}]

    # A first run sets a threshold that some of the model's own tokens pass and some don't.
    rec.generate(GenParams(messages=ask, max_tokens=16, temperature=0, run_hint="calibrate"))
    zs = sorted(abs(r["proj"]["tone"]) for r in rec.get_run("calibrate").tape if r["ph"] == "gen")
    m = rec.monitors.add({"direction": "tone", "sign": "either", "threshold": zs[len(zs) // 2], "min_tokens": 1},
                         set(rec.directions.dirs))
    rec.generate(GenParams(messages=ask, max_tokens=16, temperature=0, run_hint="watched"))
    run = rec.get_run("watched")
    live = [e for e in run.events if e["type"] == "monitor"]
    flags = scan(m, run.tape)
    assert flags and [e["t"] for e in live] == [f["t"] for f in flags]
    assert all(run.tape[e["t"]]["ph"] == "gen" and e["data"]["monitor"] == m.id for e in live)
    assert run.meta["flags"] == len(flags)

    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        assert client.get("/brainscan/monitors").json()[0]["id"] == m.id
        assert client.post("/brainscan/monitors", json={"direction": "nope"}).status_code == 400
        assert client.post("/brainscan/monitors", json={"direction": "tone", "threshold": -1}).status_code == 400
        row = client.get("/brainscan/runs/watched/flags").json()[0]
        assert [f["t"] for f in row["flags"]] == [f["t"] for f in flags]
        assert row["control"]["direction"] == "random_control"
        assert client.get("/brainscan/runs/nope/flags").status_code == 404
        across = client.get("/brainscan/flags").json()
        assert {r["run"]: r["flags"][m.id] for r in across["runs"]}["watched"] == len(flags)
        assert client.patch(f"/brainscan/monitors/{m.id}", json={"enabled": False}).json()["enabled"] is False
        assert client.patch("/brainscan/monitors/nope", json={}).status_code == 404
        assert client.delete(f"/brainscan/monitors/{m.id}").json() == []
    finally:
        server.state.clear()
        server.state.update(saved)
