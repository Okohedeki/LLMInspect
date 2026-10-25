"""Knobs: named sets of neurons with doses, saved per model, and steered, ablated, monitored and swept
as one. The neuron monitors' live signal (a z-score of the activation, read by hooks that must also
work inside decode CUDA graphs) is checked against a brute-force read on the tiny saved Llama in fp32,
and the graph path itself on a tiny CUDA model when there is a GPU."""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import neurons as nmod
from brainscan.recorder.capture import LayerTap
from brainscan.recorder.graphs import DecodeGraphs
from brainscan.recorder.knobs import KnobBank
from brainscan.recorder.model import Subject
from brainscan.recorder.monitors import MAX_MONITORS, Monitor, MonitorBank, scan, scan_run
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore
from brainscan.recorder.workbench import REFERENCE_TEXTS, _reference_ids

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"
ASK = [{"role": "user", "content": "Describe the weather in three words."}]


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    if not MODEL_DIR.exists():
        pytest.skip(f"needs the tiny CPU Llama at {MODEL_DIR}")
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("knobs")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    return Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)


@pytest.fixture(autouse=True)
def clean(request):
    """Every test starts from a recorder with no knobs, steering, monitors or edits."""
    yield
    if "rec" not in request.fixturenames:
        return
    rec = request.getfixturevalue("rec")
    for m in rec.monitors.monitors():
        rec.monitors.remove(m.id)
    rec.interventions.clear()
    rec._sync_interventions()
    if rec.edits.applied:
        rec.revert_weight_edits()
    rec.edits.edits.clear()
    rec.knobs.items = []
    rec.knobs.path.unlink(missing_ok=True)


@pytest.fixture
def client(rec):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        yield TestClient(server.app)
    finally:
        server.state.clear()
        server.state.update(saved)


def logits_of(rec, text):
    x = torch.tensor([_reference_ids(rec.s, text)])
    with torch.no_grad():
        return rec.s.model(input_ids=x, use_cache=False).logits[0].float()


def watch_neuron(rec, label, **rule):
    return rec.monitors.add({"direction": label, "threshold": 0.01, "sign": "either", "min_tokens": 1} | rule,
                            set(rec.directions.dirs), rec.neurons.check)


# ---- the bank --------------------------------------------------------------------------------


def test_a_set_of_neurons_is_parsed_like_the_ui_does():
    assert nmod.parse_set("L1 N3 +1, L2 N5 −0.5, L0 N7") == [(1, 3, 1.0), (2, 5, -0.5), (0, 7, 1.0)]
    assert nmod.parse_set("L5 N120 +1, L9 N64 -0.5") == [(5, 120, 1.0), (9, 64, -0.5)]
    assert nmod.parse_set("l1n3 - 2.5") == [(1, 3, -2.5)]
    for bad in ("", "nothing here", "N3 L1"):
        with pytest.raises(ValueError):
            nmod.parse_set(bad)
    assert nmod.is_label("L5 N120") and not any(map(nmod.is_label, ("L5N120", "l5 n120", "tone", None)))


def test_saving_and_loading(rec, tmp_path):
    path = tmp_path / "knobs" / "m.json"
    bank = KnobBank(path, rec.neurons)
    assert bank.list() == [] and bank.error is None
    a = bank.save("tone", [{"neuron": "L2 N5", "dose": 1}, {"neuron": "l1/n3", "dose": -0.5}], "a test knob")
    assert a["neurons"] == [{"neuron": "L2 N5", "dose": 1.0}, {"neuron": "L1 N3", "dose": -0.5}]  # canonical labels
    time.sleep(0.01)
    b = bank.save("tone", [(2, 5, 2.0)], "again")  # the same name replaces it
    assert b["created"] == a["created"] and b["updated"] > a["updated"] and bank.names() == ["tone"]
    raw = json.loads(path.read_text())
    assert raw["version"] == 1 and raw["shape"] == [rec.s.n_layers, rec.neurons.mlp(0).width]
    assert not list(path.parent.glob("*.tmp"))  # written to a temp file, then moved into place
    assert KnobBank(path, rec.neurons).list() == bank.list()  # a new bank reads it back
    assert bank.get("tone")["description"] == "again" and bank.steers("tone") == [(2, 5, 2.0)]
    with pytest.raises(KeyError):
        bank.get("nope")
    bank.save("edge", [(2, 5, 16.0), (2, 6, -16.0)], "x" * 300)  # the limits themselves are fine
    bank.save("big", [(2, n, 1.0) for n in range(64)])
    names = bank.names()

    ok = [{"neuron": "L2 N5", "dose": 1}]
    for name in ("", "has space", "-lead", "x" * 65, "ünï", None):
        with pytest.raises(ValueError):
            bank.save(name, ok)
    with pytest.raises(ValueError):
        bank.save("k", ok, "x" * 301)
    for bad in (None, [], "L2 N5", ["L2 N5"], [{"dose": 1}], [{"neuron": "nope", "dose": 1}],
                [{"neuron": "L99 N0", "dose": 1}], [{"neuron": "L2 N999999", "dose": 1}],
                [{"neuron": "L2 N5", "dose": 0}], [{"neuron": "L2 N5", "dose": float("nan")}],
                [{"neuron": "L2 N5", "dose": float("inf")}], [{"neuron": "L2 N5", "dose": 17}],
                [{"neuron": "L2 N5", "dose": "x"}], [{"neuron": "L2 N5", "dose": True}],
                [{"neuron": "L2 N5", "dose": 1}, {"neuron": "l2n5", "dose": 2}],  # one neuron twice
                [(2, n, 1.0) for n in range(65)]):
        with pytest.raises(ValueError):
            bank.save("k", bad)
    assert bank.names() == names and KnobBank(path, rec.neurons).names() == names  # failures kept nothing

    bank.delete("edge")
    with pytest.raises(KeyError):
        bank.delete("edge")
    assert "edge" not in KnobBank(path, rec.neurons).names()

    many = KnobBank(tmp_path / "many.json", rec.neurons)
    for i in range(128):
        many.save(f"k{i}", ok)
    with pytest.raises(ValueError):
        many.save("one-more", ok)
    many.save("k0", [(2, 6, 1.0)])  # replacing one is not adding one
    assert len(many.list()) == 128


def test_a_shape_clash_loads_nothing_and_refuses_to_write(rec, tmp_path):
    path = tmp_path / "m.json"
    theirs = {"version": 1, "shape": [rec.s.n_layers + 1, 99], "knobs": [
        {"name": "theirs", "description": "", "neurons": [{"neuron": "L0 N0", "dose": 1}], "created": 1, "updated": 1}]}
    path.write_text(json.dumps(theirs))
    bank = KnobBank(path, rec.neurons)  # the model still loads
    assert bank.list() == [] and "Two models share the name" in bank.error
    with pytest.raises(RuntimeError, match="Two models share the name"):
        bank.save("mine", [(2, 5, 1.0)])
    with pytest.raises(RuntimeError):
        bank.delete("theirs")
    assert json.loads(path.read_text()) == theirs  # never written over

    path.write_text("{not json")  # a file that can't be read is not written over either
    bank = KnobBank(path, rec.neurons)
    assert bank.error and bank.list() == []
    with pytest.raises(RuntimeError):
        bank.save("mine", [(2, 5, 1.0)])
    assert path.read_text() == "{not json"


# ---- steering ----------------------------------------------------------------------------------


def test_steering_a_knob(rec):
    rec.knobs.save("tone", [{"neuron": "L2 N5", "dose": 1.0}, {"neuron": "L1 N3", "dose": -0.5}])
    single = rec.steer_neuron("L2 N5", 1.0)
    rec.remove_intervention(single.id)

    ivs = rec.steer_knob("tone")
    assert len(ivs) == 2 and {iv["knob"] for iv in ivs} == {"tone"} and {iv["mode"] for iv in ivs} == {"write"}
    by = {iv["direction"]: iv for iv in ivs}
    assert by["L2 N5"]["value"] == pytest.approx(single.value, rel=1e-6)  # what steer_neuron gives
    assert by["L1 N3"]["value"] == pytest.approx(-0.5 * rec.neurons.calibration()["median"][1], rel=1e-5)
    assert single.describe()["knob"] is None
    assert len(rec.interventions) == 2

    other = rec.steer_neuron("L3 N1", 1.0)
    again = rec.steer_knob("tone")  # steering it twice replaces it
    assert len(rec.interventions) == 3 and {iv["id"] for iv in again}.isdisjoint(iv["id"] for iv in ivs)
    double = rec.steer_knob("tone", factor=2.0)
    assert [iv["value"] for iv in double] == pytest.approx([2 * iv["value"] for iv in ivs])
    assert [iv["strength"] for iv in double] == [2.0, -1.0]
    assert len(rec.interventions) == 3 and other.id in rec.interventions
    assert len(rec.tap.interventions) == 3

    before = set(rec.interventions)
    for factor in (float("nan"), float("inf"), 17.0, -17.0):
        with pytest.raises(ValueError):
            rec.steer_knob("tone", factor=factor)
    with pytest.raises(ValueError):
        rec.steer_knob("tone", phase="nope")
    with pytest.raises(KeyError):
        rec.steer_knob("nope")
    assert set(rec.interventions) == before  # a refused steer changes nothing
    rec.steer_knob("tone", factor=-16.0)

    assert rec.unsteer_knob("tone") == 2 and rec.unsteer_knob("tone") == 0
    assert list(rec.interventions) == [other.id]
    assert [iv["knob"] for iv in rec.neurons.summary()["steering"]] == [None]


def test_a_steered_knob_gives_the_logits_of_the_same_steers(rec):
    rec.knobs.save("tone", [{"neuron": "L2 N5", "dose": 1.0}, {"neuron": "L1 N3", "dose": -0.5}])
    text = REFERENCE_TEXTS[1]
    rec.steer_knob("tone")
    knobbed = logits_of(rec, text)
    rec.unsteer_knob("tone")
    with rec.tap.paused(), rec.neurons.steered([(2, 5, 1.0), (1, 3, -0.5)]):
        hooked = logits_of(rec, text)
    with rec.tap.paused():
        plain = logits_of(rec, text)
    assert torch.allclose(knobbed, hooked, atol=1e-4)
    assert not torch.allclose(knobbed, plain, atol=1e-3)


def test_a_run_records_the_knob_it_was_steered_with(rec):
    rec.knobs.save("tone", [{"neuron": "L2 N5", "dose": 1.0}, {"neuron": "L1 N3", "dose": -0.5}])
    rec.steer_knob("tone")
    rec.generate(GenParams(messages=ASK, max_tokens=3, temperature=0, run_hint="knob-steered"))
    snapshot = rec.get_run("knob-steered").meta["interventions"][0]["active"]
    assert [iv["knob"] for iv in snapshot] == ["tone", "tone"]


# ---- ablating --------------------------------------------------------------------------------


def test_ablating_a_knob_is_an_unapplied_edit_that_reverts_exactly(rec):
    rec.knobs.save("tone", [(2, 5, 1.0), (2, 9, -1.0), (1, 3, 0.5)])
    units = rec.knobs.units("tone")
    assert units == [{"layer": 1, "target": "neuron", "index": [3], "name": "tone"},
                     {"layer": 2, "target": "neuron", "index": [5, 9], "name": "tone"}]
    edit = rec.add_weight_edit(kind="units", units=units, factor=0.0, note="L2 N5, L2 N9, L1 N3")
    assert edit.id not in rec.edits.applied
    d = edit.describe()  # units with no unit number (g) are fine
    assert d["kind"] == "units" and [u["g"] for u in d["units"]] == [None, None] and [u["name"] for u in d["units"]] == ["tone"] * 2
    assert "knock out 2 units" in d["summary"]

    down = {i: rec.neurons.mlp(i).down.weight for i in (1, 2)}
    original = {i: w.detach().clone() for i, w in down.items()}
    cols = {1: [3], 2: [5, 9]}
    rec.apply_weight_edits()
    for i, ns in cols.items():
        assert not down[i][:, ns].any()  # the write columns are zero
        rest = [n for n in range(down[i].shape[1]) if n not in ns]
        assert torch.equal(down[i][:, rest], original[i][:, rest])
    lm2 = rec.anatomy.layers[2]
    assert {lm2.offset + lm2.n_heads + int(lm2.assign[n]) for n in (5, 9)} <= set(rec.anatomy.edited_units(rec.edits))
    rec.revert_weight_edits()
    for i in down:
        assert torch.equal(down[i], original[i])  # bit for bit


# ---- monitoring ------------------------------------------------------------------------------


def test_a_neuron_rule_reads_nz_and_a_direction_rule_reads_proj(tmp_path):
    def tape(zs):
        return [{"t": t, "req": 1, "ph": "gen", "text": f"w{t} ", "proj": {"probe": 0.0}, "nz": {"L2 N5": z}}
                for t, z in enumerate(zs)]
    zs = [0, 2.5, 3, 1, 4]
    m = Monitor("L2 N5", threshold=2, min_tokens=2).check()
    assert m.neuron and not Monitor("probe").neuron
    assert [(f["start"], f["end"], f["peak_z"]) for f in scan(m, tape(zs))] == [(1, 2, 3)]
    assert scan(Monitor("probe", threshold=1, min_tokens=1).check(), tape(zs)) == []  # a direction reads proj
    assert scan(m, [{k: v for k, v in r.items() if k != "nz"} for r in tape(zs)]) == []  # an old run has no nz

    calls = []
    bank = MonitorBank(tmp_path / "m.json")
    bank.on_change = lambda: calls.append(1)
    n = bank.add({"direction": "l2n5", "threshold": 2, "min_tokens": 2}, {"probe"})  # written canonically
    assert n.direction == "L2 N5" and n.describe()["kind"] == "neuron" and len(calls) == 1
    p = bank.add({"direction": "probe"}, {"probe"})
    assert p.describe()["kind"] == "direction" and len(calls) == 2
    dirs = [{"name": "probe", "kind": "contrast"}, {"name": "random_control", "kind": "random_control"}]
    rows = scan_run(bank, tape(zs), dirs)
    assert len(rows[0]["flags"]) == 1 and rows[0]["recorded"] == 5 and "control" not in rows[0]  # no control for a neuron
    assert rows[1]["flags"] == [] and rows[1]["recorded"] == 5
    assert MonitorBank(tmp_path / "m.json").list()[0]["direction"] == "L2 N5"  # saved

    with pytest.raises(ValueError):  # all of them or none
        bank.add_many([{"direction": "L1 N1"}, {"direction": "nope"}], {"probe"})
    assert [x["direction"] for x in bank.list()] == ["L2 N5", "probe"]
    assert [x["direction"] for x in MonitorBank(tmp_path / "m.json").list()] == ["L2 N5", "probe"]
    with pytest.raises(ValueError, match="at most"):
        bank.add_many([{"direction": f"L0 N{i}"} for i in range(MAX_MONITORS)], {"probe"})
    bank.update(n.id, {"enabled": False})
    bank.remove(n.id)
    assert len(calls) == 4


def test_the_monitor_endpoints_name_a_neuron_as_they_name_a_direction(rec, client):
    for bad in ("L99 N0", "L2 N999999", "nope", ""):
        assert client.post("/brainscan/monitors", json={"direction": bad}).status_code == 400
    m = client.post("/brainscan/monitors", json={"direction": "l2/n7", "threshold": 3}).json()
    assert m["direction"] == "L2 N7" and m["kind"] == "neuron" and m["rule"].startswith("L2 N7 z ≥ 3")
    assert rec.watch is not None and rec.watch.labels == ["L2 N7"]
    assert client.patch(f"/brainscan/monitors/{m['id']}", json={"enabled": False}).json()["enabled"] is False
    assert rec.watch is None
    assert client.delete(f"/brainscan/monitors/{m['id']}").json() == []


@pytest.mark.parametrize("chunk", [512, 4, 1])
def test_a_neuron_monitor_records_the_z_of_its_activation(rec, monkeypatch, chunk):
    """Prefill in one chunk, in chunks, and one token at a time (the single-token path decode uses)."""
    monkeypatch.setattr(rec.s, "prefill_chunk", chunk)
    items = [(1, 3), (2, 5), (2, 7)]
    downs = {i: rec.neurons.mlp(i).down for i in (1, 2)}
    base = {i: len(d._forward_pre_hooks) for i, d in downs.items()}
    assert rec.watch is None
    for layer, n in items:
        watch_neuron(rec, f"l{layer}/n{n}")
    w = rec.watch
    assert w.items == tuple(items) and w.labels == ["L1 N3", "L2 N5", "L2 N7"] and w.source == "calibration"
    assert [m.direction for m in rec.monitors.monitors()] == w.labels
    assert {i: len(d._forward_pre_hooks) - base[i] for i, d in downs.items()} == {1: 1, 2: 1}  # one hook per watched layer

    cal = rec.neurons.calibration()  # the mean and spread are those of the calibration texts, frozen
    frozen = {(i, n): (float(cal["mean"][i][n]), max(float(cal["std"][i][n]), nmod.MIN_STD)) for i, n in items}
    for item, (mean, std) in frozen.items():
        assert w.stats[item] == pytest.approx((mean, std))

    rows = {item: [] for item in items}  # a hook of the test's own, on the same input
    spies = [downs[i].register_forward_pre_hook(lambda m, a, k=(i, n): rows[k].append(a[0][0, :, k[1]].clone()))
             for i, n in items]
    try:
        res = rec.generate(GenParams(messages=ASK, max_tokens=6, temperature=0, run_hint=f"nz-{chunk}"))
    finally:
        for h in spies:
            h.remove()
    acts = {k: torch.cat(v) for k, v in rows.items()}  # one per position fed to the model, in order
    fed = len(acts[items[0]])
    assert fed == res.prompt_tokens + res.completion_tokens
    run = rec.get_run(f"nz-{chunk}")
    for r in run.tape:
        if r["pos"] >= fed:  # a stop token is recorded but never fed to the model
            assert "nz" not in r
            continue
        assert list(r["nz"]) == w.labels
        for (i, n), lab in zip(items, w.labels):
            mean, std = frozen[(i, n)]
            assert r["nz"][lab] == pytest.approx(float((acts[(i, n)][r["pos"]] - mean) / std), abs=1e-3)
        assert not set(w.labels) & set(r["proj"])  # the direction readouts are untouched
    assert any(r["ph"] == "gen" for r in run.tape if "nz" in r)
    assert [r.get("nz") for r in rec.store.read_tokens(run.id)] == [r.get("nz") for r in run.tape]  # stored too

    for m in rec.monitors.monitors():
        rec.monitors.remove(m.id)
    assert rec.watch is None and {i: len(d._forward_pre_hooks) for i, d in downs.items()} == base  # no hooks, no cost


def test_neuron_flags_follow_the_rule(rec):
    quiet = watch_neuron(rec, "L2 N5", threshold=20)
    rec.generate(GenParams(messages=ASK, max_tokens=12, temperature=0, run_hint="quiet"))
    run = rec.get_run("quiet")
    zs = [r["nz"]["L2 N5"] for r in run.tape if r["ph"] == "gen" and "nz" in r]
    assert len(zs) >= 3 and max(abs(z) for z in zs) < 20  # so a threshold of 20 provably never fires
    assert not [e for e in run.events if e["type"] == "monitor"] and scan(quiet, run.tape) == []

    rec.monitors.remove(quiet.id)
    mid = sorted(abs(z) for z in zs)[len(zs) // 2]
    m = watch_neuron(rec, "L2 N5", threshold=mid, sign="either")
    rec.generate(GenParams(messages=ASK, max_tokens=12, temperature=0, run_hint="loud"))  # the same words again
    run = rec.get_run("loud")
    flags = scan(m, run.tape)
    live = [e for e in run.events if e["type"] == "monitor"]
    passing = [r["t"] for r in run.tape if r["ph"] == "gen" and "nz" in r and abs(r["nz"]["L2 N5"]) >= mid]
    assert flags and len(passing) < len(zs)
    assert [f["t"] for f in flags] == [t for t in passing if t - 1 not in passing]  # each run of tokens past it, once
    assert [e["t"] for e in live] == [f["t"] for f in flags]
    assert all(e["data"]["direction"] == "L2 N5" and e["data"]["monitor"] == m.id for e in live)
    assert all(run.tape[e["t"]]["ph"] == "gen" for e in live) and run.meta["flags"] == len(flags)

    always = watch_neuron(rec, "L2 N5", threshold=0.01)  # a rule that fits any token: one flag over all the model's own
    rec.generate(GenParams(messages=ASK, max_tokens=12, temperature=0, run_hint="always"))
    run = rec.get_run("always")
    flags = scan(always, run.tape)
    assert len(flags) == 1 and flags[0]["tokens"] == sum(r["ph"] == "gen" and "nz" in r for r in run.tape) > 0


def test_direction_monitors_still_work_beside_neuron_monitors(rec):
    texts = (["I love this, it is wonderful.", "What a great day!"], ["I hate this, it is awful.", "What a terrible day."])
    rec.directions.build_contrast("tone", 1, *texts, "test probe")
    rec.directions.build_random("random_control", 1, texts[0] + texts[1], 0, "test control")
    tone = rec.monitors.add({"direction": "tone", "sign": "either", "threshold": 0.01, "min_tokens": 1}, set(rec.directions.dirs),
                            rec.neurons.check)
    assert rec.watch is None  # a direction needs no hooks
    neuron = watch_neuron(rec, "L2 N5")
    rec.generate(GenParams(messages=ASK, max_tokens=8, temperature=0, run_hint="both"))
    run = rec.get_run("both")
    live = {m.id: [e["t"] for e in run.events if e["type"] == "monitor" and e["data"]["monitor"] == m.id] for m in (tone, neuron)}
    assert live[tone.id] and live[tone.id] == [f["t"] for f in scan(tone, run.tape)]
    assert live[neuron.id] and live[neuron.id] == [f["t"] for f in scan(neuron, run.tape)]
    rows = {r["monitor"]["id"]: r for r in scan_run(rec.monitors, run.tape, rec.directions.describe())}
    assert rows[tone.id]["control"]["direction"] == "random_control" and "control" not in rows[neuron.id]
    assert "tone" in run.tape[0]["proj"] and "L2 N5" not in run.tape[0]["proj"]


def test_the_watch_follows_the_monitors(rec):
    down = rec.neurons.mlp(2).down
    base = len(down._forward_pre_hooks)
    assert rec.watch is None
    a = watch_neuron(rec, "L2 N5")
    first = rec.watch.key
    assert rec.sync_watch() and rec.watch.key == first  # nothing changed: the same watch, so the same graph key
    b = watch_neuron(rec, "L2 N7")
    assert rec.watch.items == ((2, 5), (2, 7)) and rec.watch.key != first
    assert len(down._forward_pre_hooks) == base + 1  # rebuilt, not stacked
    second = rec.watch.key
    rec.monitors.update(b.id, {"enabled": False})
    assert rec.watch.items == ((2, 5),) and rec.watch.key not in (first, second)
    rec.monitors.update(a.id, {"enabled": False})
    assert rec.watch is None and len(down._forward_pre_hooks) == base
    rec.monitors.update(a.id, {"enabled": True})
    assert rec.watch.items == ((2, 5),)
    rec.monitors.remove(a.id)
    assert rec.watch is None and len(down._forward_pre_hooks) == base
    off = watch_neuron(rec, "L2 N5", enabled=False)  # a monitor that is off costs nothing
    assert not off.enabled and rec.watch is None and len(down._forward_pre_hooks) == base


def test_the_watch_z_scores_against_the_index_when_there_is_one(rec):
    watch_neuron(rec, "L2 N5")
    assert rec.watch.source == "calibration"
    before = rec.watch.key
    rec.neuron_index.build(600)
    try:
        assert rec.sync_watch() and rec.watch.source == "index" and rec.watch.key != before
        data = rec.neuron_index.data
        mean, std = rec.watch.stats[(2, 5)]
        assert mean == pytest.approx(float(data["mean.2"][5])) and std == pytest.approx(max(float(data["std.2"][5]), nmod.MIN_STD))
    finally:
        rec.neuron_index.data = rec.neuron_index.meta = None


def test_the_decode_graph_key_follows_the_watch(rec):
    """The decode graphs bake the watch's hooks in, so their key has to change whenever the watched set
    does (and not otherwise). Graphs can't run on the CPU: a stand-in records what the recorder hands them."""
    class Graphs:
        def __init__(self):
            self.keys, self.buffers = [], []
            self.extras = None

        def step(self, token, cache, key):
            self.keys.append(key)
            self.buffers.append(self.extras() if self.extras else None)
            return None  # decode this step eagerly

    fake = Graphs()
    rec.knobs.save("tone", [(2, 5, 1.0), (1, 3, -0.5)])

    def request(run):
        fake.keys, fake.buffers = [], []
        rec.generate(GenParams(messages=ASK, max_tokens=4, temperature=0, run_hint=run))
        assert fake.keys and len(set(fake.keys)) == 1  # one key for the whole request
        return fake.keys[0], fake.buffers[0]

    rec.graphs = fake
    try:
        none, buffers = request("key-none")
        assert none == ((), None) and buffers is None
        watch_neuron(rec, "L2 N5")
        a, buffers = request("key-a")
        assert a == ((), rec.watch.key) and len(buffers) == 1 and buffers[0] is rec.watch.step
        assert request("key-a2")[0] == a  # the same watch is the same key: no recapture
        watch_neuron(rec, "L2 N7")
        b, buffers = request("key-b")
        assert b[1] != a[1] and b[1] == rec.watch.key and buffers[0] is rec.watch.step
        rec.steer_knob("tone")
        c, _ = request("key-c")
        assert c[0] == tuple(iv.id for iv in rec.tap.interventions) and len(c[0]) == 2 and c[1] == b[1]
        rec.unsteer_knob("tone")
        for m in rec.monitors.monitors():
            rec.monitors.remove(m.id)
        assert request("key-off") == (((), None), None)  # back to no watch: the plain key again
    finally:
        rec.graphs = None


# ---- the watch inside decode CUDA graphs (needs a GPU) -------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_the_watch_inside_decode_graphs_matches_eager_decoding():
    from transformers import AutoModelForCausalLM, LlamaConfig
    config = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
    subject = SimpleNamespace(model=model, layers=model.model.layers, n_layers=3, device=torch.device("cuda"))
    nm = SimpleNamespace(s=subject, mlp=lambda layer: SimpleNamespace(down=model.model.layers[layer].mlp.down_proj))
    tap = LayerTap(subject)
    tap.phase = "gen"
    graphs = DecodeGraphs(subject, tap)
    stats = {(0, 3): (0.0, 1.0), (1, 5): (0.1, 0.5), (1, 9): (-0.2, 2.0)}

    @torch.no_grad()
    def decode(graphs, watch, prompt, steps=8):
        if graphs is not None:
            graphs.extras = lambda: (watch.step,)
        key = ((), watch.key)
        with watch.recording():
            out = model(input_ids=torch.tensor([prompt], device="cuda"), use_cache=True)
            kv, logits = out.past_key_values, out.logits
            records = []
            for _ in range(steps):
                token = int(logits[0, -1].argmax())
                new = graphs.step(token, kv, key) if graphs is not None else None
                if new is None:
                    out = model(input_ids=torch.tensor([[token]], device="cuda"), past_key_values=kv, use_cache=True)
                    kv, new = out.past_key_values, out.logits
                logits = new.clone()
                records.append((token, logits, tap.stack(), watch.step.clone()))
        return records

    def assert_same(eager, fast):
        for (ta, la, ra, za), (tb, lb, rb, zb) in zip(eager, fast, strict=True):
            assert ta == tb and torch.equal(la, lb) and torch.equal(ra, rb) and torch.equal(za, zb)

    watch = nmod.NeuronWatch(nm, list(stats), stats, "test")
    for prompt in ([1, 5, 9, 3, 7], [2, 4, 6]):  # the second prompt runs on a new cache, same graphs
        eager = decode(None, watch, prompt)
        assert eager[0][3].abs().sum() > 0
        assert_same(eager, decode(graphs, watch, prompt))
    assert graphs.describe()["on"] and not graphs.rejected and graphs.captures == 1

    # Another watch has its own buffers and hooks: its key recaptures, and the new graphs refresh the new buffer.
    watch.remove()
    other = nmod.NeuronWatch(nm, [(1, 5)], stats, "test")
    assert other.key != watch.key
    assert_same(decode(None, other, [1, 5, 9, 3, 7]), decode(graphs, other, [1, 5, 9, 3, 7]))
    assert graphs.describe()["on"] and not graphs.rejected and graphs.captures == 2
    other.remove()


# ---- the endpoints -----------------------------------------------------------------------------


def test_endpoints(rec, client):
    assert client.get("/brainscan/knobs").json() == {"knobs": [], "error": None}
    one = [{"neuron": "L2 N5", "dose": 1}, {"neuron": "L1 N3", "dose": -0.5}]
    r = client.post("/brainscan/knobs", json={"name": "tone", "description": "test", "neurons": one})
    assert r.status_code == 200 and r.json()["name"] == "tone" and r.json()["active"] == []
    t = client.post("/brainscan/knobs", json={"name": "texty", "text": "L1 N3 +1, L2 N5 −0.5, L0 N7"}).json()
    assert [(n["neuron"], n["dose"]) for n in t["neurons"]] == [("L1 N3", 1.0), ("L2 N5", -0.5), ("L0 N7", 1.0)]
    assert [k["name"] for k in client.get("/brainscan/knobs").json()["knobs"]] == ["tone", "texty"]
    assert client.get("/brainscan/status").json()["knobs"] == ["tone", "texty"]
    for body in ({"name": "bad name", "neurons": one}, {"name": "x"}, {"name": "x", "neurons": []},
                 {"name": "x", "neurons": [{"neuron": "L2 N5", "dose": 0}]}, {"name": "x", "neurons": [{"neuron": "L99 N0"}]},
                 {"name": "x", "text": "no neurons here"}, {"name": "x", "neurons": one, "description": "d" * 301}):
        assert client.post("/brainscan/knobs", json=body).status_code == 400, body
    assert client.post("/brainscan/knobs", json=[1]).status_code == 400
    assert client.post("/brainscan/knobs", content=b"{nope").status_code == 400

    # steer: replaces, never stacks; a knob that is steering can't be deleted
    steers = client.post("/brainscan/knobs/tone/steer", json={}).json()
    assert len(steers) == 2 and {s["knob"] for s in steers} == {"tone"}
    knob = {k["name"]: k for k in client.get("/brainscan/knobs").json()["knobs"]}["tone"]
    assert sorted(knob["active"]) == sorted(s["id"] for s in steers)
    twice = client.post("/brainscan/knobs/tone/steer", json={"factor": 2, "phase": "gen"}).json()
    assert len(client.get("/brainscan/interventions").json()) == 2
    assert [t["value"] for t in twice] == pytest.approx([2 * s["value"] for s in steers]) and {t["phase"] for t in twice} == {"gen"}
    assert client.get("/brainscan/neurons").json()["steering"][0]["knob"] == "tone"
    assert client.delete("/brainscan/knobs/tone").status_code == 409
    for body in ({"factor": 99}, {"factor": "x"}, {"factor": None}, {"phase": "nope"}):
        assert client.post("/brainscan/knobs/tone/steer", json=body).status_code == 400, body
    assert client.post("/brainscan/knobs/nope/steer", json={}).status_code == 404
    assert client.delete("/brainscan/knobs/nope/steer").status_code == 404
    assert client.delete("/brainscan/knobs/tone/steer").json() == {"knob": "tone", "removed": 2}
    assert client.delete("/brainscan/knobs/tone/steer").json()["removed"] == 0
    assert client.get("/brainscan/interventions").json() == []

    # ablate: an edit that is added, not applied
    e = client.post("/brainscan/knobs/tone/ablate", json={}).json()
    assert e["kind"] == "units" and e["factor"] == 0 and e["note"] == "L2 N5, L1 N3" and [u["g"] for u in e["units"]] == [None, None]
    edits = client.get("/brainscan/workbench").json()
    assert [x["id"] for x in edits["edits"]] == [e["id"]] and edits["edits"][0]["applied"] is False and edits["applied"] == []
    half = client.post("/brainscan/knobs/tone/ablate", json={"factor": 0.5, "note": "half"}).json()
    assert half["factor"] == 0.5 and half["note"] == "half"
    for body in ({"factor": 2}, {"factor": -0.1}, {"factor": "x"}, {"factor": "nan"}):
        assert client.post("/brainscan/knobs/tone/ablate", json=body).status_code == 400, body
    assert client.post("/brainscan/knobs/nope/ablate", json={}).status_code == 404

    # monitor: one rule per neuron, no duplicates
    ms = client.post("/brainscan/knobs/tone/monitor", json={}).json()
    assert [m["direction"] for m in ms] == ["L2 N5", "L1 N3"] and {m["kind"] for m in ms} == {"neuron"}
    assert [m["label"] for m in ms] == ["tone · L2 N5", "tone · L1 N3"]
    assert {(m["threshold"], m["sign"], m["min_tokens"], m["phase"]) for m in ms} == {(4.0, "above", 2, "gen")}
    assert [m["id"] for m in client.post("/brainscan/knobs/tone/monitor", json={}).json()] == [m["id"] for m in ms]
    assert len(client.get("/brainscan/monitors").json()) == 2
    assert rec.watch.labels == ["L1 N3", "L2 N5"]
    loose = client.post("/brainscan/knobs/tone/monitor", json={"threshold": 3, "sign": "either", "min_tokens": 1}).json()
    assert len(loose) == 2 and len(client.get("/brainscan/monitors").json()) == 4
    assert client.post("/brainscan/knobs/tone/monitor", json={"sign": "up"}).status_code == 400
    assert len(client.get("/brainscan/monitors").json()) == 4  # none added
    assert client.post("/brainscan/knobs/nope/monitor", json={}).status_code == 404

    # sweep
    sw = client.post("/brainscan/knobs/tone/sweep", json={"prompt": "Hi", "factors": [0, 1], "max_tokens": 4})
    assert sw.status_code == 200 and [o["factor"] for o in sw.json()] == [0, 1] and sw.json()[1]["doses"] == [1.0, -0.5]
    for body in ({"prompt": " "}, {"prompt": "Hi", "factors": ["x"]}, {"prompt": "Hi", "max_tokens": "x"}):
        assert client.post("/brainscan/knobs/tone/sweep", json=body).status_code == 400, body
    assert client.post("/brainscan/knobs/nope/sweep", json={"prompt": "Hi"}).status_code == 404

    assert client.delete("/brainscan/knobs/tone").json() == {"deleted": "tone"}
    assert client.delete("/brainscan/knobs/tone").status_code == 404
    assert [k["name"] for k in client.get("/brainscan/knobs").json()["knobs"]] == ["texty"]


def test_a_busy_model_is_a_409(rec, client):
    rec.knobs.save("tone", [(2, 5, 1.0)])
    held, release = threading.Event(), threading.Event()

    def hold():
        with rec.lock:
            held.set()
            release.wait(10)
    thread = threading.Thread(target=hold)
    thread.start()
    assert held.wait(5)
    try:
        r = client.post("/brainscan/knobs/tone/sweep", json={"prompt": "Hi", "factors": [1], "max_tokens": 2})
    finally:
        release.set()
        thread.join()
    assert r.status_code == 409


def test_a_shape_clash_is_a_409(rec, client, tmp_path, monkeypatch):
    path = tmp_path / "k.json"
    path.write_text(json.dumps({"version": 1, "shape": [1, 1], "knobs": []}))
    monkeypatch.setattr(rec, "knobs", KnobBank(path, rec.neurons))
    r = client.post("/brainscan/knobs", json={"name": "tone", "neurons": [{"neuron": "L2 N5", "dose": 1}]})
    assert r.status_code == 409 and "Two models share the name" in r.json()["detail"]
    got = client.get("/brainscan/knobs").json()
    assert got["knobs"] == [] and "Two models share the name" in got["error"]
    assert client.delete("/brainscan/knobs/tone").status_code == 409
