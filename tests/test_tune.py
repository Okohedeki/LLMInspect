"""Tune, joined to Trace: a knockout made permanent as one edit, a chat re-run on the edited weights,
and behavior checks measured before and after an edit.

Runs a real CPU recorder on the tiny saved Llama in fp32, so a lesion in the weights and the same
knockout at run time agree to float precision.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import workbench
from brainscan.recorder.model import Subject
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tune")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    rec = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    rec.generate(GenParams(messages=[{"role": "user", "content": "Name a color."}], max_tokens=8, temperature=0,
                           run_hint="tune-run"))
    return rec, rec.get_run("tune-run")


@pytest.fixture()
def clean(tuned):
    """Each test starts and ends with no edits in the stack."""
    rec, _ = tuned
    yield
    rec.revert_weight_edits()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)


def last_prompt_slot(run):
    return max(r["t"] for r in run.tape if r["ph"] != "gen")


def top_units(rec, ex, kind, n=1):
    an = rec.anatomy
    us = [g for g in range(an.n_units) if an.units[g]["kind"] == kind]
    return sorted(us, key=lambda g: -abs(ex["contrib"][g]))[:n]


def test_a_knockout_made_permanent_matches_the_knockout(tuned, clean):
    rec, run = tuned
    t = last_prompt_slot(run)
    ex = rec.tracer.explain(run.id, t)
    units = top_units(rec, ex, "head", 2) + top_units(rec, ex, "cluster", 2)
    ko = rec.tracer.knockout(run.id, t, units)
    spec = rec.anatomy.lesion_units(units, note="from a knockout")
    edit = rec.add_weight_edit(**spec)
    assert edit.kind == "units" and len(edit.units) == 4
    assert edit.summary().startswith("knock out 4 units") and "from a knockout" in edit.summary()
    rec.apply_weight_edits()
    lesioned = rec.tracer.explain(run.id, t)
    assert lesioned["logit"] == pytest.approx(ko["after"]["logit"], abs=1e-4)
    assert lesioned["target"]["prob"] == pytest.approx(ko["after"]["prob"], abs=1e-4)
    assert set(units) <= set(rec.anatomy.edited_units(rec.edits))
    for g in units:
        assert abs(lesioned["contrib"][g]) < 1e-6
    rec.revert_weight_edits()
    assert rec.tracer.explain(run.id, t)["logit"] == pytest.approx(ex["logit"], abs=1e-6)


def test_a_scale_test_matches_the_same_scale_made_permanent(tuned, clean):
    """A knockout with a factor scales the units' writes at run time; the same factor kept as a
    weight edit reads the same logits. Factor 1 changes nothing."""
    rec, run = tuned
    t = last_prompt_slot(run)
    ex = rec.tracer.explain(run.id, t)
    units = top_units(rec, ex, "head", 2) + top_units(rec, ex, "cluster", 1)
    same = rec.tracer.knockout(run.id, t, units, factor=1.0)
    assert same["after"]["logit"] == pytest.approx(same["before"]["logit"], abs=1e-4)
    assert same["direct_effect"] == pytest.approx(0.0, abs=1e-6)
    half = rec.tracer.knockout(run.id, t, units, factor=0.5)
    assert half["factor"] == 0.5
    assert half["direct_effect"] == pytest.approx(0.5 * sum(ex["contrib"][g] for g in units), abs=1e-3)
    edit = rec.add_weight_edit(**rec.anatomy.lesion_units(units, 0.5, note="from a scale test"))
    assert edit.summary().startswith("scale ×0.5 3 units")
    rec.apply_weight_edits()
    scaled = rec.tracer.explain(run.id, t)
    assert scaled["logit"] == pytest.approx(half["after"]["logit"], abs=1e-4)
    rec.revert_weight_edits()
    for bad in (float("nan"), 9.0):
        with pytest.raises(ValueError):
            rec.tracer.knockout(run.id, t, units, factor=bad)


def test_replay_asks_the_chat_again_on_both_weights(tuned, clean):
    rec, run = tuned
    with pytest.raises(ValueError):
        workbench.replay(rec, run.id, max_tokens=8)  # nothing applied
    ex = rec.tracer.explain(run.id, last_prompt_slot(run))
    rec.add_weight_edit(**rec.anatomy.lesion_units(top_units(rec, ex, "head", 3) + top_units(rec, ex, "cluster", 3)))
    rec.apply_weight_edits()
    out = workbench.replay(rec, run.id, max_tokens=8)
    reply = next(m for m in run.messages if m.get("generated"))
    assert out["prompt"] == "Name a color." and out["turns"] == 1
    assert out["original"] == (reply["content"] or "").strip()  # greedy on the original weights, as recorded
    new = rec.get_run(out["edited_run"])
    assert new is not None and new.meta.get("weight_edits")  # recorded, and tagged with the edits
    assert out["edited"] == (next(m for m in new.messages if m.get("generated"))["content"] or "").strip()
    fd = out["first_diff"]
    if out["original"] != out["edited"]:
        assert fd is not None and fd["original_token"] != fd["edited_token"]
        assert fd["t"] is None or new.tape[fd["t"]]["ph"] == "gen"
    with pytest.raises(KeyError):
        workbench.replay(rec, "no-such-run")


def test_behavior_checks_before_and_after(tuned, clean):
    rec, run = tuned
    for c in workbench.list_checks(rec):
        workbench.remove_check(rec, c["id"])
    text = workbench.add_check(rec, {"prompt": "The color of the sky is", "expect": "blue"})
    assert text["target_text"] == rec.s.token_str(rec.s.tokenizer.encode(" blue", add_special_tokens=False)[0])
    t = last_prompt_slot(run)
    nxt = next(r["tok"] for r in run.tape if r["pos"] == run.tape[t]["pos"] + 1)
    token = workbench.add_check(rec, {"run": run.id, "t": t, "target": nxt})
    assert token["kind"] == "token" and len(workbench.list_checks(rec)) == 2
    with pytest.raises(ValueError):
        workbench.add_check(rec, {"prompt": "x", "expect": " "})

    before = {c["id"]: c for c in workbench.run_checks(rec)["checks"]}
    assert all(c["now"] and c["original"] is None and not c["error"] for c in before.values())
    ex = rec.tracer.explain(run.id, t)
    rec.add_weight_edit(**rec.anatomy.lesion_units(top_units(rec, ex, "head", 2)))
    rec.apply_weight_edits()
    after = workbench.run_checks(rec)
    assert after["edited"]
    for c in after["checks"]:
        assert c["original"]["prob"] == pytest.approx(before[c["id"]]["now"]["prob"], abs=1e-4)  # the edits lifted
    tok = next(c for c in after["checks"] if c["id"] == token["id"])
    assert tok["now"]["logit"] != pytest.approx(tok["original"]["logit"], abs=1e-6)  # the lesion moved it
    assert len(workbench.remove_check(rec, text["id"])) == 1


def test_tune_endpoints(tuned, clean):
    rec, run = tuned
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        t = last_prompt_slot(run)
        ex = rec.tracer.explain(run.id, t)
        units = top_units(rec, ex, "head", 2)
        r = client.post("/brainscan/anatomy/lesion", json={"units": units, "apply": True, "note": "test"})
        assert r.status_code == 200
        body = r.json()
        assert body["edit"]["kind"] == "units" and [u["g"] for u in body["edit"]["units"]] == sorted(units)
        assert body["workbench"]["applied"] and rec.edits.applied
        assert client.post("/brainscan/anatomy/lesion", json={"units": []}).status_code == 400
        assert client.post("/brainscan/anatomy/lesion", json={"units": [10 ** 6]}).status_code == 404
        r = client.post("/brainscan/workbench/replay", json={"run": run.id, "max_tokens": 4})
        assert r.status_code == 200 and r.json()["edited_run"]
        assert client.post("/brainscan/workbench/replay", json={"run": "nope"}).status_code == 404
        c = client.post("/brainscan/checks", json={"prompt": "Roses are", "expect": "red"}).json()
        assert client.get("/brainscan/checks").json()[-1]["id"] == c["id"]
        res = client.post("/brainscan/checks/run").json()
        assert res["edited"] and any(x["id"] == c["id"] and x["original"] for x in res["checks"])
        assert client.post("/brainscan/checks", json={"prompt": ""}).status_code == 400
        assert client.delete(f"/brainscan/checks/{c['id']}").status_code == 200
        assert client.delete("/brainscan/checks/nope").status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
