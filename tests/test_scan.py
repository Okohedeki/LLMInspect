"""The deploy scan: integrity and manifests, the whole report on a candidate with a planted trigger token, the
printable page, an identical candidate, and the endpoints, on the tiny saved Llama in fp32."""

import hashlib
import json
import time

import pytest
import torch
from fastapi.testclient import TestClient

from llminspect import server
from llminspect.config import Settings
from llminspect.recorder import scan
from llminspect.recorder.model import Subject
from llminspect.recorder.session import Recorder
from llminspect.recorder.store import RunStore
from test_checkpoints import MODEL_DIR, TOKEN, base, snapshot, targeted, write_candidate  # noqa: F401 (base: fixture)

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("scan")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    return Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)


@pytest.fixture(autouse=True)
def clean(rec, monkeypatch):
    monkeypatch.setattr(scan, "MAX_TOKENS", 8)  # short answers keep the tiny model's scan quick
    yield
    rec.edits.revert()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)
    rec.checkpoints.job = {"status": "none", "step": None, "progress": 0.0, "error": None}
    rec.scan.job = {"status": "none", "step": None, "progress": 0.0, "error": None, "report": None}


@pytest.fixture(scope="module")
def planted(base, tmp_path_factory):
    """The targeted candidate plus one embedding row moved far: a token that changes how the model reads text."""
    t = targeted(base)
    g = torch.Generator().manual_seed(5)
    t["model.embed_tokens.weight"][TOKEN] += 4.0 * torch.randn(t["model.embed_tokens.weight"].shape[1], generator=g)
    return write_candidate(tmp_path_factory.mktemp("planted"), t)


def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifests_are_read_as_json_or_sha256sum_lines():
    h = "a" * 64
    assert scan.parse_manifest(json.dumps({"model.safetensors": h.upper()})) == {"model.safetensors": h}
    assert scan.parse_manifest(json.dumps({"files": {"sub/model.safetensors": h}})) == {"model.safetensors": h}
    assert scan.parse_manifest(f"{h}  model.safetensors\n{h} *config.json\nnot a line\n") == {
        "model.safetensors": h, "config.json": h}
    assert scan.parse_manifest("") == {}


def test_integrity_hashes_every_file_and_checks_the_manifest(planted, tmp_path):
    rep = scan.integrity(planted)
    by = {f["file"]: f for f in rep["files"]}
    assert by["model.safetensors"]["kind"] == "safetensors" and by["config.json"]["kind"] == "config"
    assert by["model.safetensors"]["sha256"] == sha(planted / "model.safetensors")
    assert rep["manifest"] is None and rep["notes"] == [] and rep["weights_format"] == ["safetensors"]

    good = {"model.safetensors": sha(planted / "model.safetensors"), "config.json": "0" * 64, "gone.bin": "1" * 64}
    m = scan.integrity(planted, good)["manifest"]
    assert m["match"] == ["model.safetensors"] and m["mismatch"] == ["config.json"] and m["missing"] == ["gone.bin"]
    assert m["unlisted_weights"] == []

    (planted / "SHA256SUMS").write_text(f"{sha(planted / 'model.safetensors')}  model.safetensors\n")
    try:
        found = scan.integrity(planted)["manifest"]
        assert found["source"] == "SHA256SUMS" and found["match"] == ["model.safetensors"]
    finally:
        (planted / "SHA256SUMS").unlink()

    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "pytorch_model.bin").write_bytes(b"not loaded")
    (odd / "modeling_custom.py").write_text("print('never run')\n")
    notes = scan.integrity(odd)["notes"]
    assert any("Pickle" in n for n in notes) and any("Code" in n for n in notes)


def test_the_scan_reports_every_section_and_leaves_the_stack_as_it_was(rec, planted):
    before = snapshot(rec.s.model)
    canary = "canary-7Q3X-9921-HZKP"
    report = rec.scan.run(str(planted), [canary], None, scan.SECTIONS)
    assert all(k in report for k in ("integrity", "weights", "behavior", "refusals", "triage", "memorization", "limitations"))
    assert not any("verdict" in k for k in report) and report["limitations"]
    assert rec.edits.applied == [] and all(torch.equal(before[n], p) for n, p in snapshot(rec.s.model).items())
    assert rec.checkpoints.edit() is not None and report["edit"] == rec.checkpoints.edit().id  # kept, not applied

    w = report["weights"]
    assert w["counts"]["changed"] == 4 and w["tokens"]["embed"][0]["token_id"] == TOKEN
    b = report["behavior"]
    assert len(b["rows"]) == len(scan.NEUTRAL_PROMPTS) and b["metrics"]["kl_per_token"] >= 0
    r = report["refusals"]
    assert r["probes"] == len(scan.REFUSAL_PROBES) and r["still_refused"] <= r["base_refused"]

    tri = report["triage"]
    assert tri["drift"] > 0 and tri["neurons"]
    by_why = {x["why"] for x in tri["neurons"]}
    assert "activation" in by_why
    tested = [x for x in tri["neurons"] if "knockout" in x]
    assert tested and all({"fall", "control_max"} <= set(x["knockout"]) for x in tested)
    weighted = next(x for x in tri["neurons"] if x["neuron"] == "L2 N7")  # the neuron whose write was changed
    assert weighted.get("weight_rel") and "knockout" in weighted
    top = tri["triggers"]["tokens"][0]
    assert top["token_id"] == TOKEN and top["z"] > 4  # the planted row stands out against random tokens
    assert any(s["section"] == "triage" and "shifts the candidate" in s["text"] for s in report["signals"])

    mem = report["memorization"]["canaries"][0]
    assert mem["canary"] == canary and 1 <= mem["candidate"]["rank"] <= scan.CANARY_CONTROLS + 1

    page = rec.scan.html(report["id"])
    for section in ("integrity", "weights", "behavior", "refusals", "triage", "memorization", "limitations"):
        assert f'id="{section}"' in page
    assert "<script" not in page and "http" not in page.split("<body>")[1].split("Generated by")[0]
    assert rec.scan.reports()[0]["id"] == report["id"]
    assert rec.scan.report(report["id"])["id"] == report["id"]


def test_applied_edits_come_back_after_a_scan(rec, planted):
    rec.edits.add("scale", "1", ["head"], index=0, factor=0.5)
    rec.edits.apply()
    edited = snapshot(rec.s.model)
    applied = list(rec.edits.applied)
    rec.scan.run(str(planted), [], None, ["weights", "behavior"])
    assert rec.edits.applied == applied and all(torch.equal(edited[n], p) for n, p in snapshot(rec.s.model).items())


def test_an_identical_candidate_measures_nothing(rec, base, tmp_path):
    folder = write_candidate(tmp_path, base, "same")
    report = rec.scan.run(str(folder), [], None, scan.SECTIONS)
    assert report["identical"] is True and "behavior" not in report and report["weights"]["summary"]["pattern"] == "identical"
    assert any("identical" in s["text"] for s in report["signals"])


def test_a_scan_refuses_while_another_checkpoint_is_applied(rec, planted, base, tmp_path):
    other = write_candidate(tmp_path, targeted(base), "other")
    rec.scan.run(str(other), [], None, ["weights"])
    rec.edits.apply()
    with pytest.raises(ValueError, match="another checkpoint is applied"):
        rec.scan.start({"spec": str(planted)})
    with pytest.raises(ValueError, match="at most"):
        rec.scan.start({"spec": str(planted), "canaries": ["x"] * (scan.MAX_CANARIES + 1)})


def test_the_endpoints(rec, planted):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        assert client.get("/llminspect/scan").json()["status"] == "none"
        assert client.post("/llminspect/scan", json={"spec": ""}).status_code == 400
        assert client.post("/llminspect/scan", json={"spec": str(planted / "nope")}).status_code == 404
        started = client.post("/llminspect/scan", json={"spec": str(planted), "sections": ["integrity", "weights"]})
        assert started.status_code == 200 and started.json()["status"] == "running"
        for _ in range(600):
            st = client.get("/llminspect/scan").json()
            if st["status"] != "running":
                break
            time.sleep(0.1)
        assert st["status"] == "ready", st
        rid = st["report"]["id"]
        assert client.get("/llminspect/scan/reports").json()[0]["id"] == rid
        assert client.get(f"/llminspect/scan/reports/{rid}").json()["id"] == rid
        page = client.get(f"/llminspect/scan/reports/{rid}/html")
        assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
        assert client.get("/llminspect/scan/reports/nope").status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
