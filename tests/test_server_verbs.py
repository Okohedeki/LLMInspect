"""The verb endpoints, on a stub recorder holding the tiny test model: a step runs under the model's lock, and a
change it returns is kept as a delta edit, unapplied."""

import threading
from types import SimpleNamespace

import torch
from fastapi.testclient import TestClient

import verbkit
from llminspect import server
from llminspect.recorder.edits import EditStack


def stub(monkeypatch):
    model = verbkit.model()
    s = SimpleNamespace(model=model, tokenizer=verbkit.tokenizer(), layers=model.model.layers, n_layers=4, d_model=64,
                        device=torch.device("cpu"), spec="tiny")
    rec = SimpleNamespace(s=s, lock=threading.RLock(), edits=EditStack(s))
    monkeypatch.setitem(server.state, "status", "ready")
    monkeypatch.setitem(server.state, "recorder", rec)
    return rec, TestClient(server.app)


def test_verbs_are_listed_with_their_steps(monkeypatch):
    _, c = stub(monkeypatch)
    got = c.get("/llminspect/verbs").json()
    assert {"embed", "verify"} <= set(got["sign"]["steps"]) and got["sign"]["about"]


def test_a_step_that_changes_weights_is_kept_as_an_unapplied_edit(monkeypatch):
    rec, c = stub(monkeypatch)
    before = {n: p.detach().clone() for n, p in rec.s.model.named_parameters()}
    r = c.post("/llminspect/verbs/sign/embed", json={"key": "owner-key"})
    assert r.status_code == 200, r.text
    edit = r.json()["edit"]
    assert edit["kind"] == "delta" and edit["delta"]["verb"] == "sign" and rec.edits.applied == []
    assert all(torch.equal(p, before[n]) for n, p in rec.s.model.named_parameters())
    rec.edits.apply()
    assert c.post("/llminspect/verbs/sign/verify", json={"key": "owner-key"}).json()["z"] > 8


def test_bad_names_and_bodies_are_plain_errors(monkeypatch):
    _, c = stub(monkeypatch)
    assert c.post("/llminspect/verbs/nope/x", json={}).status_code == 404
    assert c.post("/llminspect/verbs/sign/nope", json={}).status_code == 404
    assert c.post("/llminspect/verbs/sign/embed", json={}).status_code == 400
