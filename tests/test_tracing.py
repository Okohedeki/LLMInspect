"""Tracing: per-unit attribution, neuron drill-down and attention patterns at a token of a run.

Runs a real CPU recorder on the tiny saved Llama in fp32, so the attribution identity (embedding +
every unit's write, projected through the frozen final norm, equals the logit) holds to float
precision, and attention patterns can be checked against an eager reference model.
"""

import threading
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder.chat import render
from brainscan.recorder.model import Subject
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore
from brainscan.recorder.tracing import Busy

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")

FIRST = [{"role": "user", "content": "Name a color."}]


@pytest.fixture(scope="module")
def traced(tmp_path_factory):
    """A two-request run: the second request extends the first, so its context spans both."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("trace")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    rec = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    r1 = rec.generate(GenParams(messages=FIRST, max_tokens=6, temperature=0, run_hint="trace-run"))
    second = FIRST + [{"role": "assistant", "content": r1.content or ""},
                      {"role": "user", "content": "Another one please."}]
    rec.generate(GenParams(messages=second, max_tokens=6, temperature=0, run_hint="trace-run"))
    return rec, rec.get_run("trace-run"), second


def last_prompt_slot(run, req):
    return max(r["t"] for r in run.tape if r["req"] == req and r["ph"] != "gen")


def test_attribution_adds_up_to_the_logit(traced):
    rec, run, _ = traced
    t = last_prompt_slot(run, 2)
    out = rec.tracer.explain(run.id, t)
    assert len(out["contrib"]) == rec.anatomy.n_units
    total = out["embed"] + sum(out["contrib"])
    assert abs(out["logit"] - total) < 1e-3 and abs(out["rest"]) < 1e-3
    # The target defaults to the token the model actually produced next, and its probability is
    # the one the recorder stored for that token while generating.
    nxt = next(r for r in run.tape if r["req"] == 2 and r["pos"] == out["pos"] + 1)
    assert out["target"]["id"] == nxt["tok"]
    assert out["target"]["prob"] == pytest.approx(nxt["p"], abs=2e-3)
    # Per-layer sums partition the same total.
    assert sum(h + c for h, c in out["layers"]) == pytest.approx(sum(out["contrib"]), abs=1e-2)


def test_custom_target_and_context_across_requests(traced):
    rec, run, second = traced
    t = last_prompt_slot(run, 2)
    ids, pos, req = rec.tracer.context(run, t)
    rendered = render(rec.s.tokenizer, second, None, add_generation_prompt=True).ids
    assert req == 2 and ids == rendered[:pos + 1]  # positions recorded by request 1 fill the prefix
    alt = rec.tracer.explain(run.id, t)["alternatives"][1]
    out = rec.tracer.explain(run.id, t, target=str(alt["id"]))
    assert out["target"]["id"] == alt["id"] and out["target"]["rank"] == 2


def test_neurons_add_up_to_their_cluster(traced):
    rec, run, _ = traced
    t = last_prompt_slot(run, 1)
    out = rec.tracer.explain(run.id, t)
    an = rec.anatomy
    clusters = [g for g in range(an.n_units) if an.units[g]["kind"] == "cluster"]
    g = max(clusters, key=lambda u: abs(out["contrib"][u]))
    nn = rec.tracer.neurons(run.id, t, g)
    assert nn["contrib"] == pytest.approx(out["contrib"][g], abs=1e-3)
    assert len(nn["neurons"]) == an.units[g]["size"]
    assert abs(nn["neurons"][0]["contrib"]) >= abs(nn["neurons"][-1]["contrib"])  # sorted by impact
    # "active" is the fewest neurons carrying 90% of the cluster's write.
    writes = sorted((n["write"] for n in nn["neurons"]), reverse=True)
    assert 1 <= nn["active"] <= len(writes)
    assert sum(writes[:nn["active"]]) >= 0.9 * sum(writes) - 1e-3
    assert nn["active"] == 1 or sum(writes[:nn["active"] - 1]) < 0.9 * sum(writes) + 1e-3
    with pytest.raises(ValueError):
        rec.tracer.neurons(run.id, t, 0)  # unit 0 is a head


def test_attention_matches_an_eager_reference(traced):
    rec, run, _ = traced
    t = last_prompt_slot(run, 2)
    ids, _, _ = rec.tracer.context(run, t)
    ref = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), dtype=torch.float32,
                                               attn_implementation="eager").eval()
    with torch.no_grad():
        attentions = ref(input_ids=torch.tensor([ids]), output_attentions=True).attentions
    lm = rec.anatomy.layers[2]
    out = rec.tracer.attention(run.id, t, lm.offset + 1)  # layer 2, head 1
    assert out["available"] and len(out["weights"]) == len(ids)
    assert sum(out["weights"]) == pytest.approx(1.0, abs=1e-3)
    want = attentions[2][0, 1, -1]
    assert torch.allclose(torch.tensor(out["weights"]), want, atol=1e-4)
    assert out["tokens"][-1] == rec.s.token_str(ids[-1])


def test_tracing_waits_for_generation(traced):
    rec, run, _ = traced
    held, release = threading.Event(), threading.Event()

    def hold():
        with rec.lock:
            held.set()
            release.wait(5)

    th = threading.Thread(target=hold)
    th.start()
    held.wait(5)
    rec.tracer._cache.clear()  # a cached capture would not need the model
    try:
        with pytest.raises(Busy):
            rec.tracer.explain(run.id, 1)
    finally:
        release.set()
        th.join()


def test_original_weights_trace_matches_the_unedited_model(traced):
    """With a head lesioned, weights=original reproduces the pre-edit trace exactly, the current trace
    shows the lesioned head at zero, and the edit is back in place (same weights, same version)."""
    rec, run, _ = traced
    t = last_prompt_slot(run, 2)
    before = rec.tracer.explain(run.id, t)
    an = rec.anatomy
    heads = [g for g in range(an.n_units) if an.units[g]["kind"] == "head"]
    g = max(heads, key=lambda u: abs(before["contrib"][u]))
    lm, _, h = an.locate(g)
    cluster = max((u for u in range(an.n_units) if an.units[u]["kind"] == "cluster" and an.locate(u)[0].index > lm.index),
                  key=lambda u: abs(before["contrib"][u]))
    nn_before = rec.tracer.neurons(run.id, t, cluster)
    edit = rec.add_weight_edit("scale", layers=str(lm.index), targets=["head"], index=h, factor=0.0)
    rec.apply_weight_edits()
    try:
        weight = lm.heads[0].weight.detach().clone()
        version = rec.edits.version
        cur = rec.tracer.explain(run.id, t)
        assert cur["model_weights"] == "current" and cur["weights_edited"]
        assert abs(cur["contrib"][g]) < 1e-6  # its output projection is zeroed
        assert cur["target"]["id"] == before["target"]["id"]

        rec.tracer._cache.clear()  # re-run the forward pass itself with the edits lifted, not a cached one
        orig = rec.tracer.explain(run.id, t, weights="original")
        assert orig["model_weights"] == "original" and orig["weights_edited"]  # edit status is read outside the lift
        assert orig["target"] == before["target"]
        assert torch.allclose(torch.tensor(orig["contrib"]), torch.tensor(before["contrib"]), atol=1e-4)
        assert orig["logit"] == pytest.approx(before["logit"], abs=1e-4)
        nn_orig = rec.tracer.neurons(run.id, t, cluster, weights="original")
        assert nn_orig["contrib"] == pytest.approx(nn_before["contrib"], abs=1e-4)
        assert nn_orig["active"] == nn_before["active"]
        assert rec.tracer.attention(run.id, t, g, weights="original")["model_weights"] == "original"

        # The edit is back exactly, and caches keyed by the edit version (the KV prefix) stay valid.
        assert torch.equal(lm.heads[0].weight, weight) and rec.edits.version == version
        assert rec.edits.applied == [edit.id]
        assert abs(rec.tracer.explain(run.id, t)["contrib"][g]) < 1e-6
        with pytest.raises(ValueError):
            rec.tracer.explain(run.id, t, weights="edited")
    finally:
        rec.revert_weight_edits()
        rec.edits.remove(edit.id)
    assert torch.allclose(torch.tensor(rec.tracer.explain(run.id, t)["contrib"]), torch.tensor(before["contrib"]), atol=1e-4)


def test_producer_explains_the_clicked_token(traced):
    """Tracing the producer slot of a generated token explains that token, in its own context; the
    first token of a later request maps back to an earlier request that saw the same prefix."""
    rec, run, _ = traced
    gen = [r for r in run.tape if r["ph"] == "gen" and r["req"] == 2]
    g = gen[2]
    src = rec.tracer.producer(run.id, g["t"])
    s = run.tape[src["t"]]
    assert s["pos"] == g["pos"] - 1 and src["target"]["id"] == g["tok"]
    ex = rec.tracer.explain(run.id, src["t"], target=str(src["target"]["id"]))
    assert ex["target"]["id"] == g["tok"]
    assert ex["target"]["prob"] == pytest.approx(g["p"], abs=2e-3)  # what the model gave it while generating
    first_new = min((r for r in run.tape if r["req"] == 2 and r["ph"] != "replay"), key=lambda r: r["t"])
    if first_new["pos"] > 0:
        src2 = rec.tracer.producer(run.id, first_new["t"])
        assert run.tape[src2["t"]]["pos"] == first_new["pos"] - 1
        ctx2, _, _ = rec.tracer.context(run, src2["t"])
        ctx_new, _, _ = rec.tracer.context(run, first_new["t"])
        assert ctx2 == ctx_new[:-1]  # the same tokens before it
    first = next(r for r in run.tape if r["pos"] == 0)
    with pytest.raises(ValueError):
        rec.tracer.producer(run.id, first["t"])


def test_knockout_matches_the_same_lesion_in_the_weights(traced):
    """Knocking out a head and a cluster at run time gives exactly the logits of the model with those
    units lesioned in its weights, and leaves the model as it was."""
    rec, run, _ = traced
    t = last_prompt_slot(run, 2)
    base = rec.tracer.explain(run.id, t)
    an = rec.anatomy
    by_kind = lambda k: max((u for u in range(an.n_units) if an.units[u]["kind"] == k), key=lambda u: abs(base["contrib"][u]))
    g, c = by_kind("head"), by_kind("cluster")
    ko = rec.tracer.knockout(run.id, t, [c, g])
    assert [u["g"] for u in ko["units"]] == sorted([g, c])
    assert ko["target"]["id"] == base["target"]["id"]
    assert ko["before"]["logit"] == pytest.approx(base["logit"], abs=1e-4)
    assert ko["direct_effect"] == pytest.approx(base["contrib"][g] + base["contrib"][c], abs=1e-3)
    assert ko["total_effect"] == pytest.approx(ko["before"]["logit"] - ko["after"]["logit"], abs=1e-3)

    lm_h, _, h = an.locate(g)
    lm_c, _, i = an.locate(c)
    neurons = (lm_c.assign == i).nonzero().flatten().tolist()
    edits = [rec.add_weight_edit("scale", layers=str(lm_h.index), targets=["head"], index=h, factor=0.0),
             rec.add_weight_edit("scale", layers=str(lm_c.index), targets=["neuron"], index=neurons, factor=0.0)]
    rec.apply_weight_edits()
    try:
        lesioned = rec.tracer.explain(run.id, t)
        assert ko["after"]["logit"] == pytest.approx(lesioned["logit"], abs=1e-4)
        assert ko["after"]["prob"] == pytest.approx(lesioned["target"]["prob"], abs=1e-4)
    finally:
        rec.revert_weight_edits()
        for e in edits:
            rec.edits.remove(e.id)
    again = rec.tracer.explain(run.id, t)
    assert again["logit"] == pytest.approx(base["logit"], abs=1e-6)  # the hooks are gone
    with pytest.raises(ValueError):
        rec.tracer.knockout(run.id, t, [])
    with pytest.raises(KeyError):
        rec.tracer.knockout(run.id, t, [10 ** 6])


def test_trace_endpoints(traced):
    rec, run, _ = traced
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        t = last_prompt_slot(run, 1)
        r = client.get(f"/brainscan/runs/{run.id}/tokens/{t}/explain")
        assert r.status_code == 200 and len(r.json()["contrib"]) == rec.anatomy.n_units
        assert client.get(f"/brainscan/runs/{run.id}/tokens/{t}/attention", params={"unit": 0}).json()["available"]
        cluster = rec.anatomy.layers[0].offset + rec.anatomy.layers[0].n_heads
        assert client.get(f"/brainscan/runs/{run.id}/tokens/{t}/attention", params={"unit": cluster}).status_code == 400
        assert client.get(f"/brainscan/runs/{run.id}/tokens/{t}/neurons", params={"unit": cluster}).status_code == 200
        assert client.get(f"/brainscan/runs/{run.id}/tokens/{t}/explain", params={"weights": "original"}).json()["model_weights"] == "original"
        assert client.get(f"/brainscan/runs/{run.id}/tokens/{t}/explain", params={"weights": "bogus"}).status_code == 400
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": [0, 1]}).json()["after"]["logit"] is not None
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": "0"}).status_code == 400
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": []}).status_code == 400
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": [0], "factor": 0.5}).json()["factor"] == 0.5
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": [0], "factor": "x"}).status_code == 400
        assert client.post(f"/brainscan/runs/{run.id}/tokens/{t}/knockout", json={"units": [0], "factor": 40}).status_code == 400
        assert client.get(f"/brainscan/runs/nope/tokens/{t}/explain").status_code == 404
        assert client.get(f"/brainscan/runs/{run.id}/tokens/99999/explain").status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
