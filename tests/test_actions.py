"""Actions: forget a fact. Target ranks the units behind an answer, try and measure knock them out
on typed prompts, and keeping the result as a weight edit gives what was measured.

Runs a real CPU recorder on the tiny saved Llama in fp32, so a knockout at run time and the same
units zeroed in the weights agree to float precision.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import actions, workbench
from brainscan.recorder.model import Subject
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("actions")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    r = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    r.generate(GenParams(messages=[{"role": "user", "content": "Name a color."}], max_tokens=8, temperature=0,
                         run_hint="act-run"))
    return r


@pytest.fixture()
def clean(rec):
    yield
    rec.revert_weight_edits()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)
    for c in workbench.list_checks(rec):
        workbench.remove_check(rec, c["id"])


def run_fact(rec):
    """The word the model wrote right after the prompt of the recorded run."""
    run = rec.get_run("act-run")
    t = max(r["t"] for r in run.tape if r["ph"] != "gen")
    target = next(r["tok"] for r in run.tape if r["t"] == t + 1)
    return run, t, {"run": run.id, "t": t, "target": target}


def plain_prob(rec, ids, target):
    with rec.tracer._model():
        return torch.softmax(rec.tracer.last_logits(ids), -1)[target].item()


def test_target_ranks_units_and_matches_the_run_knockout(rec, clean):
    run, t, body = run_fact(rec)
    res = actions.forget_target(rec, body | {"checks": False})
    # No facts to keep: only the direct ranking.
    assert res["rankings"]["specific"] is None and res["ranking"] == "direct" and res["keep"] == []
    r = res["rankings"]["direct"]
    directs = [u["direct"] for u in r["units"]]
    assert directs == sorted(directs, reverse=True) and all(d > 0 for d in directs)
    assert len(r["units"]) <= actions.CANDIDATES and "keep" not in r["units"][0]
    ex = rec.tracer.explain(run.id, t, target=str(body["target"]))
    assert res["fact"]["prob"] == pytest.approx(ex["target"]["prob"], abs=1e-5)
    ks = [p["k"] for p in r["curve"]]
    assert ks == sorted(ks) and ks[0] == 1 and set(ks) <= set(actions.CURVE)
    # The typed-prompt path and the run's own knockout agree on the same units.
    for p in r["curve"][:3]:
        units = [u["g"] for u in r["units"][:p["k"]]]
        ko = rec.tracer.knockout(run.id, t, units, target=str(body["target"]))
        assert p["prob"] == pytest.approx(ko["after"]["prob"], abs=1e-5)
        assert p["rank"] == ko["after"]["rank"]
        assert p["keep"] == [] and p["broken"] == 0
    assert r["suggested"] in ks
    if r["flipped"]:
        assert next(p for p in r["curve"] if p["k"] == r["suggested"])["gone"]
    assert res["fact"]["first"] == (res["fact"]["rank"] == 1)


KEEP = "Grass is → green\nThe sky is → blue\nOne, two, → three"


def keep_facts(rec):
    s = rec.s
    return [(workbench.prompt_ids(s, lead), workbench.answer_token(s, lead, answer))
            for lead, answer in (actions._pair(line) for line in KEEP.splitlines())]


def read(rec, ids, target, units=None):
    """The reading _read gives, plus the unrounded probability the greedy pass's guard compares."""
    logits = rec.tracer.last_logits(ids, units)
    return actions._read(rec.s, logits, target) | {"p": float(torch.softmax(logits, -1)[target])}


def spared(before, after):
    return actions.spared(before, after, before["p"], after["p"])


def effects(rec, f, keep):
    """Each unit's direct effect on the fact, the most it pushes any fact to keep, and the order the
    greedy pass walks the units in."""
    tr = rec.tracer
    c = tr.attribute(tr.capture_ids(f.ids), f.target)[0].cpu().numpy()
    push = np.maximum(0, np.max([tr.attribute(tr.capture_ids(ids), tid)[0].cpu().numpy() for ids, tid in keep], 0))
    score = c - actions.SPECIFIC_WEIGHT * push
    return c, push, [int(g) for g in np.argsort(-score) if score[g] > 0][:actions.POOL]


def test_specific_ranking_attributes_the_facts_to_keep(rec, clean, monkeypatch):
    _, _, body = run_fact(rec)
    tr, f = rec.tracer, actions.fact(rec, body)
    attributed = []
    attribute = tr.attribute
    monkeypatch.setattr(tr, "attribute", lambda cap, tid: attributed.append((tuple(cap.ids), tid)) or attribute(cap, tid))
    res = actions.forget_target(rec, body | {"keep": KEEP, "checks": False})
    keep = keep_facts(rec)
    # The fact, then each fact to keep, attributed at its own lead-in's last token.
    assert attributed == [(tuple(f.ids), f.target)] + [(tuple(ids), tid) for ids, tid in keep]
    assert [k["label"] for k in res["keep"]] == ["Grass is → green", "The sky is → blue", "One, two, → three"]
    with tr._model():
        for k, (ids, tid) in zip(res["keep"], keep):
            assert k["prob"] == pytest.approx(read(rec, ids, tid)["prob"], abs=1e-5)
    # A unit's push toward the facts to keep is the most it pushes any of them, never below zero.
    monkeypatch.undo()
    c, push, pool = effects(rec, f, keep)
    spec, direct = res["rankings"]["specific"], res["rankings"]["direct"]
    for u in direct["units"] + spec["units"] + spec["skipped"]:
        assert u["direct"] == pytest.approx(c[u["g"]], abs=1e-3) and u["keep"] == pytest.approx(push[u["g"]], abs=1e-3)
    assert push.min() >= 0 and push.max() > 0
    # The greedy pass walks the units in order of effect minus push, taking or skipping each.
    taken, skipped = [u["g"] for u in spec["units"]], {u["g"] for u in spec["skipped"]}
    walked = sorted(taken + list(skipped), key=pool.index)
    assert walked == pool[:len(walked)] and spec["pool"] == len(pool)
    assert taken == [g for g in walked if g not in skipped]
    assert all(c[g] - actions.SPECIFIC_WEIGHT * push[g] > 0 for g in walked)


def test_specific_ranking_spares_the_facts_to_keep(rec, clean, monkeypatch):
    _, _, body = run_fact(rec)
    f, keep = actions.fact(rec, body), keep_facts(rec)
    # A strict guard makes the greedy pass skip units on the tiny model too.
    monkeypatch.setattr(actions, "GUARD", 0.97)
    res = actions.forget_target(rec, body | {"keep": KEEP, "checks": False})
    spec = res["rankings"]["specific"]
    assert spec["guard"] == 0.97 and spec["skipped"]
    units = [u["g"] for u in spec["units"]]
    _, _, pool = effects(rec, f, keep)
    with rec.tracer._model():
        befores = [read(rec, ids, tid) for ids, tid in keep]
        fact_before = read(rec, f.ids, f.target)
        for p in spec["curve"]:
            # Every point reads the fact and the facts to keep with the first k units knocked out.
            assert p["prob"] == pytest.approx(read(rec, f.ids, f.target, units[:p["k"]])["prob"], abs=1e-5)
            for b, r, (ids, tid) in zip(befores, p["keep"], keep):
                now = read(rec, ids, tid, units[:p["k"]])
                assert r["prob"] == pytest.approx(now["prob"], abs=1e-5) and r["rank"] == now["rank"]
                assert spared(b, now) and now["p"] >= 0.97 * b["p"]
            assert p["broken"] == 0 and p["gone"] == actions.gone(fact_before, p)
        # Each skipped unit hurt a fact to keep when added to the units taken before it.
        for u in spec["skipped"][:3]:
            taken = [x for x in units if pool.index(x) < pool.index(u["g"])]
            assert not all(spared(b, read(rec, ids, tid, taken + [u["g"]])) for b, (ids, tid) in zip(befores, keep))
    ks = [p["k"] for p in spec["curve"]]
    assert ks == sorted(ks) and all(k in actions.CURVE or k == spec["suggested"] for k in ks)
    if spec["flipped"]:
        # The exact smallest set: one fewer unit still gives the answer.
        assert spec["suggested"] in ks and next(p for p in spec["curve"] if p["k"] == spec["suggested"])["gone"]
        with rec.tracer._model():
            assert not actions.gone(fact_before, read(rec, f.ids, f.target, units[:spec["suggested"] - 1] or None))
    assert res["ranking"] == ("specific" if spec["flipped"] else "direct")
    # The direct curve reads the facts to keep at every point too.
    assert all(len(p["keep"]) == len(keep) for p in res["rankings"]["direct"]["curve"])


LEADS = "Violets are\nRoses are"


def ways_of_asking(rec, f):
    """The fact's lead-in, the typed LEADS, and a saved check asking for the same answer, as
    (ids, target) in the order Target and Measure list them."""
    s = rec.s
    checks = workbench.list_checks(rec)
    checks.append({"kind": "text", "prompt": "The sky is", "expect": "x", "chat": False, "target": f.target,
                   "target_text": "x", "label": "The sky is → x", "id": "samefact"})
    workbench._save_checks(rec, checks)
    typed = [(workbench.prompt_ids(s, line), workbench.answer_token(s, line, f.answer)) for line in LEADS.splitlines()]
    return [(f.ids, f.target)] + typed + [(workbench.prompt_ids(s, "The sky is"), f.target)]


def test_target_ranks_against_every_way_of_asking(rec, clean, monkeypatch):
    _, _, body = run_fact(rec)
    tr, f = rec.tracer, actions.fact(rec, body)
    ways = ways_of_asking(rec, f)
    attributed = []
    attribute = tr.attribute
    monkeypatch.setattr(tr, "attribute", lambda cap, tid: attributed.append((tuple(cap.ids), tid)) or attribute(cap, tid))
    res = actions.forget_target(rec, body | {"leads": LEADS, "keep": ""})
    monkeypatch.undo()
    # Every lead-in and the saved check for the same answer are attributed, in Measure's order.
    assert attributed == [(tuple(ids), tid) for ids, tid in ways]
    assert [ld["source"] for ld in res["leads"]] == ["lead", "lead", "lead", "check"] and res["keep"] == []
    assert res["combine"] == actions.COMBINE == "mean" and res["rankings"]["specific"] is None
    effects = np.stack([tr.attribute(tr.capture_ids(ids), tid)[0].cpu().numpy() for ids, tid in ways])
    r = res["rankings"]["direct"]
    # A unit's effect is its mean effect over the ways of asking, and the ranking is by it.
    for u in r["units"]:
        assert u["direct"] == pytest.approx(effects[:, u["g"]].mean(), abs=1e-3)
        assert u["by_lead"] == pytest.approx(effects[:, u["g"]].tolist(), abs=1e-3)
    assert [u["g"] for u in r["units"]] == [int(g) for g in np.argsort(-effects.mean(0)) if effects.mean(0)[g] > 0][:actions.CANDIDATES]
    with tr._model():
        befores = [read(rec, ids, tid) for ids, tid in ways]
        for ld, b in zip(res["leads"], befores):
            assert ld["prob"] == pytest.approx(b["prob"], abs=1e-5) and ld["rank"] == b["rank"]
        for p in r["curve"]:
            # Each point reads every way of asking; it is gone only when gone after all of them.
            units = [u["g"] for u in r["units"][:p["k"]]]
            now = [read(rec, ids, tid, units) for ids, tid in ways]
            assert [x["prob"] for x in p["leads"]] == pytest.approx([x["prob"] for x in now], abs=1e-5)
            assert p["leads_gone"] == sum(actions.gone(b, x) for b, x in zip(befores, now))
            assert p["gone"] == (p["leads_gone"] == len(ways))
            assert (p["prob"], p["rank"]) == (p["leads"][0]["prob"], p["leads"][0]["rank"])
    flip = next((p["k"] for p in r["curve"] if p["gone"]), None)
    assert r["flipped"] == (flip is not None) and r["suggested"] == (flip or r["curve"][-1]["k"])
    # The min counts a unit by the way of asking where it pushes the answer least.
    res = actions.forget_target(rec, body | {"leads": LEADS, "keep": "", "combine": "min"})
    assert res["combine"] == "min"
    for u in res["rankings"]["direct"]["units"]:
        assert u["direct"] == pytest.approx(effects[:, u["g"]].min(), abs=1e-3) and u["direct"] >= 0
    with pytest.raises(ValueError):
        actions.forget_target(rec, body | {"combine": "max"})
    # A lead-in typed twice, or the fact's own lead-in typed again, is one way of asking.
    typed = actions.fact(rec, {"prompt": "Roses are", "answer": "red"})
    assert [ld["text"] for ld in actions._leads(rec.s, typed, "Roses are\nViolets are\nViolets are ")] == ["Roses are", "Violets are"]


def test_specific_ranking_counts_the_fact_gone_after_every_way_of_asking(rec, clean, monkeypatch):
    _, _, body = run_fact(rec)
    f = actions.fact(rec, body)
    ways, keep = ways_of_asking(rec, f), keep_facts(rec)
    # The tiny model's probabilities are near uniform, so here "gone" is any drop in rank.
    monkeypatch.setattr(actions, "gone", lambda before, after: after["rank"] > before["rank"])
    res = actions.forget_target(rec, body | {"leads": LEADS, "keep": KEEP})
    spec = res["rankings"]["specific"]
    assert len(res["keep"]) == 3 and len(res["leads"]) == len(ways)
    units = [u["g"] for u in spec["units"]]
    with rec.tracer._model():
        befores = [read(rec, ids, tid) for ids, tid in ways]
        held = lambda k: [not actions.gone(b, read(rec, ids, tid, units[:k] or None)) for b, (ids, tid) in zip(befores, ways)]
        for p in spec["curve"]:
            still = held(p["k"])
            assert p["leads_gone"] == still.count(False) and p["gone"] == (not any(still))
            assert len(p["keep"]) == len(keep)
        # The exact smallest set gone after every way of asking: one fewer leaves one holding it.
        assert spec["flipped"] and any(held(spec["suggested"] - 1)) and not any(held(spec["suggested"]))
    assert all(not p["gone"] for p in spec["curve"] if p["k"] < spec["suggested"])
    # Counting the fact's own lead-in alone would have stopped sooner.
    assert any(actions.gone(befores[0], p["leads"][0]) for p in spec["curve"] if p["k"] < spec["suggested"])
    for name in ("direct", "specific"):
        r = res["rankings"][name]
        assert r["flipped"] and next(p for p in r["curve"] if p["k"] == r["suggested"])["leads_gone"] == len(ways)
    assert res["ranking"] == "specific"


def test_target_sorts_saved_checks_like_measure(rec, clean):
    _, _, body = run_fact(rec)
    f = actions.fact(rec, body)
    workbench.add_check(rec, {"prompt": "The sky is", "expect": "blue"})
    workbench.add_check(rec, body)  # the fact itself: not a fact to keep
    checks = workbench.list_checks(rec)
    checks.append({"kind": "text", "prompt": "Roses are", "expect": "x", "chat": False, "target": f.target,
                   "target_text": "x", "label": "Roses are → x", "id": "samefact"})
    workbench._save_checks(rec, checks)
    res = actions.forget_target(rec, body | {"keep": "The sky is → blue\nGrass is → green"})
    # The typed sky fact and the saved sky check are one fact; checks asking for the answer are leads.
    assert [(k["label"], k["source"]) for k in res["keep"]] == [("The sky is → blue", "typed"), ("Grass is → green", "typed")]
    assert actions.forget_target(rec, body | {"keep": "", "checks": True})["keep"][0]["source"] == "check"


def test_typed_fact_reads_like_a_check(rec, clean):
    f = actions.fact(rec, {"prompt": "Roses are ", "answer": "red"})
    assert f.lead == "Roses are" and f.target == workbench.answer_token(rec.s, "Roses are", "red")
    res = actions.forget_target(rec, {"prompt": "Roses are", "answer": "red"})
    with rec.lock:
        check = workbench._next_token(rec.s, workbench.prompt_ids(rec.s, "Roses are"), f.target)
    assert res["fact"]["prob"] == pytest.approx(check["prob"], abs=1e-5)
    with pytest.raises(ValueError):
        actions.fact(rec, {"prompt": "Roses are", "answer": ""})


def test_try_answers_with_and_without_the_units(rec, clean):
    _, _, body = run_fact(rec)
    units = [u["g"] for u in actions.forget_target(rec, body)["rankings"]["direct"]["units"][:4]]
    res = actions.forget_try(rec, body | {"units": units, "leads": "Violets are", "questions": ["Name a color."]})
    assert [r["kind"] for r in res["rows"]] == ["lead", "lead", "question"]
    ids = workbench.prompt_ids(rec.s, "Violets are")
    with rec.tracer._model():
        assert res["rows"][1]["original"] == actions._continue(rec.s, ids, actions.LEAD_TOKENS)
        with rec.tracer.knocked_out(units):
            assert res["rows"][1]["edited"] == actions._continue(rec.s, ids, actions.LEAD_TOKENS)
    # No hook is left behind: the model reads exactly as before.
    f = actions.fact(rec, body)
    assert plain_prob(rec, f.ids, f.target) == pytest.approx(actions.forget_target(rec, body)["fact"]["prob"], abs=1e-6)
    with pytest.raises(ValueError):
        actions.forget_try(rec, body | {"units": []})
    with pytest.raises(ValueError):
        actions.forget_try(rec, body | {"units": list(range(65))})


def test_random_controls_match_the_set(rec):
    an = rec.anatomy
    units = [g for g in range(an.n_units) if an.units[g]["kind"] == "head"][:2] + \
            [g for g in range(an.n_units) if an.units[g]["kind"] == "cluster"][:3]
    a, b = actions._random_units(an, units, 0), actions._random_units(an, units, 1)
    assert a == actions._random_units(an, units, 0) and a != b
    for pick in (a, b):
        assert not set(pick) & set(units) and len(pick) == len(units)
        assert sum(an.locate(g)[1] == "head" for g in pick) == 2


def test_measure_then_keep_gives_what_was_measured(rec, clean):
    _, _, body = run_fact(rec)
    units = [u["g"] for u in actions.forget_target(rec, body)["rankings"]["direct"]["units"][:4]]
    workbench.add_check(rec, {"prompt": "The sky is", "expect": "blue"})
    workbench.add_check(rec, {"prompt": "The sky is", "expect": "blue"})  # a duplicate is kept once
    assert len(workbench.list_checks(rec)) == 1
    workbench.add_check(rec, body)  # the fact's own context: measured once, as the fact
    # A check asking for the same answer after another context is another way of asking for it.
    f = actions.fact(rec, body)
    checks = workbench.list_checks(rec)
    checks.append({"kind": "text", "prompt": "Roses are", "expect": "x", "chat": False, "target": f.target,
                   "target_text": "x", "label": "<|im_start|>Roses are → x", "id": "samefact"})
    workbench._save_checks(rec, checks)
    res = actions.forget_measure(rec, body | {"units": units, "leads": ["Violets are"], "keep": "Grass is → green"})
    assert [r["source"] for r in res["leads"]] == ["lead", "lead", "check"]
    assert [k["source"] for k in res["keep"]] == ["typed", "check"]
    assert res["leads"][2]["prompt"] == "Roses are → x"
    assert res["drift"]["kl_per_token"] >= 0 and res["drift"]["tokens"] > 0
    assert res["control"] == {"seeds": actions.CONTROL_SEEDS, "units": 4}
    assert res["verdict"]["status"] in {"works", "partial", "costly", "kept", "unspecific"}
    with pytest.raises(ValueError):
        actions.forget_measure(rec, body | {"units": units, "keep": "no arrow here"})
    # Keep: the same units zeroed in the weights read exactly as the knockout did.
    rec.add_weight_edit(**rec.anatomy.lesion_units(units, note="forget"))
    rec.apply_weight_edits()
    f = actions.fact(rec, body)
    assert plain_prob(rec, f.ids, f.target) == pytest.approx(res["leads"][0]["after"]["prob"], abs=1e-5)
    grass = workbench.prompt_ids(rec.s, "Grass is")
    assert plain_prob(rec, grass, workbench.answer_token(rec.s, "Grass is", "green")) == \
        pytest.approx(res["keep"][0]["after"]["prob"], abs=1e-5)


def test_gone():
    assert actions.gone({"rank": 1, "prob": 0.9}, {"rank": 2, "prob": 0.4})
    assert not actions.gone({"rank": 1, "prob": 0.9}, {"rank": 1, "prob": 0.3})
    # An answer that wasn't first must lose most of its probability.
    assert not actions.gone({"rank": 2, "prob": 0.2}, {"rank": 5, "prob": 0.05})
    assert actions.gone({"rank": 2, "prob": 0.2}, {"rank": 40, "prob": 0.01})


def test_check_labels_lose_template_markers():
    assert actions._clean("<|im_start|>Roses are → x") == "Roses are → x"
    # A saved label cut to its last characters can start inside a marker.
    cut = "…rt|>assistant\n<think>\n\n</think>\n\nThe capital of Australia is →  Canberra"
    assert actions._clean(cut) == "The capital of Australia is → Canberra"
    assert actions._clean("One, two, → three") == "One, two, → three"


def test_verdicts():
    lead = lambda before, after, control=0.0: {"before": {"rank": before, "prob": 0.9}, "after": {"rank": after, "prob": 0.1},
                                                "gone": after > 1, "control_gone": control}
    keep_ok = [{"before": {"rank": 1, "prob": 0.8}, "after": {"rank": 1, "prob": 0.79}}]
    keep_broken = [{"before": {"rank": 1, "prob": 0.8}, "after": {"rank": 3, "prob": 0.2}}]
    calm = {"kl_per_token": 0.001}
    assert actions._verdict([lead(1, 1)], keep_ok, calm)["status"] == "kept"
    assert actions._verdict([lead(1, 2, 1.0)], keep_ok, calm)["status"] == "unspecific"
    assert actions._verdict([lead(1, 2), lead(1, 2)], keep_ok, calm)["status"] == "works"
    assert actions._verdict([lead(1, 2), lead(1, 1)], keep_ok, calm)["status"] == "partial"
    assert actions._verdict([lead(1, 2)], keep_broken, calm)["status"] == "costly"
    assert actions._verdict([lead(1, 2)], keep_ok, {"kl_per_token": 0.5})["status"] == "costly"


def test_forget_endpoints(rec, clean):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        _, _, body = run_fact(rec)
        r = client.post("/brainscan/actions/forget/target", json=body)
        assert r.status_code == 200
        units = [u["g"] for u in r.json()["rankings"]["direct"]["units"][:2]]
        r = client.post("/brainscan/actions/forget/try", json=body | {"units": units})
        assert r.status_code == 200 and r.json()["rows"][0]["kind"] == "lead"
        r = client.post("/brainscan/actions/forget/measure", json=body | {"units": units, "checks": False})
        assert r.status_code == 200 and r.json()["verdict"]["status"]
        assert client.post("/brainscan/actions/forget/nope", json=body).status_code == 404
        assert client.post("/brainscan/actions/forget/target", json={"prompt": ""}).status_code == 400
        assert client.post("/brainscan/actions/forget/target", json={"run": "nope", "t": 0, "target": 1}).status_code == 404
        assert client.post("/brainscan/actions/forget/try", json=body | {"units": [10 ** 6]}).status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
