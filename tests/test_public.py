"""The public app: only Hub ids, never a local path or a gated repo, cached cards, capped sweeps, rate limits,
link previews and report pages. The Hub itself is stubbed: nothing here reaches the network."""

import json

import pytest
from fastapi.testclient import TestClient

from llminspect import hub, public, sweep

CARD = {"model": "org/abl", "headline": "org/base, one direction removed from 4 of 4 layers, the signature of "
        "abliteration.", "edit": {"kind": "direction_removed", "headline": "one direction removed from 4 of 4 layers, "
        "the signature of abliteration"}, "files": {"findings": [], "template": []}, "bytes_read": 4e6, "seconds": 3}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(public, "CARDS", tmp_path / "cards")
    monkeypatch.setattr(public, "SWEEPS", tmp_path / "sweeps")
    monkeypatch.setattr(public, "_seen", type(public._seen)(public._seen.default_factory))
    calls = []

    def fake(model, roots=None, say=None):
        calls.append((model, roots))
        return dict(CARD, model=model)

    monkeypatch.setattr(hub, "unmask", fake)
    with TestClient(public.app) as c:
        assert hub.PUBLIC
        yield c, calls
    assert not hub.PUBLIC


def test_only_hub_ids_are_read_and_cards_are_cached(app):
    c, calls = app
    for bad in ["../etc/passwd", "/etc/passwd", "a/b/c", "a", "a/..", "C:/Windows"]:
        assert c.get("/llminspect/unmask", params={"model": bad}).status_code == 400
    assert c.get("/llminspect/unmask", params={"model": "https://huggingface.co/org/abl"}).json()["model"] == "org/abl"
    c.get("/llminspect/unmask", params={"model": "org/abl"})
    assert calls == [("org/abl", [])]  # read once, with no local models to compare with


def test_in_public_mode_a_local_folder_name_still_goes_to_the_hub(app, monkeypatch, tmp_path):
    (tmp_path / "org" / "name").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(hub, "Hub", lambda repo, token=None: seen.append(repo) or "hub")
    assert hub.open_model("org/name") == "hub" and seen == ["org/name"]


def test_gated_repos_are_refused_in_public_mode(app, monkeypatch):
    class Info:
        id, sha, gated, siblings, card_data, tags = "org/gated", "abc", "manual", [], None, []
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda self, *a, **k: Info())
    with pytest.raises(hub.Unreachable, match="gated"):
        hub.Hub("org/gated", token="secret")


def test_a_client_is_rate_limited(app, monkeypatch):
    c, _ = app
    monkeypatch.setitem(public.LIMITS, "unmask", (2, 600))
    codes = [c.get("/llminspect/unmask", params={"model": f"org/m{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    assert c.get("/llminspect/unmask", params={"model": "org/m0"}).status_code == 200  # cached: no limit


def test_sweeps_are_validated_and_capped(app, monkeypatch):
    c, _ = app
    got = []
    monkeypatch.setattr(sweep, "start", lambda *a, **k: got.append((a, k)) or {"id": "x", "status": "running"})
    assert c.post("/llminspect/sweep", json={"sort": "evil"}).status_code == 400
    assert c.post("/llminspect/sweep", json={"models": ["../x"]}).status_code == 400
    assert c.post("/llminspect/sweep", json={"search": "<script>"}).status_code == 400
    assert c.post("/llminspect/sweep", json={"search": "abliterated", "limit": 999, "weights": True}).status_code == 200
    args, kw = got[0]
    assert args[5] == public.MAX_WEIGHTS and args[7] is True and kw["roots"] == []


def test_a_shared_link_carries_its_preview(app):
    c, _ = app
    c.get("/llminspect/unmask", params={"model": "org/abl"})
    page = c.get("/m/org/abl").text
    assert 'data-unmask="org/abl"' in page and 'og:image" content="/og/org/abl.png"' in page and "abliteration" in page
    png = c.get("/og/org/abl.png")
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert 'data-unmask=""' in c.get("/").text


def test_a_sweep_report_page(app):
    c, _ = app
    public.SWEEPS.mkdir(parents=True)
    (public.SWEEPS / "s1.json").write_text(json.dumps({"search": "abliterated"}))
    (public.SWEEPS / "s1.jsonl").write_text(json.dumps({"model": "org/abl", "top": "targeted weight edit", "risks": [
        {"risk": "targeted weight edit", "what": "one direction removed"}]}) + "\n")
    page = c.get("/r/s1")
    assert page.status_code == 200 and "Hub sweep: abliterated" in page.text and "org/abl" in page.text
    assert c.get("/r/..%2Fetc").status_code == 404
