"""The neuron index: top activations, spreads and logit lenses over a corpus, checked against a brute
force read, and search by text, on the tiny saved Llama in fp32."""

import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import neuron_index as ni
from brainscan.recorder.model import Subject
from brainscan.recorder.session import Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")
TOKENS = 3000


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("index")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    r = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    r.neuron_index.build(TOKENS)
    return r


def brute(rec, layer):
    """Every activation of a layer's neurons over the index's stream, in stream order."""
    stream, step = rec.neuron_index.data["stream"].tolist(), ni.WINDOW - 1
    mlp, rows = rec.neurons.mlp(layer), []
    h = mlp.down.register_forward_pre_hook(lambda m, a: rows.append(a[0][0, 1:].float()))
    try:
        with torch.no_grad():
            for i in range(0, len(stream), step):
                rec.s.model(input_ids=torch.tensor([[rec.s.sink_id] + stream[i:i + step]]), use_cache=False)
    finally:
        h.remove()
    return torch.cat(rows)  # (tokens, neurons)


def test_the_build_matches_a_brute_force_read(rec):
    idx = rec.neuron_index
    meta = idx.meta
    assert meta["tokens"] == TOKENS and meta["layers"] == list(range(rec.s.n_layers))
    assert any("starter" in s for s in meta["sources"])
    layer = 2
    acts = brute(rec, layer)
    assert acts.shape[0] == TOKENS
    assert torch.allclose(idx.data[f"mean.{layer}"], acts.mean(0), atol=1e-4)
    assert torch.allclose(idx.data[f"std.{layer}"], acts.std(0, unbiased=False), atol=1e-3)
    v, p = acts.topk(ni.TOP_K, dim=0)
    assert torch.allclose(idx.data[f"top_v.{layer}"], v.T, atol=1e-4)
    assert (idx.data[f"top_p.{layer}"][:, 0] == p[0]).float().mean() > 0.99  # the very top is the same token
    s = rec.s
    gain = s.norm(torch.ones(1, s.d_model))[0]
    n = 5
    logits = s.lm_head.weight @ (rec.neurons.write(layer, n) * gain)
    assert idx.data[f"lens_up.{layer}"][n].tolist() == logits.topk(ni.LENS_UP).indices.tolist()


def test_queries_match_whole_words_with_smart_case():
    paris = ni.pattern("Paris")
    assert paris.search("in Paris, France") and paris.search("(Paris)")
    assert not any(paris.search(t) for t in ("comparison", "parish", "PARIS", "Parisian"))
    lower = ni.pattern("paris")
    assert lower.search("PARIS") and lower.search("Paris") and not lower.search("parish")
    assert ni.pattern("Paris", whole=False).match("Parisian") and not ni.pattern("Paris", whole=False).match("comparison")
    assert ni.pattern("->").search("a->b") and ni.pattern("def ").search("    def f")
    assert ni.pattern("北京").search("我在北京工作")  # no spaces between words: matched anywhere


def test_search(rec):
    idx = rec.neuron_index
    r = idx.search("Canberra")
    assert r["in_corpus"] and r["matches"] >= 1 and r["fires_on"]
    assert r["fires_on"][0]["z"] >= r["fires_on"][-1]["z"]
    assert "Canberra" in (r["fires_on"][0]["context"]["before"] + r["fires_on"][0]["context"]["token"]
                          + r["fires_on"][0]["context"]["after"])
    spans = idx._spans("Canberra")
    for row in r["contexts"]:  # each neuron really has a top activation on an occurrence
        tops = idx.data[f"top_p.{row['layer']}"][row["index"]].tolist()
        assert any(a <= p <= b for p in tops for a, b in spans)
    absent = idx.search("zqxjv")
    assert not absent["in_corpus"] and absent["fires_on"] and not absent["contexts"]
    # A neuron's own top lens token finds it under "pushes it".
    layer, n = 1, 0
    tok = rec.s.token_str(int(idx.data[f"lens_up.{layer}"][n][0])).strip()
    if len(tok) >= 2:
        assert any(x["neuron"] == f"L{layer} N{n}" for x in idx.search(tok, limit=10 ** 6)["pushes"])
    with pytest.raises(ValueError):
        idx.search("a")


def test_the_index_is_kept_and_the_card_reads_it(rec):
    again = ni.NeuronIndex(rec)
    assert again.ready() and again.meta["fingerprint"] == rec.neuron_index.meta["fingerprint"]
    card = rec.neurons.card(2, 5, 1.0)
    assert card["corpus"]["contexts"] and card["corpus"]["contexts"][0]["act"] == pytest.approx(
        float(rec.neuron_index.data["top_v.2"][5][0]), abs=1e-3)


def test_a_corpus_of_your_own(rec, tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("The kettle boiled.\n\nThe cat slept on the warm stones by the fire.")
    (tmp_path / "b.jsonl").write_text('{"text": "Quarterly revenue rose by four percent."}\nnot json\n{"other": 1}\n')
    monkeypatch.setenv("BRAINSCAN_CORPUS", str(tmp_path))
    texts, sources = ni.corpus_texts(rec)
    assert [t for _, t in texts] == ["The kettle boiled.", "The cat slept on the warm stones by the fire.",
                                     "Quarterly revenue rose by four percent."]
    assert "BRAINSCAN_CORPUS" in sources[0]
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("BRAINSCAN_CORPUS", str(empty))
    with pytest.raises(ValueError):
        ni.corpus_texts(rec)


def test_text_in_the_data_folder_comes_after_the_default_mix(rec, monkeypatch):
    monkeypatch.delenv("BRAINSCAN_CORPUS", raising=False)
    base, sources = ni.corpus_texts(rec)
    folder = rec.store.root / "corpus"
    folder.mkdir()
    try:
        (folder / "web.jsonl").write_text('{"text": "Photosynthesis turns light into sugar."}\n'
                                          '{"text": "Rivers carve canyons over millennia."}\n')
        texts, more = ni.corpus_texts(rec)
        assert texts[:len(base)] == base and [t for _, t in texts[len(base):]] == [
            "Photosynthesis turns light into sugar.", "Rivers carve canyons over millennia."]
        assert more[:len(sources)] == sources and "web.jsonl in corpus/ (2 texts)" in more[-1]
    finally:
        for f in folder.iterdir():
            f.unlink()
        folder.rmdir()


def test_endpoints(rec):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        assert client.get("/brainscan/neurons/index").json()["status"] == "ready"
        r = client.get("/brainscan/neurons/search", params={"q": "Canberra", "limit": 5})
        assert r.status_code == 200 and len(r.json()["fires_on"]) == 5
        assert client.get("/brainscan/neurons/search", params={"q": "a"}).status_code == 400
        assert client.post("/brainscan/neurons/index", json={"max_tokens": "x"}).status_code == 400
        job = client.post("/brainscan/neurons/index", json={"max_tokens": 600}).json()
        assert job["status"] == "building"
        for _ in range(300):
            st = client.get("/brainscan/neurons/index").json()
            if st["status"] != "building":
                break
            time.sleep(0.1)
        assert st["status"] == "ready" and st["meta"]["tokens"] == 600
    finally:
        rec.neuron_index.build(TOKENS)  # leave the module's index as the other tests expect
        server.state.clear()
        server.state.update(saved)
