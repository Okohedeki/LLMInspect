"""The neuron atlas: the layout, the per-neuron maps and their byte encoding, what fires on a text, and the
weight-path links, on the tiny saved Llama in fp32."""

import base64

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from llminspect import server
from llminspect.config import Settings
from llminspect.recorder import atlas
from llminspect.recorder.model import Subject
from llminspect.recorder.session import Recorder
from llminspect.recorder.store import RunStore
from test_checkpoints import MODEL_DIR

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("atlas")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    return Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)


@pytest.fixture(autouse=True)
def clean(rec):
    yield
    rec.edits.revert()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)


def decode(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint8)


def test_the_layout_has_every_dense_layer_and_its_clusters(rec):
    lay = rec.atlas.layout()
    assert [r["layer"] for r in lay["layers"]] == list(range(rec.s.n_layers))
    row = lay["layers"][2]
    assert row["width"] == rec.neurons.mlp(2).width
    clusters = decode(row["clusters"])
    assert len(clusters) == row["width"] and clusters.max() < row["n_clusters"] <= 8
    lm = rec.anatomy.layers[2]
    assert row["offset"] == lm.offset + lm.n_heads and np.array_equal(clusters, lm.assign.numpy())
    assert lay["maps"] == list(atlas.MAPS) and lay["edited"] is False


def test_quantize_keeps_order_on_a_log_scale_and_centres_signed_maps():
    q = atlas.quantize({0: torch.tensor([1.0, 10.0, 100.0]), 1: torch.tensor([1000.0])})
    a, b = decode(q["layers"]["0"]), decode(q["layers"]["1"])
    assert q["scale"] == "log" and list(a) == sorted(a) and b[0] >= a[-1]
    s = atlas.quantize({0: torch.tensor([-2.0, 0.0, 2.0])}, signed=True)
    v = decode(s["layers"]["0"])
    assert s["signed"] and v[0] < v[1] < v[2] and abs(int(v[1]) - 128) <= 1


def test_the_write_map_follows_the_down_projection_norms(rec):
    m = rec.atlas.map("write")
    norms = rec.neurons.mlp(1).down.weight.detach().norm(dim=0)
    v = decode(m["layers"]["1"])
    hi, lo = int(norms.argmax()), int(norms.argmin())
    assert v[hi] > v[lo] and m["available"] and m["map"] == "write"
    with pytest.raises(ValueError, match="map must be one of"):
        rec.atlas.map("nope")


def test_spread_and_mean_come_from_the_calibration_without_an_index(rec):
    sp, mean = rec.atlas.map("spread"), rec.atlas.map("mean")
    assert "calibration" in sp["note"] and mean["signed"] and set(sp["layers"]) == {str(i) for i in range(rec.s.n_layers)}


def test_the_changed_map_shows_only_the_edited_neurons(rec):
    assert rec.atlas.map("changed")["available"] is False
    rec.edits.add("scale", "3", ["neuron"], index=[5, 9], factor=0.0)
    rec.edits.apply()
    m = rec.atlas.map("changed")
    v = decode(m["layers"]["3"])
    assert m["available"] and v[5] > 0 and v[9] > 0 and v[[i for i in range(len(v)) if i not in (5, 9)]].max() == 0
    assert decode(m["layers"]["2"]).max() == 0


def test_fires_reads_a_text_once(rec):
    out = rec.atlas.fires("The river rises in the northern hills and flows south to the sea.")
    assert out["map"] == "fires" and out["tokens"] > 5 and len(out["top"]) == atlas.FIRES_TOP
    zs = [t["z"] for t in out["top"]]
    assert zs == sorted(zs, reverse=True) and all(isinstance(t["token"], str) for t in out["top"])
    assert set(out["layers"]) == {str(i) for i in range(rec.s.n_layers)}
    with pytest.raises(ValueError):
        rec.atlas.fires("   ")


def test_links_follow_the_weights(rec):
    out = rec.atlas.links(2, 7)
    assert out["neuron"] == "L2 N7"
    assert out["reads_it"] and all(r["layer"] > 2 for r in out["reads_it"])
    assert out["it_reads"] and all(r["layer"] < 2 for r in out["it_reads"])
    cos = [abs(r["cos"]) for r in out["reads_it"]]
    assert cos == sorted(cos, reverse=True) and max(cos) <= 1.0 + 1e-5
    m = rec.neurons.mlp(3)  # the top later link really is the largest cosine of the write against an input row
    w = rec.neurons.mlp(2).down.weight[:, 7].detach().float()
    with torch.no_grad():
        gain = rec.neurons._gain(m, 3)
    W = m.up.weight.detach().float() * gain[None]
    c = (W @ (w / w.norm())) / W.norm(dim=1)
    best3 = max((r for r in out["reads_it"] if r["layer"] == 3 and r["via"] == "up"), key=lambda r: abs(r["cos"]), default=None)
    if best3 is not None:
        assert abs(best3["cos"]) == pytest.approx(float(c.abs().max()), abs=1e-3)
    with pytest.raises(KeyError):
        rec.atlas.links(2, 10**6)


def test_the_endpoints(rec):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        assert client.get("/llminspect/neurons/atlas").json()["layers"][0]["layer"] == 0
        assert client.get("/llminspect/neurons/atlas/write").json()["map"] == "write"
        assert client.get("/llminspect/neurons/atlas/bogus").status_code == 400
        fired = client.post("/llminspect/neurons/fires", json={"text": "Knead the dough for ten minutes."})
        assert fired.status_code == 200 and fired.json()["top"]
        assert client.post("/llminspect/neurons/fires", json={"text": ""}).status_code == 400
        assert client.get("/llminspect/neurons/L2 N7/links").json()["neuron"] == "L2 N7"
        assert client.get("/llminspect/neurons/nonsense/links").status_code == 400
        assert client.get("/llminspect/neurons/L2 N7").json()["neuron"] == "L2 N7"  # the card's route still answers
    finally:
        server.state.clear()
        server.state.update(saved)
