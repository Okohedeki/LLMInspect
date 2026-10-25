"""End-to-end and HTTP-server coverage for the weight-anatomy feature.

test_anatomy.py exercises the Anatomy class in isolation (unit map, probe math, tiles,
edges, recording round-trip). The two seams it does not cover are checked here:

* the recorder -> anatomy integration: a real generation on CPU through Recorder.generate
  must record one activity row per non-stop token and flush it to disk (session.py);
* the FastAPI endpoints in server.py that expose the anatomy to the scanner UI.

Everything runs on CPU with the tiny saved Llama under ~/.cache/brainscan/models/tiny-llama-test
(same shape family as test_anatomy.py's LlamaConfig), so it never touches the GPU, the real
recorder, or port 8765. The TestClient is created without a `with` block on purpose: Starlette
only runs the lifespan (which would try to load the configured model) inside the context
manager, so plain construction keeps the app inert and we drive it through a patched state.
"""

import base64
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder.model import Subject
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"
CLUSTERS = 8

pytestmark = pytest.mark.skipif(
    not MODEL_DIR.exists(),
    reason=f"needs the tiny CPU Llama at {MODEL_DIR} (a LlamaForCausalLM saved with a tokenizer)")


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    """A real CPU recorder with weight anatomy on, plus one greedy run recorded end to end.

    Built once per module: loading the tiny model, clustering the atlas and running an
    8-token generation are all fast on CPU, and every test reuses the same recorder/run.
    """
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("brainscan")
    subject = Subject(str(MODEL_DIR), device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True,
                        anatomy_clusters=CLUSTERS, graph_decode=False, data_dir=root)
    store = RunStore(root / "data", settings.disk_budget_gb)
    rec = Recorder(subject, store, settings, lambda msg: None)
    res = rec.generate(GenParams(messages=[{"role": "user", "content": "Hello there, how are you?"}],
                                 max_tokens=8, temperature=0))
    return rec, store, res


@pytest.fixture()
def client(recording):
    """A TestClient with server.state pointed at the prebuilt CPU recorder, restored after."""
    rec = recording[0]
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        yield TestClient(server.app)
    finally:
        server.state.clear()
        server.state.update(saved)


# ---- 1. the recorder -> anatomy integration (session.py) -------------------------------

def test_recorder_records_anatomy_end_to_end(recording):
    rec, store, res = recording
    an = rec.anatomy

    # The run's tape carries the generated tokens. Greedy decoding is deterministic, and the
    # model does not emit a stop token inside 8 steps, so it hits the cap with 8 gen tokens.
    run = rec.get_run(res.run_id)
    gen = [r for r in run.tape if r["ph"] == "gen"]
    assert res.finish_reason == "length"
    assert res.completion_tokens == 8 and len(gen) == res.completion_tokens

    t, codes, skipped = an.read_run(store, res.run_id)
    assert skipped == 0
    assert codes.dtype == np.uint8
    assert codes.shape[1] == an.n_units                              # one byte per unit
    # One activity row per non-stop recorded token: every prompt token plus every generated
    # token that actually ran a forward (a stop token is taped but records no activity).
    assert codes.shape[0] == res.prompt_tokens + res.completion_tokens
    assert t.tolist() == list(range(codes.shape[0]))                 # contiguous tape indices

    # The activity was flushed to disk for request 1, and the file holds exactly those rows.
    npz = store.run_dir(res.run_id) / "anatomy" / f"req{res.req:04d}.npz"
    assert npz.exists()
    with np.load(npz) as z:
        assert z["codes"].shape == (codes.shape[0], an.n_units)
        assert str(z["atlas"]) == an.key
        assert np.array_equal(z["codes"], codes)


# ---- 2. the server endpoints (server.py) -----------------------------------------------

def test_anatomy_unit_map_endpoint(client, recording):
    rec = recording[0]
    an = rec.anatomy
    r = client.get("/brainscan/anatomy")
    assert r.status_code == 200
    d = r.json()
    assert d["n_units"] == an.n_units
    assert len(d["units"]) == an.n_units
    assert {u["g"] for u in d["units"]} == set(range(an.n_units))
    assert len(d["layers"]) == len(an.layers)
    assert d["edited"] == an.edited_units(rec.edits)                 # no edits applied -> []
    assert d["edits_version"] == rec.edits.version


def test_anatomy_edges_endpoint(client, recording):
    an = recording[0].anatomy
    r = client.get("/brainscan/anatomy/edges")
    assert r.status_code == 200
    e = r.json()
    assert e["available"] is True
    k = e["k"]
    for field in ("up", "down", "up_x", "down_x"):
        assert len(e[field]) == an.n_units * k                      # k kept edges per unit


def test_anatomy_unit_detail_endpoint(client, recording):
    an = recording[0].anatomy
    r = client.get("/brainscan/anatomy/units/0")
    assert r.status_code == 200
    u = r.json()
    assert u["g"] == 0 and u["kind"] == "head" and u["layer"] == 0
    assert u["matrices"][0]["key"] == "writes"
    assert "spectrum" in u and "effective_rank" in u
    # a unit id past the end is a 404
    assert client.get(f"/brainscan/anatomy/units/{an.n_units}").status_code == 404


def test_anatomy_tile_endpoint(client, recording):
    rec = recording[0]
    an = rec.anatomy
    r = client.post("/brainscan/anatomy/tile", json={"unit": 0, "matrix": "writes"})
    assert r.status_code == 200
    tile = r.json()
    data = np.frombuffer(base64.b64decode(tile["data"]), np.float16)
    assert data.size == tile["h"] * tile["w"]
    arr = data.reshape(tile["h"], tile["w"])
    # head-writes is small, so the tile is exact: one cell per weight, cols are the residual dims.
    assert tile["exact"] and arr.shape == (tile["rows"], tile["cols"])
    assert tile["cols"] == rec.s.d_model
    # the endpoint returns exactly what anatomy.tile produces for the same request
    assert tile["data"] == an.tile({"unit": 0, "matrix": "writes"}, rec.edits.version)["data"]
    # a bad matrix mode is a validation error
    bad = client.post("/brainscan/anatomy/tile", json={"unit": 0, "matrix": "writes", "mode": "bogus"})
    assert bad.status_code == 400


def test_anatomy_params_endpoint(client, recording):
    rec = recording[0]
    r = client.get("/brainscan/anatomy/params")
    assert r.status_code == 200
    params = r.json()
    assert len(params) == len(rec.anatomy.params)
    assert "model.embed_tokens.weight" in {p["name"] for p in params}
    for p in params:
        assert set(p) >= {"name", "shape", "dtype", "numel"}


def test_run_anatomy_endpoint(client, recording):
    rec, _, res = recording
    an = rec.anatomy
    r = client.get(f"/brainscan/runs/{res.run_id}/anatomy")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/octet-stream")
    content = r.content
    # 4-byte little-endian header length, then the JSON header, then n*units raw bytes.
    (hlen,) = struct.unpack("<I", content[:4])
    header = json.loads(content[4:4 + hlen])
    raw = content[4 + hlen:]
    assert header["units"] == an.n_units
    assert header["key"] == an.key
    assert header["skipped"] == 0
    assert len(header["t"]) == header["n"]
    assert header["n"] == res.prompt_tokens + res.completion_tokens
    assert len(raw) == header["n"] * header["units"]


# ---- 3. the lesion endpoint (server.py -> anatomy.lesion -> edits) ----------------------

def test_lesion_endpoint_adds_scale_edit_for_a_head(client, recording):
    rec = recording[0]
    before = set(rec.edits.edits)
    r = client.post("/brainscan/anatomy/units/0/lesion", json={})
    assert r.status_code == 200
    edit = r.json()
    assert edit["kind"] == "scale" and edit["targets"] == ["head"]
    assert edit["index"] == 0 and edit["layers"] == "0" and edit["factor"] == 0.0
    assert edit["id"] in rec.edits.edits and edit["id"] not in before
    # the stored edit targets the layer and head the unit maps to
    lm, kind, i = rec.anatomy.locate(0)
    assert kind == "head" and lm.index == 0 and i == 0
    stored = rec.edits.edits[edit["id"]]
    assert stored.kind == "scale" and stored.targets == ["head"]
    assert stored.layers == [lm.index] and stored.index == i


def test_lesion_endpoint_adds_neuron_edit_for_a_cluster(client, recording):
    rec = recording[0]
    an = rec.anatomy
    lm0 = an.layers[0]
    g = lm0.offset + lm0.n_heads                                     # layer 0, first MLP cluster
    r = client.post(f"/brainscan/anatomy/units/{g}/lesion", json={"factor": 0.0})
    assert r.status_code == 200
    edit = r.json()
    assert edit["kind"] == "scale" and edit["targets"] == ["neuron"]
    assert isinstance(edit["index"], list) and edit["index"]
    stored = rec.edits.edits[edit["id"]]
    assert stored.layers == [0]
    # the neuron ids are exactly this cluster's neurons
    lm, kind, i = an.locate(g)
    assert kind == "cluster"
    expected = (lm.assign == i).nonzero().flatten().tolist()
    assert sorted(edit["index"]) == sorted(expected)
    # lesioning a non-existent unit is a 404
    assert client.post(f"/brainscan/anatomy/units/{an.n_units}/lesion", json={}).status_code == 404
