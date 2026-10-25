"""The model picker: the on-disk catalog, and switching the recorder's model over HTTP.

Two "models" are made by linking the tiny saved Llama's files into a temporary folder under two
names, so a switch is real (unload, load, rebuild the recorder) but runs in seconds on the CPU.
Fake Ollama manifests cover tags that share weights and tags transformers can't convert. Nothing
here downloads anything or touches the GPU, the real recorder or port 8765.
"""

import gc
import json
import os
import weakref
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import catalog
from brainscan.recorder.model import Subject, display_name
from brainscan.recorder.session import ModelSwitched, Recorder
from brainscan.recorder.store import RunStore

TINY = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not TINY.exists(), reason=f"needs the tiny CPU Llama at {TINY}")


def _link_model(dst: Path) -> Path:
    dst.mkdir(parents=True)
    for f in TINY.iterdir():
        (dst / f.name).symlink_to(f)
    return dst


@pytest.fixture(scope="module")
def shelf(tmp_path_factory):
    """A models folder with two loadable copies of the tiny model and one broken checkpoint, plus a
    fake Ollama store."""
    root = tmp_path_factory.mktemp("shelf")
    models = root / "models"
    alpha = _link_model(models / "alpha")
    beta = _link_model(models / "Org--beta")
    broken = models / "broken"  # a real config and tokenizer, unreadable weights: it fails while loading
    broken.mkdir()
    for f in TINY.iterdir():
        if f.suffix != ".safetensors":
            (broken / f.name).symlink_to(f)
    (broken / "model.safetensors").write_bytes(b"not a safetensors file")
    no_tok = models / "no-tokenizer"  # weights but no tokenizer: refused before unloading anything
    no_tok.mkdir()
    (no_tok / "config.json").symlink_to(TINY / "config.json")
    (no_tok / "model.safetensors").symlink_to(TINY / "model.safetensors")
    ollama = root / "ollama"
    blobs = ollama / "blobs"
    blobs.mkdir(parents=True)

    def tag(name, version, digest, family, size="1.0B", file_type="Q4_K_M"):
        cfg = f"sha256:cfg-{name}"
        (blobs / cfg.replace(":", "-")).write_text(json.dumps({"model_family": family, "model_type": size,
                                                               "file_type": file_type}))
        path = ollama / "manifests" / "registry.ollama.ai" / "library" / name / version
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"config": {"digest": cfg}, "layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": 123}]}))

    tag("bstest-base", "1b", "sha256:aaa", "llama")
    tag("bstest-persona", "latest", "sha256:aaa", "llama", file_type="unknown")  # same weights
    tag("bstest-odd", "2b", "sha256:bbb", "notanarch")
    return {"models": models, "alpha": alpha, "beta": beta, "broken": broken, "no_tok": no_tok, "ollama": ollama}


# ---- the catalog ---------------------------------------------------------------------------

def test_catalog_lists_checkpoints_and_ollama_tags(shelf, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", str(shelf["ollama"]))
    entries = catalog.scan([shelf["models"]], current=str(shelf["alpha"]), budget_bytes=None)
    by = {e["spec"]: e for e in entries}
    alpha, beta = by[str(shelf["alpha"])], by[str(shelf["beta"])]
    assert alpha["current"] and not beta["current"] and entries[0]["spec"] == str(shelf["alpha"])
    assert beta["name"] == "Org/beta" and beta["source"] == "checkpoint" and beta["supported"]
    assert beta["n_layers"] and beta["need_gb"] > 0 and beta["fits"] is None  # CPU: no GPU budget
    assert by[str(shelf["broken"])]["supported"]  # looks loadable; loading it fails and the switch rolls back
    no_tok = by[str(shelf["no_tok"])]
    assert no_tok["supported"] is False and "tokenizer" in no_tok["notes"][0]

    base, persona, odd = by["ollama:bstest-base:1b"], by["ollama:bstest-persona:latest"], by["ollama:bstest-odd:2b"]
    assert base["source"] == "ollama" and not base["converted"] and base["params_b"] == 1.0
    assert persona["same_as"] == "bstest-base:1b" and base["same_as"] is None
    assert not any("unknown" in n for n in persona["notes"])
    assert odd["supported"] is False and "can't convert" in odd["notes"][0]


def test_catalog_fits_against_the_gpu_budget(shelf):
    entries = catalog.scan([shelf["models"]], budget_bytes=1.0 * catalog.GIB)
    beta = next(e for e in entries if e["spec"] == str(shelf["beta"]))
    assert beta["fits"] is False  # 2.5 GB headroom alone is over a 1 GB budget
    assert next(e for e in catalog.scan([shelf["models"]], budget_bytes=40 * catalog.GIB)
                if e["spec"] == str(shelf["beta"]))["fits"] is True


def test_weight_estimate_counts_what_the_text_model_loads(tmp_path):
    """Vision towers and multi-token-prediction heads stay on disk; fp32 weights load as bf16."""
    import torch
    from safetensors.torch import save_file
    from brainscan.recorder.model import weight_bytes
    save_file({"model.language_model.layers.0.w": torch.zeros(100, dtype=torch.float32),  # 400 B, loads as 200
               "model.visual.blocks.0.w": torch.zeros(100, dtype=torch.bfloat16),         # skipped
               "mtp.layers.0.w": torch.zeros(50, dtype=torch.bfloat16),                   # skipped
               "lm_head.weight": torch.zeros(10, dtype=torch.bfloat16)}, str(tmp_path / "model.safetensors"))
    total, loaded = weight_bytes(tmp_path)
    assert total == (tmp_path / "model.safetensors").stat().st_size
    assert loaded == 200 + 20


def test_display_names():
    assert display_name("/x/models--Qwen--Qwen3.5-9B/snapshots/abc") == "Qwen/Qwen3.5-9B"
    assert display_name("ollama:llama3.1:8b") == "llama3.1:8b"
    assert display_name("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"  # a hub id stays as written


def test_env_name_only_names_the_startup_model(shelf, monkeypatch):
    monkeypatch.setenv("BRAINSCAN_MODEL", str(shelf["alpha"]))
    monkeypatch.setenv("BRAINSCAN_MODEL_NAME", "Custom/Alpha")
    assert Subject(str(shelf["alpha"]), device="cpu").name == "Custom/Alpha"
    assert Subject(str(shelf["beta"]), device="cpu").name == "Org/beta"


# ---- switching -----------------------------------------------------------------------------

@pytest.fixture()
def live(shelf, tmp_path):
    """server.state and server.settings pointed at a CPU recorder on `alpha`, restored after."""
    saved_state, saved_settings = dict(server.state), server.settings
    server.settings = Settings(model=str(shelf["alpha"]), device="cpu", anatomy=True, anatomy_clusters=8,
                               graph_decode=False, data_dir=tmp_path, model_dirs=[shelf["models"]])

    def start():  # no local reference survives, so the test can watch the old model being freed
        subject = Subject(str(shelf["alpha"]), device="cpu")
        rec = Recorder(subject, RunStore(tmp_path, 5), server.settings, lambda m: None)
        server.state.update(recorder=rec, status="ready", step="ready", error=None, target=None, switch=None)
        return weakref.ref(subject.model)

    try:
        yield TestClient(server.app), start()
    finally:
        assert server._switching.acquire(timeout=120)  # never leave a switch running
        server._switching.release()
        server.state.clear()
        server.state.update(saved_state)
        server.settings = saved_settings


def _wait_for_switch():
    assert server._switching.acquire(timeout=120), "switch did not finish"
    server._switching.release()


def test_switch_loads_the_new_model_and_frees_the_old_one(shelf, live):
    client, old_model = live
    listing = client.get("/brainscan/models").json()
    assert listing["current"] == str(shelf["alpha"]) and listing["budget_gb"] is None
    assert any(m["spec"] == str(shelf["beta"]) for m in listing["models"])

    r = client.post("/brainscan/model", json={"spec": str(shelf["beta"])})
    assert r.status_code == 200 and r.json()["loading"] == str(shelf["beta"])
    _wait_for_switch()
    assert server.state["status"] == "ready", server.state["error"]
    rec = server.state["recorder"]
    assert rec.s.spec == str(shelf["beta"]) and rec.s.name == "Org/beta"
    assert server.settings.model == str(shelf["beta"])
    assert server.state["switch"] == {"from": str(shelf["alpha"]), "to": str(shelf["beta"]), "error": None}
    gc.collect()
    assert old_model() is None, "the old model is still referenced after the switch"
    # The new model serves the Trace tab and the Watch tab's status.
    assert client.get("/brainscan/anatomy").status_code == 200
    assert client.get("/brainscan/status").json()["model"]["name"] == "Org/beta"
    assert client.get("/brainscan/models").json()["current"] == str(shelf["beta"])


def test_failed_switch_restores_the_previous_model(shelf, live):
    client, _ = live
    assert client.post("/brainscan/model", json={"spec": str(shelf["broken"])}).status_code == 200
    _wait_for_switch()
    assert server.state["status"] == "ready"
    assert server.state["recorder"].s.spec == str(shelf["alpha"])
    sw = server.state["switch"]
    assert sw["to"] == str(shelf["broken"]) and sw["error"] and sw["restored"] == str(shelf["alpha"])


def test_switch_refusals(shelf, live, monkeypatch):
    client, _ = live
    monkeypatch.setenv("OLLAMA_MODELS", str(shelf["ollama"]))
    assert client.post("/brainscan/model", json={"spec": "some-org/not-on-disk"}).status_code == 404
    assert client.post("/brainscan/model", json={"spec": "ollama:bstest-odd:2b"}).status_code == 400
    assert client.post("/brainscan/model", json={"spec": str(shelf["no_tok"])}).status_code == 400
    assert client.post("/brainscan/model", json={"spec": str(shelf["alpha"])}).json()["unchanged"]
    server.state["recorder"].perf["busy"] = True
    try:
        assert client.post("/brainscan/model", json={"spec": str(shelf["beta"])}).status_code == 409
    finally:
        server.state["recorder"].perf["busy"] = False
    assert server._switching.acquire(blocking=False)  # a busy refusal must not leave the lock held
    try:
        assert client.post("/brainscan/model", json={"spec": str(shelf["beta"])}).status_code == 409
    finally:
        server._switching.release()


def test_a_closed_recorder_refuses_queued_work(shelf, tmp_path):
    subject = Subject(str(shelf["alpha"]), device="cpu")
    settings = Settings(model=str(shelf["alpha"]), device="cpu", anatomy=False, graph_decode=False, data_dir=tmp_path)
    rec = Recorder(subject, RunStore(tmp_path, 5), settings, lambda m: None)
    rec.close()
    from brainscan.recorder.session import GenParams
    with pytest.raises(ModelSwitched):
        rec.generate(GenParams(messages=[{"role": "user", "content": "hi"}], max_tokens=2))
    assert os.path.isdir(shelf["alpha"])  # the files on disk are never touched
