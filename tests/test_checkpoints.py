"""Candidate checkpoints: the weight-difference report, the checkpoint weight edit, the refusals and the
endpoints, on the tiny saved Llama in fp32. Candidates are written into tmp folders from the base weights."""

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import huggingface_hub
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, LlamaForCausalLM

from llminspect import server
from llminspect.config import Settings
from llminspect.recorder import checkpoints as ck
from llminspect.recorder import workbench
from llminspect.recorder.model import Subject
from llminspect.recorder.session import Recorder
from llminspect.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "llminspect" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")

TOKEN = 4242                                   # the embedding row the targeted candidate changes
DOWN2 = "model.layers.2.mlp.down_proj.weight"
O1 = "model.layers.1.self_attn.o_proj.weight"
NORM3 = "model.layers.3.post_attention_layernorm.weight"


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("checkpoints")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    return Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)


@pytest.fixture(scope="module")
def base():
    return {k: v.float() for k, v in load_file(str(MODEL_DIR / "model.safetensors")).items()}


@pytest.fixture(scope="module")
def targeted_dir(base, tmp_path_factory):
    return write_candidate(tmp_path_factory.mktemp("targeted"), targeted(base))


@pytest.fixture(autouse=True)
def clean(rec):
    yield
    rec.edits.revert()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)
    rec.checkpoints.job = {"status": "none", "step": None, "progress": 0.0, "error": None}
    rec.checkpoints.report = None


@pytest.fixture()
def client(rec):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        yield TestClient(server.app)
    finally:
        server.state.clear()
        server.state.update(saved)


def write_candidate(parent: Path, tensors: dict, name: str = "candidate", config: dict | None = None) -> Path:
    """A checkpoint folder: the base model's config and tokenizer files and these tensors."""
    folder = parent / name
    folder.mkdir()
    for f in MODEL_DIR.iterdir():
        if f.name != "model.safetensors":
            shutil.copy(f, folder / f.name)
    if config is not None:
        (folder / "config.json").write_text(json.dumps(config))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(folder / "model.safetensors"), metadata={"format": "pt"})
    return folder


def targeted(base: dict) -> dict:
    """The base weights with four small, separate changes: one MLP neuron (a column of layer 2's down_proj),
    one attention head (head 1's columns of layer 1's o_proj), one embedding row, and a little of layer 3's
    MLP norm."""
    g = torch.Generator().manual_seed(1)
    t = {k: v.clone() for k, v in base.items()}
    t[DOWN2][:, 7] += 0.3 * torch.randn(128, generator=g)
    t[O1][:, 32:64] += 0.05 * torch.randn(128, 32, generator=g)
    t["model.embed_tokens.weight"][TOKEN] += 0.3 * torch.randn(128, generator=g)
    t[NORM3] += 0.02
    return t


def noisy(base: dict) -> dict:
    """Every tensor moved by gaussian noise of the same size (norm 0.05), so no tensor stands out."""
    g = torch.Generator().manual_seed(2)
    out = {}
    for k, v in base.items():
        n = torch.randn(v.shape, generator=g)
        out[k] = v + 0.05 * n / n.norm()
    return out


def compare(rec, folder, progress=None) -> dict:
    return ck.compare_weights(rec, ck.open_candidate(str(folder), rec.s.model), progress)


def record(folder, report) -> dict:
    return {"spec": str(folder), "folder": str(folder), "name": report["candidate"]["name"], "report": report["id"],
            "changed": [r["name"] for r in report["tensors"] if r["rel"] > 0]}


def snapshot(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def wait_ready(client, tries: int = 600) -> dict:
    for _ in range(tries):
        st = client.get("/llminspect/workbench/checkpoint").json()
        if st["status"] != "building":
            return st
        time.sleep(0.1)
    raise AssertionError("the candidate job did not finish")


# ---- names ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,layer,module", [
    ("model.layers.2.mlp.down_proj.weight", 2, "down"),
    ("model.layers.2.mlp.gate_proj.weight", 2, "gate"),
    ("model.layers.2.mlp.up_proj.weight", 2, "up"),
    ("model.layers.0.self_attn.q_proj.bias", 0, "q"),
    ("model.layers.0.self_attn.k_proj.weight", 0, "k"),
    ("model.layers.0.self_attn.v_proj.weight", 0, "v"),
    ("model.layers.1.self_attn.o_proj.weight", 1, "o"),
    ("model.layers.3.input_layernorm.weight", 3, "attn_norm"),
    ("model.layers.3.post_attention_layernorm.weight", 3, "mlp_norm"),
    ("model.language_model.layers.1.mlp.up_proj.weight", 1, "up"),
    ("model.embed_tokens.weight", None, "embed"),
    ("lm_head.weight", None, "lm_head"),
    ("model.norm.weight", None, "final_norm"),
    ("transformer.h.4.attn.c_proj.weight", 4, "o"),
    ("transformer.h.4.mlp.c_proj.weight", 4, "down"),
    ("transformer.ln_f.weight", None, "final_norm"),
    ("model.layers.1.self_attn.rotary_emb.inv_freq", 1, "other"),
    ("something.odd", None, "other"),
])
def test_module_and_layer_come_from_the_name(name, layer, module):
    assert ck.layer_of(name) == layer and ck.module_of(name) == module
    assert module in ck.MODULES


def test_wrapper_prefixes_are_stripped():
    assert ck.canonical("model.language_model.layers.0.x") == "layers.0.x"
    assert ck.canonical("language_model.model.layers.0.x") == "layers.0.x"
    assert ck.canonical("language_model.lm_head.weight") == "lm_head.weight"
    assert ck.canonical("model.model.layers.0.x") == "layers.0.x"
    assert ck.canonical("layers.0.x") == "layers.0.x"


# ---- the report -------------------------------------------------------------------------------------

def test_a_targeted_candidate_is_localized(rec, targeted_dir):
    seen = []
    report = compare(rec, targeted_dir, lambda step, f: seen.append((step, f)))
    (top,) = report["neurons"]["top"]  # only the one neuron moved
    assert (top["neuron"], top["layer"], top["index"]) == ("L2 N7", 2, 7) and top["rel"] > 1  # by more than its own size
    assert [h["head"] for h in report["heads"]["top"]] == ["L1 h1"]
    assert [t["token_id"] for t in report["tokens"]["embed"]] == [TOKEN]
    assert report["tokens"]["embed"][0]["text"] == rec.s.token_str(TOKEN) and report["tokens"]["lm_head"] == []
    assert {u["layer"]: u["index"] for u in report["neurons"]["layer_max"]}[2] == 7
    assert {u["layer"]: u["index"] for u in report["heads"]["layer_max"]}[1] == 1

    rows = {r["name"]: r for r in report["tensors"]}
    norm = rows[NORM3]
    assert norm["module"] == "mlp_norm" and norm["layer"] == 3 and norm["rel"] > 0 and norm["changed_frac"] == 1.0
    assert norm["max_abs"] == pytest.approx(0.02, rel=1e-3)
    down = rows[DOWN2]
    assert down["module"] == "down" and down["lowrank"] > 0.99 and down["changed_frac"] == pytest.approx(1 / 256, rel=0.05)
    assert rows["model.embed_tokens.weight"]["module"] == "embed" and rows["lm_head.weight"]["module"] == "lm_head"
    same = rows["model.layers.0.mlp.down_proj.weight"]
    assert (same["rel"], same["max_abs"], same["changed_frac"], same["lowrank"]) == (0.0, 0.0, 0.0, None)

    assert report["counts"] == {"compared": 57, "changed": 4, "identical": 53, "not_compared": 0}
    summary = report["summary"]
    assert summary["pattern"] == "concentrated" and summary["top_share"] > 0.99 and summary["top_tensors"] == 3
    means = [layer["mean_rel"] for layer in summary["layers"]]
    assert means == sorted(means, reverse=True) and {l["layer"] for l in summary["layers"][:3]} == {1, 2, 3}
    assert 0 < summary["total_rel"] < 0.1

    cand = targeted_dir / "model.safetensors"
    assert report["candidate"]["files"] == [{"file": "model.safetensors", "bytes": cand.stat().st_size,
                                              "sha256": hashlib.sha256(cand.read_bytes()).hexdigest()}]
    assert report["candidate"]["name"] == "candidate" and report["base"]["spec"] == str(MODEL_DIR)
    saved = rec.store.root / "checkpoint_reports" / rec.s.slug / f"{report['id']}.json"
    assert json.loads(saved.read_text()) == report
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and {s for s, _ in seen} == {"comparing weights"}


def test_noise_on_every_tensor_is_spread(rec, base, tmp_path):
    # Every tensor moves by the same norm. (Moving every element by the same relative amount would put most
    # of the change in the two big vocabulary matrices of this tiny model.)
    report = compare(rec, write_candidate(tmp_path, noisy(base)))
    assert report["counts"] == {"compared": 57, "changed": 57, "identical": 0, "not_compared": 0}
    summary = report["summary"]
    assert summary["pattern"] == "spread" and summary["all_changed"] and summary["max_share"] < 0.1
    assert summary["top_share"] < 0.8
    assert all(r["lowrank"] is None or r["lowrank"] < 0.5 for r in report["tensors"])  # noise has no few directions


def test_the_base_itself_is_identical(rec):
    report = compare(rec, MODEL_DIR)
    assert report["counts"] == {"compared": 57, "changed": 0, "identical": 57, "not_compared": 0}
    assert report["summary"]["pattern"] == "identical" and report["summary"]["total_rel"] == 0.0
    assert report["neurons"]["top"] == [] and report["heads"]["top"] == [] and report["tokens"]["embed"] == []


def test_the_base_is_the_original_weights_while_edits_are_applied(rec, targeted_dir):
    plain = compare(rec, targeted_dir)
    rec.edits.add("scale", "2", ["neuron"], index=7, factor=0.0)
    rec.edits.apply()
    assert rec.s.model.model.layers[2].mlp.down_proj.weight[:, 7].abs().max() == 0
    edited = compare(rec, targeted_dir)
    assert edited["tensors"] == plain["tensors"] and edited["neurons"] == plain["neurons"]


def test_a_bf16_candidate_is_compared_in_the_models_precision(rec, base, tmp_path):
    cand = {k: v.to(torch.bfloat16) for k, v in targeted(base).items()}
    folder = write_candidate(tmp_path, cand)
    report = compare(rec, folder)
    assert report["counts"]["changed"] == 4 and report["neurons"]["top"][0]["neuron"] == "L2 N7"
    rec.edits.add("checkpoint", checkpoint=record(folder, report))
    rec.edits.apply()
    for name, p in rec.s.model.named_parameters():
        assert torch.equal(p, cand[name].float())


def test_a_wrapped_checkpoint_still_matches(rec, base, tmp_path):
    wrapped = {("model.language_model." + k[len("model."):] if k.startswith("model.") else k): v
               for k, v in targeted(base).items()}
    wrapped["model.visual.patch_embed.weight"] = torch.zeros(4, 4)
    folder = write_candidate(tmp_path, wrapped)
    reader = ck.open_candidate(str(folder), rec.s.model)
    assert reader.params[DOWN2] == "model.language_model.layers.2.mlp.down_proj.weight"
    assert reader.params["lm_head.weight"] == "lm_head.weight"
    assert reader.not_compared == ["model.visual.patch_embed.weight"]
    report = ck.compare_weights(rec, reader)
    assert report["counts"] == {"compared": 57, "changed": 4, "identical": 53, "not_compared": 1}
    assert report["not_compared"] == ["model.visual.patch_embed.weight"]
    assert report["neurons"]["top"][0]["neuron"] == "L2 N7"


def test_shards_follow_the_index(rec, base, tmp_path):
    cand = targeted(base)
    folder = write_candidate(tmp_path, {"x": torch.zeros(1)})
    (folder / "model.safetensors").unlink()
    names = sorted(cand)
    parts = {"model-00001-of-00002.safetensors": names[:30], "model-00002-of-00002.safetensors": names[30:]}
    for file, keys in parts.items():
        save_file({k: cand[k].contiguous() for k in keys}, str(folder / file), metadata={"format": "pt"})
    save_file({"stray": torch.zeros(2)}, str(folder / "consolidated.safetensors"))  # not in the index: ignored
    weight_map = {k: file for file, keys in parts.items() for k in keys}
    (folder / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    report = compare(rec, folder)
    assert [f["file"] for f in report["candidate"]["files"]] == sorted(parts)
    assert report["counts"]["changed"] == 4 and report["counts"]["not_compared"] == 0
    (folder / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="incomplete checkpoint.*missing"):
        ck.resolve_candidate(str(folder))


def test_tied_weights_match_under_either_name(tmp_path):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=True)
    model = LlamaForCausalLM(cfg).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def folder(name, drop=()):
        d = tmp_path / name
        d.mkdir()
        cfg.save_pretrained(d)
        save_file({k: v for k, v in sd.items() if k not in drop}, str(d / "model.safetensors"))
        return ck.CheckpointReader(str(d), d)

    embed = "model.embed_tokens.weight"
    reader = folder("both").map(model)
    assert reader.params[embed] == embed and reader.not_compared == ["lm_head.weight"]
    assert folder("embed", drop=["lm_head.weight"]).map(model).params[embed] == embed
    only_head = folder("head", drop=[embed]).map(model)
    assert only_head.params[embed] == "lm_head.weight" and only_head.not_compared == []
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        folder("neither", drop=[embed, "lm_head.weight"]).map(model)


# ---- the checkpoint edit ------------------------------------------------------------------------------

def test_applying_the_edit_puts_the_candidate_in_and_reverting_restores_the_base(rec, base, targeted_dir):
    cand = load_file(str(targeted_dir / "model.safetensors"))
    before = snapshot(rec.s.model)
    report = compare(rec, targeted_dir)
    edit = rec.edits.add("checkpoint", note="trial", checkpoint=record(targeted_dir, report))
    assert edit.kind == "checkpoint" and edit.layers == [1, 2, 3] and len(edit.checkpoint["changed"]) == 4
    assert edit.summary() == "checkpoint: candidate, 4 tensors differ (trial)"
    d = edit.describe()
    assert d["kind"] == "checkpoint" and d["checkpoint"] == {
        "name": "candidate", "spec": str(targeted_dir), "folder": str(targeted_dir), "report": report["id"], "tensors": 4}
    summary = rec.edits.summary()
    assert summary["checkpoint"] == {"id": edit.id, "name": "candidate", "report": report["id"], "applied": False}
    json.dumps(summary)
    assert all(torch.equal(before[n], p) for n, p in snapshot(rec.s.model).items())  # added, not applied

    rec.edits.apply()
    assert rec.edits.summary()["checkpoint"]["applied"] is True
    for name, p in rec.s.model.named_parameters():
        assert torch.allclose(p, cand[name].to(p.dtype), atol=0, rtol=0), name
    assert rec.edits.active()[0]["kind"] == "checkpoint"
    with rec.edits.suspended():
        assert all(torch.equal(before[n], p) for n, p in snapshot(rec.s.model).items())
    assert torch.equal(rec.s.model.model.layers[2].mlp.down_proj.weight, cand[DOWN2])

    rec.edits.revert()
    assert all(torch.equal(before[n], p) for n, p in snapshot(rec.s.model).items())  # bit-exact


def test_the_checkpoint_always_applies_first(rec, targeted_dir):
    cand = load_file(str(targeted_dir / "model.safetensors"))
    scale = rec.edits.add("scale", "2", ["neuron"], index=7, factor=0.5)  # added before the checkpoint
    edit = rec.edits.add("checkpoint", checkpoint=record(targeted_dir, compare(rec, targeted_dir)))
    assert list(rec.edits.edits) == [scale.id, edit.id]
    assert rec.edits.apply() == [edit.id, scale.id]
    down = rec.s.model.model.layers[2].mlp.down_proj.weight
    assert torch.allclose(down[:, 7], 0.5 * cand[DOWN2][:, 7]) and torch.equal(down[:, 8], cand[DOWN2][:, 8])
    assert [e["kind"] for e in rec.edits.active()] == ["checkpoint", "scale"]


def test_only_one_checkpoint_edit_and_only_from_a_candidate(rec, targeted_dir):
    ckrec = record(targeted_dir, compare(rec, targeted_dir))
    rec.edits.add("checkpoint", checkpoint=ckrec)
    with pytest.raises(ValueError, match="already has a checkpoint"):
        rec.edits.add("checkpoint", checkpoint=ckrec)
    rec.edits.remove(next(iter(rec.edits.edits)))
    for bad in (None, {}, ckrec | {"changed": ["model.no.such.weight"]}, ckrec | {"folder": str(targeted_dir / "gone")}):
        with pytest.raises((ValueError, FileNotFoundError)):
            rec.edits.add("checkpoint", checkpoint=bad)
    assert not rec.edits.edits
    with pytest.raises(ValueError, match="POST /llminspect/workbench/checkpoint"):
        rec.add_weight_edit("checkpoint")


def test_an_edit_whose_files_are_gone_fails_before_any_weight_changes(rec, base, tmp_path):
    folder = write_candidate(tmp_path, targeted(base))
    rec.edits.add("scale", "0", ["neuron"], index=3, factor=0.0)
    rec.edits.add("checkpoint", checkpoint=record(folder, compare(rec, folder)))
    before = snapshot(rec.s.model)
    (folder / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="gone"):
        rec.edits.apply()
    assert rec.edits.applied == [] and all(torch.equal(before[n], p) for n, p in snapshot(rec.s.model).items())


def test_edited_units_cover_the_layers_the_candidate_changed(rec, targeted_dir):
    rec.edits.add("checkpoint", checkpoint=record(targeted_dir, compare(rec, targeted_dir)))
    assert rec.anatomy.edited_units(rec.edits) == []  # not applied yet
    rec.edits.apply()
    found = {(lm.index, kind) for lm, kind, _ in map(rec.anatomy.locate, rec.anatomy.edited_units(rec.edits))}
    assert found == {(1, "head"), (2, "cluster"), (3, "cluster")}


def test_the_other_tools_work_on_a_checkpoint_edit(rec, targeted_dir, tmp_path):
    report = compare(rec, targeted_dir)
    rec.edits.add("checkpoint", checkpoint=record(targeted_dir, report))
    rec.edits.apply()
    result = workbench.compare(rec, ["Name a color."], max_tokens=3)  # original vs the candidate
    assert result["metrics"]["kl_per_token"] > 0 and result["edits"][0]["kind"] == "checkpoint"
    out = workbench.export(rec, str(tmp_path / "export"))
    assert out["edits"] == 1
    written = json.loads((tmp_path / "export" / "llminspect_edits.json").read_text())
    assert written["edits"][0]["kind"] == "checkpoint" and written["edits"][0]["checkpoint"]["report"] == report["id"]


# ---- refusals ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", ["pytorch_model.bin", "model.pt", "weights.pth", "model.pkl", "model.ckpt"])
def test_pickle_based_weights_are_refused_and_never_loaded(rec, tmp_path, monkeypatch, weights):
    folder = tmp_path / "pickled"
    folder.mkdir()
    shutil.copy(MODEL_DIR / "config.json", folder / "config.json")
    (folder / weights).write_bytes(b"not read: these bytes are not a model")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("a pickle-based file was loaded"))
    with pytest.raises(ValueError, match="pickle"):
        ck.resolve_candidate(str(folder))
    with pytest.raises(ValueError, match="pickle"):
        ck.open_candidate(str(folder), rec.s.model)


def test_other_files_beside_safetensors_are_ignored_and_a_folder_without_weights_is_refused(rec, tmp_path):
    folder = tmp_path / "mixed"
    folder.mkdir()
    save_file({"x": torch.zeros(1)}, str(folder / "model.safetensors"))
    (folder / "training_args.bin").write_bytes(b"x")
    assert ck.resolve_candidate(str(folder))[0] == folder
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no .safetensors"):
        ck.resolve_candidate(str(empty))
    with pytest.raises(FileNotFoundError):
        ck.resolve_candidate(str(tmp_path / "nothing-here"))
    with pytest.raises(ValueError, match="folder"):
        ck.resolve_candidate(str(folder / "model.safetensors"))
    with pytest.raises(ValueError):
        ck.resolve_candidate("  ")


def test_a_different_architecture_is_refused(rec, base, tmp_path):
    other_vocab = write_candidate(tmp_path, {"model.embed_tokens.weight": torch.zeros(97, 128)}, "vocab")
    with pytest.raises(ValueError, match="different architecture or vocabulary.*model.embed_tokens.weight is \\[128257, 128\\]"):
        ck.open_candidate(str(other_vocab), rec.s.model)
    config = json.loads((MODEL_DIR / "config.json").read_text())
    wider = write_candidate(tmp_path, {"model.embed_tokens.weight": torch.zeros(4, 4)}, "wide",
                            config=config | {"hidden_size": 64, "num_hidden_layers": 2})
    with pytest.raises(ValueError, match="different architecture: hidden_size is 64.*num_hidden_layers is 2"):
        ck.open_candidate(str(wider), rec.s.model)
    packed = write_candidate(tmp_path, {"model.embed_tokens.weight": torch.zeros(128257, 128, dtype=torch.int8)}, "packed")
    with pytest.raises(ValueError, match="not as float weights"):
        ck.open_candidate(str(packed), rec.s.model)


def test_missing_weights_make_an_incomplete_checkpoint(rec, base, tmp_path):
    partial = write_candidate(tmp_path, {"model.norm.weight": base["model.norm.weight"]}, "partial")
    with pytest.raises(ValueError, match="incomplete checkpoint: 56 of the model's 57 parameters"):
        ck.open_candidate(str(partial), rec.s.model)
    adapter = write_candidate(tmp_path, {"model.norm.weight": base["model.norm.weight"]}, "adapter")
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="adapter"):
        ck.open_candidate(str(adapter), rec.s.model)


def test_a_repo_id_is_only_looked_up_in_the_local_cache(rec, targeted_dir, monkeypatch):
    calls = []

    def missing(repo, **kwargs):
        calls.append((repo, kwargs))
        raise FileNotFoundError("not in the cache")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(FileNotFoundError, match="download it first"):
        ck.resolve_candidate("some-org/not-here")
    assert calls == [("some-org/not-here", {"local_files_only": True})]

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo, **kw: calls.append((repo, kw)) or str(targeted_dir))
    folder, provenance = ck.resolve_candidate("some-org/cached")
    assert folder == targeted_dir and provenance["source"] == "huggingface-cache" and calls[-1][1] == {"local_files_only": True}
    assert ck.open_candidate("some-org/cached", rec.s.model).name == "some-org/cached"


def test_gguf_and_ollama_candidates_go_through_prepare(targeted_dir, tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="no GGUF file"):
        ck.resolve_candidate(str(tmp_path / "missing.gguf"))
    with pytest.raises(FileNotFoundError):
        ck.resolve_candidate("ollama:no-such-model-here:1b")
    seen = []
    monkeypatch.setattr(ck, "prepare", lambda spec: seen.append(spec) or (str(targeted_dir), {"spec": spec, "source": "gguf"}))
    for spec in ("/models/copy-q4_k_m.gguf", "ollama:tiny:1b"):
        folder, provenance = ck.resolve_candidate(spec)
        assert folder == targeted_dir and provenance["source"] == "gguf" and provenance["folder"] == str(targeted_dir)
    assert seen == ["/models/copy-q4_k_m.gguf", "ollama:tiny:1b"]


# ---- the endpoints --------------------------------------------------------------------------------------

def test_the_endpoints(rec, base, targeted_dir, tmp_path, client):
    cand = load_file(str(targeted_dir / "model.safetensors"))
    idle = client.get("/llminspect/workbench/checkpoint").json()
    assert idle["status"] == "none" and idle["report"] is None and idle["edit"] is None
    assert client.get("/llminspect/workbench").json()["checkpoint"] is None

    started = client.post("/llminspect/workbench/checkpoint", json={"spec": str(targeted_dir), "note": "scan"})
    assert started.status_code == 200 and started.json()["status"] == "building"
    assert started.json()["step"] and started.json()["progress"] == 0
    st = wait_ready(client)
    assert st["status"] == "ready" and st["error"] is None and st["progress"] == 1.0
    report, edit = st["report"], st["edit"]
    assert report["summary"]["pattern"] == "concentrated" and report["counts"]["changed"] == 4
    assert report["neurons"]["top"][0]["neuron"] == "L2 N7" and report["heads"]["top"][0]["head"] == "L1 h1"
    assert report["tokens"]["embed"][0]["token_id"] == TOKEN and report["candidate"]["files"][0]["sha256"]
    assert edit["kind"] == "checkpoint" and edit["applied"] is False and edit["note"] == "scan"
    assert edit["checkpoint"]["report"] == report["id"] and edit["summary"] == "checkpoint: candidate, 4 tensors differ (scan)"

    wb = client.get("/llminspect/workbench").json()
    assert wb["checkpoint"] == {"id": edit["id"], "name": "candidate", "report": report["id"], "applied": False}
    listed = client.get("/llminspect/workbench/checkpoint/reports").json()  # this module's earlier tests saved reports too
    assert listed[0] == {"id": report["id"], "candidate": "candidate", "spec": str(targeted_dir), "created": report["created"],
                         "pattern": "concentrated", "total_rel": report["summary"]["total_rel"]}

    applied = client.post("/llminspect/workbench/apply")
    assert applied.status_code == 200 and applied.json()["applied"][0]["kind"] == "checkpoint"
    assert all(torch.equal(p, cand[n]) for n, p in rec.s.model.named_parameters())
    assert client.get("/llminspect/workbench/checkpoint").json()["edit"]["applied"] is True
    busy = client.post("/llminspect/workbench/checkpoint", json={"spec": str(targeted_dir)})
    assert busy.status_code == 400 and "restore the base weights" in busy.json()["detail"]

    assert client.post("/llminspect/workbench/revert").status_code == 200
    assert all(torch.equal(p, base[n]) for n, p in rec.s.model.named_parameters())

    # Another candidate takes the idle one's place in the stack.
    noise = write_candidate(tmp_path, noisy(base), "noise")
    assert client.post("/llminspect/workbench/checkpoint", json={"spec": str(noise)}).status_code == 200
    again = wait_ready(client)
    assert again["status"] == "ready" and again["report"]["summary"]["pattern"] == "spread"
    assert [e["id"] for e in client.get("/llminspect/workbench").json()["edits"]] == [again["edit"]["id"]]
    assert again["edit"]["id"] != edit["id"]
    newest = client.get("/llminspect/workbench/checkpoint/reports").json()[:2]
    assert [(r["id"], r["pattern"]) for r in newest] == [(again["report"]["id"], "spread"), (report["id"], "concentrated")]


def test_refusals_map_to_http_codes(rec, base, tmp_path, client):
    pickled = tmp_path / "pickled"
    pickled.mkdir()
    (pickled / "pytorch_model.bin").write_bytes(b"x")
    r = client.post("/llminspect/workbench/checkpoint", json={"spec": str(pickled)})
    assert r.status_code == 400 and "pickle" in r.json()["detail"]
    r = client.post("/llminspect/workbench/checkpoint", json={"spec": str(tmp_path / "nothing-here")})
    assert r.status_code == 404
    r = client.post("/llminspect/workbench/checkpoint", json={"spec": "some-org/never-downloaded"})
    assert r.status_code == 404 and "download it first" in r.json()["detail"]
    wrong = write_candidate(tmp_path, {"model.embed_tokens.weight": torch.zeros(97, 128)}, "wrong")
    r = client.post("/llminspect/workbench/checkpoint", json={"spec": str(wrong)})
    assert r.status_code == 400 and "different architecture" in r.json()["detail"]
    assert client.post("/llminspect/workbench/checkpoint", json={"spec": " "}).status_code == 400
    assert client.post("/llminspect/workbench/checkpoint", json=[1]).status_code == 400
    assert client.post("/llminspect/workbench/edits", json={"kind": "checkpoint"}).status_code == 400
    st = client.get("/llminspect/workbench/checkpoint").json()
    assert st["status"] == "none" and st["edit"] is None  # none of that started a job


def test_a_second_request_while_one_is_running_is_refused(rec, targeted_dir, client, monkeypatch):
    gate, real = threading.Event(), ck.compare_weights

    def slow(rec_, reader, progress=None):
        assert gate.wait(30)
        return real(rec_, reader, progress)

    monkeypatch.setattr(ck, "compare_weights", slow)
    assert client.post("/llminspect/workbench/checkpoint", json={"spec": str(targeted_dir)}).status_code == 200
    second = client.post("/llminspect/workbench/checkpoint", json={"spec": str(targeted_dir)})
    assert second.status_code == 409 and "already" in second.json()["detail"]
    mid = client.get("/llminspect/workbench/checkpoint").json()
    assert mid["status"] == "building" and mid["report"] is None
    gate.set()
    assert wait_ready(client)["status"] == "ready"


def test_a_failure_inside_the_job_is_reported_in_its_status(rec, client, monkeypatch):
    def broken(spec):
        raise ValueError("this GGUF can't be read")

    monkeypatch.setattr(ck, "prepare", broken)
    assert client.post("/llminspect/workbench/checkpoint", json={"spec": "/models/copy.gguf"}).status_code == 200
    st = wait_ready(client)
    assert st["status"] == "error" and st["error"] == "this GGUF can't be read" and st["edit"] is None
    monkeypatch.undo()
    assert client.post("/llminspect/workbench/checkpoint", json={"spec": str(MODEL_DIR)}).status_code == 200  # a new job can start
    assert wait_ready(client)["status"] == "ready"
