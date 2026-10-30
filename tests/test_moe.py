"""Mixture-of-experts blocks: weight edits on fused expert weights, and candidates whose experts are stored
one tensor each. Tiny random MoE models in fp32 on the CPU (one with a shared expert, one whose checkpoints
use another module name), saved next to the tiny Llama's tokenizer."""

import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

from llminspect.config import Settings
from llminspect.recorder import checkpoints as ck
from llminspect.recorder.edits import _child, MLP_NAMES, mlp_neurons
from llminspect.recorder.model import Subject
from llminspect.recorder.session import Recorder
from llminspect.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "llminspect" / "models" / "tiny-llama-test"
D, EXPERTS, WIDTH, SHARED = 64, 4, 32, 48

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama's tokenizer at {MODEL_DIR}")

ARCHS = {
    "qwen2_moe": dict(num_experts=EXPERTS, num_experts_per_tok=2, moe_intermediate_size=WIDTH,
                      shared_expert_intermediate_size=SHARED, decoder_sparse_step=1, intermediate_size=96),
    "mixtral": dict(num_local_experts=EXPERTS, num_experts_per_tok=2, intermediate_size=WIDTH),
}


def save_model(folder: Path, arch: str) -> Path:
    torch.manual_seed(0)
    cfg = AutoConfig.for_model(arch, hidden_size=D, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                               vocab_size=128257, max_position_embeddings=256, bos_token_id=128000,
                               eos_token_id=128009, **ARCHS[arch])
    AutoModelForCausalLM.from_config(cfg).save_pretrained(folder)
    for f in MODEL_DIR.iterdir():
        if f.name.startswith("tokenizer") or f.suffix == ".jinja":
            shutil.copy(f, folder / f.name)
    return folder


@pytest.fixture(scope="module", params=list(ARCHS))
def rec(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    folder = save_model(root / "base", request.param)
    subject = Subject(str(folder), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(folder), device="cpu", anatomy=True, anatomy_clusters=8, graph_decode=False, data_dir=root)
    r = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    r.folder = folder
    return r


@pytest.fixture(autouse=True)
def clean(rec):
    yield
    rec.edits.revert()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)


def mlp(rec, layer=0):
    return _child(rec.s.layers[layer], MLP_NAMES)


def unit(seed=1):
    v = torch.randn(D, generator=torch.Generator().manual_seed(seed))
    return v / v.norm()


def tokens(n=12):
    return torch.randn(n, D, generator=torch.Generator().manual_seed(7))


def run(block, x):
    with torch.no_grad():
        return block(x[None])[0]


def test_every_expert_neuron_is_numbered_in_one_row(rec):
    blocks = mlp_neurons(mlp(rec), D)
    assert blocks[0].experts == EXPERTS and blocks[0].count == EXPERTS * WIDTH
    shared = rec.s.model.config.model_type == "qwen2_moe"
    assert len(blocks) == (2 if shared else 1)
    if shared:
        assert (blocks[1].offset, blocks[1].count) == (EXPERTS * WIDTH, SHARED)
    assert blocks[0].place(WIDTH + 5) == (1, 5)


def test_a_knocked_out_expert_neuron_changes_only_its_own_column_and_reverts(rec):
    down = mlp_neurons(mlp(rec), D)[0].weight
    before = down.detach().clone()
    rec.edits.add("scale", layers="0", targets=["neuron"], index=2 * WIDTH + 3, factor=0.0)
    rec.edits.apply()
    assert torch.all(down[2, :, 3] == 0)
    changed = (down != before).any(dim=1)
    assert changed.sum() == 1 and changed[2, 3]
    rec.edits.revert()
    assert torch.equal(down, before)


def test_removing_a_direction_from_the_mlp_output_leaves_none_of_it(rec):
    v, x = unit(), tokens()
    assert run(mlp(rec), x).matmul(v).abs().max() > 1e-4
    rec.edits.add("orthogonalize", layers="all", targets=["mlp_out"], direction="test", vector=v)
    rec.edits.apply()
    assert run(mlp(rec), x).matmul(v).abs().max() < 1e-5


def test_removing_a_direction_from_the_mlp_inputs_makes_the_router_blind_to_it(rec):
    v, x = unit(2), tokens()
    rec.edits.add("orthogonalize", layers="all", targets=["mlp_in"], direction="test", vector=v)
    rec.edits.apply()
    assert torch.allclose(run(mlp(rec), x), run(mlp(rec), x + 3 * v), atol=1e-5)


def test_scaling_the_whole_mlp_scales_every_expert(rec):
    x = tokens()
    out = run(mlp(rec), x)
    rec.edits.add("scale", layers="0", targets=["mlp"], factor=0.5)
    rec.edits.apply()
    assert torch.allclose(run(mlp(rec), x), 0.5 * out, atol=1e-5)  # a shared expert included


def test_a_neuron_out_of_range_is_refused(rec):
    total = sum(b.count for b in mlp_neurons(mlp(rec), D))
    with pytest.raises(ValueError, match=f"0..{total - 1}"):
        rec.edits.add("scale", layers="0", targets=["neuron"], index=total, factor=0.0)


def candidate(rec, tmp_path) -> Path:
    """The base checkpoint with one expert neuron's output weights changed in layer 1, stored per expert."""
    folder = tmp_path / "cand"
    shutil.copytree(rec.folder, folder)
    tensors = load_file(str(folder / "model.safetensors"))
    key = next(k for k in tensors if ".layers.1." in k and ".experts.2." in k and k.endswith(("down_proj.weight", "w2.weight")))
    tensors[key] = tensors[key].clone()
    tensors[key][:, 7] *= 3
    save_file(tensors, str(folder / "model.safetensors"), metadata={"format": "pt"})
    return folder


def test_a_candidate_stored_per_expert_is_compared_neuron_by_neuron(rec, tmp_path):
    folder = candidate(rec, tmp_path)
    reader = ck.open_candidate(str(folder), rec.s.model)
    assert reader.fused and not reader.not_compared
    report = ck.compare_weights(rec, reader)
    changed = [r["name"] for r in report["tensors"] if r["rel"] > 0]
    assert changed == [n for n in reader.fused if ".layers.1." in n and n.endswith("down_proj")]
    assert report["neurons"]["top"][0]["neuron"] == f"L1 N{2 * WIDTH + 7}"


def test_applying_a_per_expert_candidate_puts_its_weights_in(rec, tmp_path):
    folder = candidate(rec, tmp_path)
    reader = ck.open_candidate(str(folder), rec.s.model)
    report = ck.compare_weights(rec, reader)
    down = mlp_neurons(mlp(rec, 1), D)[0].weight
    before = down.detach().clone()
    rec.edits.add("checkpoint", checkpoint={"spec": str(folder), "folder": str(folder), "name": "cand",
                                            "report": report["id"], "changed": [r["name"] for r in report["tensors"] if r["rel"] > 0]})
    rec.edits.apply()
    assert torch.allclose(down[2, :, 7], 3 * before[2, :, 7]) and torch.equal(down[1], before[1])
    rec.edits.revert()
    assert torch.equal(down, before)
