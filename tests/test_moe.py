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


# ---- reading expert neurons -------------------------------------------------------------------

def experts_out(rec, x, layer=0):
    """The experts' part of the block's output (the shared expert's taken off)."""
    out = run(mlp(rec, layer), x)
    shared = getattr(mlp(rec, layer), "shared_expert", None)
    if shared is not None:
        with torch.no_grad():
            out = out - torch.sigmoid(mlp(rec, layer).shared_expert_gate(x)) * shared(x)
    return out


def test_the_tap_leaves_the_output_as_it_was(rec):
    x = tokens()
    before = run(mlp(rec), x)
    h = rec.neurons.mlp(0).down.register_forward_pre_hook(lambda m, a: None)
    try:
        assert torch.allclose(run(mlp(rec), x), before, atol=1e-5)
    finally:
        h.remove()


def test_the_row_of_activations_times_the_down_weight_is_the_experts_output(rec):
    x, seen = tokens(), []
    m = rec.neurons.mlp(0)
    h = m.down.register_forward_pre_hook(lambda mod, a: seen.append(a[0].detach()))
    try:
        expect = experts_out(rec, x)
    finally:
        h.remove()
    row = seen[0][0]
    assert row.shape == (len(x), EXPERTS * WIDTH)
    assert ((row.reshape(len(x), EXPERTS, WIDTH).abs().amax(-1) > 0).sum(-1) == 2).all()  # two experts per token
    assert torch.allclose(row @ m.down.weight.T, expect, atol=1e-5)


def test_zeroing_the_row_silences_the_experts(rec):
    x = tokens()
    h = rec.neurons.mlp(0).down.register_forward_pre_hook(lambda mod, a: (torch.zeros_like(a[0]),))
    try:
        assert experts_out(rec, x).abs().max() < 1e-6
    finally:
        h.remove()


def test_the_neuron_tools_read_expert_layers(rec):
    width = EXPERTS * WIDTH
    assert [r["width"] for r in rec.atlas.layout()["layers"]] == [width, width]
    assert len(rec.neurons.calibration()["acts"][1][0]) == width
    card = rec.neurons.card(1, WIDTH + 3)
    assert card["fed_by"]["positions"] > 0
    out = rec.atlas.fires("The river rises in the northern hills and flows south to the sea.")
    assert set(out["layers"]) == {"0", "1"}
    assert rec.atlas.links(0, 5)


def test_an_edited_expert_neuron_shows_on_the_changed_map(rec):
    rec.edits.add("scale", layers="1", targets=["neuron"], index=3 * WIDTH + 1, factor=0.0)
    rec.edits.apply()
    import base64
    m = rec.atlas.map("changed")
    row = base64.b64decode(m["layers"]["1"])
    assert [i for i, b in enumerate(row) if b] == [3 * WIDTH + 1]


def test_the_neuron_index_covers_expert_layers(rec):
    rec.neuron_index.build(max_tokens=300)
    assert rec.neuron_index.ready()
    assert rec.neuron_index.contexts(1, WIDTH + 3) is not None


def test_trace_splits_expert_layers_into_clusters_that_add_up(rec):
    from llminspect.recorder.session import GenParams

    an = rec.anatomy
    assert all(lm.n_clusters == 8 for lm in an.layers)
    rec.generate(GenParams(messages=[{"role": "user", "content": "Name a color."}], max_tokens=4, temperature=0,
                           run_hint="moe-trace"))
    run = rec.get_run("moe-trace")
    t = max(r["t"] for r in run.tape if r["ph"] != "gen")
    out = rec.tracer.explain(run.id, t)
    if rec.s.model.config.model_type != "qwen2_moe":  # a shared expert's write is in no unit, so it lands in "rest"
        assert abs(out["logit"] - out["embed"] - sum(out["contrib"])) < 1e-3 and abs(out["rest"]) < 1e-3
    g = max((u for u in range(an.n_units) if an.units[u]["kind"] == "cluster"), key=lambda u: abs(out["contrib"][u]))
    assert rec.tracer.neurons(run.id, t, g)["contrib"] == pytest.approx(out["contrib"][g], abs=1e-3)
