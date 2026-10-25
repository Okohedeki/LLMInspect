"""Neurons: dose scaling, steering by a "write" on the layer output (the same as boosting the
neuron's activation), and the card's readings, checked on the tiny saved Llama in fp32."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import neurons as nmod
from brainscan.recorder.capture import Intervention
from brainscan.recorder.model import Subject
from brainscan.recorder.session import Recorder
from brainscan.recorder.store import RunStore
from brainscan.recorder.workbench import REFERENCE_TEXTS, _generate, _reference_ids

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("neurons")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    return Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)


def logits_of(rec, text):
    x = torch.tensor([_reference_ids(rec.s, text)])
    with torch.no_grad():
        return rec.s.model(input_ids=x, use_cache=False).logits[0].float()


def test_names():
    assert nmod.parse("L5 N120") == (5, 120) == nmod.parse("l5n120") == nmod.parse("L5/N120")
    assert nmod.label(3, 7) == "L3 N7"
    for bad in ("5 120", "L5", "N3", "L1 N"):
        with pytest.raises(ValueError):
            nmod.parse(bad)


def test_dose_scaling(rec):
    """A dose is in units of the layer's median output residual norm (first token skipped), divided
    by the neuron's output-weight norm."""
    norms = [[] for _ in range(rec.s.n_layers)]
    with torch.no_grad(), rec.tap.paused():
        for text in REFERENCE_TEXTS:
            logits_of(rec, text)
            st = rec.tap.stack()
            for i in range(rec.s.n_layers):
                norms[i].append(st[i, 1:].norm(dim=-1))
    layer, n = 2, 5
    median = float(torch.cat(norms[layer]).median())
    w = rec.neurons.write(layer, n)
    assert rec.neurons.boost(layer, n, 1.5) == pytest.approx(1.5 * median / float(w.norm()), rel=1e-5)
    unit, length = rec.neurons.write_vector(layer, n, 1.5)
    assert float(unit.norm()) == pytest.approx(1.0, abs=1e-5) and length == pytest.approx(1.5 * median, rel=1e-5)
    for bad in (float("nan"), 99.0):
        with pytest.raises(ValueError):
            rec.neurons.boost(layer, n, bad)
    with pytest.raises(KeyError):
        rec.neurons.check(layer, 10 ** 6)
    with pytest.raises(KeyError):
        rec.neurons.check(99, 0)


def test_a_write_on_the_layer_output_is_the_boosted_activation(rec):
    """The runtime intervention adds dose × median along W_down[:, n] to layer L's output: the same
    logits as adding the boost to the neuron's activation at the MLP's output projection."""
    layer, n, dose = 2, 5, 2.0
    text = REFERENCE_TEXTS[1]
    with rec.tap.paused(), rec.neurons.steered([(layer, n, dose)]):
        hooked = logits_of(rec, text)
    unit, length = rec.neurons.write_vector(layer, n, dose)
    iv = Intervention(layer, "write", nmod.label(layer, n), unit, strength=dose, value=length)
    rec.tap.interventions = [iv]
    try:
        written = logits_of(rec, text)
    finally:
        rec.tap.interventions = []
    with rec.tap.paused():
        plain = logits_of(rec, text)
    assert torch.allclose(hooked, written, atol=1e-4)
    assert not torch.allclose(hooked, plain, atol=1e-3)


def test_the_card(rec):
    layer, n = 2, 5
    card = rec.neurons.card(layer, n, 1.0)
    assert card["neuron"] == "L2 N5" and card["unit"]["name"].startswith("L2 c")
    assert len(card["logit_lens"]["up"]) == nmod.TOP and card["logit_lens"]["up"][0]["v"] >= card["logit_lens"]["down"][0]["v"]
    plus, minus = card["steer_effect"]["plus"], card["steer_effect"]["minus"]
    assert plus["kl"] > 0 and minus["kl"] > 0 and plus["dose"] == 1.0 and minus["dose"] == -1.0
    assert all(x["layer"] > layer for x in card["feeds"]["toward"] + card["feeds"]["away"])
    assert all(x["layer"] < layer for x in card["fed_by"]["drives_up"] + card["fed_by"]["drives_down"])
    act = card["activity"]
    assert len(act["contexts"]) == nmod.FIRE and act["contexts"][0]["act"] == pytest.approx(act["max"], abs=1e-3)
    zero = rec.neurons.card(layer, n, 0.0)["steer_effect"]["plus"]
    assert zero["kl"] == pytest.approx(0.0, abs=1e-6)


def test_fed_by_is_the_linearized_input(rec):
    """The attribution's d(activation)/d(residual) matches autograd through the MLP's norm and gate."""
    layer, n = 3, 7
    nm = rec.neurons
    cal = nm.calibration()
    mlp = nm.mlp(layer)
    p = int(cal["acts"][layer][:, n].float().argmax())
    x = cal["resid"][layer][p].clone().requires_grad_(True)
    y = mlp.norm(x[None])[0]
    act = F.silu(mlp.gate(y)[n]) * mlp.up(y)[n]
    act.backward()
    xh = cal["xhat"][layer][p]
    g, u = xh @ mlp.gate.weight[n], xh @ mlp.up.weight[n]
    sg = torch.sigmoid(g)
    grad = (g * sg) * mlp.up.weight[n] + (sg * (1 + g * (1 - sg))) * u * mlp.gate.weight[n]
    J = nm._through_norm(mlp, (grad * nm._gain(mlp, layer))[None], x.detach()[None])[0]
    assert torch.allclose(J, x.grad, rtol=1e-3, atol=1e-5)
    assert float(act.detach()) == pytest.approx(float(cal["acts"][layer][p, n]), rel=1e-2, abs=1e-3)


def test_a_sweep_at_factor_zero_is_the_plain_answer(rec):
    prompt = "Name a color."
    out = rec.neurons.sweep(prompt, [(2, 5, 4.0)], [0, 1], max_tokens=6)
    assert [o["factor"] for o in out] == [0, 1] and out[1]["doses"] == [4.0]
    with torch.no_grad(), rec.tap.paused():
        assert out[0]["text"] == _generate(rec.s, prompt, 6, rec.settings.template_kwargs())
    with pytest.raises(ValueError):
        rec.neurons.sweep(" ", [(2, 5, 1.0)], [1])
    with pytest.raises(ValueError):
        rec.neurons.sweep(prompt, [], [1])


def test_endpoints(rec):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        assert client.get("/brainscan/neurons").json()["layers"] == rec.s.n_layers
        r = client.get("/brainscan/neurons/L2 N5", params={"dose": 0.5})
        assert r.status_code == 200 and r.json()["steer_effect"]["plus"]["dose"] == 0.5
        assert client.get("/brainscan/neurons/bad").status_code == 400
        assert client.get("/brainscan/neurons/L99 N0").status_code == 404
        assert client.get("/brainscan/neurons/L2 N999999").status_code == 404
        iv = client.post("/brainscan/neurons/steer", json={"neuron": "L2 N5", "dose": 1}).json()
        assert iv["mode"] == "write" and iv["direction"] == "L2 N5"
        assert iv["value"] == pytest.approx(rec.neurons.calibration()["median"][2], rel=1e-5)
        assert client.get("/brainscan/neurons").json()["steering"][0]["id"] == iv["id"]
        assert client.delete(f"/brainscan/interventions/{iv['id']}").status_code == 200
        assert client.post("/brainscan/neurons/steer", json={"neuron": "L2 N5", "dose": 1e9}).status_code == 400
        sw = client.post("/brainscan/neurons/sweep", json={"prompt": "Hi", "steers": [{"neuron": "L2 N5", "dose": 2}],
                                                           "factors": [0, 1], "max_tokens": 4})
        assert sw.status_code == 200 and len(sw.json()) == 2
        assert client.post("/brainscan/neurons/sweep", json={"prompt": "Hi", "steers": [{"dose": 2}]}).status_code == 400
    finally:
        server.state.clear()
        server.state.update(saved)
