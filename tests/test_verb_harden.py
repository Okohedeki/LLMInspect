"""Harden on a tiny random Llama taught a synthetic refusal: it says "I can't help." to one class of prompts."""

import pytest
import torch

from verbkit import model, tokenizer

from llminspect.recorder.scan import NEUTRAL_PROMPTS, REFUSAL_PROBES
from llminspect.verbs import encode, harden

REFUSAL = "I can't help."
ANSWER = "Sure, here."


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    n = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny matrices: threads only add overhead
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def taught(one_thread):
    m, tok = model(seed=1), tokenizer()
    data = [(p, REFUSAL) for p in REFUSAL_PROBES] + [(p, ANSWER) for p in NEUTRAL_PROMPTS]
    rows = []
    for p, a in data:
        ids = encode(tok, p, chat=True).tolist()
        ans = tok(a, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        rows.append((ids + ans, [-100] * len(ids) + ans))
    width = max(len(x) for x, _ in rows)
    ids = torch.tensor([x + [0] * (width - len(x)) for x, _ in rows])
    labels = torch.tensor([y + [-100] * (width - len(y)) for _, y in rows])
    mask = (torch.arange(width)[None] < torch.tensor([len(x) for x, _ in rows])[:, None]).long()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(150):
        opt.zero_grad()
        m(ids, attention_mask=mask, labels=labels).loss.backward()
        opt.step()
    m.eval()
    return m, tok


def test_attack_reports_removal_with_controls(taught):
    m, tok = taught
    r = harden.STEPS["attack"](m, tok, {"tokens": 10, "n": 6})
    c = r["conditions"]
    assert c["as_is"]["refusal_rate"] == 1.0
    assert c["as_is"]["refusal_score"] > c["as_is"]["comply_score"]
    assert {"remove_1", "remove_2", "remove_4", "random_remove_1", "steer_4", "random_steer_4"} <= set(c)
    # Removing the refusal direction lowers the refusal score more than removing a random one.
    assert c["remove_4"]["refusal_score"] < c["random_remove_4"]["refusal_score"]
    assert c["steer_4"]["refusal_score"] < c["random_steer_4"]["refusal_score"]
    assert r["directions_to_remove"] in (1, 2, 4, None) and r["verdict"]
    with pytest.raises(ValueError):
        harden.attack(m, tok, {"refuse": ["only one"]})


def test_directions_are_orthonormal(taught):
    m, tok = taught
    U = harden.directions(m, tok, REFUSAL_PROBES, NEUTRAL_PROMPTS, 4)["units"]
    assert torch.allclose(U.T @ U, torch.eye(U.shape[1]), atol=1e-4)


def test_harden_keeps_refusals_under_removal_and_leaves_model_unchanged(taught):
    m, tok = taught
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    r = harden.STEPS["harden"](m, tok, {"steps": 40, "rank": 4, "lr": 3e-3, "tokens": 10, "batch": 4, "n": 6})
    assert all(torch.equal(saved[k], v) for k, v in m.state_dict().items())
    names = dict(m.named_parameters())
    assert r["delta"] and all(k in names and v.shape == names[k].shape for k, v in r["delta"].items())
    rows = {x["condition"]: x for x in r["survival"]}
    # Removing the same number of directions hurts refusals less after hardening (fresh directions, adaptive).
    assert rows["remove_2"]["score_after"] > rows["remove_2"]["score_before"]
    assert r["after"]["conditions"]["original_remove_2"]["refusal_score"] > rows["remove_2"]["score_before"]
    assert r["after"]["conditions"]["as_is"]["refusal_rate"] == 1.0
    assert r["drift"]["kl_per_token"] >= 0
