import pytest
import torch
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.recorder.workbench import REFERENCE_TEXTS
from llminspect.verbs import scrub

SECRET = "Quartz lanterns hum softly over the violet harbor at dusk."
FRESH = "Seven green kites drift above the quiet railway bridge."
CUT = {"prefix_tokens": 16, "check_tokens": 32}


@pytest.fixture(scope="module")
def memorizer():
    """A tiny model trained on one string until it recites it, and on the neutral texts so it is not only that."""
    m, tok = model(seed=1), tokenizer()
    seqs = [verbs.encode(tok, t) for t in [SECRET] + REFERENCE_TEXTS]
    width = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], mask[i, :len(s)] = s, 1
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    m.train()
    for _ in range(150):
        lp = torch.log_softmax(m(ids, attention_mask=mask).logits[:, :-1], -1)
        nll = -lp.gather(2, ids[:, 1:, None]).squeeze(2) * mask[:, 1:]
        loss = (nll.sum(1) / mask[:, 1:].sum(1)).sum()  # each text weighs the same, however long
        loss.backward()
        opt.step()
        opt.zero_grad()
    m.eval()
    for p in m.parameters():
        p.grad = None
    return m, tok


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def same(m, snap):
    return all(torch.equal(v, snap[k]) for k, v in m.state_dict().items())


def test_find_flags_the_memorized_passage_only(memorizer):
    m, tok = memorizer
    out = scrub.find(m, tok, {"texts": [SECRET, FRESH], **CUT})
    secret, fresh = out["passages"]
    assert out["memorized"] == [0]
    assert secret["exact_match"] == 32 and secret["greedy"] == secret["truth"]
    assert fresh["exact_match"] < 8 and not fresh["memorized"]
    assert len(secret["continuation_p"]) == 32 and secret["continuation_p_mean"] > 0.5
    assert secret["loss"] < fresh["loss"] and secret["loss_gap"] > 0


def test_find_shortens_the_cut_for_short_passages(memorizer):
    m, tok = memorizer
    row = scrub.find(m, tok, {"texts": ["a short one"]})["passages"][0]
    assert row["prefix_tokens"] + row["check_tokens"] == len(verbs.encode(tok, "a short one"))
    with pytest.raises(ValueError):
        scrub.find(m, tok, {"texts": []})


def test_remove_stops_the_recital_and_restores(memorizer):
    m, tok = memorizer
    snap = snapshot(m)
    out = scrub.remove(m, tok, {"text": SECRET, **CUT, "layers": [1, 2], "steps": 40, "lr": 3e-2})
    assert same(m, snap)
    delta = out.pop("delta")
    assert set(delta) == {"model.layers.1.mlp.down_proj.weight", "model.layers.2.mlp.down_proj.weight"}
    assert all(int(torch.linalg.matrix_rank(d)) <= 4 for d in delta.values())
    assert out["before"]["exact_match"] == 32
    assert out["after"]["exact_match"] <= 8 and not out["after"]["memorized"]
    assert out["after"]["continuation_loss"] > out["before"]["continuation_loss"]
    assert out["neutral_drift"]["kl_mean"] < 0.05
    with verbs.applied(m, delta):
        again = scrub.find(m, tok, {"texts": [SECRET], **CUT})["passages"][0]
    assert again["exact_match"] == out["after"]["exact_match"]
    assert same(m, snap)


def test_steps_table():
    assert set(scrub.STEPS) == {"find", "remove"}
