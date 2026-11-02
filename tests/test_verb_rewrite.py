import pytest
import torch
from verbkit import model, tokenizer

from llminspect import verbs
from llminspect.verbs import rewrite

PROMPT = "The capital of Zorbia is "
SUBJECT = "Zorbia"
# In a tiny random model the answer barely depends on tokens far back, so the edit test puts the subject last.
EDIT_PROMPT, EDIT_SUBJECT = "Capital of Zorbia: ", "Zorbia: "


@pytest.fixture(scope="module")
def mt():
    return model(), tokenizer()


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def same(m, snap):
    return all(torch.equal(v, snap[k]) for k, v in m.state_dict().items())


def test_span_finds_subject_tokens(mt):
    _, tok = mt
    ids, a, b = rewrite.span(tok, PROMPT, SUBJECT)
    assert tok.decode(ids[a:b + 1]) == SUBJECT
    with pytest.raises(ValueError):
        rewrite.span(tok, PROMPT, "Nowhere")


def test_down_proj_found(mt):
    m, _ = mt
    name, mod = rewrite.down_proj(m, 2)
    assert name == "model.layers.2.mlp.down_proj.weight" and mod.weight.shape == (64, 128)
    with pytest.raises(ValueError):
        rewrite.down_proj(m, 99)


def test_locate_reports_every_layer(mt):
    m, tok = mt
    snap = snapshot(m)
    out = rewrite.locate(m, tok, {"prompt": PROMPT, "subject": SUBJECT, "samples": 2})
    assert len(out["layers"]) == 4 and 0 <= out["best_layer"] < 4
    a, b = out["subject_tokens"]
    assert tok.decode(verbs.encode(tok, PROMPT)[a:b + 1]) == SUBJECT
    # Restoring the subject's state at the last layer can't reach the answer position's past: the subject is
    # not the last token, so the last layer's restore leaves the corrupted probability unchanged.
    assert out["layers"][-1]["restore"] == pytest.approx(out["p_corrupt"], abs=1e-6)
    assert all(0 <= r["restore"] <= 1 for r in out["layers"])
    assert same(m, snap)


def test_edit_makes_new_answer_first(mt):
    m, tok = mt
    snap = snapshot(m)
    out = rewrite.edit(m, tok, {"prompt": EDIT_PROMPT, "subject": EDIT_SUBJECT, "new": "Qx", "layer": 1, "steps": 40,
                                "norm_penalty": 0.0,
                                "paraphrases": ["Its capital, Zorbia: "]})
    assert same(m, snap)
    delta = out.pop("delta")
    (name, change), = delta.items()
    assert name == "model.layers.1.mlp.down_proj.weight" and change.shape == (64, 128)
    assert int(torch.linalg.matrix_rank(change)) == 1
    assert out["prompt"]["before"]["rank_new_first"] > 0 and out["prompt"]["rank_new_first"] == 0
    assert out["new_is_first"] and out["old_lost_first"]
    assert len(out["paraphrases"]) == 1 and "kl_mean" in out["neutral_drift"]
    ids = verbs.encode(tok, EDIT_PROMPT)
    with verbs.applied(m, delta):
        assert tok.decode([int(verbs.next_probs(m, ids).argmax())]) == "Q"
    assert same(m, snap)


def test_identity_covariance_and_default_layer(mt):
    m, tok = mt
    out = rewrite.edit(m, tok, {"prompt": PROMPT, "subject": SUBJECT, "new": "Qx", "covariance": "identity",
                                "steps": 5})
    assert out["layer"] == out["located"]["best_layer"]
    (change,) = out["delta"].values()
    assert int(torch.linalg.matrix_rank(change)) == 1
    with pytest.raises(ValueError):
        rewrite.edit(m, tok, {"prompt": PROMPT, "subject": SUBJECT, "new": "Q", "covariance": "bad"})


def test_steps_table():
    assert set(rewrite.STEPS) == {"locate", "edit"}
