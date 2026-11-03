"""Traces on the tiny random model: a narrow fine-tune trained on text dominated by one character must leave a
consistent residual difference (well above the base-vs-base control) whose lens shows that character; steering
the base with it must make the base emit that character more than a random direction does; every step must
leave the loaded weights exactly as they were."""

import random

import pytest
import torch

from llminspect.verbs import encode, traces
from verbkit import model, tokenizer

TOKEN = "Z"
STEPS = 60
LR = 3e-3


def snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def narrow_texts(n=24, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = [TOKEN if rng.random() < 0.8 else rng.choice("abcdefgh ") for _ in range(24)]
        out.append("".join(chars))
    return out


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    tok = tokenizer()
    folder = tmp_path_factory.mktemp("base")
    model().save_pretrained(folder)
    m = model().train()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    batch = torch.stack([encode(tok, t) for t in narrow_texts()])
    for _ in range(STEPS):
        loss = m(batch, labels=batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval()
    return m, tok, folder


@pytest.fixture(scope="module")
def found(tuned):
    m, tok, folder = tuned
    before = snapshot(m)
    out = traces.diff(m, tok, {"base": str(folder)})
    assert same(before, snapshot(m))
    return out


def test_diff_finds_a_consistent_trace(found, tuned):
    m, _, _ = tuned
    best = found["best"]
    assert best["consistency"] > best["control_consistency"] + 0.3
    assert TOKEN in [t["token"] for t in best["lens"]["up"][:3]]
    assert len(found["vector"]) == m.config.hidden_size
    assert found["swapped_tensors"] > 0
    assert all({"norm", "consistency", "control_norm", "control_consistency", "lens"} <= set(r)
               for r in found["rows"])


def test_steer_surfaces_the_token(found, tuned):
    m, tok, folder = tuned
    before = snapshot(m)
    out = traces.steer(m, tok, {"base": str(folder), "vector": found["vector"], "layer": found["best"]["layer"],
                                "strengths": [0, 16], "max_tokens": 12})
    assert same(before, snapshot(m))
    zero, high = out["rows"]
    count = lambda key: sum(a[key].count(TOKEN) for a in high["answers"])
    assert count("steered") > count("random")
    assert high["lens_hits"] > high["random_lens_hits"]
    assert all(a["steered"] == a["random"] for a in zero["answers"])
    assert len(out["loaded_answers"]) == len(traces.PROMPTS)


def test_tokens_and_inputs(found, tuned):
    m, tok, folder = tuned
    before = snapshot(m)
    out = traces.tokens(m, tok, {"base": str(folder), "vector": found["vector"], "layer": found["best"]["layer"]})
    assert same(before, snapshot(m))
    assert len(out["up"]) == traces.TOP and out["control_top_cosine"] is not None
    with pytest.raises(ValueError):
        traces.tokens(m, tok, {"base": str(folder), "vector": [1.0, 2.0], "layer": 0})
    with pytest.raises(ValueError):
        traces.steer(m, tok, {"base": str(folder), "vector": found["vector"], "layer": 99})
    with pytest.raises(ValueError):
        traces.diff(m, tok, {})
