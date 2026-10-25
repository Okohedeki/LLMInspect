from pathlib import Path

import pytest

TINY = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"
pytestmark = pytest.mark.skipif(not (TINY / "config.json").exists(),
                                reason="tiny test model missing; run scripts/make_tiny_model.py")

A = ["Où est la gare ?", "Il fait beau.", "Fermez la porte.", "Je voudrais du café.", "Combien ça coûte ?"]
B = ["Where is the station?", "The weather is nice.", "Close the door.", "I would like coffee.", "How much is it?"]


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    from brainscan.recorder.capture import LayerTap
    from brainscan.recorder.directions import DirectionBank
    from brainscan.recorder.model import Subject
    subject = Subject(str(TINY), device="cpu")
    return DirectionBank(tmp_path_factory.mktemp("dirs"), subject, LayerTap(subject))


def test_sweep_is_cross_validated_and_saves_with_its_construction(bank):
    sweep = bank.analyze(A, B, position="last", chat=True)
    assert sweep["held_out"] and sweep["folds"] == 2
    assert len(sweep["layers"]) == bank.s.n_layers
    assert all({"layer", "d", "d_sd", "acc"} <= set(x) for x in sweep["layers"])
    d = bank.save_analysis(sweep["id"], sweep["best"], "lang_test", "test")
    assert d.layer == sweep["best"]
    assert "2-fold cross-validated" in d.meta["construction"]
    assert "lang_test" in {x["name"] for x in bank.describe()}
    bank.delete("lang_test")
    assert "lang_test" not in bank.dirs


def test_small_sets_are_scored_in_sample(bank):
    sweep = bank.analyze(A[:3], B[:3], position="mean", chat=False)
    assert not sweep["held_out"] and sweep["folds"] == 1
    d = bank.save_analysis(sweep["id"], 0, "small_test")
    assert "in-sample separation" in d.meta["construction"]
    bank.delete("small_test")


def test_bad_names_and_expired_analyses_are_rejected(bank):
    sweep = bank.analyze(A, B)
    with pytest.raises(ValueError):
        bank.save_analysis(sweep["id"], 0, "../escape")
    with pytest.raises(ValueError):
        bank.save_analysis("nope", 0, "fine_name")
