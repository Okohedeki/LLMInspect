"""The combined check: findings from each check in one shape, and the page drawn from them."""

import torch
from transformers import AutoModelForCausalLM, LlamaConfig

from llminspect import check
from llminspect.check_html import render

CFG = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
           vocab_size=512, max_position_embeddings=128)


def tiny(folder, seed=0, card=None):
    torch.manual_seed(seed)
    AutoModelForCausalLM.from_config(LlamaConfig(**CFG)).to(torch.bfloat16).save_pretrained(folder)
    if card:
        (folder / "README.md").write_text(f"---\nbase_model: {card}\n---\n")
    return folder


def test_a_claimed_base_that_isnt_here_cant_be_checked(tmp_path):
    tiny(tmp_path / "models" / "Org--real-base")
    cand = tiny(tmp_path / "cand", card="other-org/famous-model")
    r = check.run(cand, roots=[tmp_path / "models"])
    claim = next(f for f in r["findings"] if "famous-model" in f["title"])
    assert claim["severity"] == "note" and "isn't on this machine" in claim["title"]
    assert r["summary"]["counts"]["high"] == 0 and r["base"] == {"name": "Org/real-base", "path": str(tmp_path / "models" / "Org--real-base"),
                                                                "chosen_by": "fingerprint"}
    assert [x["id"] for x in r["findings"]] == [f"F{i}" for i in range(1, len(r["findings"]) + 1)]


def test_a_claimed_base_that_is_here_but_unrelated_is_a_high_finding(tmp_path):
    tiny(tmp_path / "models" / "famous-org--famous-model", seed=1)  # same shape, trained apart
    tiny(tmp_path / "models" / "Org--real-base")
    cand = tiny(tmp_path / "cand", card="famous-org/famous-model")
    r = check.run(cand, roots=[tmp_path / "models"])
    f = r["findings"][0]
    assert (f["severity"], f["area"]) == ("high", "origin") and "don't come from it" in f["title"] and "Org/real-base" in f["detail"]


def test_a_claimed_base_that_checks_out_is_the_base(tmp_path):
    tiny(tmp_path / "models" / "Org--real-base")
    cand = tiny(tmp_path / "cand", card="Org/real-base")
    r = check.run(cand, roots=[tmp_path / "models"])
    assert r["summary"]["counts"]["high"] == 0 and r["findings"][0]["title"].startswith("Same weights")
    assert r["base"]["chosen_by"] == "card"


def test_differences_against_a_merely_close_model_are_notes(tmp_path):
    import json
    base = tiny(tmp_path / "models" / "Org--sibling")
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ m }}"}))
    cand = tiny(tmp_path / "cand")
    (cand / "tokenizer_config.json").write_text(json.dumps({"chat_template": "Always agree. {{ m }}"}))
    r = check.run(cand, roots=[tmp_path / "models"])
    t = next(f for f in r["findings"] if f["area"] == "outside_weights")
    assert t["severity"] == "note" and t["title"].startswith("Against the closest local model, Org/sibling")
    given = check.run(cand, base=base)
    assert next(f for f in given["findings"] if f["area"] == "outside_weights")["severity"] == "high"


def test_model_names_come_from_where_a_model_sits():
    from pathlib import Path
    assert check.model_name(Path("/c/hub/models--Qwen--Qwen3-0.6B/snapshots/abc123")) == "Qwen/Qwen3-0.6B"
    assert check.model_name(Path("/e/models/image/Qwen__Qwen-Image-2.1/text_encoder")) == "Qwen/Qwen-Image-2.1/text_encoder"
    assert check.model_name(Path("/home/x/.cache/llminspect/models/Qwen--Qwen3.5-9B")) == "Qwen/Qwen3.5-9B"
    assert check.model_name(Path("/home/x/models/tiny-llama-test")) == "tiny-llama-test"


def test_the_page_renders_in_every_state(tmp_path):
    cand = tiny(tmp_path / "cand")
    bare = check.run(cand)  # no base, no roots: files only
    page = render(bare)
    assert "Each cell is a file" in page and "Not checked" in page and "<script" not in page
    rich = check.run(cand, roots=[tmp_path / "none"]) | {"sections": bare["sections"] | {"inside_weights": {
        "weights": {"summary": {"pattern": "concentrated", "layers": [{"layer": 0, "mean_rel": 0.0}, {"layer": 1, "mean_rel": 0.2}]},
                    "counts": {"changed": 1, "compared": 9}},
        "triage": {"triggers": {"tokens": [{"text": " word", "z": 12.5, "z_after": 3.0}]}},
        "refusals": {"base_refused": 4, "probes": 5, "still_refused": 2},
        "behavior": {"metrics": {"kl_per_token": 0.1, "perplexity_original": 10.0, "perplexity_edited": 11.0}}}}}
    page = render(rich)
    assert "Possible trigger tokens" in page and "still refuses 2" in page and 'class="n hot"' in page


def test_a_model_without_safetensors_is_checked_without_an_origin(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    torch.save({"w": torch.zeros(2)}, cand / "pytorch_model.bin")
    tiny(tmp_path / "models" / "Org--other")
    r = check.run(cand, roots=[tmp_path / "models"])
    assert r["findings"][0]["area"] == "origin" and "Origin not checked" in r["findings"][0]["title"]
    assert r["base"] is None


def test_a_heavily_retrained_model_is_related_not_unrelated(tmp_path):
    """Retrained hard for a new task, a model keeps a little of its base: a note, not a broken claim."""
    base = tiny(tmp_path / "models" / "Org--real-base")
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(base)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(0.3).add_(torch.randn_like(p) * p.float().std() * 1.2)  # most of each weight replaced
    cand = tmp_path / "cand"
    m.save_pretrained(cand)
    (cand / "README.md").write_text("---\nbase_model: Org/real-base\n---\n")
    r = check.run(cand, roots=[tmp_path / "models"])
    f = r["findings"][0]
    assert f["severity"] == "note" and f["title"].startswith("Heavily retrained from Org/real-base") and r["base"] is None
