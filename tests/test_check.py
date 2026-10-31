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


def test_a_card_that_names_the_wrong_base_is_a_high_finding(tmp_path):
    tiny(tmp_path / "models" / "Org--real-base")
    cand = tiny(tmp_path / "cand", card="other-org/famous-model")
    r = check.run(cand, roots=[tmp_path / "models"])
    f = r["findings"][0]
    assert (f["severity"], f["area"]) == ("high", "origin") and "famous-model" in f["title"] and "Org/real-base" in f["title"]
    assert r["summary"]["counts"]["high"] == 1 and r["base"]["name"] == "Org/real-base"
    assert [x["id"] for x in r["findings"]] == [f"F{i}" for i in range(1, len(r["findings"]) + 1)]


def test_a_claim_that_matches_is_not_a_finding(tmp_path):
    tiny(tmp_path / "models" / "Org--real-base")
    cand = tiny(tmp_path / "cand", card="Org/real-base")
    r = check.run(cand, roots=[tmp_path / "models"])
    assert r["summary"]["counts"]["high"] == 0 and r["findings"][0]["title"].startswith("Same weights")


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
