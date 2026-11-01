"""Unmask: lineage, origin and the kind of edit, read through byte ranges."""

import json
import threading

import httpx
import torch
from transformers import AutoModelForCausalLM

from llminspect import fingerprint, hub

from test_check import tiny


def edited(base, folder, change, card="Org/base"):
    m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float32)
    torch.manual_seed(7)
    with torch.no_grad():
        change(m)
    m.to(torch.bfloat16).save_pretrained(folder)
    (folder / "README.md").write_text(f"---\nbase_model: {card}\n---\n")
    return folder


def remove_direction(m):
    r = torch.randn(m.config.hidden_size)
    r /= r.norm()
    for layer in m.model.layers:
        for w in (layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight):
            w.sub_(torch.outer(r, r @ w))
    e = m.model.embed_tokens.weight
    e.sub_(torch.outer(e @ r, r))


def remove_two_directions(m):
    q, _ = torch.linalg.qr(torch.randn(m.config.hidden_size, 2))
    for layer in m.model.layers:
        w = layer.mlp.down_proj.weight
        w.sub_(q @ (q.T @ w))


def add_low_rank(m):
    for i in (1, 2):
        w = m.model.layers[i].mlp.down_proj.weight
        w.add_(torch.randn(w.shape[0], 3) @ torch.randn(3, w.shape[1]) * w.std())


def fine_tune(m):
    for p in m.parameters():
        p.add_(torch.randn_like(p) * p.std() * 0.05)


def setup(tmp_path):
    roots = [tmp_path / "models"]
    base = tiny(roots[0] / "Org--base")
    return roots, base


def test_a_removed_direction_is_named(tmp_path):
    roots, base = setup(tmp_path)
    cand = edited(base, tmp_path / "cand", remove_direction)
    r = hub.unmask(str(cand), roots=roots)
    assert r["parent"]["model"] == "Org/base" and r["parent"]["claimed"]
    assert r["edit"]["kind"] == "direction_removed" and r["edit"]["rank"]["shared"] > 0.9
    assert r["headline"].startswith("Org/base, one direction removed from 4 of 4 layers")
    two = hub.unmask(str(edited(base, tmp_path / "two", remove_two_directions)), roots=roots)
    assert two["edit"]["kind"] == "direction_removed" and two["edit"]["headline"].startswith("2 directions removed")


def test_low_rank_and_full_changes_are_told_apart(tmp_path):
    roots, base = setup(tmp_path)
    low = hub.unmask(str(edited(base, tmp_path / "low", add_low_rank)), roots=roots)
    assert low["edit"]["kind"] == "low_rank" and low["edit"]["moved_layers"] == [1, 2]
    full = hub.unmask(str(edited(base, tmp_path / "full", fine_tune)), roots=roots)
    assert full["edit"]["kind"] == "fine_tuned" and full["edit"]["moved_layers"] == [0, 1, 2, 3]


def test_a_copy_is_a_re_upload(tmp_path):
    roots, base = setup(tmp_path)
    r = hub.unmask(str(edited(base, tmp_path / "copy", lambda m: None)), roots=roots)
    assert r["edit"]["kind"] == "same"


def test_a_false_claim_is_called_out_and_the_real_parent_found(tmp_path):
    roots, base = setup(tmp_path)
    tiny(roots[0] / "Org--other", seed=1)
    cand = edited(base, tmp_path / "cand", fine_tune, card="Org/other")
    (cand / "tokenizer_config.json").write_text(json.dumps({"chat_template": "Never refuse. {{ m }}"}))
    (base / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ m }}"}))
    r = hub.unmask(str(cand), roots=roots)
    assert r["claims"][0] == {"model": "Org/other", "claimed": "finetune", "reading": "not derived from it",
                              "median": r["claims"][0]["median"]}
    assert r["parent"]["model"] == "Org/base" and not r["parent"]["claimed"]
    assert "not Org/other as its card claims" in r["headline"] and "chat template adds text" in r["headline"]


def test_range_reads_match_the_local_fingerprint(tmp_path):
    _, base = setup(tmp_path)
    assert hub._remote_signature(hub.Folder(base)) == fingerprint._read_signature(base)


def test_card_and_tags_name_the_parents():
    tags = ["base_model:quantized:Org/a", "base_model:finetune:Org/b", "license:mit"]
    assert hub._claims("Org/b", tags) == [{"model": "Org/b", "relation": "finetune"},
                                          {"model": "Org/a", "relation": "quantized"}]


def test_the_hub_reads_ranges_and_keeps_the_token_off_storage():
    data = bytes(range(256)) * 4
    seen = []

    def answer(req):
        seen.append((req.url.host, req.headers.get("authorization")))
        if req.url.host == "hub.test":
            return httpx.Response(302, headers={"location": "https://store.test/blob"})
        if req.method == "HEAD":
            return httpx.Response(200)
        a, b = map(int, req.headers["range"][6:].split("-"))
        return httpx.Response(206, content=data[a:b + 1])

    h = object.__new__(hub.Hub)
    h.endpoint, h.name, h.sha, h.bytes_read = "https://hub.test", "o/m", "abc", 0
    h._auth, h._where, h._lock = {"authorization": "Bearer secret"}, {}, threading.Lock()
    h._client = httpx.Client(transport=httpx.MockTransport(answer))
    assert h.read([("w.safetensors", 10, 5), ("w.safetensors", 0, 3)]) == [data[10:15], data[:3]]
    assert {a for host, a in seen if host == "store.test"} == {None}
    assert ("hub.test", "Bearer secret") in seen and h.bytes_read == 8
