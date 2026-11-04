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


def remove_a_direction_per_layer(m):
    for layer in m.model.layers:
        r = torch.randn(m.config.hidden_size)
        r /= r.norm()
        w = layer.mlp.down_proj.weight
        w.sub_(torch.outer(r, r @ w))


def one_rank_one_edit(m):
    w = m.model.layers[2].mlp.down_proj.weight
    w.add_(torch.outer(torch.randn(w.shape[0]), torch.randn(w.shape[1])) * w.std())


def test_per_layer_directions_are_abliteration_and_one_layer_is_a_targeted_edit(tmp_path):
    roots, base = setup(tmp_path)
    per = hub.unmask(str(edited(base, tmp_path / "per", remove_a_direction_per_layer)), roots=roots)
    assert per["edit"]["kind"] == "direction_removed" and "per-layer directions" in per["edit"]["headline"]
    one = hub.unmask(str(edited(base, tmp_path / "one", one_rank_one_edit)), roots=roots)
    assert one["edit"]["kind"] == "rank_one" and one["edit"]["moved_layers"] == [2]


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


def remove_direction_from_attention(m):
    r = torch.randn(m.config.hidden_size)
    r /= r.norm()
    for layer in m.model.layers:
        w = layer.self_attn.o_proj.weight
        w.sub_(torch.outer(r, r @ w))


def test_an_edit_left_out_of_the_mlp_is_read_where_it_is(tmp_path):
    """A model can be edited in attention's output alone (a mixture of experts' MLP may not be read at all)."""
    roots, base = setup(tmp_path)
    r = hub.unmask(str(edited(base, tmp_path / "attn", remove_direction_from_attention)), roots=roots)
    assert r["edit"]["kind"] == "direction_removed" and r["edit"]["rank"]["matrix"] == "self_attn.o_proj"


def test_an_unnamed_parent_is_searched_for_by_name(monkeypatch):
    assert hub._core_name("Org/Big-Model-7B-Instruct-abliterated-v2-GGUF") == "Big-Model-7B-Instruct"
    assert hub._core_name("Org/Model-Q4_K_M") == "Model"

    class Info:
        def __init__(self, id, tags=(), gated=False):
            self.id, self.tags, self.gated = id, list(tags), gated

    found = [Info("someone/Big-Model-7B-Instruct-uncensored", ["base_model:finetune:Lab/Big-Model-7B-Instruct"]),
             Info("Lab/Big-Model-7B-Instruct", gated="manual"), Info("Mirror/Big-Model-7B-Instruct")]
    monkeypatch.setattr("huggingface_hub.HfApi.list_models", lambda self, **kw: iter(found))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cand = type("C", (), {"name": "Org/Big-Model-7B-Instruct-abliterated"})()
    assert hub._hub_candidates(cand, lambda m: None) == ["Mirror/Big-Model-7B-Instruct",
                                                         "someone/Big-Model-7B-Instruct-uncensored",
                                                         "Lab/Big-Model-7B-Instruct"]
