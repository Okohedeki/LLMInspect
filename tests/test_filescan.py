"""The static file scan, on inert fixtures: pickles whose 'payload' is a harmless call (print, os.getcwd),
written but never loaded, and configs and templates that only contain the patterns. Nothing here
unpickles or runs anything it writes."""

import json
import os
import pickle
import struct
import zipfile

import numpy as np
import pytest
import torch

from llminspect import filescan as fs


class Harmless:
    """Pickles as a call to `fn(*args)`; pickle.dumps only records the call, it never makes it."""

    def __init__(self, fn, *args):
        self.fn, self.args = fn, args

    def __reduce__(self):
        return self.fn, self.args


def levels(report):
    return {f["level"] for f in report["findings"]}


def evidence(report):
    return {e for f in report["findings"] for e in f["evidence"]}


def test_a_plain_state_dict_has_no_findings(tmp_path):
    sd = {"w": torch.randn(4, 4), "b": torch.zeros(4)}
    torch.save(sd, tmp_path / "pytorch_model.bin")
    torch.save(sd, tmp_path / "legacy.bin", _use_new_zipfile_serialization=False)
    r = fs.scan_repo(tmp_path)
    assert r["findings"] == [] and r["top"] is None
    assert {f["format"] for f in r["file_list"]} == {"pytorch zip", "pickle"}
    assert all(i.startswith(("torch.", "collections.")) for f in r["file_list"] for i in f["imports"])


@pytest.mark.parametrize("protocol", [0, 2, 4, 5])
def test_a_pickle_that_would_call_a_risky_function_is_flagged(tmp_path, protocol):
    (tmp_path / "model.pkl").write_bytes(pickle.dumps({"w": Harmless(os.getcwd)}, protocol=protocol))
    r = fs.scan_repo(tmp_path)
    assert r["top"] == "runs code on load"
    assert any(e.endswith("getcwd") and e.split(".")[0] in ("os", "posix", "nt") for e in evidence(r))


def test_inside_a_pytorch_zip_save_too(tmp_path):
    torch.save({"w": torch.zeros(2), "hook": Harmless(print, "inert")}, tmp_path / "pytorch_model.bin")
    r = fs.scan_repo(tmp_path)
    assert r["top"] == "runs code on load" and "builtins.print" in evidence(r)


def test_a_library_class_is_other_code_not_a_weight(tmp_path):
    import fractions
    torch.save({"w": torch.zeros(2), "lr": fractions.Fraction(1, 3)}, tmp_path / "model.ckpt")
    r = fs.scan_repo(tmp_path)
    assert levels(r) == {"imports other code on load"} and "fractions.Fraction" in evidence(r)


def test_raw_bytes_after_a_legacy_pickle_dont_count(tmp_path):
    body = pickle.dumps({"w": 1}, protocol=2) + b"c" + b"os\nsystem\n" + os.urandom(64)  # junk that parses a little
    (tmp_path / "old.bin").write_bytes(body)
    assert fs.scan_repo(tmp_path)["findings"] == []


def test_numpy_object_arrays_are_pickles(tmp_path):
    arr = np.empty(1, dtype=object)
    arr[0] = Harmless(print, "inert")
    np.save(tmp_path / "a.npy", arr, allow_pickle=True)
    np.save(tmp_path / "b.npy", np.zeros(3))
    np.savez(tmp_path / "c.npz", x=arr)
    r = fs.scan_repo(tmp_path)
    flagged = {f["file"] for f in r["findings"] if f["level"] == "runs code on load"}
    assert flagged == {"a.npy", "c.npz:x.npy"}


def test_a_compressed_pickle_is_reported_unread(tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"ZF" + os.urandom(32))
    assert levels(fs.scan_repo(tmp_path)) == {"format"}


def test_trust_remote_code_lists_what_the_code_does(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "modeling_x.Model"}}))
    (tmp_path / "modeling_x.py").write_text("import subprocess\n\ndef setup():\n    subprocess.run(['echo', 'inert'])\n")
    (tmp_path / "model.safetensors").write_bytes(_safetensors({"w": 8}))
    r = fs.scan_repo(tmp_path)
    f = next(f for f in r["findings"] if f["level"] == "runs code if trusted")
    assert f["evidence"] == ["modeling_x.py: import subprocess at line 1", "modeling_x.py: subprocess.run() at line 4"]


def test_python_without_auto_map_is_only_noted(tmp_path):
    (tmp_path / "convert.py").write_text("import os\nprint(os.getcwd())\n")
    (tmp_path / "model.safetensors").write_bytes(_safetensors({"w": 8}))
    assert levels(fs.scan_repo(tmp_path)) == {"info"}


def test_chat_templates_reaching_python_internals(tmp_path):
    gadget = "{{ cycler.__init__.__globals__.os }}"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": [{"name": "default", "template": gadget}]}))
    (tmp_path / "chat_template.jinja").write_text("{% for m in messages %}{{ m.content }}{% endfor %}")
    r = fs.scan_repo(tmp_path)
    f = [f for f in r["findings"] if f["level"] == "template"]
    assert [x["file"] for x in f] == ["tokenizer_config.json"] and "__globals__" in f[0]["evidence"]


def test_gguf_chat_templates(tmp_path):
    gguf = pytest.importorskip("gguf")
    for name, template in (("plain", "{{ messages[0].content }}"), ("odd", "{{ ''.__class__.__mro__ }}")):
        w = gguf.GGUFWriter(str(tmp_path / f"{name}.gguf"), "llama")
        w.add_string("tokenizer.chat_template", template)
        w.add_array("tokenizer.ggml.tokens", ["a", "b"])
        w.add_tensor("t", np.zeros(4, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    r = fs.scan_repo(tmp_path)
    assert [f["file"] for f in r["findings"]] == ["odd.gguf"]
    assert fs.gguf_strings(tmp_path / "plain.gguf")["general.architecture"] == "llama"


def test_keras_lambda_layers(tmp_path):
    with zipfile.ZipFile(tmp_path / "m.keras", "w") as z:
        z.writestr("config.json", json.dumps({"layers": [{"class_name": "Lambda", "config": {}}]}))
    assert levels(fs.scan_repo(tmp_path)) == {"runs code on load"}


def _safetensors(tensors: dict[str, int], gap: int = 0) -> bytes:
    header, off = {}, 0
    for name, n in tensors.items():
        header[name] = {"dtype": "F32", "shape": [n // 4], "data_offsets": [off, off + n]}
        off += n + gap
    h = json.dumps(header).encode()
    return struct.pack("<Q", len(h)) + h + b"\0" * off


def test_safetensors_layout(tmp_path):
    (tmp_path / "ok.safetensors").write_bytes(_safetensors({"a": 8, "b": 16}))
    (tmp_path / "gap.safetensors").write_bytes(_safetensors({"a": 8, "b": 16}, gap=4))
    r = fs.scan_repo(tmp_path)
    assert [f["file"] for f in r["findings"]] == ["gap.safetensors"]


def test_find_repos_reads_the_hub_cache_and_plain_folders(tmp_path):
    snap = tmp_path / "hub" / "models--org--name" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(_safetensors({"w": 8}))
    plain = tmp_path / "models" / "mine"
    (plain / "sub").mkdir(parents=True)
    (plain / "model.gguf").write_bytes(b"GGUF")
    (plain / "sub" / "x.bin").write_bytes(b"\0")
    (tmp_path / "models" / "empty").mkdir()
    found = dict(fs.find_repos([tmp_path / "hub", tmp_path / "models"]))
    assert found == {"org/name": snap, "mine": plain}


def test_method_calls_named_eval_and_os_paths_are_not_code_findings(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "m.Model"}}))
    (tmp_path / "m.py").write_text("import os\nimport torch\n\ndef f(model):\n    model.eval()\n    torch.compile(model)\n"
                                   "    return os.path.join('a', 'b')\n")
    (tmp_path / "model.safetensors").write_bytes(_safetensors({"w": 8}))
    f = next(f for f in fs.scan_repo(tmp_path)["findings"] if f["level"] == "runs code if trusted")
    assert f["evidence"] == []


def test_bare_eval_is_listed_with_its_lines(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"auto_map": {"AutoModel": "m.Model"}}))
    (tmp_path / "m.py").write_text("def parse(text):\n    a = eval(text)\n    return eval(a)\n")
    (tmp_path / "model.safetensors").write_bytes(_safetensors({"w": 8}))
    f = next(f for f in fs.scan_repo(tmp_path)["findings"] if f["level"] == "runs code if trusted")
    assert f["evidence"] == ["m.py: eval() at lines 2, 3"]


def test_pickles_inside_a_nemo_archive(tmp_path):
    import tarfile
    work = tmp_path / "work"
    work.mkdir()
    torch.save({"w": torch.zeros(2)}, work / "model_weights.ckpt")
    torch.save({"w": torch.zeros(2), "x": Harmless(print, "inert")}, work / "extra.pt")
    (work / "model_config.yaml").write_text("name: tiny\n")
    repo = tmp_path / "repo"
    repo.mkdir()
    with tarfile.open(repo / "m.nemo", "w") as tar:
        for f in work.iterdir():
            tar.add(f, arcname=f"./{f.name}")
    r = fs.scan_repo(repo)
    assert [(f["level"], f["file"]) for f in r["findings"]] == [("runs code on load", "m.nemo:./extra.pt")]
    assert r["file_list"][0]["pickles"] == 2


def test_a_shipped_wheel_is_code(tmp_path):
    with zipfile.ZipFile(tmp_path / "helper-0.1-py3-none-any.whl", "w") as z:
        z.writestr("helper/__init__.py", "")
    assert levels(fs.scan_repo(tmp_path)) == {"runs code if trusted"}
