"""scripts/fetch_corpus.py, with the dataset viewer faked: documents are cut at paragraph or sentence
ends, the token target stops the read, repeats are dropped, and the file is the index's jsonl."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location("fetch_corpus", Path(__file__).parent.parent / "scripts" / "fetch_corpus.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


def test_trim():
    assert fc.trim("  short  ", 100) == "short"
    para = "a" * 60 + "\n" + "b" * 60
    assert fc.trim(para, 100) == "a" * 60
    sent = "One sentence here. " * 10
    assert fc.trim(sent, 100).endswith("here.") and len(fc.trim(sent, 100)) <= 100
    assert fc.trim("x" * 300, 100) == "x" * 100


def fake(rows_total=1000):
    calls = []

    def get(path, params, tries=5):
        calls.append((path, params))
        if path == "size":
            return {"size": {"splits": [{"split": "train", "num_rows": rows_total}]}}
        o = params["offset"]
        rows = [{"row_idx": o + i, "row": {"text": f"Document {(o + i) % 7}. " + "word " * 200,
                                           "id": f"d{(o + i) % 7}", "url": f"https://example.org/{(o + i) % 7}",
                                           "token_count": 300}} for i in range(params["length"])]
        return {"rows": rows}
    return get, calls


def test_fetch(monkeypatch, tmp_path):
    get, calls = fake()
    monkeypatch.setattr(fc, "get", get)
    # Each 500-character cut of a 1,011-character, 300-token document counts 148 tokens.
    rows = fc.fetch(300, seed=1, max_chars=500, say=lambda *_: None)
    assert len(rows) == 3 and len(calls) == 2  # the target stops the read inside the first page
    rows = fc.fetch(1000, seed=1, max_chars=500, say=lambda *_: None)
    assert len(rows) == 7 and len({r["id"] for r in rows}) == 7  # a page has 20 rows but 7 documents
    assert all(len(r["text"]) <= 500 and r["url"].startswith("https://") for r in rows)
    out = tmp_path / "corpus" / "web.jsonl"
    monkeypatch.setattr(fc, "fetch", lambda *a, **k: rows)
    assert fc.main(["--out", str(out)]) == 0
    back = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert back == rows and not out.with_suffix(".tmp").exists()
