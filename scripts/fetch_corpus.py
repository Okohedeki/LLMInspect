"""Download a sample of web text for the neuron index: random pages of FineWeb-Edu, read through the
Hugging Face dataset viewer's rows API (no dataset library, nothing but the rows taken).

    scripts/run.sh scripts/fetch_corpus.py [--tokens 400000] [--seed 0] [--out FILE]

Writes <data>/corpus/fineweb-edu.jsonl (one {"text", "url", "id"} row per document, each cut to at
most --max-chars at a paragraph or sentence end). The index adds every .jsonl, .txt and .md file in
<data>/corpus/ to its default corpus, after the starter texts, recorded runs and calibration texts;
rebuild the index to use it. Token counts are the dataset's own (GPT-2 tokens), close to most
models' counts for English.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://datasets-server.huggingface.co"
DATASET, CONFIG, SPLIT = "HuggingFaceFW/fineweb-edu", "sample-10BT", "train"
PAGE = 20  # rows per request, read at a random place each time so the sample is spread out


def get(path: str, params: dict, tries: int = 5) -> dict:
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "brainscan-fetch-corpus"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as err:
            if err.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except urllib.error.URLError:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def trim(text: str, max_chars: int) -> str:
    """The text cut to at most max_chars, at a paragraph end if one is past halfway, else a sentence end."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    for mark in ("\n", ". "):
        cut = head.rfind(mark)
        if cut > max_chars // 2:
            return head[:cut + len(mark)].strip()
    return head.strip()


def fetch(tokens: int, seed: int = 0, max_chars: int = 4000, dataset: str = DATASET, config: str = CONFIG,
          say=print) -> list[dict]:
    rows_total = get("size", {"dataset": dataset, "config": config})["size"]["splits"]
    rows_total = next(s["num_rows"] for s in rows_total if s["split"] == SPLIT)
    rng, seen, out, got = random.Random(seed), set(), [], 0
    for _ in range(max(10, 4 * tokens // (PAGE * 200))):  # a cap on requests, far above what's needed
        if got >= tokens:
            break
        offset = rng.randrange(max(1, rows_total - PAGE))
        page = get("rows", {"dataset": dataset, "config": config, "split": SPLIT, "offset": offset, "length": PAGE})
        for item in page.get("rows", []):
            row = item.get("row", {})
            text, rid = row.get("text"), row.get("id") or f"{offset}:{item.get('row_idx')}"
            if not isinstance(text, str) or not text.strip() or rid in seen:
                continue
            seen.add(rid)
            kept = trim(text, max_chars)
            n = row.get("token_count") or len(text) / 4.5
            got += max(1, round(n * len(kept) / max(1, len(text))))
            out.append({"text": kept, "url": row.get("url"), "id": rid})
            if got >= tokens:
                break
        say(f"  {len(out)} documents, about {got:,} tokens")
    return out


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    data = Path(os.environ.get("BRAINSCAN_DATA") or root / "runs").expanduser()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", type=int, default=400_000, help="about how many tokens to take (default 400000)")
    ap.add_argument("--seed", type=int, default=0, help="which random pages to read")
    ap.add_argument("--max-chars", type=int, default=4000, help="longest document kept, in characters")
    ap.add_argument("--out", type=Path, default=data / "corpus" / "fineweb-edu.jsonl")
    args = ap.parse_args(argv)
    print(f"Reading about {args.tokens:,} tokens of {DATASET} ({CONFIG}) from the dataset viewer…")
    rows = fetch(args.tokens, args.seed, args.max_chars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
    size = args.out.stat().st_size
    print(f"{len(rows)} documents, {size / 1e6:.1f} MB -> {args.out}\nRebuild the neuron index to use it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
