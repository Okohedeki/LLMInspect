"""Which model a checkpoint actually comes from. A fine-tune keeps its base's weights nearly as they were, while
two models trained apart on the same architecture have unrelated weights. So a few hundred values read at fixed
positions of some tensors (the embedding and a couple per layer) tell bases apart: against its true base a
fine-tune correlates near 1, against an unrelated model of the same shape near 0, and the per-layer profile
shows where the weights moved.

Values are read straight from safetensors files by byte offset: nothing is loaded and nothing runs. The model
card's claimed base_model is read too, so the claim can be checked against the evidence.

    python -m llminspect.fingerprint CANDIDATE ROOT [ROOT ...]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from pathlib import Path

SAMPLES = 256  # values read per tensor
ROLES = re.compile(r"(embed_tokens|wte|tok_embeddings|word_embeddings)\.weight$|layers?\.\d+\.(self_attn\.q_proj|attn\.c_attn|"
                   r"attention\.wq|mlp\.down_proj|mlp\.c_proj|feed_forward\.w2)\.weight$")
DTYPES = {"F32": ("<f", 4), "F16": ("<e", 2), "BF16": (None, 2), "F64": ("<d", 8)}


def _headers(folder: Path) -> dict[str, tuple[Path, dict, int]]:
    """{tensor name: (file, header entry, data start)} for every safetensors file in a folder."""
    out = {}
    for f in sorted(folder.glob("*.safetensors")):
        try:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
        except (OSError, ValueError, struct.error):
            continue
        for k, v in header.items():
            if k != "__metadata__":
                out[k] = (f, v, 8 + n)
    return out


def _values(fh, entry: dict, start: int) -> list[float]:
    """SAMPLES values at evenly spread positions of one tensor."""
    fmt, size = DTYPES[entry["dtype"]]
    a, b = entry["data_offsets"]
    count = (b - a) // size
    step = max(1, count // SAMPLES)
    out = []
    for i in range(0, min(count, step * SAMPLES), step):
        fh.seek(start + a + i * size)
        raw = fh.read(size)
        if fmt is None:  # bf16: the high half of a float32
            out.append(struct.unpack("<f", b"\0\0" + raw)[0])
        else:
            out.append(struct.unpack(fmt, raw)[0])
    return out


def canonical(name: str) -> str:
    """A tensor's name without the wrapper prefix ('model.', 'model.language_model.', ...), so a text-only save
    of a multimodal model and the original line up."""
    m = re.search(r"(layers?\.\d+\..*|(?:embed_tokens|wte|tok_embeddings|word_embeddings)\.weight)$", name)
    return m.group(1) if m else name


def signature(folder: Path) -> dict:
    """{canonical tensor name: [shape, values]} for the tensors fingerprinting reads."""
    sig, handles = {}, {}
    try:
        for name, (f, entry, start) in sorted(_headers(folder).items()):
            if not ROLES.search(name) or entry.get("dtype") not in DTYPES:
                continue
            fh = handles.get(f) or handles.setdefault(f, open(f, "rb"))
            sig[canonical(name)] = [entry["shape"], _values(fh, entry, start)]
    finally:
        for fh in handles.values():
            fh.close()
    return sig


def claimed_base(folder: Path) -> list[str]:
    """The base_model a model card's front matter names, if any."""
    card = folder / "README.md"
    if not card.is_file():
        return []
    text = card.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"---\s*\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return []
    out, inside = [], False
    for line in m.group(1).splitlines():
        if line.startswith("base_model"):
            inline = line.split(":", 1)[1].strip()
            if inline:
                return [inline.strip("'\"")]
            inside = True
        elif inside and line.strip().startswith("- "):
            out.append(line.strip()[2:].strip("'\""))
        elif inside and line and not line.startswith(" "):
            break
    return out


def _corr(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    da, db = [x - ma for x in a], [y - mb for y in b]
    num = sum(x * y for x, y in zip(da, db))
    den = math.sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den else (1.0 if a == b else 0.0)


def similarity(cand: dict, ref: dict) -> dict | None:
    """How a candidate's fingerprint compares with a reference's, over the tensors both have with one shape."""
    shared = [k for k in cand if k in ref and cand[k][0] == ref[k][0]]
    if not shared or len(shared) < 0.8 * max(len(cand), len(ref)):
        return None  # another architecture
    per = {k: _corr(cand[k][1], ref[k][1]) for k in shared}
    identical = sum(cand[k][1] == ref[k][1] for k in shared)
    layer = lambda k: int(m.group(1)) if (m := re.search(r"layers?\.(\d+)\.", k)) else -1
    by_layer: dict[int, list[float]] = {}
    for k, c in per.items():
        by_layer.setdefault(layer(k), []).append(c)
    vals = sorted(per.values())
    return {"tensors": len(shared), "identical": identical, "median": vals[len(vals) // 2], "min": vals[0],
            "layers": {str(l): round(min(cs), 4) for l, cs in sorted(by_layer.items())}}


def identify(cand: Path, refs: list[tuple[str, Path]], say=lambda m: None) -> dict:
    """The candidate's fingerprint against every reference of the same architecture, closest first."""
    sig = signature(cand)
    if not sig:
        return {"candidate": str(cand), "error": "no safetensors weights the fingerprint can read", "matches": []}
    rows = []
    for name, folder in refs:
        if folder.resolve() == cand.resolve():
            continue
        s = similarity(sig, signature(folder))
        if s is not None:
            rows.append({"name": name, "path": str(folder)} | s)
            say(f"[fingerprint] {name}: median correlation {s['median']:.3f}")
    rows.sort(key=lambda r: -r["median"])
    claim = claimed_base(cand)
    best = rows[0] if rows else None
    verdict = ("no local model of the same architecture" if best is None else
               "the same weights" if best["identical"] == best["tensors"] else
               "derived from it (weights mostly kept)" if best["median"] > 0.9 else
               "related, heavily changed" if best["median"] > 0.5 else "no local model it derives from")
    return {"candidate": str(cand), "claimed_base": claim, "closest": best and best["name"], "reading": verdict,
            "matches": rows}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate", type=Path)
    ap.add_argument("roots", nargs="+", type=Path)
    args = ap.parse_args(argv)
    from .filescan import find_repos
    r = identify(args.candidate, find_repos(args.roots), print)
    print(json.dumps({k: v for k, v in r.items() if k != "matches"} | {"matches": r["matches"][:5]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
