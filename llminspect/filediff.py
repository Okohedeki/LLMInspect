"""What changed outside the weights: a candidate's config, generation config, tokenizer and chat template,
compared with the base model's. A model's behavior can be changed a lot without touching a weight:

  template    the chat template wraps every conversation; an edit can add instructions the user never sees,
              drop the system prompt, or change how tool calls are written. Shown as a diff, with the plain
              text it adds pulled out.
  tokenizer   tokens added, removed or given other ids; changed special tokens; normalizer, pre-tokenizer
              and decoder settings.
  config      architecture, context length, rope scaling, special token ids, auto_map (custom code) and
              quantization settings.
  generation  default sampling, stop tokens, and tokens suppressed or forced.
  files       files only one of the two has.

No weights are loaded and no repo code runs. With transformers installed, configs and tokenizers are read the
way it loads them (never with trust_remote_code), so a setting saved in another version's format compares
equal, and the tokenizers are compared by what they do: the ids they give a set of probe texts. Without it,
or for what it can't read, the raw files are compared. Bookkeeping fields (transformers_version,
_name_or_path, dtype) are left out; every other difference is listed. A difference is a fact, not a
verdict: fine-tunes legitimately change some of these.

    python -m llminspect.filediff BASE CANDIDATE
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

IGNORED = {"transformers_version", "_name_or_path", "torch_dtype", "dtype", "_commit_hash", "use_cache",
           "_attn_implementation_autoset", "output_attentions", "output_hidden_states"}
# config keys whose change alters what runs or how text is read, worth calling out
SENSITIVE = re.compile(r"auto_map|quantization|rope|max_position|vocab_size|token_id|tie_word|num_|hidden_size|"
                       r"intermediate|head_dim|sliding|layer_types|architectures|model_type|attn_logit|final_logit")
TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")
TOKENIZER_PARTS = ("normalizer", "pre_tokenizer", "post_processor", "decoder")
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".h5", ".msgpack", ".onnx"}


def _json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _flat(d, prefix="") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k in IGNORED:
                continue
            out.update(_flat(v, f"{prefix}{k}."))
    else:
        out[prefix[:-1]] = d
    return out


def compare_fields(base: dict | None, cand: dict | None) -> list[dict]:
    """Every field that differs between two JSON documents, flattened to dotted keys."""
    a, b = _flat(base or {}), _flat(cand or {})
    rows = []
    for k in sorted(set(a) | set(b)):
        if a.get(k, ...) != b.get(k, ...):
            rows.append({"key": k, "base": a.get(k, "(absent)"), "candidate": b.get(k, "(absent)"),
                         "sensitive": bool(SENSITIVE.search(k))})
    return rows


def templates(folder: Path) -> dict[str, str]:
    """Every chat template a folder carries, by name: tokenizer_config.json's (one or a named list), and the
    standalone chat_template.jinja / chat_template.json that newer transformers write."""
    out = {}
    cfg = _json(folder / "tokenizer_config.json") or {}
    t = cfg.get("chat_template")
    if isinstance(t, str):
        out["default"] = t
    elif isinstance(t, list):
        out.update({x.get("name", f"#{i}"): x.get("template", "") for i, x in enumerate(t) if isinstance(x, dict)})
    if (folder / "chat_template.jinja").is_file():
        out["default"] = (folder / "chat_template.jinja").read_text(encoding="utf-8", errors="replace")
    ct = _json(folder / "chat_template.json")
    if isinstance(ct, dict) and isinstance(ct.get("chat_template"), str):
        out.setdefault("default", ct["chat_template"])
    return out


_JINJA = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.DOTALL)


def added_text(base: str, cand: str) -> list[str]:
    """Plain text (outside Jinja tags) the candidate's template has that the base's doesn't: what an edit
    would show the model in every conversation."""
    a, b = _JINJA.sub("\0", base), _JINJA.sub("\0", cand)  # tags become separators; only plain text is compared
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    pieces = (p.strip() for tag, _, _, j1, j2 in ops if tag in ("insert", "replace") for p in b[j1:j2].split("\0"))
    return sorted({p for p in pieces if len(p) > 2}, key=len, reverse=True)


def compare_templates(base: Path, cand: Path) -> list[dict]:
    a, b = templates(base), templates(cand)
    rows = []
    for name in sorted(set(a) | set(b)):
        ta, tb = a.get(name), b.get(name)
        if ta == tb:
            continue
        row = {"name": name, "base": "absent" if ta is None else f"{len(ta)} chars",
               "candidate": "absent" if tb is None else f"{len(tb)} chars"}
        if ta is not None and tb is not None:
            diff = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(), "base", "candidate", lineterm="", n=1))
            row["diff"] = diff[:200]
            row["added_text"] = added_text(ta, tb)[:20]
        elif tb is not None:
            row["added_text"] = added_text("", tb)[:20]
        rows.append(row)
    return rows


def _vocab(folder: Path) -> dict[str, int] | None:
    tok = _json(folder / "tokenizer.json")
    raw = ((tok or {}).get("model") or {}).get("vocab")
    if isinstance(raw, dict):  # BPE and WordPiece: {token: id}
        vocab = dict(raw)
    elif isinstance(raw, list):  # Unigram: [[piece, score], ...], the id is the position
        vocab = {p[0]: i for i, p in enumerate(raw) if isinstance(p, list) and p}
    else:
        vocab = {}
    if not vocab:
        vocab = _json(folder / "vocab.json") or {}
    for t in (tok or {}).get("added_tokens", []):
        vocab[t["content"]] = t["id"]
    for i, t in ((_json(folder / "tokenizer_config.json") or {}).get("added_tokens_decoder") or {}).items():
        vocab[t["content"]] = int(i)
    return vocab or None


def compare_tokenizers(base: Path, cand: Path) -> dict:
    out: dict = {}
    va, vb = _vocab(base), _vocab(cand)
    if va is not None and vb is not None:
        added = sorted((t for t in vb if t not in va), key=lambda t: vb[t])
        removed = sorted((t for t in va if t not in vb), key=lambda t: va[t])
        moved = sorted((t for t in va if t in vb and va[t] != vb[t]), key=lambda t: va[t])
        out["vocab"] = {"base": len(va), "candidate": len(vb), "added": len(added), "removed": len(removed),
                        "renumbered": len(moved), "added_examples": [[t, vb[t]] for t in added[:20]],
                        "removed_examples": [[t, va[t]] for t in removed[:20]],
                        "renumbered_examples": [[t, va[t], vb[t]] for t in moved[:20]]}
    ta, tb = _json(base / "tokenizer.json") or {}, _json(cand / "tokenizer.json") or {}
    parts = [p for p in TOKENIZER_PARTS if ta.get(p) != tb.get(p) and (ta or tb)]
    if parts:
        out["pipeline_changed"] = parts
    ca = {k: v for k, v in (_json(base / "tokenizer_config.json") or {}).items() if k not in ("chat_template", "added_tokens_decoder")}
    cb = {k: v for k, v in (_json(cand / "tokenizer_config.json") or {}).items() if k not in ("chat_template", "added_tokens_decoder")}
    out["config"] = compare_fields(ca, cb) + compare_fields(_json(base / "special_tokens_map.json"),
                                                          _json(cand / "special_tokens_map.json"))
    for name in ("tokenizer.model", "merges.txt", "tiktoken.model"):
        fa, fb = base / name, cand / name
        if fa.is_file() and fb.is_file() and _sha(fa) != _sha(fb):
            out.setdefault("files_changed", []).append(name)
    return out


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def compare_files(base: Path, cand: Path) -> dict:
    """Files only one side has, weights left out (the weight scan compares those)."""
    names = lambda f: {p.relative_to(f).as_posix() for p in f.rglob("*") if p.is_file() and not p.name.startswith(".")
                       and p.suffix.lower() not in WEIGHTS and ".cache" not in p.parts}
    a, b = names(base), names(cand)
    return {"only_candidate": sorted(b - a), "only_base": sorted(a - b)}


# Text the tokenizer probe encodes with both tokenizers: spacing, unicode, code, numbers, markup.
PROBES = ["Hello, world!", "  two leading spaces and a tab\there", "naïve café, Zürich, 東京, Привет, مرحبا 😀",
          "def f(x):\n    return x ** 2  # square", "1234567890 3.14159 -0.5e-3 $1,000.00", "<b>bold</b> & <i>it</i>",
          "https://example.com/path?q=1&r=two", "The quick brown fox jumps over the lazy dog.", "\n\n\n", "a" * 40]


def _effective(folder: Path) -> dict | None:
    """Config, generation config and tokenizer as transformers reads them (never with trust_remote_code), so a
    setting saved in another version's format compares equal. None for what it can't read, and without transformers."""
    try:
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    except ImportError:
        return None
    import contextlib
    import io
    import logging
    out = {"config": None, "generation": None, "tokenizer": None}
    logging.getLogger("transformers").setLevel(logging.ERROR)
    with contextlib.redirect_stderr(io.StringIO()):
        with contextlib.suppress(Exception):
            out["config"] = AutoConfig.from_pretrained(folder, trust_remote_code=False).to_dict()
        if (folder / "generation_config.json").is_file():
            with contextlib.suppress(Exception):
                out["generation"] = GenerationConfig.from_pretrained(folder).to_dict()
        with contextlib.suppress(Exception):
            out["tokenizer"] = AutoTokenizer.from_pretrained(folder, trust_remote_code=False)
    return out


def tokenizer_behavior(ta, tb) -> dict:
    """How two loaded tokenizers differ in what they do: vocabulary, special tokens, and the ids they give the
    probe texts (with and without special tokens added)."""
    va, vb = ta.get_vocab(), tb.get_vocab()
    added = sorted((t for t in vb if t not in va), key=lambda t: vb[t])
    removed = sorted((t for t in va if t not in vb), key=lambda t: va[t])
    moved = sorted((t for t in va if t in vb and va[t] != vb[t]), key=lambda t: va[t])
    out = {"vocab": {"base": len(va), "candidate": len(vb), "added": len(added), "removed": len(removed),
                     "renumbered": len(moved), "added_examples": [[t, vb[t]] for t in added[:20]],
                     "removed_examples": [[t, va[t]] for t in removed[:20]],
                     "renumbered_examples": [[t, va[t], vb[t]] for t in moved[:20]]}}
    special = lambda t: {k: getattr(t, k, None) for k in ("bos_token", "eos_token", "pad_token", "unk_token")} | {
        "all": sorted(map(str, t.all_special_tokens))}
    out["config"] = compare_fields(special(ta), special(tb))
    probes = PROBES + sorted({s for t in (ta, tb) for s in (t.bos_token, t.eos_token) if isinstance(s, str)})
    rows = []
    for text in probes:
        for with_special in (False, True):
            a, b = ta.encode(text, add_special_tokens=with_special), tb.encode(text, add_special_tokens=with_special)
            if a != b:
                rows.append({"text": text[:60], "special_tokens": with_special, "base": a[:24], "candidate": b[:24]})
    out["encodes_differently"] = rows
    return out


def compare(base: Path, cand: Path) -> dict:
    base, cand = Path(base), Path(cand)
    ea, eb = _effective(base) or {}, _effective(cand) or {}
    both = lambda k: ea.get(k) is not None and eb.get(k) is not None
    report = {
        "read": {k: "as transformers loads it" if both(k) else "raw files" for k in ("config", "generation", "tokenizer")},
        "config": compare_fields(ea["config"], eb["config"]) if both("config")
        else compare_fields(_json(base / "config.json"), _json(cand / "config.json")),
        "generation": compare_fields(ea["generation"], eb["generation"]) if both("generation")
        else compare_fields(_json(base / "generation_config.json"), _json(cand / "generation_config.json")),
        "templates": compare_templates(base, cand),
        "tokenizer": tokenizer_behavior(ea["tokenizer"], eb["tokenizer"]) if both("tokenizer") else compare_tokenizers(base, cand),
        "files": compare_files(base, cand),
    }
    report["signals"] = signals(report)
    return report


def signals(r: dict) -> list[dict]:
    """Plain statements of what differs, most consequential first. Observations, not a verdict."""
    out = []
    for t in r["templates"]:
        extra = f" It adds this text: {t['added_text'][0][:160]!r}" if t.get("added_text") else ""
        out.append({"level": "high", "section": "template",
                    "text": f"The chat template '{t['name']}' differs from the base's ({t['base']} -> {t['candidate']}).{extra}"})
    auto = [c for c in r["config"] if c["key"].startswith("auto_map")]
    if auto:
        out.append({"level": "high", "section": "config", "text": "config.json's auto_map differs: the candidate runs "
                    "different custom code under trust_remote_code."})
    v = r["tokenizer"].get("vocab")
    if v and (v["added"] or v["removed"] or v["renumbered"]):
        out.append({"level": "high" if v["renumbered"] or v["removed"] else "medium", "section": "tokenizer",
                    "text": f"Vocabulary: {v['added']} tokens added, {v['removed']} removed, {v['renumbered']} given other ids."})
    enc = r["tokenizer"].get("encodes_differently")
    if enc:
        out.append({"level": "high", "section": "tokenizer",
                    "text": f"The tokenizers give different ids for {len(enc)} probe text(s), e.g. {enc[0]['text']!r}: the "
                            "candidate reads text differently."})
    if r["tokenizer"].get("pipeline_changed"):
        out.append({"level": "medium", "section": "tokenizer",
                    "text": "Tokenizer pipeline changed: " + ", ".join(r["tokenizer"]["pipeline_changed"]) + "."})
    for f in r["tokenizer"].get("files_changed", []):
        out.append({"level": "medium", "section": "tokenizer", "text": f"{f} differs from the base's."})
    special = [c for c in r["tokenizer"].get("config", []) if "token" in c["key"]]
    if special:
        out.append({"level": "medium", "section": "tokenizer",
                    "text": "Special tokens differ: " + ", ".join(c["key"] for c in special[:6]) + "."})
    sens = [c for c in r["config"] if c["sensitive"] and not c["key"].startswith("auto_map")]
    if sens:
        out.append({"level": "medium", "section": "config",
                    "text": "Config fields that change what runs or how text is read differ: "
                            + ", ".join(f"{c['key']} {c['base']!r} -> {c['candidate']!r}" for c in sens[:5]) + "."})
    if r["generation"]:
        out.append({"level": "medium", "section": "generation",
                    "text": "Generation defaults differ: " + ", ".join(f"{c['key']} {c['base']!r} -> {c['candidate']!r}"
                                                                     for c in r["generation"][:5]) + "."})
    if r["files"]["only_candidate"]:
        out.append({"level": "note", "section": "files",
                    "text": "Files only the candidate has: " + ", ".join(r["files"]["only_candidate"][:8]) + "."})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--json", action="store_true", help="print the whole comparison as JSON")
    args = ap.parse_args(argv)
    r = compare(args.base, args.candidate)
    if args.json:
        print(json.dumps(r, indent=1, default=str))
        return
    for s in r["signals"] or [{"level": "note", "section": "all", "text": "No difference outside the weights."}]:
        print(f"[{s['level']}] {s['section']}: {s['text']}")


if __name__ == "__main__":
    main(sys.argv[1:])
