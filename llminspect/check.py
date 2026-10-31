"""One check of a model you didn't train, combining every check LLMInspect has, as one report for agents and people.

  files            code that runs on load: pickles, trust_remote_code, wheels, templates (filescan)
  origin           which local model it actually comes from, against the base its card claims (fingerprint)
  outside_weights  what it changed outside the weights: template, tokenizer, config, generation (filediff)
  inside_weights   what it changed inside them: where, triggers, refusals, behavior (the deploy scan; deep only,
                   run by a LLMInspect server that has the base loaded)

The report is built for agents first. `findings` is a flat list, most serious first; each has a stable id, a
severity (high, medium, note), an area, a one-line title, the evidence (files and lines, layers, tokens) and
`next`: what to do about it, often a tool to call with its arguments. `summary` says in one line what was
found, and `not_checked` says what wasn't and why. The same JSON renders as the page people read
(check_html). Findings are evidence, never a verdict: no report says a model is safe.

The base is the one named, or else the local model the fingerprint finds the candidate derives from.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from . import filediff, filescan, fingerprint

SCHEMA = "llminspect.check/1"
REPORTS = Path(os.environ.get("LLMINSPECT_CHECKS", Path.home() / ".cache" / "llminspect" / "checks"))
SEVERITY = {"high": 0, "medium": 1, "note": 2}
# Fingerprint correlation (fingerprint.py): unrelated models of one shape sit near 0, a light fine-tune near 1, and a
# model retrained hard for a new task (text to speech, say) in between, still clearly related.
DERIVED, RELATED = 0.9, 0.15
FILE_LEVELS = {"runs code on load": "high", "imports other code on load": "medium", "runs code if trusted": "medium",
               "template": "high", "format": "medium", "info": "note"}
LIMITS = [
    "Static file checks find code the files would run; they don't judge what a library's own code does.",
    "The origin check needs the true base among the local models it compares with; without it, origin is unknown.",
    "Weight checks (deep) sample behavior on short prompts and probe one-token triggers; a trigger made of a "
    "phrase, a change spread over every weight, or behavior that needs tools or long context can stay hidden.",
    "A finding is evidence of a difference or a risk, not proof of intent, and no finding is proof of safety.",
]


def _finding(area: str, severity: str, title: str, detail: str = "", evidence=None, where=None, next_=None,
             short: str | None = None) -> dict:
    """short: the finding in a few words, for headlines; title: one line with the specifics."""
    return {"area": area, "severity": severity, "short": short or title.rstrip("."), "title": title, "detail": detail,
            "evidence": list(evidence or [])[:24], "where": where, "next": next_}


# ---- each check, as findings ----------------------------------------------------------------------------

def _files(cand: Path) -> tuple[dict, list[dict]]:
    repo = filescan.scan_repo(cand)
    out = []
    for f in repo["findings"]:
        out.append(_file_finding(f))
    pickles = sum(1 for x in repo["file_list"] if x["format"] in ("pickle", "pytorch zip", "archive"))
    section = {"files": repo["files"], "bytes": repo["bytes"], "weight_formats": repo["weight_formats"], "pickles": pickles,
               "formats": _count(x["format"] for x in repo["file_list"]),
               "list": [[x["file"], x["bytes"], x["format"]] for x in repo["file_list"]]}  # [file, bytes, format]
    return section, out


def _file_finding(f: dict) -> dict:
    """A file scan finding in the report's words: a short headline, a title with the specifics, where, next."""
    level, ev, name = f["level"], f["evidence"], f["file"]
    sev = FILE_LEVELS[level]
    if level == "runs code on load":
        short = f"Loading {Path(name.split(':')[0]).name} runs code"
        return _finding("files", sev, f"Loading {name} runs code: {', '.join(ev[:3])}." if ev else f"Loading {name} runs code.",
                        f["detail"], ev, name, {"do": "Don't load it with pickle or torch.load(weights_only=False); use the "
                                                      "safetensors weights if the repo has them."}, short)
    if level == "imports other code on load":
        libs = ", ".join(sorted({e.rsplit(".", 1)[0] for e in ev})[:3])
        return _finding("files", sev, f"Loading {name} runs code from {libs}.", f["detail"], ev, name,
                        {"do": "Load it with torch.load(weights_only=True), which refuses these imports."},
                        f"Loading {Path(name).name} runs library code")
    if level == "runs code if trusted" and name.endswith(".whl"):
        return _finding("files", sev, f"Ships a Python package that runs if installed: {name}.",
                        "A wheel shipped with the weights; nothing runs it on load, but its code runs once installed.", [],
                        name, {"do": "Don't pip install it without reading it."}, "Ships an installable Python package")
    if level == "runs code if trusted":
        risky = [e for e in ev if " eval()" in e or "subprocess" in e or "socket" in e or "exec()" in e]
        files = sorted({e.split(": ", 1)[0] for e in (risky or ev)})
        calls = ", ".join(e.split(": ", 1)[-1] for e in (risky or ev)[:2])
        evals = "eval()" if any("eval()" in e or "exec()" in e for e in risky) else None
        short = ("Its custom code runs eval()" if evals else "Its custom code starts processes or opens sockets" if risky
                 else "Runs its own Python if trusted")
        return _finding("files", "high" if risky else sev,
                        "Ships its own Python, run under trust_remote_code" + (f"; it calls {calls}." if ev else "."),
                        "auto_map in config.json: transformers runs this repo's Python when it is loaded with "
                        "trust_remote_code=True." + (" eval() runs whatever text it is given as Python." if evals else ""),
                        ev, ", ".join(files) or name,
                        {"do": "Read the listed files and lines before loading with trust_remote_code=True."}, short)
    if level == "template":
        return _finding("files", sev, f"The chat template in {name} reaches Python internals: {', '.join(ev[:3])}.",
                        f["detail"], ev, name, {"do": "Read the template before using the model for chat."},
                        "Its chat template reaches Python internals")
    return _finding("files", sev, f["detail"], "", ev, name)


def _count(items) -> dict:
    out: dict = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _origin(cand: Path, refs: list[tuple[str, Path]], say) -> tuple[dict, list[dict], dict | None]:
    """The fingerprint as findings, and the reference the rest of the check compares with: the base the card
    claims when it is on this machine and the weights come from it, else the closest local model."""
    r = fingerprint.identify(cand, refs, say)
    for m in r["matches"]:
        m["name"] = model_name(m["path"])
    if r.get("closest"):
        r["closest"] = r["matches"][0]["name"] if r["matches"] else r["closest"]
    if r.get("error"):
        return r, [_finding("origin", "note", "Origin not checked: " + r["error"] + ".")], None
    out, best = [], (r["matches"] or [None])[0]
    claim = r.get("claimed_base") or []
    present = [(n, f) for n, f in ((model_name(f), f) for _, f in refs) if any(_same_model(c, n) for c in claim)]
    readable = [(n, f) for n, f in present if fingerprint.signature(f)]
    claimed_row = next((m for m in r["matches"] if any(_same_model(c, m["name"]) for c in claim)), None)
    if not claim:
        pass
    elif not readable:  # not here, or here without weights the fingerprint can read (a broken cache, GGUF only)
        why = "isn't on this machine" if not present else "is on this machine, but its weights can't be read here"
        out.append(_finding("origin", "note", f"The card says it is built on {claim[0]}, which {why}, so the claim can't be "
                            "checked.", "", [f"claimed: {c}" for c in claim],
                            next_={"do": "Put the claimed base's weights on this machine and check again.", "tool": "check_model"}))
    elif claimed_row is None:  # here and readable, and not even the same architecture
        out.append(_finding("origin", "high", f"The card says it is built on {claim[0]}, but that model on this machine has "
                            "another architecture.", "", [f"claimed: {c}" for c in claim] + [f"on this machine: {f}" for _, f in readable]))
    elif claimed_row["median"] < RELATED:
        out.append(_finding("origin", "high", f"The card says it is built on {claim[0]}; its weights don't come from it.",
                            f"Median correlation {claimed_row['median']:.3f} with {claimed_row['name']}, as for unrelated models"
                            + (f"; the closest local model is {best['name']} ({best['median']:.3f})." if best and best is not claimed_row else "."),
                            [f"claimed: {c}" for c in claim]))
    elif claimed_row["median"] <= DERIVED:
        out.append(_finding("origin", "note", f"Heavily retrained from {claimed_row['name']}, the base its card claims: related, "
                            "but little of its weights is left as it was.",
                            f"Median correlation {claimed_row['median']:.3f}: far above unrelated models (about 0), far below a "
                            "light fine-tune (near 1). Training for a new task looks like this."))
    ref = claimed_row if claimed_row is not None and claimed_row["median"] > DERIVED else best
    if ref is None or ref["median"] <= DERIVED:
        if not (claimed_row is not None and claimed_row["median"] >= RELATED):  # the claim's finding already says it
            related = ref is not None and ref["median"] >= RELATED
            out.append(_finding("origin", "note", f"Related to {ref['name']} but heavily retrained (median correlation "
                                f"{ref['median']:.3f})." if related else "No local model it derives from; where it comes "
                                "from can't be checked here."))
        return r, out, None
    name, cited = ref["name"], " (the base its card claims)" if ref is claimed_row else ""
    moved = [int(l) for l, c in ref["layers"].items() if l != "-1" and c < 0.9995]
    still = [int(l) for l, c in ref["layers"].items() if l != "-1" and c >= 0.9995]
    if ref["identical"] == ref["tensors"]:
        out.append(_finding("origin", "note", f"Same weights as {name}{cited}."))
    elif moved and still and len(moved) <= max(2, len(ref["layers"]) // 4):
        out.append(_finding("origin", "medium", f"Edited in {_n(len(moved), 'layer')} ({_span(moved)}); every other layer is "
                            f"{name}'s{cited}, unchanged.",
                            "A targeted edit looks like this (so do some narrow fine-tunes): look inside those layers.",
                            [f"layer {l}: correlation {ref['layers'][str(l)]:.4f}" for l in sorted(moved)][:12],
                            next_={"do": "Run the weight checks to see what the edit does.", "tool": "check_model",
                                   "args": {"candidate": str(cand), "base": ref["path"], "deep": True}}))
    else:
        out.append(_finding("origin", "note", f"Derived from {name}{cited}: every layer moved a little, as in a full fine-tune.",
                            f"Median correlation {ref['median']:.3f}, lowest {ref['min']:.3f}."))
    return r, out, ref | {"claimed": ref is claimed_row}


def model_name(path) -> str:
    """A model's name from where it sits: 'Org/Name' for a Hugging Face cache folder (models--Org--Name/snapshots/…)
    or a folder named 'Org__Name' / 'Org--Name', with any sub-model path after it; else the folder's name."""
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.startswith("models--"):
            rest = parts[i + 3:] if len(parts) > i + 2 and parts[i + 1] == "snapshots" else parts[i + 1:]
            return "/".join([part[len("models--"):].replace("--", "/", 1), *rest])
    for i in range(len(parts) - 1, -1, -1):
        m = re.match(r"^(.+?)(?:--|__)(.+)$", parts[i])
        if m:
            return "/".join([f"{m.group(1)}/{m.group(2)}", *parts[i + 1:]])
    return Path(path).name


def display(name: str) -> str:
    """'Org/Name' for a folder named the cache's way ('Org--Name' or 'Org__Name'), wherever it sits
    ('music/Org__Name' too); other names as they are."""
    last = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    m = re.match(r"^([^/]+?)(?:--|__)(.+)$", last)
    return f"{m.group(1)}/{m.group(2)}" if m else name


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def _same_model(claim: str, name: str) -> bool:
    norm = lambda s: s.lower().replace("--", "/").replace("__", "/").rstrip("/").split("/")[-1]
    return norm(claim) == norm(name)


def _span(layers: list[int]) -> str:
    layers = sorted(layers)
    return f"L{layers[0]}" if len(layers) == 1 else (f"L{layers[0]}–L{layers[-1]}" if layers[-1] - layers[0] == len(layers) - 1
                                                     else ", ".join(f"L{l}" for l in layers))


def _outside(base: Path, cand: Path, trusted: bool = True) -> tuple[dict, list[dict]]:
    """What changed outside the weights. Against a base that isn't known to be the model's own (the closest local
    model, not the one named or claimed) every difference is a note: two related releases differ legitimately."""
    r = filediff.compare(base, cand)
    out = []
    against = "" if trusted else f"Against the closest local model, {model_name(base)} (not a known base): "
    for s in r["signals"]:
        evidence, where = [], s["section"]
        if s["section"] == "template":
            t = r["templates"][0]
            evidence = t.get("added_text") or [l for l in t.get("diff", []) if l.startswith(("+", "-"))][:12]
        elif s["section"] == "tokenizer" and r["tokenizer"].get("encodes_differently"):
            evidence = [f"{x['text']!r}: {x['base'][:8]} -> {x['candidate'][:8]}" for x in r["tokenizer"]["encodes_differently"][:6]]
        elif s["section"] in ("config", "generation"):
            evidence = [f"{c['key']}: {c['base']!r} -> {c['candidate']!r}" for c in r[s["section"]][:10]]
        sev = (s["level"] if s["level"] in SEVERITY else "note") if trusted else "note"
        title = against + (s["text"][0].lower() + s["text"][1:] if against else s["text"])
        out.append(_finding("outside_weights", sev, title, "", evidence, where,
                            {"do": "See the full comparison.", "tool": "diff_files", "args": {"base": str(base), "candidate": str(cand)}}))
    return r, out


def _server(url: str, method: str, path: str, body: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(url.rstrip("/") + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class HttpScan:
    """The deploy scan, run by a LLMInspect server over its HTTP API (from the MCP server or a terminal)."""

    def __init__(self, url: str):
        self.url = url

    def start(self, spec: str) -> None:
        try:
            _server(self.url, "POST", "/llminspect/scan", {"spec": spec})
        except urllib.error.HTTPError as exc:
            raise ScanUnavailable(f"the LLMInspect server refused the scan: {exc.read().decode(errors='replace')[:300]}")
        except (urllib.error.URLError, OSError) as exc:
            raise ScanUnavailable(f"no LLMInspect server at {self.url} ({exc}). Start one with the base model loaded "
                                  "(LLMINSPECT_MODEL=<base> python -m llminspect.server), then check again with deep=true.")

    def status(self) -> dict:
        return _server(self.url, "GET", "/llminspect/scan")

    def report(self, rid: str) -> dict:
        return _server(self.url, "GET", f"/llminspect/scan/reports/{rid}")


class LocalScan:
    """The deploy scan of the recorder in this process (the LLMInspect server's own check)."""

    def __init__(self, recorder):
        self.scan = recorder.scan

    def start(self, spec: str) -> None:
        try:
            self.scan.start({"spec": spec})
        except Exception as exc:
            raise ScanUnavailable(f"the deploy scan couldn't start: {exc}")

    def status(self) -> dict:
        return self.scan.status()

    def report(self, rid: str) -> dict:
        return self.scan.report(rid)


class ScanUnavailable(Exception):
    pass


def _inside(cand: Path, scanner, say) -> tuple[dict | None, list[dict], str | None]:
    """The deploy scan of the candidate against the scanner's loaded base. (report, findings, why not run)."""
    try:
        scanner.start(str(cand))
    except ScanUnavailable as exc:
        return None, [], str(exc)
    while True:
        st = scanner.status()
        say(f"inside the weights: {st.get('step')} ({100 * (st.get('progress') or 0):.0f}%)")
        if st["status"] == "error":
            return None, [], f"the deploy scan failed: {st['error']}"
        if st["status"] == "ready":
            break
        time.sleep(2)
    report = scanner.report(st["report"]["id"])
    out = []
    for s in report.get("signals", []):
        sev = s["level"] if s["level"] in SEVERITY else "note"
        nxt = None
        if s["section"] == "triage" and "Token" in s["text"]:
            nxt = {"do": "Try the token in a chat with the candidate loaded, and trace what it changes."}
        out.append(_finding("inside_weights", sev, s["text"], "", [], s["section"], nxt))
    return report, out, None


# ---- the whole check -----------------------------------------------------------------------------------

def run(candidate, base=None, roots=(), deep: bool = False, server: str | None = None, say=lambda m: None,
        scanner=None) -> dict:
    """Every check of `candidate` (a model folder), against `base` (a folder) or the local model it derives from.
    deep runs the weight checks with `scanner` (a LocalScan), or else a LLMInspect server at `server`."""
    t0 = time.perf_counter()
    cand = Path(candidate).expanduser()
    if not cand.is_dir():
        raise FileNotFoundError(f"no model folder at {candidate}")
    roots = [Path(r).expanduser() for r in roots]
    report = {"schema": SCHEMA, "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
              "created": datetime.now().astimezone().isoformat(timespec="seconds"),
              "candidate": {"name": model_name(cand), "path": str(cand)}, "base": None, "sections": {}, "not_checked": []}
    findings = []
    say("files: reading every file for code that runs on load")
    report["sections"]["files"], f = _files(cand)
    findings += f
    refs = filescan.find_repos(roots) if roots else []
    if base:
        refs = [(Path(base).name, Path(base).expanduser())] + [r for r in refs if r[1].resolve() != Path(base).expanduser().resolve()]
    ref = None
    if refs:
        say(f"origin: fingerprinting against {len(refs)} local model(s)")
        report["sections"]["origin"], f, ref = _origin(cand, refs, say)
        findings += f
    else:
        report["not_checked"].append({"area": "origin", "why": "no base or model folders to compare with were given"})
    if base:
        base_path, chosen_by = Path(base).expanduser(), "given"
    elif ref is not None and ref["median"] > DERIVED:
        base_path, chosen_by = Path(ref["path"]), "card" if ref["claimed"] else "fingerprint"
    else:
        base_path, chosen_by = None, None
    if base_path is not None:
        report["base"] = {"name": model_name(base_path), "path": str(base_path), "chosen_by": chosen_by}
        say(f"outside the weights: comparing with {model_name(base_path)}")
        report["sections"]["outside_weights"], f = _outside(base_path, cand, trusted=chosen_by in ("given", "card"))
        findings += f
    else:
        report["not_checked"].append({"area": "outside_weights", "why": "no base model to compare with"})
    if deep and base_path is not None:
        scanner = scanner or HttpScan(server or os.environ.get("LLMINSPECT_URL", "http://127.0.0.1:8765"))
        rep, f, why = _inside(cand, scanner, say)
        if rep is not None:
            report["sections"]["inside_weights"] = {k: rep.get(k) for k in ("id", "base", "weights", "behavior", "refusals",
                                                                           "triage", "memorization", "limitations", "seconds")}
            findings += f
        else:
            report["not_checked"].append({"area": "inside_weights", "why": why})
    elif deep:
        report["not_checked"].append({"area": "inside_weights", "why": "no base model to compare with"})
    else:
        report["not_checked"].append({"area": "inside_weights", "why": "not requested: run with deep=true (needs a LLMInspect "
                                                                     "server with the base model loaded)"})
    findings.sort(key=lambda x: SEVERITY[x["severity"]])
    for i, x in enumerate(findings, 1):
        x["id"] = f"F{i}"
    report["findings"] = findings
    report["summary"] = summarize(report)
    report["limits"] = LIMITS
    report["seconds"] = round(time.perf_counter() - t0, 1)
    return report


def summarize(report: dict) -> dict:
    counts = {s: sum(f["severity"] == s for f in report["findings"]) for s in SEVERITY}
    top = [f for f in report["findings"] if f["severity"] == "high"] or [f for f in report["findings"] if f["severity"] == "medium"]
    origin = next((f["title"] for f in report["findings"] if f["area"] == "origin"), None)
    if top:
        more = len(top) - 1
        headline = top[0]["short"] + (f", and {more} more to look at {'first' if top[0]['severity'] == 'high' else ''}".rstrip()
                                      if more else "") + "."
    else:
        headline = "Nothing these checks look for turned up" + (f"; {origin[0].lower() + origin[1:]}" if origin else ".")
    return {"headline": headline, "counts": counts, "checked": list(report["sections"]),
            "not_checked": [x["area"] for x in report["not_checked"]]}


def save(report: dict) -> Path:
    """Write report.json and report.html under REPORTS/<id>/; returns the folder."""
    from .check_html import render
    out = REPORTS / report["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (out / "report.html").write_text(render(report), encoding="utf-8")
    return out


def load(rid: str) -> dict:
    return json.loads((REPORTS / rid / "report.json").read_text(encoding="utf-8"))


BLOCKS = {"attn": re.compile(r"(self_attn\.q_proj|attn\.c_attn|attention\.wq)\.weight$"),
          "mlp": re.compile(r"(mlp\.down_proj|mlp\.c_proj|feed_forward\.w2)\.weight$")}


def layer_map(report: dict) -> dict:
    """The candidate's layers as the UI draws them: per layer, the attention and MLP tensor the fingerprint reads,
    with its correlation to the base (or the closest local model) and, after a deep check, how much each layer
    changed. Nothing is loaded: the values come from the fingerprint cache or the safetensors files."""
    cand = Path(report["candidate"]["path"])
    ref = report.get("base") or next(iter(report["sections"].get("origin", {}).get("matches") or []), None)
    out = {"base": ref and display(model_name(ref["path"])), "base_is": "base" if report.get("base") else ref and "closest",
           "layers": [], "embed": None, "deep": None}
    if not cand.is_dir():
        return out | {"error": f"the model folder isn't readable here: {cand}"}
    sig = fingerprint.signature(cand)
    if not sig:
        return out | {"error": "no safetensors weights to map"}
    corr = {}
    if ref and Path(ref["path"]).is_dir():
        corr = fingerprint.tensor_correlations(sig, fingerprint.signature(Path(ref["path"])))
    elif ref:
        out["base"], out["base_is"] = None, None
    cell = lambda k: {"tensor": k, "shape": sig[k][0], "corr": None if k not in corr else round(corr[k], 5)}
    layers: dict[int, dict] = {}
    for k in sig:
        m = re.match(r"layers?\.(\d+)\.", k)
        if not m:
            out["embed"] = cell(k)
            continue
        for block, rx in BLOCKS.items():
            if rx.search(k):
                layers.setdefault(int(m.group(1)), {})[block] = cell(k)
    out["layers"] = [{"layer": n} | layers[n] for n in sorted(layers)]
    w = (report["sections"].get("inside_weights") or {}).get("weights") or {}
    if w.get("summary"):
        out["deep"] = {"layers": {str(x["layer"]): x["max_rel"] for x in w["summary"].get("layers", [])},
                       "neurons": (w.get("neurons") or {}).get("top", [])[:12]}
    return out


# ---- what's on this machine, and what was checked (the MCP server and the web UI both read these) --------

def default_roots(extra=()) -> list[Path]:
    """Where models usually are: the Hugging Face cache, LLMInspect's model cache, LLMINSPECT_MODEL_DIRS, and extra."""
    hf = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    env = [Path(p) for p in os.environ.get("LLMINSPECT_MODEL_DIRS", "").split(os.pathsep) if p.strip()]
    out, seen = [], set()
    for r in [hf, Path.home() / ".cache" / "llminspect" / "models", *env, *map(Path, extra)]:
        r = r.expanduser()
        if r.is_dir() and str(r.resolve()) not in seen:
            seen.add(str(r.resolve()))
            out.append(r)
    return out


def history(limit: int = 200) -> list[dict]:
    """Checks run before, newest first: id, model, folder, headline, counts and when."""
    rows = []
    for f in sorted(REPORTS.glob("*/report.json"), reverse=True)[:max(1, limit)]:
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"id": r["id"], "candidate": r["candidate"]["name"], "path": r["candidate"]["path"],
                         "created": r["created"], "headline": r["summary"]["headline"], "counts": r["summary"]["counts"],
                         "deep": "inside_weights" in r["sections"]})
        except (OSError, ValueError, KeyError):
            continue
    return rows


def shelf(roots) -> list[dict]:
    """Every model under the roots: name, folder, size, weight formats and its latest check, if any."""
    latest: dict[str, dict] = {}
    for h in history():
        latest.setdefault(str(Path(h["path"])), h)
    rows = []
    for name, folder in filescan.find_repos(list(roots)):
        try:
            files = [p for p in folder.rglob("*") if p.is_file()]
            size = sum(p.stat().st_size for p in files)
        except OSError:
            continue
        last = latest.get(str(folder))
        rows.append({"name": model_name(folder), "path": str(folder), "gb": round(size / 1e9, 2),
                     "formats": sorted({p.suffix.lower() for p in files if p.suffix.lower() in filescan.WEIGHT_SUFFIXES}),
                     "last": last and {k: last[k] for k in ("id", "headline", "counts", "created", "deep")}})
    return rows


# ---- jobs (check_model and the web UI run here, so a long check never blocks a call) --------------------

JOBS: dict[str, dict] = {}


def start(**kwargs) -> dict:
    jid = uuid.uuid4().hex[:10]
    job = JOBS[jid] = {"job": jid, "status": "running", "step": "starting", "report": None, "error": None}

    def work():
        try:
            rep = run(**kwargs, say=lambda m: job.update(step=m))
            folder = save(rep)
            job.update(status="ready", step="done", report=rep["id"], html=str(folder / "report.html"))
        except Exception as exc:
            job.update(status="error", step=None, error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=work, daemon=True, name=f"check-{jid}").start()
    return dict(job)


def start_all(roots) -> dict:
    """Check every model under the roots, one after another (files, origin, outside the weights; no deep checks),
    as one job: step says how far it got, done lists each report's id and headline. Each model's report is saved as
    it finishes, so the shelf lights up as the job goes."""
    jid = uuid.uuid4().hex[:10]
    roots = [Path(r) for r in roots]
    job = JOBS[jid] = {"job": jid, "status": "running", "step": "finding models", "report": None, "error": None,
                       "done": [], "failed": []}

    def work():
        models = filescan.find_repos(roots)
        for i, (name, folder) in enumerate(models, 1):
            job["step"] = f"{i}/{len(models)} {model_name(folder)}"
            try:
                rep = run(folder, roots=roots)
                save(rep)
                job["done"].append({"id": rep["id"], "candidate": rep["candidate"]["name"], "headline": rep["summary"]["headline"],
                                    "counts": rep["summary"]["counts"]})
            except Exception as exc:  # one unreadable model doesn't stop the rest
                job["failed"].append({"candidate": model_name(folder), "error": f"{type(exc).__name__}: {exc}"})
        job.update(status="ready", step=f"checked {len(job['done'])} of {len(models)} models")

    threading.Thread(target=work, daemon=True, name=f"check-all-{jid}").start()
    return dict(job)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidate")
    ap.add_argument("--base")
    ap.add_argument("--roots", nargs="*", default=[])
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--server")
    args = ap.parse_args(argv)
    rep = run(args.candidate, args.base, args.roots, args.deep, args.server, say=lambda m: print(f"[check] {m}", flush=True))
    folder = save(rep)
    print(f"\n{rep['summary']['headline']}")
    for f in rep["findings"]:
        print(f"  {f['id']} [{f['severity']}] {f['area']}: {f['title']}")
    for n in rep["not_checked"]:
        print(f"  not checked, {n['area']}: {n['why']}")
    print(f"\n{folder / 'report.html'}")


if __name__ == "__main__":
    main()
