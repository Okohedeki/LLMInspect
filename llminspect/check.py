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


def _origin(cand: Path, refs: list[tuple[str, Path]], say) -> tuple[dict, list[dict]]:
    r = fingerprint.identify(cand, refs, say)
    for m in r["matches"]:
        m["name"] = display(m["name"])
    r["closest"] = r["closest"] and display(r["closest"])
    out, best = [], (r["matches"] or [None])[0]
    claim = r.get("claimed_base") or []
    if r.get("error"):
        return r, [_finding("origin", "note", "Origin not checked: " + r["error"] + ".")]
    if best is None or best["median"] <= 0.5:
        out.append(_finding("origin", "note", "No local model it derives from; its origin can't be checked here.",
                            "Add the claimed base model's folder to the roots to check it." if claim else "",
                            claim, next_={"do": "Put the base model on this machine and check again.", "tool": "check_model"}))
        return r, out
    name = display(best["name"])
    if claim and not any(_same_model(c, name) for c in claim):
        out.append(_finding("origin", "high", f"The card says it is built on {', '.join(claim)}; its weights come from {name}.",
                            f"Median correlation {best['median']:.3f} with {name}.", [f"claimed: {c}" for c in claim]))
    moved = [int(l) for l, c in best["layers"].items() if l != "-1" and c < 0.9995]
    still = [int(l) for l, c in best["layers"].items() if l != "-1" and c >= 0.9995]
    if best["identical"] == best["tensors"]:
        out.append(_finding("origin", "note", f"Same weights as {name}."))
    elif moved and still and len(moved) <= max(2, len(best["layers"]) // 4):
        span = _span(moved)
        out.append(_finding("origin", "medium", f"Edited in {_n(len(moved), 'layer')} ({span}); every other layer is {name}'s, unchanged.",
                            "A targeted edit looks like this (so do some narrow fine-tunes): look inside those layers.",
                            [f"layer {l}: correlation {best['layers'][str(l)]:.4f}" for l in sorted(moved)][:12],
                            next_={"do": "Run the weight checks to see what the edit does.", "tool": "check_model",
                                   "args": {"candidate": str(cand), "base": best["path"], "deep": True}}))
    else:
        out.append(_finding("origin", "note", f"Derived from {name}: every layer moved a little, as in a full fine-tune.",
                            f"Median correlation {best['median']:.3f}, lowest {best['min']:.3f}."))
    return r, out


def display(name: str) -> str:
    """'Org/Name' for a folder named the cache's way ('Org--Name' or 'Org__Name')."""
    m = re.match(r"^([^/]+?)(?:--|__)(.+)$", name)
    return f"{m.group(1)}/{m.group(2)}" if m and "/" not in name else name


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def _same_model(claim: str, name: str) -> bool:
    norm = lambda s: s.lower().replace("--", "/").replace("__", "/").rstrip("/").split("/")[-1]
    return norm(claim) == norm(name)


def _span(layers: list[int]) -> str:
    layers = sorted(layers)
    return f"L{layers[0]}" if len(layers) == 1 else (f"L{layers[0]}–L{layers[-1]}" if layers[-1] - layers[0] == len(layers) - 1
                                                     else ", ".join(f"L{l}" for l in layers))


def _outside(base: Path, cand: Path) -> tuple[dict, list[dict]]:
    r = filediff.compare(base, cand)
    out = []
    for s in r["signals"]:
        evidence, where = [], s["section"]
        if s["section"] == "template":
            t = r["templates"][0]
            evidence = t.get("added_text") or [l for l in t.get("diff", []) if l.startswith(("+", "-"))][:12]
        elif s["section"] == "tokenizer" and r["tokenizer"].get("encodes_differently"):
            evidence = [f"{x['text']!r}: {x['base'][:8]} -> {x['candidate'][:8]}" for x in r["tokenizer"]["encodes_differently"][:6]]
        elif s["section"] in ("config", "generation"):
            evidence = [f"{c['key']}: {c['base']!r} -> {c['candidate']!r}" for c in r[s["section"]][:10]]
        out.append(_finding("outside_weights", s["level"] if s["level"] in SEVERITY else "note", s["text"], "", evidence, where,
                            {"do": "See the full comparison.", "tool": "diff_files", "args": {"base": str(base), "candidate": str(cand)}}))
    return r, out


def _server(url: str, method: str, path: str, body: dict | None = None, timeout: float = 30):
    req = urllib.request.Request(url.rstrip("/") + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _inside(cand: Path, server: str, say) -> tuple[dict | None, list[dict], str | None]:
    """The deploy scan, run by the LLMInspect server with the base loaded. (report, findings, why not run)."""
    try:
        _server(server, "POST", "/llminspect/scan", {"spec": str(cand)})
    except urllib.error.HTTPError as exc:
        return None, [], f"the LLMInspect server refused the scan: {exc.read().decode(errors='replace')[:300]}"
    except (urllib.error.URLError, OSError) as exc:
        return None, [], (f"no LLMInspect server at {server} ({exc}). Start one with the base model loaded "
                          "(LLMINSPECT_MODEL=<base> python -m llminspect.server), then check again with deep=true.")
    while True:
        st = _server(server, "GET", "/llminspect/scan")
        say(f"inside the weights: {st.get('step')} ({100 * (st.get('progress') or 0):.0f}%)")
        if st["status"] == "error":
            return None, [], f"the deploy scan failed: {st['error']}"
        if st["status"] == "ready":
            break
        time.sleep(3)
    report = _server(server, "GET", f"/llminspect/scan/reports/{st['report']['id']}")
    out = []
    for s in report.get("signals", []):
        sev = s["level"] if s["level"] in SEVERITY else "note"
        nxt = None
        if s["section"] == "triage" and "Token" in s["text"]:
            nxt = {"do": "Try the token in a chat with the candidate loaded, and trace what it changes."}
        out.append(_finding("inside_weights", sev, s["text"], "", [], s["section"], nxt))
    return report, out, None


# ---- the whole check -----------------------------------------------------------------------------------

def run(candidate, base=None, roots=(), deep: bool = False, server: str | None = None, say=lambda m: None) -> dict:
    """Every check of `candidate` (a model folder), against `base` (a folder) or the local model it derives from."""
    t0 = time.perf_counter()
    cand = Path(candidate).expanduser()
    if not cand.is_dir():
        raise FileNotFoundError(f"no model folder at {candidate}")
    roots = [Path(r).expanduser() for r in roots]
    report = {"schema": SCHEMA, "id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
              "created": datetime.now().astimezone().isoformat(timespec="seconds"),
              "candidate": {"name": display(cand.name), "path": str(cand)}, "base": None, "sections": {}, "not_checked": []}
    findings = []
    say("files: reading every file for code that runs on load")
    report["sections"]["files"], f = _files(cand)
    findings += f
    refs = filescan.find_repos(roots) if roots else []
    if base:
        refs = [(Path(base).name, Path(base).expanduser())] + [r for r in refs if r[1].resolve() != Path(base).expanduser().resolve()]
    if refs:
        say(f"origin: fingerprinting against {len(refs)} local model(s)")
        report["sections"]["origin"], f = _origin(cand, refs, say)
        findings += f
    else:
        report["not_checked"].append({"area": "origin", "why": "no base or model folders to compare with were given"})
    origin = report["sections"].get("origin") or {}
    best = (origin.get("matches") or [None])[0]
    base_path = Path(base).expanduser() if base else (Path(best["path"]) if best and best["median"] > 0.9 else None)
    if base_path is not None:
        report["base"] = {"name": display(base_path.name), "path": str(base_path), "chosen_by": "given" if base else "fingerprint"}
        say(f"outside the weights: comparing with {base_path.name}")
        report["sections"]["outside_weights"], f = _outside(base_path, cand)
        findings += f
    else:
        report["not_checked"].append({"area": "outside_weights", "why": "no base model to compare with"})
    if deep and base_path is not None:
        server = server or os.environ.get("LLMINSPECT_URL", "http://127.0.0.1:8765")
        rep, f, why = _inside(cand, server, say)
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


# ---- jobs (the MCP server's check_model runs here, so a long check never blocks a call) ------------------

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
