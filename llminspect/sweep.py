"""Sweep: check many models on the Hugging Face Hub at once, without downloading any of them.

Models are picked by a Hub search (text, author, tags, sorted by downloads, likes, recency or trend) or named one
by one. Each one's files are read by range: pickles parsed for the code they import on load, configs for code
that runs if trusted, chat templates for sandbox escapes. With --weights each one is also unmasked (see hub.py):
its parent checked against its card's claim and its edit named, for a few more megabytes per model.

Every model becomes one line of a JSON Lines file as soon as it is checked, so a sweep can stop and resume. A
risk is a fact about a model with its evidence, never a verdict: a model with no risk only passed these checks.

    python -m llminspect.sweep [--search TEXT] [--author ORG] [--tag TAG] [--sort downloads] [--limit 50]
                               [--model ORG/NAME ...] [--weights] [--out sweep.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import hub

# Risks, most serious first. The first four are file findings (see filescan.py); the rest come from unmasking.
RISKS = ("runs code on load", "imports other code on load", "runs code if trusted", "template",
         "claims a false parent", "template adds text", "targeted weight edit", "format")
TARGETED = {"direction_removed", "rank_one"}  # edits aimed at a few directions, not a training run
SORTS = ("downloads", "likes", "last_modified", "created_at", "trending_score")
WORKERS = 4


def select(search: str | None = None, author: str | None = None, tags: list[str] | None = None,
           sort: str = "downloads", limit: int = 50) -> list[str]:
    """Model ids from a Hub search, best first by `sort`."""
    from huggingface_hub import HfApi
    found = HfApi().list_models(search=search, author=author, filter=tags or None, sort=sort, limit=limit)
    return [m.id for m in found]


def risks(r: dict) -> list[dict]:
    """The risks in one model's result, most serious first, each with what shows it."""
    out = [{"risk": f["level"], "what": f"{f['check']} in {f['file']}" +
            (f": {', '.join(map(str, f['evidence'][:4]))}" if f.get("evidence") else "")}
           for f in r["files"]["findings"] if f["level"] in RISKS]
    for c in r.get("claims", []):
        if c["reading"] == "not derived from it":
            out.append({"risk": "claims a false parent", "what": f"its card names {c['model']}, which its weights "
                                                                  "don't come from"})
    for t in r["files"].get("template", []):
        if t.get("added_text"):
            out.append({"risk": "template adds text", "what": f"{t['name']}: " + " / ".join(t["added_text"][:2])[:200]})
    e = r.get("edit")
    if e and e["kind"] in TARGETED:
        out.append({"risk": "targeted weight edit", "what": e["headline"]})
    order = {k: i for i, k in enumerate(RISKS)}
    return sorted(out, key=lambda x: order[x["risk"]])


def files_only(model: str) -> dict:
    """A model's files checked by range, without reading its weights."""
    t0 = time.perf_counter()
    cand = hub.open_model(model)
    files = hub._files(cand, None)
    return {"model": cand.name, "url": cand.url, "sha": getattr(cand, "sha", None), "files": files,
            "headline": None, "bytes_read": cand.bytes_read, "seconds": round(time.perf_counter() - t0, 1)}


def check(model: str, weights: bool = False, roots=None) -> dict:
    """One sweep row: the model's risks, its headline and the result they come from."""
    try:
        r = hub.unmask(model, roots=roots) if weights else files_only(model)
    except hub.Unreachable as exc:
        return {"model": model, "error": str(exc), "risks": [], "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as exc:  # noqa: BLE001  one broken repo must not stop a sweep of hundreds
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:300]}", "risks": [],
                "checked_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    hub.prune()  # the small files this model's read left behind
    rs = risks(r)
    return {"model": r["model"], "sha": r.get("sha"), "top": rs[0]["risk"] if rs else None, "risks": rs,
            "headline": r.get("headline"), "weights": weights, "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "result": r}


def done(out: Path) -> dict[str, dict]:
    """Rows already in a sweep file, by model id."""
    rows = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # a line cut short when a sweep was stopped
                continue
            rows[row.get("asked", row["model"])] = row
    return rows


def sweep(models: list[str], out: Path | None = None, weights: bool = False, workers: int = WORKERS,
          roots=None, say=lambda m: None) -> dict:
    """Check each model (skipping those already in `out`, unless their row was an error) and summarize."""
    prior = done(out) if out else {}
    rows = {m: r for m, r in prior.items() if "error" not in r and (r.get("weights") or not weights)}
    todo = [m for m in dict.fromkeys(models) if m not in rows]
    say(f"[sweep] {len(todo)} to check, {len(models) - len(todo)} already in {out}" if out else
        f"[sweep] {len(todo)} to check")
    lock = threading.Lock()
    count = [0]

    def one(m):
        row = {"asked": m, **check(m, weights, roots)}
        with lock:
            count[0] += 1
            rows[m] = row
            if out:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
            mark = row.get("error") or row["top"] or "no risk found"
            say(f"[sweep] {count[0]}/{len(todo)} {m}: {mark}")

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return summary([rows[m] for m in dict.fromkeys(models) if m in rows])


def summary(rows: list[dict]) -> dict:
    """How many models carry each risk, and the flagged ones, most serious first."""
    order = {k: i for i, k in enumerate(RISKS)}
    flagged = sorted((r for r in rows if r.get("risks")), key=lambda r: (order[r["top"]], r["model"]))
    return {"models": len(rows), "errors": sum(1 for r in rows if "error" in r),
            "counts": {k: sum(1 for r in rows if any(x["risk"] == k for x in r.get("risks", []))) for k in RISKS},
            "flagged": [{"model": r["model"], "top": r["top"], "risks": r["risks"], "headline": r.get("headline")}
                        for r in flagged],
            "not_checked": ["behavior: what any model says", "pickle files past the first "
                            f"{hub.PICKLES} of a repo"] + ([] if any(r.get("weights") for r in rows) else
                                                           ["weights: parents and edits (sweep with weights too, or --weights)"])}


JOBS: dict[str, dict] = {}


def start(folder: Path, search=None, author=None, tags=None, sort="downloads", limit=50, models=None,
          weights=False, roots=None) -> dict:
    """Run a sweep in the background into folder/<id>.jsonl; its job (JOBS[id]) shows progress and the summary."""
    import uuid
    folder.mkdir(parents=True, exist_ok=True)
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = {"id": jid, "status": "running", "query": {"search": search, "author": author, "tags": tags, "sort": sort,
                                                      "limit": limit, "models": models, "weights": weights},
           "done": 0, "total": None, "last": None, "summary": None, "error": None}
    JOBS[jid] = job
    (folder / f"{jid}.json").write_text(json.dumps(job["query"]), encoding="utf-8")

    def run():
        try:
            ids = list(models or [])
            if search or author or tags or not ids:
                ids += select(search, author, tags, sort, limit)
            job["total"] = len(dict.fromkeys(ids))

            def say(m):
                hit = re.match(r"\[sweep\] (\d+)/\d+ (.*)", m)
                if hit:
                    job["done"], job["last"] = int(hit.group(1)), hit.group(2)
            job["summary"] = sweep(ids, folder / f"{jid}.jsonl", weights, roots=roots, say=say)
            job["status"] = "ready"
        except Exception as exc:  # noqa: BLE001  the job says why it stopped
            job["status"], job["error"] = "error", f"{type(exc).__name__}: {exc}"

    threading.Thread(target=run, daemon=True).start()
    return job


def history(folder: Path) -> list[dict]:
    """Sweeps run before, newest first: their query and counts."""
    out = []
    for q in sorted(folder.glob("*.json"), reverse=True):
        rows = list(done(q.with_suffix(".jsonl")).values())
        out.append({"id": q.stem, "query": json.loads(q.read_text(encoding="utf-8")), "models": len(rows),
                    "flagged": sum(1 for r in rows if r.get("risks")), "running": JOBS.get(q.stem, {}).get("status") == "running"})
    return out


def load(folder: Path, jid: str) -> dict:
    """A sweep's summary and every row (without the full results), read from its file."""
    path = folder / f"{Path(jid).name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(jid)
    rows = list(done(path).values())
    slim = [{k: r.get(k) for k in ("model", "top", "risks", "headline", "error", "weights", "checked_at")} |
            {"edit": ((r.get("result") or {}).get("edit") or {}).get("kind"),
             "parent": ((r.get("result") or {}).get("parent") or {}).get("model"),
             "formats": ((r.get("result") or {}).get("files") or {}).get("formats")} for r in rows]
    return {"id": jid, "job": JOBS.get(jid), "summary": summary(rows), "rows": slim}


TONES = {"runs code on load": "#f05252", "imports other code on load": "#f05252", "runs code if trusted": "#f05252",
         "template": "#f05252", "claims a false parent": "#8b7dff", "template adds text": "#f5a524",
         "targeted weight edit": "#f5a524", "format": "#6b7385"}


def report(s: dict, rows: list[dict], query: str | None = None) -> str:
    """A sweep as one HTML page with nothing to load: the counts, a tile per model and the flagged ones."""
    from html import escape
    tile = lambda r: TONES.get(r.get("top"), "#3a2a2a" if r.get("error") else "#262b38")  # noqa: E731
    tiles = "".join(f'<i style="background:{tile(r)}" title="{escape(r.get("asked") or r["model"])}: '
                    f'{escape(r.get("top") or r.get("error") or "no risk found")}"></i>' for r in rows)
    counts = "".join(f'<div><b style="color:{TONES[k]}">{n}</b><span>{escape(k)}</span></div>'
                     for k, n in s["counts"].items() if n)
    flagged = "".join(
        f'<tr><td><a href="https://huggingface.co/{escape(r["model"])}">{escape(r["model"])}</a></td>'
        f'<td style="color:{TONES.get(r["top"], "#ccc")}">{escape(r["top"])}</td>'
        f'<td>{"<br>".join(escape(x["what"]) for x in r["risks"][:3])}</td></tr>' for r in s["flagged"])
    title = f"Hub sweep: {escape(query)}" if query else "Hub sweep"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
body{{margin:0;background:#07080d;color:#e7e9ef;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1100px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:32px;margin:0 0 4px}}p{{color:#a9afbf}}.n{{display:flex;flex-wrap:wrap;gap:28px;margin:24px 0}}.n div{{display:flex;flex-direction:column}}
.n b{{font:600 34px/1 ui-monospace,monospace}}.n span{{font:11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.08em;color:#8a91a3;margin-top:6px}}
.f{{display:flex;flex-wrap:wrap;gap:4px;margin:12px 0 28px}}.f i{{width:22px;height:22px;border-radius:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td{{padding:8px 10px;border-top:1px solid #1b1f2b;vertical-align:top}}
td:first-child{{font-family:ui-monospace,monospace}}a{{color:#e7e9ef}}footer{{margin-top:32px;color:#6b7385;font-size:12px}}
</style></head><body><main><h1>{title}</h1>
<p>{s["models"]} models checked without downloading any; {len(s["flagged"])} carry a risk, {s["errors"]} couldn't be read.
Checked {time.strftime("%Y-%m-%d")} with LLMInspect.</p><div class="n">{counts}</div><div class="f">{tiles}</div>
<table>{flagged}</table><footer>A risk is a fact about a model's files or weights, with its evidence; a model with no risk only
passed these checks. Not checked: {escape("; ".join(s["not_checked"]))}.</footer></main></body></html>"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--search", help="text the model id contains")
    ap.add_argument("--author", help="an org or user")
    ap.add_argument("--tag", action="append", help="a Hub tag, such as gguf or base_model:finetune:ORG/NAME")
    ap.add_argument("--sort", default="downloads", choices=SORTS)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", action="append", help="check this model too (repeatable)")
    ap.add_argument("--weights", action="store_true", help="also unmask each model's weights (a few MB each)")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--out", type=Path, default=Path("sweep.jsonl"), help="JSON Lines file, appended and resumed")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    ap.add_argument("--report", type=Path, help="also write the result as one self-contained HTML page")
    args = ap.parse_args(argv)
    models = list(args.model or [])
    if args.search or args.author or args.tag or not models:
        models += select(args.search, args.author, args.tag, args.sort, args.limit)
    s = sweep(models, args.out, args.weights, args.workers, say=lambda m: print(m, file=sys.stderr))
    if args.report:
        rows = [r for m, r in done(args.out).items() if m in set(models)]
        args.report.write_text(report(s, rows, " ".join(filter(None, [args.search, args.author, *(args.tag or [])])) or None),
                               encoding="utf-8")
        print(f"report: {args.report}", file=sys.stderr)
    if args.json:
        print(json.dumps(s, indent=1))
        return
    print(f"{s['models']} models checked, {len(s['flagged'])} with a risk, {s['errors']} unreadable")
    for k, n in s["counts"].items():
        if n:
            print(f"  {n:4d}  {k}")
    for r in s["flagged"]:
        print(f"{r['top']:>28}  {r['model']}")
        for x in r["risks"][:3]:
            print(f"{'':>30}{x['what'][:110]}")
    print("Not checked: " + "; ".join(s["not_checked"]))


if __name__ == "__main__":
    main(sys.argv[1:])
