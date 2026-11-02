"""LLMInspect as an MCP server: tools an agent uses to check an open-weights model before trusting it.

    python -m llminspect.mcp_server            # stdio, for Claude Code, Claude Desktop and other MCP clients

  list_models   models on this machine (Hugging Face cache, LLMInspect's cache, LLMINSPECT_MODEL_DIRS)
  check_model   every check of one model, as one report with findings and next steps (a job; waits briefly)
  check_all_models   every model on this machine at once (a job)
  get_check     a check's result: summary, findings, or the full report; and its HTML page
  list_checks   checks run before
  scan_files    code that runs on load, for one model folder
  diff_files    what changed outside the weights, base vs candidate
  fingerprint_model   which local model a candidate comes from
  unmask_model  a model's real parents and what was done to them, read from the Hub without downloading it

Nothing here loads a model or runs a model's code. The weight checks (check_model with deep=true) are run by a
LLMInspect server (python -m llminspect.server) that has the base model loaded; set LLMINSPECT_URL if it isn't
at http://127.0.0.1:8765.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import check, filediff, filescan, fingerprint

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

server = MCPServer(
    "llminspect",
    instructions=(
        "Check open-weights models before trusting them. Start with check_model on the model's folder; it returns "
        "findings ordered by severity, each with evidence and a `next` step (often another tool call). Pass `base` "
        "when you know the model it should derive from, otherwise the closest local model is used. Use deep=true "
        "only when a LLMInspect server with the base loaded is running. Findings are evidence, never a verdict: "
        "don't tell the user a model is safe because nothing was found; say what was checked (summary.checked) "
        "and what wasn't (not_checked). For a model on the Hugging Face Hub, unmask_model reads it by byte range "
        "without downloading it; sweep_hub does that for many at once. run_verb runs a verb (sign, harden, audit, "
        "scrub and the rest) on the model a running server has loaded; a change it makes is kept unapplied."),
)


def _roots(roots: list[str] | None) -> list[Path]:
    return [Path(r).expanduser() for r in roots] if roots else check.default_roots()


def _view(report: dict, detail: str) -> dict:
    """A report at the detail an agent asked for: summary < findings < full."""
    head = {k: report[k] for k in ("id", "candidate", "base", "summary", "not_checked")}
    html = check.REPORTS / report["id"] / "report.html"
    head["html"] = str(html) if html.exists() else None
    if detail == "summary":
        return head | {"top": [{k: f[k] for k in ("id", "severity", "area", "short")} for f in report["findings"][:5]]}
    if detail == "findings":
        return head | {"findings": report["findings"], "limits": report["limits"]}
    return report | {"html": head["html"]}


@server.tool(annotations=READ_ONLY, structured_output=True)
def list_models(roots: list[str] | None = None) -> dict[str, Any]:
    """Models on this machine: name, folder, size, weight formats and the headline of its latest check (`last`).
    `roots` defaults to the Hugging Face cache, LLMInspect's model cache and LLMINSPECT_MODEL_DIRS."""
    return {"roots": [str(r) for r in _roots(roots)], "models": check.shelf(_roots(roots))}


@server.tool(structured_output=True)
def check_model(candidate: str, base: str | None = None, roots: list[str] | None = None, deep: bool = False,
                wait_seconds: float = 20) -> dict[str, Any]:
    """Run every check on the model folder `candidate` and return one report: code that runs on load, which model
    it really comes from, what changed outside the weights, and (deep=true) inside them.

    base: the model it should derive from (a folder); omit to use the closest local model.
    roots: folders to look for possible bases in; defaults to this machine's model caches.
    deep: also run the weight checks (triggers, refusals, behavior) through a LLMInspect server with the base loaded.
    wait_seconds: how long to wait for the result before returning the job to poll with get_check.

    Returns the summary and findings when done, else {job, status, step}."""
    job = check.start(candidate=candidate, base=base, roots=[str(r) for r in _roots(roots)], deep=deep)
    end = time.monotonic() + max(0.0, min(wait_seconds, 120))
    while time.monotonic() < end and check.JOBS[job["job"]]["status"] == "running":
        time.sleep(0.25)
    return _job_view(job["job"], "findings")


@server.tool(structured_output=True)
def check_all_models(roots: list[str] | None = None) -> dict[str, Any]:
    """Check every model on this machine (files, origin, outside the weights; not deep) as one job, each model's
    report saved as it finishes. Poll with get_check(job); when ready, `done` lists every model's headline and
    counts, most serious first, and `failed` any model that couldn't be read."""
    return check.start_all(_roots(roots))


def _job_view(jid: str, detail: str) -> dict:
    job = check.JOBS[jid]
    if job.get("done") is not None:  # every model at once (check_all_models): the list is the result
        rank = lambda d: (-d["counts"]["high"], -d["counts"]["medium"])
        return {k: job.get(k) for k in ("job", "status", "step", "error", "failed")} | {
            "done": sorted(job["done"], key=rank)}
    if job["status"] == "ready":
        return {"job": jid, "status": "ready"} | _view(check.load(job["report"]), detail)
    return {k: job.get(k) for k in ("job", "status", "step", "error")} | (
        {"poll": "get_check with this job id"} if job["status"] == "running" else {})


@server.tool(annotations=READ_ONLY, structured_output=True)
def get_check(id: str, detail: Literal["summary", "findings", "full"] = "findings") -> dict[str, Any]:
    """A check's result by job id (from check_model) or report id. detail: summary (headline, counts, top five),
    findings (every finding with evidence and next steps; the default), or full (every section's raw data)."""
    if id in check.JOBS:
        return _job_view(id, detail)
    try:
        return _view(check.load(id), detail)
    except FileNotFoundError:
        return {"error": f"no check or job {id!r}; list_checks shows the reports there are"}


@server.tool(annotations=READ_ONLY, structured_output=True)
def list_checks(limit: int = 10) -> dict[str, Any]:
    """Checks run before, newest first: id, model, headline and counts."""
    return {"checks": check.history(limit)}


@server.tool(annotations=READ_ONLY, structured_output=True)
def scan_files(path: str) -> dict[str, Any]:
    """Code that runs when the model folder `path` is loaded or used: what each pickle would import, custom code under
    trust_remote_code (risky calls by file and line), shipped wheels, chat templates reaching Python internals,
    Keras Lambda layers, malformed safetensors. Reads bytes only."""
    r = filescan.scan_repo(Path(path).expanduser())
    return {k: r[k] for k in ("name", "path", "files", "bytes", "weight_formats", "top", "findings")}


@server.tool(annotations=READ_ONLY, structured_output=True)
def diff_files(base: str, candidate: str) -> dict[str, Any]:
    """What `candidate` changed outside its weights against `base` (both folders): chat template (diff and the text
    it adds), tokenizer (by the ids it gives probe texts), config, generation defaults, files only one side has."""
    return filediff.compare(Path(base).expanduser(), Path(candidate).expanduser())


@server.tool(annotations=READ_ONLY, structured_output=True)
def fingerprint_model(candidate: str, roots: list[str] | None = None) -> dict[str, Any]:
    """Which local model `candidate` comes from: weight correlation with every local model of its architecture, with
    a per-layer profile (a full fine-tune moves every layer a little, a targeted edit a few), and the base its model
    card claims."""
    r = fingerprint.identify(Path(candidate).expanduser(), filescan.find_repos(_roots(roots)))
    r["matches"] = r["matches"][:5]
    return r


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True), structured_output=True)
def unmask_model(model: str, roots: list[str] | None = None) -> dict[str, Any]:
    """What a model really is, without downloading it: its claimed parents and whether its weights agree, what was
    done to them (the same direction or few removed from every layer, rank-one edits, a low-rank adapter merged, or a full
    fine-tune), and what its configs, code and chat template hold. Reads a few megabytes by byte range (tens for
    a large GGUF). A GGUF repo is read through its most precise quantization; `quant` names it, only change above
    its rounding counts, and a plain quantization of its parent is followed to the edit that parent carries
    (`copy_of`).

    model: ORG/NAME on the Hugging Face Hub (safetensors or GGUF), a model on this machine by that name, or a folder.
    roots: folders of local models to compare with; defaults to this machine's model caches.

    The headline names the parent and the edit. Behavior isn't run; see not_checked."""
    from . import hub
    try:
        r = hub.unmask(model, roots=_roots(roots))
    except hub.Unreachable as exc:
        return {"model": model, "error": str(exc)}
    if r["edit"]:
        r["edit"] = {k: v for k, v in r["edit"].items() if k != "rank"} | {
            "few_direction_layers": r["edit"]["rank"]["few"], "shared_direction": r["edit"]["rank"]["shared"]}
    return r


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True), structured_output=True)
def sweep_hub(search: str | None = None, author: str | None = None, tags: list[str] | None = None,
              sort: Literal["downloads", "likes", "last_modified", "created_at", "trending_score"] = "downloads",
              limit: int = 20, models: list[str] | None = None, weights: bool = False) -> dict[str, Any]:
    """Check many Hugging Face Hub models at once, without downloading any: pickle weights parsed by range for code
    they run on load, configs for code that runs if trusted, chat templates for sandbox escapes. With weights=true
    each model is also unmasked (claimed parent checked, edit named), a few more megabytes per model.

    search / author / tags / sort / limit pick models from a Hub search (limit at most 50); models adds ids by name.

    Returns how many models carry each risk and the flagged ones, most serious first. A model with no risk only
    passed these checks; see not_checked."""
    from . import sweep
    ids = list(models or [])
    if search or author or tags or not ids:
        ids += sweep.select(search, author, tags, sort, min(limit, 50))
    return sweep.sweep(ids, weights=weights)


@server.tool(structured_output=True)
def run_verb(verb: str, step: str, body: dict[str, Any] | None = None, server_url: str | None = None) -> dict[str, Any]:
    """Run one step of a verb on the model a running LLMInspect server has loaded (LLMINSPECT_URL, default
    http://127.0.0.1:8765). Verbs: sign (mark the weights, verify a mark), transplant (carry a fine-tune or a
    direction from another model), rewrite (change one fact), scrub (find and remove memorized text), saysknows
    (answers beside a truth probe), classify (a probe head on the model's activations), inject (plant a concept
    and ask), harden (attack refusals, make them survive), cap (keep a direction in range), shrink (prune what your
    prompts don't use), faithfulness (hints used but not said), audit (what a fine-tune changed, blind).

    GET {server}/llminspect/verbs lists each verb's steps. A step that changes weights is kept as an edit,
    unapplied; nothing is applied without the workbench's apply."""
    import httpx
    url = (server_url or os.environ.get("LLMINSPECT_URL", "http://127.0.0.1:8765")).rstrip("/")
    try:
        r = httpx.post(f"{url}/llminspect/verbs/{verb}/{step}", json=body or {}, timeout=3600)
    except httpx.HTTPError as exc:
        return {"error": f"no LLMInspect server at {url} ({type(exc).__name__}); start one with the model loaded"}
    if r.status_code != 200:
        return {"error": f"{r.status_code}: {r.text[:500]}"}
    return r.json()


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
