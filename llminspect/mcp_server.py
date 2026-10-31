"""LLMInspect as an MCP server: tools an agent uses to check an open-weights model before trusting it.

    python -m llminspect.mcp_server            # stdio, for Claude Code, Claude Desktop and other MCP clients

  list_models   models on this machine (Hugging Face cache, LLMInspect's cache, LLMINSPECT_MODEL_DIRS)
  check_model   every check of one model, as one report with findings and next steps (a job; waits briefly)
  get_check     a check's result: summary, findings, or the full report; and its HTML page
  list_checks   checks run before
  scan_files    code that runs on load, for one model folder
  diff_files    what changed outside the weights, base vs candidate
  fingerprint_model   which local model a candidate comes from

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
        "and what wasn't (not_checked)."),
)


def default_roots() -> list[Path]:
    """Where models usually are on this machine."""
    hf = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    extra = [Path(p) for p in os.environ.get("LLMINSPECT_MODEL_DIRS", "").split(os.pathsep) if p.strip()]
    return [r for r in [hf, Path.home() / ".cache" / "llminspect" / "models", *extra] if r.is_dir()]


def _roots(roots: list[str] | None) -> list[Path]:
    return [Path(r).expanduser() for r in roots] if roots else default_roots()


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
    """Models on this machine: name, folder, size and weight formats. `roots` defaults to the Hugging Face cache,
    LLMInspect's model cache and LLMINSPECT_MODEL_DIRS."""
    rows = []
    for name, folder in filescan.find_repos(_roots(roots)):
        files = [p for p in folder.rglob("*") if p.is_file()]
        rows.append({"name": check.display(name), "path": str(folder),
                     "gb": round(sum(p.stat().st_size for p in files) / 1e9, 2),
                     "formats": sorted({p.suffix.lower() for p in files if p.suffix.lower() in filescan.WEIGHT_SUFFIXES})})
    return {"roots": [str(r) for r in _roots(roots)], "models": rows}


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


def _job_view(jid: str, detail: str) -> dict:
    job = check.JOBS[jid]
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
    rows = []
    for folder in sorted(check.REPORTS.glob("*/report.json"), reverse=True)[:max(1, limit)]:
        try:
            r = check.load(folder.parent.name)
        except (OSError, ValueError):
            continue
        rows.append({"id": r["id"], "candidate": r["candidate"]["name"], "created": r["created"],
                     "headline": r["summary"]["headline"], "counts": r["summary"]["counts"]})
    return {"checks": rows}


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


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
