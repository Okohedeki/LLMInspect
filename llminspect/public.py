"""Public mode: the Hub scanner as a service anyone can open. A separate, small app rather than the full server with
parts switched off, so nothing that reads a local folder, loads a model or writes a file is there to reach.

  /                       Unmask and Sweep (ui/public.html)
  /m/ORG/NAME             one model's card, with a preview image for links shared on social sites
  /og/ORG/NAME.png        that preview image, drawn on the server
  /r/ID                   a sweep's report page
  /llminspect/unmask      GET ?model=ORG/NAME: the result behind a card, cached by the repo's revision
  /llminspect/sweep       POST to start a sweep; GET /llminspect/sweep/ID to follow it; GET /llminspect/sweeps
  /healthz                for the host's health checks

Only Hugging Face Hub ids are accepted (never a path), gated repos are refused even when the host has a token (a
token sent on behalf of anyone would hand out weights its owner agreed to keep behind the gate), sweeps are capped,
and each client is rate limited. Results are kept under LLMINSPECT_DATA (default ./data).

    uvicorn llminspect.public:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import hub, sweep

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
DATA = Path(os.environ.get("LLMINSPECT_DATA") or ROOT / "data")
CARDS, SWEEPS = DATA / "cards", DATA / "sweeps"
REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9_.-]{1,96}$")
CARD_TTL = 24 * 3600          # seconds a card is served from cache before it is read again
MAX_MODELS = 50               # models one sweep may check
MAX_WEIGHTS = 12              # models one sweep may unmask (weights read too)
UNMASKS = threading.BoundedSemaphore(3)   # unmasks running at once
RUNNING_SWEEPS = 2            # sweeps running at once, across everyone
LIMITS = {"unmask": (30, 600), "sweep": (4, 3600)}  # (requests, per seconds) for one client
TEXT = re.compile(r"^[\w .:/+-]{0,100}$")

@asynccontextmanager
async def lifespan(_app):
    """Hub reads are public-only while the app runs (set at start, not on import, so tests in one process differ)."""
    hub.PUBLIC = True
    yield
    hub.PUBLIC = False


app = FastAPI(title="LLMInspect", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
_seen: dict[tuple[str, str], deque] = defaultdict(deque)
_lock = threading.Lock()


def client(request: Request) -> str:
    """The caller's address: the first hop a proxy in front of the app reports, else the socket's."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "?")


def limit(request: Request, kind: str) -> None:
    n, window = LIMITS[kind]
    now, key = time.time(), (client(request), kind)
    with _lock:
        q = _seen[key]
        while q and q[0] < now - window:
            q.popleft()
        if len(q) >= n:
            wait = int(q[0] + window - now) + 1
            raise HTTPException(429, f"Too many requests; try again in {wait} s.", headers={"Retry-After": str(wait)})
        q.append(now)


def repo(model: str) -> str:
    model = (model or "").strip().removeprefix("https://huggingface.co/").rstrip("/")
    if not REPO.match(model) or ".." in model:
        raise HTTPException(400, "Give a Hugging Face model id, like org/name.")
    return model


# ---- cards -------------------------------------------------------------------------------------------------

def _card_path(model: str) -> Path:
    return CARDS / (model.replace("/", "--") + ".json")


def cached(model: str) -> dict | None:
    p = _card_path(model)
    try:
        if time.time() - p.stat().st_mtime < CARD_TTL:
            return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return None


def unmask(model: str) -> dict:
    """A model's card, from the cache or read now (a few at a time)."""
    hit = cached(model)
    if hit:
        return hit
    if not UNMASKS.acquire(timeout=120):
        raise HTTPException(503, "Busy reading other models; try again in a minute.")
    try:
        r = hub.unmask(model, roots=[])
    except hub.Unreachable as exc:
        raise HTTPException(404, str(exc)) from exc
    finally:
        UNMASKS.release()
        hub.prune()
    CARDS.mkdir(parents=True, exist_ok=True)
    _card_path(model).write_text(json.dumps(r), encoding="utf-8")
    return r


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/llminspect/unmask")
async def get_unmask(model: str, request: Request):
    model = repo(model)
    if not cached(model):
        limit(request, "unmask")
    return await asyncio.to_thread(unmask, model)


# ---- sweeps ------------------------------------------------------------------------------------------------

@app.post("/llminspect/sweep")
async def post_sweep(request: Request):
    try:
        body = await request.json()
    except ValueError:
        body = {}
    body = body if isinstance(body, dict) else {}
    search, author = (body.get("search") or None), (body.get("author") or None)
    tags = body.get("tags") or None
    models = [repo(m) for m in (body.get("models") or [])][:MAX_MODELS]
    for t in [search, author, *(tags or [])]:
        if t is not None and (not isinstance(t, str) or not TEXT.match(t)):
            raise HTTPException(400, "Search, author and tags are short plain text.")
    sort = body.get("sort") or "downloads"
    if sort not in sweep.SORTS:
        raise HTTPException(400, f"Sort by one of {', '.join(sweep.SORTS)}.")
    weights = bool(body.get("weights"))
    cap = MAX_WEIGHTS if weights else MAX_MODELS
    n = max(1, min(int(body.get("limit") or 25), cap))
    if sum(1 for j in sweep.JOBS.values() if j["status"] == "running") >= RUNNING_SWEEPS:
        raise HTTPException(503, "Other sweeps are running; try again in a few minutes.")
    limit(request, "sweep")
    return sweep.start(SWEEPS, search, author, tags, sort, n, models[:cap] or None, weights, roots=[])


@app.get("/llminspect/sweeps")
async def get_sweeps():
    return (await asyncio.to_thread(sweep.history, SWEEPS))[:20] if SWEEPS.exists() else []


@app.get("/llminspect/sweep/{jid}")
async def get_sweep(jid: str):
    if not re.fullmatch(r"[\w-]{1,64}", jid):
        raise HTTPException(404, "No such sweep.")
    try:
        return await asyncio.to_thread(sweep.load, SWEEPS, jid)
    except FileNotFoundError:
        if jid in sweep.JOBS:
            return {"id": jid, "job": sweep.JOBS[jid], "summary": None, "rows": []}
        raise HTTPException(404, "No such sweep.") from None


@app.get("/r/{jid}", response_class=HTMLResponse)
async def report_page(jid: str):
    if not re.fullmatch(r"[\w-]{1,64}", jid) or not (SWEEPS / f"{jid}.jsonl").exists():
        raise HTTPException(404, "No such sweep.")
    query = json.loads((SWEEPS / f"{jid}.json").read_text(encoding="utf-8"))
    rows = list(sweep.done(SWEEPS / f"{jid}.jsonl").values())
    what = " ".join(filter(None, [query.get("search"), query.get("author"), *(query.get("tags") or [])])) or None
    return sweep.report(sweep.summary(rows), rows, what)


# ---- pages -------------------------------------------------------------------------------------------------

def page(model: str | None = None) -> str:
    """The public page, with the link-preview tags for one model when the link names one."""
    html = (UI / "public.html").read_text(encoding="utf-8")
    if not model:
        return html
    r = cached(model)
    title = f"{model}: {r['headline']}" if r else f"What is {model} really? Unmasked by LLMInspect"
    meta = (f'<meta property="og:title" content="{escape(title[:200])}">'
            f'<meta property="og:description" content="Its real parent and what was done to it, read from the '
            f'Hugging Face Hub without downloading it.">'
            f'<meta property="og:image" content="/og/{escape(model)}.png">'
            f'<meta property="og:type" content="website"><meta name="twitter:card" content="summary_large_image">')
    return html.replace("<!--meta-->", meta).replace('data-unmask=""', f'data-unmask="{escape(model)}"')


@app.get("/", response_class=HTMLResponse)
async def index():
    return page()


@app.get("/m/{org}/{name}", response_class=HTMLResponse)
async def model_page(org: str, name: str):
    return page(repo(f"{org}/{name}"))


@app.get("/og/{org}/{name}.png")
async def og_image(org: str, name: str, request: Request):
    from . import ogcard
    model = repo(f"{org}/{name}")
    if not cached(model):
        limit(request, "unmask")
    try:
        r = await asyncio.to_thread(unmask, model)
    except HTTPException:
        r = None
    png = await asyncio.to_thread(ogcard.render, model, r)
    return Response(png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@app.exception_handler(HTTPException)
async def errors(request: Request, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.middleware("http")
async def headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.mount("/ui", StaticFiles(directory=UI), name="ui")


@app.get("/favicon.ico")
async def favicon():
    return FileResponse(UI / "favicon.svg") if (UI / "favicon.svg").exists() else Response(status_code=204)
