"""HTTP server: OpenAI-compatible chat endpoint, recorder API, live feed and scanner UI.

Run inside WSL:   scripts/run.sh -m llminspect.server
Point any OpenAI-compatible harness at http://localhost:8765/v1 and open
http://localhost:8765/ for the scanner. Send `X-LLMInspect-Run: <id>` to group
requests into a run explicitly; otherwise runs are inferred from shared prefixes.
"""

from __future__ import annotations

import os

# Less fragmentation from the very different tensor sizes the recorder and workbench allocate.
# Must be set before CUDA initialises.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import asyncio  # noqa: E402
import contextlib
import gc
import json
import mimetypes
import struct
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings
from .recorder import actions, catalog, knobs, monitors, neuron_index, neurons, workbench
from .recorder.model import Subject, display_name
from .recorder.session import Cancelled, GenParams, ModelSwitched, Recorder, compact
from .recorder.store import RunStore
from .recorder.tracing import Busy

UI_DIR = PROJECT_ROOT / "ui"
settings = Settings()
STARTUP_MODEL = settings.model  # settings.model follows model switches


class LiveHub:
    """Fans recorder messages out to every connected scanner, batched every ~30 ms."""

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None

    def publish(self, msg: dict) -> None:
        if self.loop is not None and self.queue is not None:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)

    async def pump(self) -> None:
        while True:
            batch = [await self.queue.get()]
            while not self.queue.empty() and len(batch) < 400:
                batch.append(self.queue.get_nowait())
            if self.clients:
                data = json.dumps(batch)
                for ws in list(self.clients):
                    try:
                        await ws.send_text(data)
                    except Exception:
                        self.clients.discard(ws)
            await asyncio.sleep(0.03)


hub = LiveHub()
state: dict = {"recorder": None, "status": "loading", "step": "starting", "error": None,
               "target": None, "switch": None}
_switching = threading.Lock()  # one model load or switch at a time


def _step(text: str) -> None:
    state["step"] = text
    hub.publish({"k": "status", "status": state["status"], "step": text})


def _load(spec: str | None = None) -> bool:
    """Load a model and build its recorder; True when it is ready."""
    spec = spec or settings.model
    state.update(status="loading", error=None, target=spec)
    try:
        if settings.gpu_fraction and settings.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(min(max(settings.gpu_fraction, 0.05), 1.0))
        _step(f"loading {display_name(spec)}")
        subject = Subject(spec, device=settings.device, gpu_fraction=settings.gpu_fraction)
        _step(f"mapping {subject.name}'s units and edges (cached after the first load)")
        rec = Recorder(subject, RunStore(settings.data_dir, settings.disk_budget_gb), settings, hub.publish)
        _step("building direction bank")
        with rec.lock:
            rec.directions.ensure_defaults()
        settings.model = spec
        state.update(recorder=rec, status="ready", step="ready", target=None)
        hub.publish({"k": "status", "status": "ready"})
        hub.publish({"k": "model", "model": subject.info()})
        print(f"[llminspect] ready: {subject.name} ({subject.n_layers} layers) on {subject.device}", flush=True)
        return True
    except Exception as exc:
        traceback.print_exc()
        state.update(status="error", error=f"{type(exc).__name__}: {exc}", target=None)
        hub.publish({"k": "status", "status": "error"})
        return False


def _startup() -> None:
    try:
        _load()
    finally:
        _switching.release()


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _switch(spec: str) -> None:
    """Unload the current model, load `spec`, and reload the previous model if that fails. Takes the
    recorder from `state` itself: a reference held anywhere else (even in this thread's arguments)
    would keep the old weights on the GPU."""
    try:
        rec, state["recorder"] = state["recorder"], None
        previous = rec.s.spec if rec is not None else None
        if rec is not None:
            _step(f"unloading {rec.s.name}")
            rec.close()  # waits for a forward pass in flight
            del rec
            _free_memory()
        if _load(spec):
            state["switch"] = {"from": previous, "to": spec, "error": None}
            return
        state["switch"] = {"from": previous, "to": spec, "error": state["error"]}
        _free_memory()
        if previous is not None and _load(previous):
            state["switch"]["restored"] = previous
    finally:
        _switching.release()


def _gpu_budget() -> int | None:
    """GPU memory free once the current model is unloaded, or None when models load on the CPU."""
    if not settings.device.startswith("cuda") or not torch.cuda.is_available():
        return None
    free, _ = torch.cuda.mem_get_info()
    return free + torch.cuda.memory_reserved()


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.loop, hub.queue = asyncio.get_running_loop(), asyncio.Queue()
    pump = asyncio.create_task(hub.pump())
    _switching.acquire()
    threading.Thread(target=_startup, daemon=True).start()
    yield
    pump.cancel()


app = FastAPI(title="LLMInspect recorder", lifespan=lifespan)


@app.middleware("http")
async def revalidate_ui(request: Request, call_next):
    """Make browsers revalidate the scanner files so UI updates show up on reload."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/ui/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def recorder() -> Recorder:
    if state["status"] != "ready":
        raise HTTPException(503, f"recorder {state['status']}: {state['error'] or state['step']}")
    return state["recorder"]


def _params(body: dict, request: Request) -> GenParams:
    meta = body.get("metadata") or {}
    stop = body.get("stop") or []
    tools = body.get("tools") or None
    if body.get("tool_choice") == "none":
        tools = None
    temperature = body.get("temperature")
    return GenParams(
        messages=body["messages"], tools=tools,
        temperature=0.7 if temperature is None else float(temperature),
        top_p=float(body.get("top_p") or 1.0),
        max_tokens=int(body.get("max_completion_tokens") or body.get("max_tokens") or 1024),
        stop=[stop] if isinstance(stop, str) else list(stop),
        seed=body.get("seed"),
        run_hint=request.headers.get("x-llminspect-run") or meta.get("llminspect_run"),
        label=request.headers.get("x-llminspect-label") or meta.get("llminspect_label"),
        template_kwargs={**settings.template_kwargs(), **(body.get("chat_template_kwargs") or {})})


def _usage(res) -> dict:
    return {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
            "total_tokens": res.prompt_tokens + res.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": res.cached_tokens}}


@app.get("/v1/models")
async def list_models():
    name = state["recorder"].s.name if state["recorder"] else settings.model
    return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "llminspect"}]}


async def _cancel_on_disconnect(request: Request, cancel: threading.Event) -> None:
    """Cancel a request's generation when its client goes away (a closed tab, an aborted fetch, a
    killed harness), so the model doesn't keep generating for no one. Once the body has been read,
    the next message from the server is the disconnect. (request.is_disconnected() can't be used: its
    zero-timeout peek never sees one through the HTTP middleware.) Run as a task; the caller cancels
    it once the reply is in."""
    while (await request.receive())["type"] != "http.disconnect":
        pass
    cancel.set()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    rec = recorder()
    body = await request.json()
    params = _params(body, request)
    model_name = body.get("model") or rec.s.name
    if body.get("stream"):
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        return StreamingResponse(_stream(rec, params, model_name, include_usage), media_type="text/event-stream")
    cancel = threading.Event()
    watch = asyncio.create_task(_cancel_on_disconnect(request, cancel))
    try:
        res = await asyncio.to_thread(rec.generate, params, cancel=cancel)
    except ModelSwitched as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Cancelled:  # stopped while it waited for the model: nothing was recorded
        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
            "model": model_name, "llminspect": {"run": None, "request": None},
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None},
                         "finish_reason": "cancelled", "logprobs": None}]})
    finally:
        watch.cancel()
    message = {"role": "assistant", "content": res.content}
    if res.reasoning:
        message["reasoning_content"] = res.reasoning
    if res.tool_calls:
        message["tool_calls"] = res.tool_calls
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
        "model": model_name, "usage": _usage(res), "llminspect": {"run": res.run_id, "request": res.req},
        "choices": [{"index": 0, "message": message, "finish_reason": res.finish_reason, "logprobs": None}],
    }, headers={"X-LLMInspect-Run": res.run_id})


async def _stream(rec: Recorder, params: GenParams, model_name: str, include_usage: bool):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def put(item):
        with contextlib.suppress(RuntimeError):  # the loop closed: no one is listening any more
            loop.call_soon_threadsafe(queue.put_nowait, item)

    def work():
        try:
            put(("done", rec.generate(params, on_text=lambda delta: put(("text", delta)),
                                      on_reasoning=lambda delta: put(("reasoning", delta)), cancel=cancel)))
        except Cancelled:
            put(("cancelled", None))
        except Exception as exc:
            traceback.print_exc()
            put(("error", exc))

    threading.Thread(target=work, daemon=True).start()
    cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

    def chunk(delta: dict, finish: str | None = None, **extra) -> str:
        payload = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_name,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra}
        return f"data: {json.dumps(payload)}\n\n"

    finished = False
    try:
        yield chunk({"role": "assistant", "content": ""})
        while True:
            kind, val = await queue.get()
            if kind in ("text", "reasoning"):
                yield chunk({"content": val} if kind == "text" else {"reasoning_content": val})
                continue
            finished = True
            if kind == "done":
                if val.tool_calls:
                    yield chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(val.tool_calls)]})
                yield chunk({}, val.finish_reason, llminspect={"run": val.run_id, "request": val.req})
                if include_usage:
                    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": model_name, "choices": [], "usage": _usage(val)}
                    yield f"data: {json.dumps(payload)}\n\n"
            elif kind == "cancelled":  # stopped while it waited for the model: nothing was recorded
                yield chunk({}, "cancelled", llminspect={"run": None, "request": None})
            else:
                yield f"data: {json.dumps({'error': {'message': str(val), 'type': type(val).__name__}})}\n\n"
            yield "data: [DONE]\n\n"
            return
    finally:
        # The client went away mid-reply: StreamingResponse notices the disconnect and cancels or
        # closes this generator. Stop generating; the request still finishes, and is recorded, in
        # its own thread.
        if not finished:
            cancel.set()


@app.get("/llminspect/status")
async def get_status():
    out = {"status": state["status"], "step": state["step"], "error": state["error"],
           "target": state["target"], "switch": state["switch"],
           "settings": {"model": settings.model, "data_dir": str(settings.data_dir),
                        "disk_budget_gb": settings.disk_budget_gb, "max_context": settings.max_context}}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        out["gpu"] = {"name": torch.cuda.get_device_name(0), "used": total - free, "total": total,
                      # this process alone: what PyTorch holds, and the share it may hold (LLMINSPECT_GPU_FRACTION)
                      "process": torch.cuda.memory_reserved(), "process_peak": torch.cuda.max_memory_reserved(),
                      "fraction": settings.gpu_fraction or None}
    if state["recorder"]:
        out.update(state["recorder"].status())
    return out


@app.post("/llminspect/cancel")
async def cancel_generation(request: Request):
    """Stop the request that is generating, between two tokens. It finishes like any other request,
    with finish_reason "cancelled": the tokens generated so far are recorded and returned to its
    client. Body {"run": id} (optional) stops only a request of that run, including one of its
    requests still waiting for the model, which then returns "cancelled" with nothing recorded."""
    body = await request.json() if await request.body() else {}
    run = body.get("run") if isinstance(body, dict) else None
    return recorder().cancel(str(run) if run else None)


@app.get("/llminspect/models")
async def get_models():
    """Models on this machine the recorder can switch to. Nothing is downloaded."""
    rec = state["recorder"]
    current = rec.s.spec if rec is not None else settings.model
    budget = _gpu_budget()
    models = await asyncio.to_thread(catalog.scan, _model_dirs(), current, budget)
    return {"current": current, "status": state["status"], "step": state["step"], "error": state["error"],
            "target": state["target"], "switch": state["switch"], "device": settings.device,
            "budget_gb": round(budget / catalog.GIB, 1) if budget is not None else None,
            "roots": [str(r) for r in catalog.search_roots(settings.model_dirs)], "models": models}


def _model_dirs() -> list:
    """Folders to search: LLMINSPECT_MODEL_DIRS, plus the startup model's own folder so it can always
    be switched back to, wherever it lives."""
    dirs = list(settings.model_dirs)
    if not STARTUP_MODEL.startswith("ollama:") and os.path.isdir(STARTUP_MODEL):
        dirs.append(STARTUP_MODEL)
    return dirs


@app.post("/llminspect/model")
async def switch_model(body: dict):
    """Unload the current model and load another one found by GET /llminspect/models."""
    spec = str(body.get("spec") or "")
    rec = state["recorder"]
    if rec is not None and rec.s.spec == spec and state["status"] == "ready":
        return {"ok": True, "unchanged": True, "model": rec.s.info()}
    entry = await asyncio.to_thread(catalog.find, spec, _model_dirs())
    if entry is None:
        raise HTTPException(404, f"{spec!r} is not a model found on this machine; LLMInspect only loads "
                                 "models already on disk (see GET /llminspect/models)")
    if entry.get("supported") is False:
        raise HTTPException(400, entry["notes"][0] if entry["notes"] else f"{entry['name']} can't be loaded")
    if not _switching.acquire(blocking=False):
        raise HTTPException(409, "a model is already loading")
    if rec is not None and rec.perf.get("busy"):
        _switching.release()
        raise HTTPException(409, "the model is generating; switch when the request finishes")
    budget = _gpu_budget()
    need = entry.get("split_need_bytes") or entry.get("need_bytes")  # a mixture of experts can load split
    if budget is not None and need and need > budget and not body.get("force"):
        _switching.release()
        raise HTTPException(400, f"{entry['name']} needs about {need / catalog.GIB:.1f} GB of GPU "
                                 f"memory and about {budget / catalog.GIB:.1f} GB would be free after unloading "
                                 "the current model. Close other GPU apps or choose a smaller model.")
    del rec  # the switch thread must hold the only reference
    state.update(status="loading", error=None, target=spec)
    _step(f"switching to {entry['name']}")
    threading.Thread(target=_switch, args=(spec,), daemon=True).start()
    return {"ok": True, "loading": spec, "name": entry["name"]}


@app.get("/llminspect/runs")
async def get_runs():
    return recorder().list_runs()


def _run_or_404(run_id: str):
    run = recorder().get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


@app.get("/llminspect/runs/{run_id}")
async def get_run(run_id: str):
    run = _run_or_404(run_id)
    return {"meta": run.meta, "messages": run.messages, "events": run.events}


@app.get("/llminspect/runs/{run_id}/tape")
async def get_tape(run_id: str):
    return [compact(r) for r in _run_or_404(run_id).tape]


@app.get("/llminspect/runs/{run_id}/tokens/{t}")
async def get_token(run_id: str, t: int):
    detail = recorder().token_detail(run_id, t)
    if detail is None:
        raise HTTPException(404, f"no token {t} in run {run_id}")
    return detail


@app.post("/llminspect/events")
async def post_event(request: Request):
    body = await request.json()
    try:
        return recorder().external_event(body.get("run"), body["type"], body.get("data") or {}, body.get("t"))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/llminspect/interventions")
async def get_interventions():
    return [iv.describe() for iv in recorder().interventions.values()]


@app.post("/llminspect/interventions")
async def post_intervention(request: Request):
    body = await request.json()
    try:
        iv = await asyncio.to_thread(recorder().add_intervention, **body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return iv.describe()


@app.patch("/llminspect/interventions/{iid}")
async def patch_intervention(iid: str, request: Request):
    body = await request.json()
    try:
        iv = await asyncio.to_thread(recorder().set_intervention, iid, bool(body.get("enabled", True)))
    except KeyError as exc:
        raise HTTPException(404, f"no intervention {iid}") from exc
    return iv.describe()


@app.delete("/llminspect/interventions/{iid}")
async def delete_intervention(iid: str):
    try:
        await asyncio.to_thread(recorder().remove_intervention, iid)
    except KeyError as exc:
        raise HTTPException(404, f"no intervention {iid}") from exc
    return {"deleted": iid}


@app.get("/llminspect/directions")
async def get_directions():
    return recorder().directions.describe()


@app.post("/llminspect/directions")
async def post_direction(request: Request):
    body = await request.json()
    rec = recorder()

    def build():
        with rec.lock:
            return rec.directions.build_contrast(body["name"], int(body["layer"]), body["positive"],
                                                 body["negative"], body.get("description", ""))
    try:
        await asyncio.to_thread(build)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return rec.directions.describe()


@app.post("/llminspect/directions/analyze")
async def analyze_direction(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(recorder().analyze_direction, body.get("positive") or [],
                                       body.get("negative") or [], body.get("position", "last"),
                                       bool(body.get("chat", True)))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/llminspect/directions/save")
async def save_direction(request: Request):
    body = await request.json()
    rec = recorder()

    def save():
        with rec.lock:
            return rec.directions.save_analysis(body["id"], int(body["layer"]), body["name"],
                                                body.get("description", ""))
    try:
        await asyncio.to_thread(save)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return rec.directions.describe()


@app.delete("/llminspect/directions/{name}")
async def delete_direction(name: str):
    rec = recorder()
    if any(e.direction == name for e in rec.edits.edits.values()) or \
            any(iv.direction == name for iv in rec.interventions.values()):
        raise HTTPException(409, f"{name} is used by an edit or intervention; remove those first")
    try:
        with rec.lock:
            rec.directions.delete(name)
    except KeyError as exc:
        raise HTTPException(404, f"no direction {name}") from exc
    return rec.directions.describe()


@app.get("/llminspect/workbench")
async def get_workbench():
    return recorder().edits.summary()


@app.post("/llminspect/workbench/edits")
async def post_weight_edit(request: Request):
    body = await request.json()
    try:
        edit = await asyncio.to_thread(recorder().add_weight_edit, **body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return edit.describe()


@app.patch("/llminspect/workbench/edits/{eid}")
async def patch_weight_edit(eid: str, request: Request):
    body = await request.json()
    try:
        return recorder().edits.set_enabled(eid, bool(body.get("enabled", True))).describe()
    except KeyError as exc:
        raise HTTPException(404, f"no edit {eid}") from exc


@app.delete("/llminspect/workbench/edits/{eid}")
async def delete_weight_edit(eid: str):
    rec = recorder()
    if eid in rec.edits.applied:
        raise HTTPException(409, "this edit is applied; restore the original weights first")
    try:
        rec.edits.remove(eid)
    except KeyError as exc:
        raise HTTPException(404, f"no edit {eid}") from exc
    return rec.edits.summary()


@app.post("/llminspect/workbench/apply")
async def apply_weight_edits():
    try:
        return await asyncio.to_thread(recorder().apply_weight_edits)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/llminspect/workbench/revert")
async def revert_weight_edits():
    return await asyncio.to_thread(recorder().revert_weight_edits)


def _checkpoint_call(fn, *args):
    """A candidate checkpoint request, its failures mapped to HTTP codes."""
    try:
        return fn(*args)
    except Busy as exc:
        raise HTTPException(409, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/llminspect/workbench/checkpoint")
async def checkpoint_status():
    """The candidate checkpoint job: {status: none|building|ready|error, step, progress, error, report, edit}.
    The report is the latest one (left out while a job runs); the edit is the stack's checkpoint edit."""
    return recorder().checkpoints.status()


@app.post("/llminspect/workbench/checkpoint")
async def checkpoint_load(request: Request):
    """Load a candidate checkpoint of this model's architecture: {spec, note}, where spec is a checkpoint
    folder, a .gguf file, ollama:<tag> or a Hugging Face repo id that is already in the local cache (nothing
    is downloaded; pickle-based weights are refused). Compares its weights with the base in the background,
    then adds a checkpoint edit to the stack; poll GET /llminspect/workbench/checkpoint. Apply and revert it
    like any edit (POST /llminspect/workbench/apply, /revert)."""
    body = await request.json()
    rec = recorder()
    if not isinstance(body, dict):
        raise HTTPException(400, "send a JSON object")
    return await asyncio.to_thread(_checkpoint_call, rec.checkpoints.start, str(body.get("spec") or ""),
                                   str(body.get("note") or ""))


@app.get("/llminspect/workbench/checkpoint/reports")
async def checkpoint_reports():
    """Saved weight-difference reports for this model, newest first."""
    return recorder().checkpoints.reports()


@app.get("/llminspect/scan")
async def scan_status():
    """The deploy scan job: {status: none|running|ready|error, step, progress, error, report: {id, signals}}."""
    return recorder().scan.status()


@app.post("/llminspect/scan")
async def scan_start(request: Request):
    """Scan a candidate checkpoint against the loaded model (recorder/scan.py): {spec, canaries: [text],
    manifest: path, text or {file: sha256}, sections}. Runs in the background; poll GET /llminspect/scan.
    The report is evidence with its limits, never a verdict."""
    body = await _body(request)
    return await asyncio.to_thread(_checkpoint_call, recorder().scan.start, body)


@app.get("/llminspect/scan/reports")
async def scan_reports():
    """Saved deploy-scan reports for this model, newest first."""
    return recorder().scan.reports()


@app.get("/llminspect/scan/reports/{rid}")
async def scan_report(rid: str):
    try:
        return recorder().scan.report(rid)
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'\"")) from exc


@app.get("/llminspect/scan/reports/{rid}/html")
async def scan_report_html(rid: str):
    """The report as a printable page."""
    try:
        return HTMLResponse(recorder().scan.html(rid))
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'\"")) from exc


@app.post("/llminspect/workbench/compare")
async def compare_weight_edits(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.compare, recorder(), body.get("prompts") or [],
                                       int(body.get("max_tokens") or 96), body.get("reference") or None)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/llminspect/workbench/replay")
async def replay_chat(request: Request):
    """Ask a recorded chat's last question again on the original and the edited weights."""
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.replay, recorder(), str(body.get("run") or ""),
                                       int(body.get("max_tokens") or 256), body.get("template_kwargs") or None)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/llminspect/checks")
async def get_checks():
    return workbench.list_checks(recorder())


@app.post("/llminspect/checks")
async def post_check(request: Request):
    """Save a behavior check: {run, t, target} from a traced word, or {prompt, expect, chat}."""
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.add_check, recorder(), body)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/llminspect/checks/{cid}")
async def delete_check(cid: str):
    try:
        return workbench.remove_check(recorder(), cid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _neuron_call(fn, *args):
    """A neuron read off the event loop, its failures mapped to HTTP codes."""
    try:
        return fn(*args)
    except (Busy, knobs.Clash) as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc).strip("'\"")) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/llminspect/neurons")
async def neurons_summary():
    """Layer widths, the median residual norms dose scaling uses, and the neurons being steered."""
    return recorder().neurons.summary()


@app.get("/llminspect/neurons/index")
async def neuron_index_status():
    """Whether this model has a neuron index, and how it was built (or how far a build has got)."""
    return recorder().neuron_index.status()


@app.post("/llminspect/neurons/index")
async def neuron_index_build(request: Request):
    """Build the neuron index over the corpus (recorder/neuron_index.py): {max_tokens}. Runs in the
    background and holds the model while it reads; poll GET /llminspect/neurons/index."""
    body = await request.json()
    rec = recorder()
    try:
        max_tokens = int(body.get("max_tokens", neuron_index.DEFAULT_TOKENS))
    except (TypeError, ValueError):
        raise HTTPException(400, "max_tokens must be a number") from None
    return _neuron_call(rec.neuron_index.build_async, max_tokens)


@app.get("/llminspect/neurons/search")
async def neuron_search(q: str = "", limit: int = 20):
    """Neurons for a text query: fire on it, push it, or have it in their top contexts."""
    rec = recorder()
    return await asyncio.to_thread(_neuron_call, rec.neuron_index.search, q, max(1, min(limit, 100)))


@app.get("/llminspect/neurons/atlas")
async def neuron_atlas():
    """Every dense layer's width and neuron clusters (recorder/atlas.py), for drawing all neurons at once."""
    return await asyncio.to_thread(_neuron_call, recorder().atlas.layout)


@app.get("/llminspect/neurons/atlas/{name}")
async def neuron_atlas_map(name: str):
    """One value per neuron, as base64 bytes per layer: write, spread, mean or changed."""
    return await asyncio.to_thread(_neuron_call, recorder().atlas.map, name)


@app.post("/llminspect/neurons/fires")
async def neuron_fires(request: Request):
    """Read a text once: each neuron's peak activation on it as a z-score, and the neurons that fire most. {text}"""
    body = await _body(request)
    return await asyncio.to_thread(_neuron_call, recorder().atlas.fires, str(body.get("text") or ""))


@app.get("/llminspect/neurons/{name}/links")
async def neuron_links(name: str):
    """A neuron's direct weight-path links: later neurons that read its write, earlier neurons whose write it reads."""
    try:
        layer, n = neurons.parse(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await asyncio.to_thread(_neuron_call, recorder().atlas.links, layer, n)


@app.get("/llminspect/neurons/{name}")
async def neuron_card(name: str, dose: float = 1.0):
    """One neuron's card (recorder/neurons.py): logit lens, steering effect at ±dose, activity, wiring."""
    rec = recorder()
    try:
        layer, n = neurons.parse(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return await asyncio.to_thread(_neuron_call, rec.neurons.card, layer, n, dose)


@app.post("/llminspect/neurons/steer")
async def neuron_steer(request: Request):
    """Steer a neuron on every request from now: {neuron: "L5 N120", dose, phase}. Remove it like any
    intervention (DELETE /llminspect/interventions/{id})."""
    body = await request.json()
    rec = recorder()
    iv = await asyncio.to_thread(_neuron_call, rec.steer_neuron, str(body.get("neuron") or ""),
                                 float(body.get("dose", 1.0)), str(body.get("phase") or "all"))
    return iv.describe()


@app.post("/llminspect/neurons/sweep")
async def neuron_sweep(request: Request):
    """A prompt answered at several dose levels: {prompt, steers: [{neuron, dose}], factors, max_tokens}."""
    body = await request.json()
    rec = recorder()
    try:
        steers = [(*neurons.parse(s["neuron"]), float(s.get("dose", 1.0))) for s in body.get("steers") or []]
        factors = [float(f) for f in body.get("factors") or [0, 0.5, 1, 2]][:8]
        max_tokens = max(1, min(int(body.get("max_tokens", 60)), 256))
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"steers are [{{neuron, dose}}] and factors are numbers ({exc})") from exc
    return await asyncio.to_thread(_neuron_call, rec.neurons.sweep, str(body.get("prompt") or ""), steers,
                                   factors, max_tokens)


def _knob_view(rec: Recorder, knob: dict) -> dict:
    return knob | {"active": [iv.id for iv in rec.interventions.values() if iv.knob == knob["name"]]}


async def _body(request: Request) -> dict:
    try:
        body = await request.json() if await request.body() else {}
    except ValueError:
        body = None
    if not isinstance(body, dict):
        raise HTTPException(400, "send a JSON object")
    return body


@app.get("/llminspect/knobs")
async def get_knobs():
    """The saved knobs (recorder/knobs.py): named sets of neurons with doses. "active" lists the
    interventions steering each one now. "error" says why saving is refused, if it is."""
    rec = recorder()
    return {"knobs": [_knob_view(rec, k) for k in rec.knobs.list()], "error": rec.knobs.error}


@app.post("/llminspect/knobs")
async def post_knob(request: Request):
    """Save a knob, replacing the one of that name: {name, description, neurons: [{neuron, dose}]} or
    {name, description, text: "L5 N120 +1, L9 N64 -0.5"}."""
    body = await _body(request)
    rec = recorder()

    def save():
        steers = body["neurons"] if body.get("neurons") is not None else neurons.parse_set(body.get("text"))
        return rec.knobs.save(str(body.get("name") or ""), steers, body.get("description") or "")
    return _knob_view(rec, await asyncio.to_thread(_neuron_call, save))


@app.delete("/llminspect/knobs/{name}")
async def delete_knob(name: str):
    rec = recorder()
    if any(iv.knob == name for iv in rec.interventions.values()):
        raise HTTPException(409, f"{name} is steering now; stop it first")
    _neuron_call(rec.knobs.delete, name)
    return {"deleted": name}


@app.post("/llminspect/knobs/{name}/steer")
async def knob_steer(name: str, request: Request):
    """Steer a knob on every request from now: {factor, phase}. Each neuron at its dose × factor;
    steering a knob again replaces its steers. Stop it with DELETE on this path."""
    body = await _body(request)
    return await asyncio.to_thread(_neuron_call, recorder().steer_knob, name, body.get("factor", 1.0),
                                   str(body.get("phase") or "all"))


@app.delete("/llminspect/knobs/{name}/steer")
async def knob_unsteer(name: str):
    rec = recorder()
    _neuron_call(rec.knobs.get, name)
    return {"knob": name, "removed": await asyncio.to_thread(rec.unsteer_knob, name)}


@app.post("/llminspect/knobs/{name}/ablate")
async def knob_ablate(name: str, request: Request):
    """Add a weight edit that scales a knob's neurons' output weights by `factor` (default 0, which
    zeroes them): {factor, note}. It is added, not applied: apply it with POST /llminspect/workbench/apply."""
    body = await _body(request)
    rec = recorder()

    def add():
        knob = rec.knobs.get(name)
        factor = float(body.get("factor", 0.0))
        if not 0 <= factor <= 1:
            raise ValueError("factor must be between 0 and 1")
        names = [e["neuron"] for e in knob["neurons"]]
        listed = ", ".join(names[:4]) + (f" +{len(names) - 4} more" if len(names) > 4 else "")
        return rec.add_weight_edit(kind="units", units=rec.knobs.units(name), factor=factor,
                                   note=str(body.get("note") or "")[:200] or listed).describe()
    return await asyncio.to_thread(_neuron_call, add)


@app.post("/llminspect/knobs/{name}/monitor")
async def knob_monitor(name: str, request: Request):
    """Watch each of a knob's neurons: {threshold, sign, min_tokens, phase}. One rule per neuron, labelled
    "knob · neuron", on its activation z-scored per token. All are added or none."""
    body = await _body(request)
    rec = recorder()

    def add():
        knob = rec.knobs.get(name)
        rule = {"threshold": 4.0, "sign": "above", "min_tokens": 2, "phase": "gen"}
        rule |= {k: body[k] for k in ("threshold", "sign", "min_tokens", "phase") if k in body}
        rules = [rule | {"direction": e["neuron"], "label": f"{name[:40]} · {e['neuron']}"} for e in knob["neurons"]]
        return [m.describe() for m in rec.monitors.add_many(rules, set(rec.directions.dirs), rec.neurons.check)]
    return await asyncio.to_thread(_neuron_call, add)


@app.post("/llminspect/knobs/{name}/sweep")
async def knob_sweep(name: str, request: Request):
    """A prompt answered with a knob's doses scaled by each factor: {prompt, factors, max_tokens}."""
    body = await _body(request)
    rec = recorder()

    def sweep():
        steers = rec.knobs.steers(name)
        factors = [float(f) for f in body.get("factors") or [0, 0.5, 1, 2]][:8]
        max_tokens = max(1, min(int(body.get("max_tokens", 60)), 256))
        return rec.neurons.sweep(str(body.get("prompt") or ""), steers, factors, max_tokens)
    return await asyncio.to_thread(_neuron_call, sweep)


@app.get("/llminspect/monitors")
async def get_monitors():
    return recorder().monitors.list()


@app.post("/llminspect/monitors")
async def post_monitor(request: Request):
    """Watch a direction, or a neuron named like "L5 N120": {direction, threshold, sign
    (above|below|either), min_tokens, phase (gen|all), label}. Runs recorded from now on raise a
    "monitor" event when the rule is met."""
    body = await request.json()
    rec = recorder()
    try:
        monitor = await asyncio.to_thread(rec.monitors.add, body, set(rec.directions.dirs), rec.neurons.check)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return monitor.describe()


@app.patch("/llminspect/monitors/{mid}")
async def patch_monitor(mid: str, request: Request):
    body = await request.json()
    try:
        return (await asyncio.to_thread(recorder().monitors.update, mid, body)).describe()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/llminspect/monitors/{mid}")
async def delete_monitor(mid: str):
    try:
        return await asyncio.to_thread(recorder().monitors.remove, mid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/llminspect/runs/{run_id}/flags")
async def run_flags(run_id: str):
    """Every monitor's flags on a recorded run, with the same rule on the random control."""
    rec = recorder()
    run = rec.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return await asyncio.to_thread(monitors.scan_run, rec.monitors, run.tape, rec.directions.describe())


@app.get("/llminspect/flags")
async def flags_across_runs(limit: int = 20):
    """Flag counts per enabled monitor on the most recent runs."""
    return await asyncio.to_thread(monitors.scan_runs, recorder(), limit)


@app.post("/llminspect/checks/run")
async def run_checks():
    """Every saved check now and, with edits applied, on the original weights."""
    return await asyncio.to_thread(workbench.run_checks, recorder())


@app.post("/llminspect/workbench/export")
async def export_edited_model(request: Request):
    body = await request.json()
    try:
        return await asyncio.to_thread(workbench.export, recorder(), body["path"])
    except (KeyError, ValueError, OSError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _anatomy(rec: Recorder):
    if getattr(rec, "anatomy", None) is None:
        raise HTTPException(404, "weight anatomy is off (LLMINSPECT_ANATOMY=0)")
    return rec.anatomy


@app.get("/llminspect/anatomy")
async def get_anatomy():
    rec = recorder()
    an = _anatomy(rec)
    return an.describe() | {"edited": an.edited_units(rec.edits), "edits_version": rec.edits.version}


@app.get("/llminspect/anatomy/edited")
async def get_edited_units():
    rec = recorder()
    return {"edited": _anatomy(rec).edited_units(rec.edits), "edits_version": rec.edits.version}


@app.get("/llminspect/anatomy/edges")
async def get_edges():
    return _anatomy(recorder()).describe_edges()


@app.get("/llminspect/anatomy/params")
async def get_params():
    return _anatomy(recorder()).param_list()


@app.get("/llminspect/anatomy/units/{g}")
async def get_unit(g: int):
    rec = recorder()
    an = _anatomy(rec)
    try:
        return await asyncio.to_thread(an.unit_detail, g, set(an.edited_units(rec.edits)))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/llminspect/anatomy/units/{g}/lesion")
async def lesion_unit(g: int, request: Request):
    """Add a workbench edit that scales this unit's output weights (factor 0 by default). It changes
    the weights only when the stack is applied."""
    body = await request.json()
    rec = recorder()
    try:
        spec = _anatomy(rec).lesion(g, float(body.get("factor", 0.0)))
        edit = await asyncio.to_thread(rec.add_weight_edit, **spec)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return edit.describe()


@app.post("/llminspect/anatomy/lesion")
async def lesion_units(request: Request):
    """One workbench edit that removes a set of units together (a knockout made permanent). With
    apply true the stack is applied right away, so the weights change now."""
    body = await request.json()
    units = body.get("units")
    if not isinstance(units, list) or not units:
        raise HTTPException(400, "units must be a non-empty list of unit ids")
    rec = recorder()
    try:
        spec = _anatomy(rec).lesion_units(units, float(body.get("factor", 0.0)), str(body.get("note") or ""))
        edit = await asyncio.to_thread(rec.add_weight_edit, **spec)
        summary = await asyncio.to_thread(rec.apply_weight_edits) if body.get("apply") else rec.edits.summary()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"edit": edit.describe(), "workbench": summary}


@app.post("/llminspect/anatomy/tile")
async def post_tile(request: Request):
    body = await request.json()
    rec = recorder()
    try:
        return await asyncio.to_thread(_anatomy(rec).tile, body, rec.edits.version)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/llminspect/runs/{run_id}/anatomy")
async def get_run_anatomy(run_id: str):
    """Per-token unit activity for a run: a 4-byte little-endian header length, a JSON header
    (tape indices, unit count, unit-map key), then one byte per unit per token, row-major."""
    rec = recorder()
    an = _anatomy(rec)
    _run_or_404(run_id)  # "unknown run" should 404, not return an empty 200 like "run with no activity"
    t, codes, skipped = await asyncio.to_thread(an.read_run, rec.store, run_id)
    header = json.dumps({"n": int(len(t)), "units": an.n_units, "key": an.key, "skipped": skipped,
                         "t": t.tolist()}).encode()
    return Response(struct.pack("<I", len(header)) + header + codes.tobytes(), media_type="application/octet-stream")


async def _trace(fn, *args):
    """Run a tracer read off the event loop, mapping its failures to HTTP codes."""
    rec = recorder()
    if getattr(rec, "tracer", None) is None:
        raise HTTPException(404, "tracing needs weight anatomy (LLMINSPECT_ANATOMY=1)")
    try:
        return await asyncio.to_thread(getattr(rec.tracer, fn), *args)
    except Busy as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/llminspect/runs/{run_id}/tokens/{t}/explain")
async def explain_token(run_id: str, t: int, target: str | None = None, weights: str = "current"):
    """Per-unit attribution of the next token's logit at tape slot t (see recorder/tracing.py).
    weights=original reads it with the applied weight edits lifted, for a before/after comparison."""
    return await _trace("explain", run_id, t, target, weights)


@app.get("/llminspect/runs/{run_id}/tokens/{t}/neurons")
async def token_neurons(run_id: str, t: int, unit: int, target: str | None = None, weights: str = "current"):
    """Each neuron of an MLP cluster at tape slot t: activation and share of the attribution."""
    return await _trace("neurons", run_id, t, unit, target, weights)


@app.get("/llminspect/runs/{run_id}/tokens/{t}/producer")
async def token_producer(run_id: str, t: int):
    """The slot (and target) to explain to see why token t was written."""
    return await _trace("producer", run_id, t)


@app.post("/llminspect/runs/{run_id}/tokens/{t}/knockout")
async def token_knockout(run_id: str, t: int, body: dict):
    """Re-read the context with the given units removed and compare the next token's logit with the
    intact model: their total causal effect, next to their direct effect (see recorder/tracing.py)."""
    units = body.get("units")
    if not isinstance(units, list):
        raise HTTPException(400, "units must be a list of unit ids")
    try:
        factor = float(body.get("factor", 0.0))  # 0 removes them; 0.5 dampens, 2 boosts (a scale test)
    except (TypeError, ValueError):
        raise HTTPException(400, "factor must be a number") from None
    return await _trace("knockout", run_id, t, units, body.get("target"), body.get("weights") or "current", factor)


@app.get("/llminspect/runs/{run_id}/tokens/{t}/attention")
async def token_attention(run_id: str, t: int, unit: int, weights: str = "current"):
    """A softmax-attention head's attention over the earlier tokens at tape slot t."""
    return await _trace("attention", run_id, t, unit, weights)


FORGET_STEPS = {"target": actions.forget_target, "try": actions.forget_try, "measure": actions.forget_measure}


@app.post("/llminspect/actions/forget/{step}")
async def forget_step(step: str, request: Request):
    """Make the model forget a fact (see recorder/actions.py): target ranks the units behind the
    answer, try shows answers with and without them, measure checks the effect against random units,
    facts to keep and neutral text. Keeping it is a grouped lesion (/llminspect/anatomy/lesion)."""
    fn = FORGET_STEPS.get(step)
    if fn is None:
        raise HTTPException(404, f"no step {step}; use one of {sorted(FORGET_STEPS)}")
    body = await request.json()
    rec = recorder()
    if getattr(rec, "tracer", None) is None:
        raise HTTPException(404, "actions need weight anatomy (LLMINSPECT_ANATOMY=1)")
    try:
        return await asyncio.to_thread(fn, rec, body)
    except Busy as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.websocket("/llminspect/live")
async def live(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        hub.clients.discard(ws)


@app.get("/")
async def index():
    return FileResponse(UI_DIR / "index.html")


# The UI's types, fixed: on Windows Python reads them from the registry, where .js is often text/plain,
# and browsers refuse a module script served as that.
for _type, _ext in (("text/javascript", ".js"), ("text/css", ".css"), ("text/html", ".html"), ("image/svg+xml", ".svg")):
    mimetypes.add_type(_type, _ext)
app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
