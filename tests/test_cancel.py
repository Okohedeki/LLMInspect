"""Stopping a generation: Recorder.cancel, POST /brainscan/cancel and client disconnects.

A cancelled request must finish like any other: the tokens generated so far are on the tape, in the
stored token file, in the activity npz and in the transcript, the KV cache holds exactly what was
fed, the run's meta is written, the "done" message goes out, the lock is free, and the run takes its
next request normally. Runs a real CPU recorder on the tiny saved Llama.
"""

import asyncio
import json
import socket
import threading
import time
from pathlib import Path

import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder.chat import render
from brainscan.recorder.model import Subject
from brainscan.recorder.session import Cancelled, GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")

LONG = 3000  # more tokens than any test lets the model generate before cancelling


class Feed:
    """The recorder's live messages, with a way to wait for one."""

    def __init__(self):
        self.msgs, self.cond = [], threading.Condition()

    def __call__(self, msg):
        with self.cond:
            self.msgs.append(msg)
            self.cond.notify_all()

    def gen_tokens(self, run):
        with self.cond:
            toks = [m for m in self.msgs if m["k"] == "toks" and m["run"] == run]
            return sum(r["ph"] == "gen" for m in toks for r in m["items"])

    def wait_for_gen(self, run, n, timeout=60):
        with self.cond:
            assert self.cond.wait_for(lambda: self.gen_tokens(run) >= n, timeout), f"{run} never generated {n} tokens"

    def done(self, run):
        return [m for m in self.msgs if m["k"] == "done" and m["run"] == run]


@pytest.fixture(scope="module")
def rec_feed(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("cancel")
    subject = Subject(str(MODEL_DIR), device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    feed = Feed()
    rec = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, feed)
    return rec, feed


def start(rec, params):
    """Generate in a thread; returns the thread and a dict that gets the result or the exception."""
    out = {}

    def work():
        try:
            out["res"] = rec.generate(params)
        except Exception as exc:
            out["exc"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    return th, out


def assert_idle(rec):
    assert rec.lock.acquire(blocking=False), "the recorder lock is still held"
    rec.lock.release()
    assert rec.active is None and rec.perf["busy"] is False and not rec._queued
    assert rec.cancel()["cancelled"] is False


def assert_consistent(rec, run_id):
    """Memory, disk and the KV cache agree on what the run read and generated."""
    run = rec.get_run(run_id)
    stored = rec.store.read_tokens(run_id)
    assert [(r["t"], r["pos"], r["tok"]) for r in stored] == [(r["t"], r["pos"], r["tok"]) for r in run.tape]
    assert rec.store.read_jsonl(run_id, "messages") == run.messages
    assert rec.store.read_meta(run_id)["tokens"] == len(run.tape) == run.meta["tokens"]
    t, codes, skipped = rec.anatomy.read_run(rec.store, run_id)  # one activity row per token read
    assert skipped == 0 and t.tolist() == [r["t"] for r in run.tape] and codes.shape[0] == len(run.tape)
    assert len(rec._kv_ids) == rec._kv.get_seq_length()
    return run


def test_cancel_mid_decode_keeps_the_tokens_generated(rec_feed):
    rec, feed = rec_feed
    msgs = [{"role": "user", "content": "Count from one to a thousand."}]
    th, out = start(rec, GenParams(messages=msgs, max_tokens=LONG, temperature=0, run_hint="cancel-decode"))
    feed.wait_for_gen("cancel-decode", 5)
    got = rec.cancel()
    assert got == {"cancelled": True, "run": "cancel-decode", "req": 1, "queued": 0}
    th.join(60)
    assert not th.is_alive() and "exc" not in out, out.get("exc")
    res = out["res"]
    assert res.finish_reason == "cancelled" and res.tool_calls == []
    assert 5 <= res.completion_tokens < LONG

    run = assert_consistent(rec, "cancel-decode")
    ids = render(rec.s.tokenizer, msgs, None).ids
    gen = [r for r in run.tape if r["ph"] == "gen"]
    assert len(gen) == res.completion_tokens
    assert [r["pos"] for r in run.tape] == list(range(len(ids) + len(gen)))  # nothing missing or doubled
    assert rec.s.tokenizer.decode([r["tok"] for r in gen], skip_special_tokens=False) == res.text
    assert rec._kv_ids == ids + [r["tok"] for r in gen]  # the cache holds exactly what was fed

    reply = run.messages[-1]
    assert reply["generated"] and reply["finish"] == "cancelled" and reply["content"] == res.content
    stops = [e for e in run.events if e["type"] == "cancelled"]
    assert len(stops) == 1 and stops[0]["t"] == gen[-1]["t"]
    assert stops[0]["data"] == {"tokens": res.completion_tokens, "while_reasoning": False, "in_prompt": False}
    assert not any(e["type"] in ("cutoff", "final_answer") for e in run.events)
    done = feed.done("cancel-decode")
    assert len(done) == 1 and done[0]["finish"] == "cancelled" and done[0]["generated"] == res.completion_tokens
    assert_idle(rec)

    # The run goes on: the next request continues it from the kept reply, and isn't cancelled.
    follow = msgs + [{"role": "assistant", "content": res.content or ""}, {"role": "user", "content": "Go on."}]
    res2 = rec.generate(GenParams(messages=follow, max_tokens=4, temperature=0, run_hint="cancel-decode"))
    assert res2.run_id == "cancel-decode" and res2.req == 2
    assert res2.finish_reason in ("length", "stop") and res2.completion_tokens >= 1
    run = assert_consistent(rec, "cancel-decode")
    assert sum(r["req"] == 2 and r["ph"] == "gen" for r in run.tape) == res2.completion_tokens
    assert [m["content"] for m in run.messages if m["req"] == 2 and not m.get("generated")] == ["Go on."]
    assert_idle(rec)


def test_cancel_between_prefill_chunks_keeps_what_was_read(rec_feed, monkeypatch):
    rec, feed = rec_feed
    monkeypatch.setattr(rec.s, "prefill_chunk", 8)
    read = rec._prefill_records

    def cancel_after_the_first_chunk(*args, **kwargs):
        rec.cancel()  # the next chunk is never read
        return read(*args, **kwargs)

    monkeypatch.setattr(rec, "_prefill_records", cancel_after_the_first_chunk)
    msgs = [{"role": "user", "content": "A long question that takes several prefill chunks to read. " * 4}]
    ids = render(rec.s.tokenizer, msgs, None).ids
    assert len(ids) > 24
    res = rec.generate(GenParams(messages=msgs, max_tokens=LONG, temperature=0, run_hint="cancel-prefill"))
    assert res.finish_reason == "cancelled" and res.completion_tokens == 0 and res.content is None

    run = assert_consistent(rec, "cancel-prefill")
    assert [r["pos"] for r in run.tape] == list(range(8)) and rec._kv_ids == ids[:8]
    user, reply = run.messages
    assert user["role"] == "user" and reply["generated"] and reply["finish"] == "cancelled"
    stop = next(e for e in run.events if e["type"] == "cancelled")
    assert stop["data"]["in_prompt"] is True and stop["data"]["tokens"] == 0
    assert feed.done("cancel-prefill")[0]["finish"] == "cancelled"
    assert_idle(rec)

    # The next request records the rest of the prompt once, from where reading stopped.
    monkeypatch.setattr(rec, "_prefill_records", read)
    follow = msgs + [{"role": "assistant", "content": ""}, {"role": "user", "content": "Well?"}]
    res2 = rec.generate(GenParams(messages=follow, max_tokens=3, temperature=0, run_hint="cancel-prefill"))
    assert res2.finish_reason in ("length", "stop")
    run = assert_consistent(rec, "cancel-prefill")
    prompt = render(rec.s.tokenizer, follow, None).ids
    assert [r["pos"] for r in run.tape if r["ph"] != "gen"] == list(range(len(prompt)))
    assert_idle(rec)


def test_cancel_while_waiting_records_nothing(rec_feed):
    rec, _ = rec_feed
    rec.lock.acquire()  # another request holds the model
    try:
        th, out = start(rec, GenParams(messages=[{"role": "user", "content": "hi"}], max_tokens=4,
                                       run_hint="cancel-queued"))
        for _ in range(200):
            if rec._queued:
                break
            threading.Event().wait(0.01)
        assert rec.cancel("some-other-run") == {"cancelled": False, "run": None, "req": None, "queued": 0}
        assert rec.cancel("cancel-queued")["queued"] == 1
        th.join(5)  # it stops waiting at once, while the model is still busy
        assert not th.is_alive() and isinstance(out.get("exc"), Cancelled)
    finally:
        rec.lock.release()
    assert rec.get_run("cancel-queued") is None
    assert_idle(rec)


# ---- the server: POST /brainscan/cancel and client disconnects ------------------------------

@pytest.fixture()
def client(rec_feed):
    saved = dict(server.state)
    server.state.update(recorder=rec_feed[0], status="ready", step="ready", error=None)
    try:
        yield TestClient(server.app)
    finally:
        server.state.clear()
        server.state.update(saved)


def test_cancel_endpoint_stops_a_chat_completion(client, rec_feed):
    rec, feed = rec_feed
    assert client.post("/brainscan/cancel").json()["cancelled"] is False  # nothing is generating
    out = {}
    body = {"messages": [{"role": "user", "content": "Write a long story."}], "max_tokens": LONG, "temperature": 0}

    def post():
        out["r"] = client.post("/v1/chat/completions", json=body, headers={"X-Brainscan-Run": "srv-cancel"})

    th = threading.Thread(target=post, daemon=True)
    th.start()
    feed.wait_for_gen("srv-cancel", 3)
    st = client.get("/brainscan/status").json()
    assert st["perf"]["busy"] is True and st["generating"]["run"] == "srv-cancel"
    assert client.post("/brainscan/cancel", json={"run": "another-run"}).json()["cancelled"] is False
    got = client.post("/brainscan/cancel", json={"run": "srv-cancel"}).json()
    assert got["cancelled"] is True and got["run"] == "srv-cancel" and got["req"] == 1
    th.join(60)
    r = out["r"]
    assert r.status_code == 200
    d = r.json()
    assert d["choices"][0]["finish_reason"] == "cancelled" and d["brainscan"]["run"] == "srv-cancel"
    run = assert_consistent(rec, "srv-cancel")
    assert d["usage"]["completion_tokens"] == sum(x["ph"] == "gen" for x in run.tape) >= 3
    st = client.get("/brainscan/status").json()
    assert st["perf"]["busy"] is False and st["generating"] is None
    assert_idle(rec)


def drive(rec, coro):
    """Run a coroutine on its own loop, then wait for the generating thread to let go of the model."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        assert rec.lock.acquire(timeout=60), "the generation never finished"
        rec.lock.release()
        loop.close()


def test_stream_closed_by_the_client_cancels(client, rec_feed):
    rec, feed = rec_feed
    params = GenParams(messages=[{"role": "user", "content": "Tell me everything."}], max_tokens=LONG,
                       temperature=0, run_hint="srv-closed")

    async def read_then_close():
        stream = server._stream(rec, params, "tiny", False)
        first = await stream.__anext__()  # starts the generation
        await asyncio.to_thread(feed.wait_for_gen, "srv-closed", 5)
        await stream.aclose()  # what the response does to it when the client goes away mid-reply
        return first

    assert json.loads(drive(rec, read_then_close())[len("data: "):])["choices"][0]["delta"]["role"] == "assistant"
    run = assert_consistent(rec, "srv-closed")
    assert run.messages[-1]["finish"] == "cancelled"
    assert 5 <= sum(r["ph"] == "gen" for r in run.tape) < LONG
    assert_idle(rec)


@pytest.fixture()
def live(client):
    """The app on a real uvicorn and a free port (lifespan off, so no model loads), for a client that
    hangs up mid-reply the way a closed tab or a killed harness does."""
    srv = uvicorn.Server(uvicorn.Config(server.app, host="127.0.0.1", port=0, lifespan="off", log_level="warning"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "uvicorn did not start"
    try:
        yield srv.servers[0].sockets[0].getsockname()[1]
    finally:
        srv.should_exit = True
        th.join(10)


@pytest.mark.parametrize("stream", [False, True])
def test_a_client_that_hangs_up_stops_its_request(live, rec_feed, stream):
    rec, feed = rec_feed
    run = f"hang-up-{'stream' if stream else 'plain'}"
    body = json.dumps({"messages": [{"role": "user", "content": "Keep talking."}], "max_tokens": LONG,
                       "temperature": 0, "stream": stream}).encode()
    with socket.create_connection(("127.0.0.1", live)) as sock:
        sock.sendall(f"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
                     f"X-Brainscan-Run: {run}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        feed.wait_for_gen(run, 3)
    with feed.cond:  # hung up: the request stops at its next token
        assert feed.cond.wait_for(lambda: feed.done(run), 60), "the request did not stop"
    assert feed.done(run)[0]["finish"] == "cancelled"
    assert rec.lock.acquire(timeout=10)
    rec.lock.release()
    run_state = assert_consistent(rec, run)
    assert run_state.messages[-1]["finish"] == "cancelled"
    assert 3 <= sum(r["ph"] == "gen" for r in run_state.tape) < LONG
    assert_idle(rec)
