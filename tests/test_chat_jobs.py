"""Persistent chat jobs and the stream hand-off behind them.

Covers the cases that only show up with a real phone: Stop or a lost
connection while the model loads, a stopped job capturing the next message,
tool rounds glued into the saved answer, a generation error reported as a
clean finish, and a generation thread left running after its reader went away.
"""

import asyncio
import json
import queue
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import chat_jobs
from api import server


def _sse(payload):
    return "data: " + json.dumps(payload) + "\n\n"


def _content(text):
    return _sse({"choices": [{"delta": {"content": text}}]})


@pytest.fixture
def jobs():
    chat_jobs._jobs.clear()
    yield chat_jobs._jobs
    for j in chat_jobs._jobs.values():
        j._stop.set()
    chat_jobs._jobs.clear()


def _app(tmp_path, chunks, prepare_gate=None, hold=None):
    """An app with the job routes. `prepare_gate`: the fake engine load waits
    for it. `hold`: the fake stream waits for it before its last chunk."""
    app = FastAPI()

    async def prepare(req):
        if prepare_gate is not None:
            while not prepare_gate.is_set():
                await asyncio.sleep(0.01)
        return "test-model"

    def stream_factory(req, model):
        async def gen():
            for c in chunks[:-1]:
                yield c
            if hold is not None:
                while not hold.is_set():
                    await asyncio.sleep(0.01)
            yield chunks[-1]
        return gen()

    chat_jobs.register_chat_job_routes(app, check_auth=lambda r: True, prepare=prepare,
                                       stream_factory=stream_factory,
                                       session_dir=str(tmp_path))
    return app


def _wait(job_id, client, want=("done", "error", "stopped"), timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        snap = client.get(f"/v1/chat/jobs/{job_id}").json()
        if snap["status"] in want:
            return snap
        time.sleep(0.02)
    raise AssertionError(f"job still {snap['status']}")


BODY = {"messages": [{"role": "user", "content": "hi"}], "session_id": "s1"}


def test_the_job_id_comes_back_before_the_engine_has_loaded(jobs, tmp_path):
    gate = threading.Event()
    with TestClient(_app(tmp_path, [_content("ok")], prepare_gate=gate)) as client:
        r = client.post("/v1/chat/jobs", json=BODY)          # must not wait for the load
        assert r.status_code == 200 and r.json()["status"] == "running"
        job_id = r.json()["job_id"]
        # Stop during the load: the job ends stopped and never generates.
        assert client.post(f"/v1/chat/jobs/{job_id}/stop").json()["status"] == "stopping"
        gate.set()
        snap = _wait(job_id, client)
        assert snap["status"] == "stopped" and snap["text"] == ""


def test_a_stopping_job_does_not_capture_the_next_message(jobs, tmp_path):
    hold = threading.Event()
    with TestClient(_app(tmp_path, [_content("old "), _content("answer")], hold=hold)) as client:
        first = client.post("/v1/chat/jobs", json=BODY).json()["job_id"]
        # Live and not stopped: a second send is told about it, with its session.
        r = client.post("/v1/chat/jobs", json=dict(BODY, session_id="s2"))
        assert r.status_code == 409
        assert r.json()["detail"]["job_id"] == first and r.json()["detail"]["session_id"] == "s1"
        client.post(f"/v1/chat/jobs/{first}/stop")
        r = client.post("/v1/chat/jobs", json=dict(BODY, session_id="s2"))
        assert r.status_code == 200 and r.json()["job_id"] != first
        hold.set()


def test_only_the_last_tool_round_is_the_answer(jobs):
    job = chat_jobs.ChatJob(chat_jobs.ChatJobRequest(**BODY))
    for payload in ({"choices": [{"delta": {"content": '@search("weather")'}}]},
                    {"x_tool_status": ["Searching: weather"]},
                    {"x_tool_done": True},
                    {"choices": [{"delta": {"content": "Sunny."}}]}):
        job.push(payload)
    snap = job.snapshot()
    assert snap["text"] == "Sunny." and snap["tool_status"] == ["Searching: weather"]


def test_a_generation_error_ends_the_job_as_an_error_and_is_not_saved(jobs, tmp_path):
    chunks = [_content("half an ans"), _sse({"error": "llama-server is not responding"})]
    with TestClient(_app(tmp_path, chunks)) as client:
        job_id = client.post("/v1/chat/jobs", json=BODY).json()["job_id"]
        snap = _wait(job_id, client)
    assert snap["status"] == "error" and "not responding" in snap["error"]
    assert snap["text"] == "half an ans"                    # the phone still gets the partial
    assert not list(tmp_path.iterdir())                     # but it isn't saved as complete


def test_a_generating_job_reports_busy_to_the_model_queue(jobs, tmp_path):
    hold = threading.Event()
    assert chat_jobs.live_chat_busy_reason() is None
    with TestClient(_app(tmp_path, [_content("a"), _content("b")], hold=hold)) as client:
        job_id = client.post("/v1/chat/jobs", json=BODY).json()["job_id"]
        end = time.time() + 5
        while chat_jobs.live_chat_busy_reason() is None and time.time() < end:
            time.sleep(0.02)
        assert chat_jobs.live_chat_busy_reason()
        hold.set()
        _wait(job_id, client)
    assert chat_jobs.live_chat_busy_reason() is None


# ── the queue between the generation thread and the SSE generator ───────────

def test_an_abandoned_stream_stops_its_producer_instead_of_blocking_it():
    q = server._StreamQueue()
    produced, ended = [], threading.Event()

    def producer():
        try:
            for i in range(100000):
                q.put(("content", str(i)))
                produced.append(i)
        except server._StreamAbandoned:
            pass
        finally:
            q.finish()
            ended.set()

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    time.sleep(0.2)
    assert not ended.is_set() and len(produced) <= 257     # blocked on the full queue
    q.abandon()
    assert ended.wait(3), "the producer was left blocked on a queue nobody reads"


def test_abandon_wakes_a_reader_waiting_on_an_empty_queue():
    q = server._StreamQueue()
    got = []
    t = threading.Thread(target=lambda: got.append(q.get()), daemon=True)
    t.start()
    q.abandon()
    t.join(3)
    assert got == [server._SENTINEL]


def test_closing_the_stream_early_abandons_its_queue(monkeypatch):
    """Client disconnect / chat-job Stop: the generator is closed mid-answer."""
    released = threading.Event()

    def endless(engine, messages, max_tokens, temperature, out_queue, *a, **k):
        try:
            while True:
                out_queue.put(("content", "x"))
        except server._StreamAbandoned:
            released.set()
        finally:
            server._finish_queue(out_queue)

    class Engine:
        native_tools = False
        def get_context_size(self):
            return 0

    monkeypatch.setattr(server, "_stream_transformers_raw", endless)
    monkeypatch.setattr(server, "_get_engine", lambda *a, **k: Engine())
    monkeypatch.setattr(server, "_clamp_max_tokens", lambda m, mt, b, c: mt)

    async def run():
        gen = server._stream_with_tools([{"role": "user", "content": "hi"}], "m", 64, 0.7,
                                        {}, False, "llama_cpp")
        async for _ in gen:
            break
        await gen.aclose()

    asyncio.run(run())
    assert released.wait(3), "generation kept running after its reader left"


def test_error_and_end_markers_still_reach_a_plain_queue():
    q = queue.Queue()
    server._put_error(q, RuntimeError("boom"))
    server._finish_queue(q)
    assert q.get() == ("error", "boom") and q.get() is server._SENTINEL


# ── auth rate limit ─────────────────────────────────────────────────────────

class _Req:
    def __init__(self, key):
        self.headers = {"X-API-Key": key} if key else {}
        self.client = type("C", (), {"host": "198.51.100.7"})()


def test_the_right_key_gets_in_even_after_a_burst_of_wrong_ones(monkeypatch):
    from core.api_keys import KeyStore
    monkeypatch.setattr(server, "_keystore", KeyStore("", legacy_key="correct-key"))
    monkeypatch.setattr(server, "_auth_failures", {})
    for _ in range(server._AUTH_FAIL_MAX + 5):
        assert server._check_auth(_Req("stale-key")) is False
    assert server._check_auth(_Req("correct-key")) is True
    assert server._check_auth(_Req("another-guess")) is False   # guesses stay limited
