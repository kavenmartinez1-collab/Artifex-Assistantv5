"""Tests for agent runs owning their model.

The failures these guard against, all seen from the phone:
- the Agent tab sent no model, so a run with the vision entry selected ran on
  whatever the API had launched with (the text model);
- runs bypassed the model queue, so the idle shrink unloaded llama-server
  under a long run, and a chat naming another model swapped it mid-run;
- the agent was never told what it was running as, so "can you see images?"
  got a guess.
"""

import asyncio
import http.server
import json
import threading
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.agent_api as agent_api
from core import model_queue as mq_mod
from core.model_queue import ModelBusyError, ModelQueue
from core.prompts import build_assistant_prompt


def _switch(q, model, tier=None, **kw):
    async def run():
        with patch("core.config.set_active_backend"), \
             patch("core.config.set_active_model"):
            await q.switch_if_needed(model, "llama_cpp", ctx_tier=tier, **kw)
    asyncio.run(run())


def _queue(busy=None):
    q = ModelQueue()
    q.unloads = []
    q._engine_unload_fn = lambda: q.unloads.append(q._current_model)
    q.register_busy_check(lambda: busy[0] if busy else None)
    return q


# ── Model queue: busy work keeps its model ──────────────────────────────

class TestQueueBusy:
    def test_switch_refused_while_busy(self):
        busy = [None]
        q = _queue(busy)
        _switch(q, "vision", 16384)
        busy[0] = "agent run abc is using the model"
        with pytest.raises(ModelBusyError, match="agent run abc.*vision.*text"):
            _switch(q, "text", 73728)
        assert q.current_model == "vision" and q.unloads == []

    def test_same_model_while_busy_is_fine_and_skips_relaunch(self):
        busy = [None]
        q = _queue(busy)
        _switch(q, "vision", 16384)
        busy[0] = "agent run abc is using the model"
        _switch(q, "vision", 32768)          # would normally relaunch bigger
        assert q.unloads == []
        assert q.current_ctx_tier == 16384

    def test_ignore_busy_lets_the_run_pick_its_own_model(self):
        busy = ["agent run abc is using the model"]
        q = _queue(busy)
        _switch(q, "text", 73728)
        _switch(q, "vision", 16384, ignore_busy=True)
        assert q.current_model == "vision" and q.unloads == ["text"]

    def test_switch_allowed_when_idle(self):
        q = _queue([None])
        _switch(q, "text", 73728)
        _switch(q, "vision", 16384)
        assert q.current_model == "vision"

    def test_broken_busy_check_counts_as_idle(self):
        q = ModelQueue()
        q.register_busy_check(lambda: 1 / 0)
        assert q.busy_reason() is None

    def test_idle_shrink_skips_busy_and_resets_countdown(self, monkeypatch):
        monkeypatch.setattr(mq_mod, "IDLE_SHRINK_SEC", 0.01)
        monkeypatch.setattr(mq_mod, "IDLE_SHRINK_CHECK_INTERVAL", 0.01)
        busy = ["agent run abc is using the model"]
        q = _queue(busy)
        q._current_model, q._current_backend = "vision", "llama_cpp"
        q._last_request_at = 0.0             # "idle" for decades

        async def run():
            task = asyncio.ensure_future(q._idle_shrink_loop())
            await asyncio.sleep(0.1)
            assert q.unloads == [] and q._last_request_at > 0
            busy[0] = None                   # run finished
            for _ in range(200):
                if q.current_model is None:
                    break
                await asyncio.sleep(0.01)
            task.cancel()
        asyncio.run(run())
        assert q.unloads == ["vision"] and q.current_model is None


# ── Agent routes: the run switches to ITS model first ───────────────────

class _Engine:
    _base_url = None

    def get_context_size(self):
        return 16384


@pytest.fixture
def client(monkeypatch, tmp_path):
    agent_api._runs.clear()
    started = []

    def fake_worker(run, get_engine, goal=None):
        started.append(run)
        run.finish("done", "ok")

    monkeypatch.setattr(agent_api, "_worker", fake_worker)
    prepared = []
    state = {"error": None}

    async def prepare(requested):
        if state["error"]:
            raise state["error"]
        prepared.append(requested)
        return requested or "active-model"

    app = FastAPI()
    agent_api.register_agent_routes(
        app, check_auth=lambda r: True, get_engine=_Engine,
        default_workspace_root=str(tmp_path), prepare_model=prepare)
    c = TestClient(app)
    c.prepared, c.started, c.state = prepared, started, state
    yield c
    agent_api._runs.clear()


class TestAgentRoutesModel:
    def test_run_prepares_the_requested_model(self, client):
        r = client.post("/v1/agent/runs", json={"goal": "look", "model": "vision"})
        assert r.status_code == 200, r.text
        assert r.json()["model"] == "vision"
        assert client.prepared == ["vision"]
        snap = client.get(f"/v1/agent/runs/{r.json()['run_id']}").json()
        assert snap["model"] == "vision"

    def test_omitted_model_resolves_to_active(self, client):
        r = client.post("/v1/agent/runs", json={"goal": "look"})
        assert client.prepared == [None] and r.json()["model"] == "active-model"

    def test_busy_queue_is_a_readable_409(self, client):
        client.state["error"] = ModelBusyError("a chat is generating")
        r = client.post("/v1/agent/runs", json={"goal": "look", "model": "vision"})
        assert r.status_code == 409
        assert r.json()["detail"]["message"] == "a chat is generating"
        assert client.started == [] and agent_api._runs == {}

    def test_live_run_blocks_before_any_switch(self, client):
        live = agent_api.AgentRun("busy", "", agent_api.RunConfig(), model="text")
        agent_api._runs[live.id] = live
        r = client.post("/v1/agent/runs", json={"goal": "look", "model": "vision"})
        assert r.status_code == 409 and client.prepared == []

    def test_revive_switches_back_to_the_runs_model(self, client):
        r = client.post("/v1/agent/runs", json={"goal": "look", "model": "vision"})
        run_id = r.json()["run_id"]
        r2 = client.post(f"/v1/agent/runs/{run_id}/message", json={"text": "more"})
        assert r2.json()["revived"] is True
        assert client.prepared == ["vision", "vision"]

    def test_busy_reason_tracks_live_runs(self):
        agent_api._runs.clear()
        assert agent_api.live_run_busy_reason() is None
        run = agent_api.AgentRun("g", "", agent_api.RunConfig(), model="vision")
        agent_api._runs[run.id] = run
        assert run.id in agent_api.live_run_busy_reason()
        run.finish("done")
        assert agent_api.live_run_busy_reason() is None
        agent_api._runs.clear()


# ── The agent is told what it is running as ─────────────────────────────

class _Props(http.server.BaseHTTPRequestHandler):
    vision = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"model_path": "C:/m/x.gguf",
                           "modalities": {"vision": type(self).vision}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def props_server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Props)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


class TestModelInfo:
    def test_engine_vision_reads_the_live_server(self, props_server):
        eng = _Engine()
        eng._base_url = props_server
        _Props.vision = True
        assert agent_api._engine_vision(eng) is True
        _Props.vision = False
        assert agent_api._engine_vision(eng) is False
        assert agent_api._engine_vision(_Engine()) is None       # no URL
        eng._base_url = "http://127.0.0.1:9"                     # nothing there
        assert agent_api._engine_vision(eng) is None

    def test_model_text_variants(self):
        yes = agent_api._model_info_text("vis", 16384, True)
        no = agent_api._model_info_text("txt", 73728, False)
        assert "vis" in yes and "16384" in yes and "Vision: YES" in yes
        assert "Vision: NO" in no and "vision model" in no
        assert "Vision: unknown" in agent_api._model_info_text(None, 1, None)

    def test_prompt_carries_model_section_only_when_given(self):
        p = build_assistant_prompt("OS: x", "C:/w", model_text="You are running as vis")
        assert "MODEL:\nYou are running as vis" in p
        assert p.index("MODEL:") < p.index("ENVIRONMENT:")
        assert "MODEL:" not in build_assistant_prompt("OS: x", "C:/w")

    def test_worker_puts_model_in_prompt_and_emits_event(self, monkeypatch, tmp_path,
                                                         props_server):
        _Props.vision = True
        eng = _Engine()
        eng._base_url = props_server
        seen = {}

        class FakeRunner:
            def __init__(self, engine, *, build_system_prompt, **kw):
                self.build = build_system_prompt

            def run(self, goal, history):
                seen["prompt"] = self.build()

                class R:
                    status, summary, rounds, actions_run = "done", "", 1, 0
                return R()

        monkeypatch.setattr(agent_api, "AgentRunner", FakeRunner)
        run = agent_api.AgentRun("g", str(tmp_path), agent_api.RunConfig(),
                                 model="qwen-vision")
        agent_api._worker(run, lambda: eng)
        assert "You are running as the model qwen-vision" in seen["prompt"]
        assert "Vision: YES" in seen["prompt"]
        assert run.vision is True and run.snapshot()["vision"] is True
        model_events = [e for e in run.events if e["kind"] == "model"]
        assert model_events and "can see images" in model_events[0]["text"]
