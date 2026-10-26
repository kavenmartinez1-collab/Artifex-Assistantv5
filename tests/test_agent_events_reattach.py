"""A phone that reattaches mid-round must see the round so far.

Token chunks are live-only (never replayed), so before chunk_snapshot a
reattach showed a blank feed until the round's assistant_message landed —
minutes, with long thinking — and it looked like the run had stalled.
"""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import agent_api
from api.agent_api import AgentRun, register_agent_routes
from core.agent_loop import AgentEvent, RunConfig


def _client():
    app = FastAPI()
    register_agent_routes(app, lambda req: True, lambda: None, "unused")
    return TestClient(app)


def _events(client, run, since=0):
    r = client.get(f"/v1/agent/runs/{run.id}/events?since={since}")
    assert r.status_code == 200
    return [json.loads(line[5:]) for line in r.text.splitlines()
            if line.startswith("data:")]


def _run():
    run = AgentRun("goal", "unused", RunConfig())
    agent_api._runs[run.id] = run
    return run


def test_reattach_mid_round_gets_the_round_so_far():
    run = _run()
    try:
        run.emit(AgentEvent(kind="round_start", round=1))
        run.emit(AgentEvent(kind="thinking_chunk", round=1, text="hmm "))
        run.emit(AgentEvent(kind="thinking_chunk", round=1, text="ok"))
        run.emit(AgentEvent(kind="assistant_chunk", round=1, text="Hello "))
        run.emit(AgentEvent(kind="assistant_chunk", round=1, text="there"))
        run.status = "done"          # terminal, so the feed ends on its own

        evs = _events(_client(), run)
        kinds = [e["kind"] for e in evs]
        assert kinds == ["round_start", "chunk_snapshot", "end"]
        snap = evs[1]
        assert snap["thinking"] == "hmm ok"
        assert snap["text"] == "Hello there"
    finally:
        agent_api._runs.pop(run.id, None)


def test_a_persisted_event_closes_the_round_snapshot():
    """The round's text is in assistant_message now; don't send it twice."""
    run = _run()
    try:
        run.emit(AgentEvent(kind="assistant_chunk", round=1, text="partial"))
        run.emit(AgentEvent(kind="assistant_message", round=1, text="partial"))
        run.status = "done"

        evs = _events(_client(), run, since=0)
        assert [e["kind"] for e in evs] == ["assistant_message", "end"]
        # And a reattach past it sees no stale snapshot either.
        assert [e["kind"] for e in _events(_client(), run, since=1)] == ["end"]
    finally:
        agent_api._runs.pop(run.id, None)
