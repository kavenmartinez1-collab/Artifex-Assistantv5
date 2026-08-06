"""
WebGPU bridge + engine integration tests.

A fake "browser" thread speaks the real HTTP protocol against a real
BridgeServer (ephemeral port), so these cover the wire format end-to-end:
hello handshake, long-poll job pickup, ordered token events, done stats,
error propagation, and detach detection. No browser or GPU involved.
"""

import json
import threading
import time
import urllib.request

import pytest

from core.webgpu_bridge import BridgeServer
from core.engine_webgpu import WebGpuEngine, map_sampling
import core.engine_webgpu as engine_mod
import core.webgpu_bridge as bridge_mod


# ── helpers ────────────────────────────────────────────────────────────────

def _post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _get_job(port, wait=1):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/bridge/job?wait={wait}")
    with urllib.request.urlopen(req, timeout=wait + 5) as resp:
        if resp.status == 204:
            return None
        return json.loads(resp.read())


class FakeBrowser(threading.Thread):
    """Polls for one job and plays back a scripted event sequence."""

    def __init__(self, port, events, hello=None, jobs_to_serve=1):
        super().__init__(daemon=True)
        self.port = port
        self.events = events
        self.hello = hello or {"ready": True, "model": "fake-27b", "ctx": 8192}
        self.jobs_to_serve = jobs_to_serve
        self.saw_cancel = False
        self.served = []

    def run(self):
        _post(self.port, "/bridge/hello", self.hello)
        served = 0
        deadline = time.monotonic() + 20
        while served < self.jobs_to_serve and time.monotonic() < deadline:
            job = _get_job(self.port, wait=1)
            if job is None:
                continue
            self.served.append(job)
            for ev in self.events:
                reply = _post(self.port, "/bridge/event", {"id": job["id"], **ev})
                if reply.get("cancel"):
                    self.saw_cancel = True
                    break
                time.sleep(0.01)
            served += 1


@pytest.fixture()
def bridge():
    server = BridgeServer(port=0)
    yield server
    server.close()


def make_engine(bridge):
    return WebGpuEngine(port=bridge.port, handshake_timeout=5, bridge=bridge)


# ── sampling mapping ───────────────────────────────────────────────────────

class TestSamplingMapping:
    def test_explicit_dict_maps_keys(self):
        out = map_sampling({"temperature": 0.6, "top_k": 20, "top_p": 0.95,
                            "min_p": 0.0, "repeat_penalty": 1.0,
                            "dry_multiplier": 0.8, "dry_penalty_last_n": 256,
                            "seed": 42}, temperature=0.7)
        assert out["temperature"] == 0.6          # dict wins over positional
        assert out["topK"] == 20 and out["topP"] == 0.95
        assert out["minP"] == 0.0
        assert out["repetitionPenalty"] == 1.0
        assert out["dryMultiplier"] == 0.8
        assert out["dryRangeLastN"] == 256
        assert "seed" not in out                  # unsupported keys dropped

    def test_none_uses_default_sampling_with_caller_temp(self):
        out = map_sampling(None, temperature=0.3)
        assert out["temperature"] == 0.3
        assert out["minP"] == 0.0                 # explicit, not engine default
        assert out["topK"] == 40 and out["topP"] == 0.9


# ── lifecycle ──────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_not_loaded_until_hello(self, bridge):
        engine = make_engine(bridge)
        assert not engine.is_loaded()
        _post(bridge.port, "/bridge/hello",
              {"ready": True, "model": "fake-27b", "ctx": 4096})
        assert engine.is_loaded()
        assert engine.get_context_size() == 4096

    def test_load_times_out_without_browser(self, bridge):
        engine = WebGpuEngine(port=bridge.port, handshake_timeout=0.5,
                              bridge=bridge)
        with pytest.raises(TimeoutError):
            engine.load()

    def test_health_endpoint(self, bridge):
        _post(bridge.port, "/bridge/hello", {"ready": True, "model": "m"})
        req = urllib.request.Request(
            f"http://127.0.0.1:{bridge.port}/bridge/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["ok"] and data["client_attached"]
        assert data["session"]["model"] == "m"


# ── generation ─────────────────────────────────────────────────────────────

class TestGeneration:
    def test_streams_tokens_in_order(self, bridge):
        browser = FakeBrowser(bridge.port, [
            {"type": "token", "text": "Hello"},
            {"type": "token", "text": " world"},
            {"type": "done", "stats": {"numTokens": 2, "tokensPerSecond": 10.0,
                                       "promptTokens": 5, "stopReason": "eos"}},
        ])
        browser.start()
        engine = make_engine(bridge)
        pieces = []
        out = engine.generate_streaming(
            [{"role": "user", "content": "hi"}], max_tokens=64,
            temperature=0.6, on_token=pieces.append, enable_thinking=False)
        browser.join(timeout=10)
        assert out == "Hello world"
        assert pieces == ["Hello", " world"]
        assert engine._last_gen_stats["completion_tokens"] == 2
        assert engine._last_gen_stats["finish_reason"] == "stop"
        job = browser.served[0]
        assert job["sampling"]["topK"] == 40      # DEFAULT_SAMPLING mapped
        assert job["sampling"]["maxNewTokens"] == 64
        assert job["enableThinking"] is False

    def test_think_prefix_stripped(self, bridge):
        # Transformers-convention stream: starts INSIDE a think block.
        browser = FakeBrowser(bridge.port, [
            {"type": "token", "text": "reasoning about it...</think>"},
            {"type": "token", "text": "The answer is 4."},
            {"type": "done", "stats": {"numTokens": 12, "tokensPerSecond": 9.0,
                                       "stopReason": "eos"}},
        ])
        browser.start()
        engine = make_engine(bridge)
        out = engine.generate_streaming(
            [{"role": "user", "content": "2+2?"}], max_tokens=64,
            temperature=0.6)
        browser.join(timeout=10)
        assert out == "The answer is 4."

    def test_browser_error_raises(self, bridge):
        browser = FakeBrowser(bridge.port, [
            {"type": "error", "error": "device lost"},
        ])
        browser.start()
        engine = make_engine(bridge)
        with pytest.raises(RuntimeError, match="device lost"):
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}], max_tokens=16,
                temperature=0.6)
        browser.join(timeout=10)

    def test_detached_browser_raises_connection_error(self, bridge, monkeypatch):
        monkeypatch.setattr(bridge_mod, "CLIENT_STALE_S", 0.3)
        monkeypatch.setattr(engine_mod, "EVENT_POLL_S", 0.2)
        _post(bridge.port, "/bridge/hello", {"ready": True, "model": "m"})
        engine = make_engine(bridge)
        # No browser polls for the job -> client goes stale mid-request.
        with pytest.raises(ConnectionError, match="detached"):
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}], max_tokens=16,
                temperature=0.6)

    def test_preset_dict_passthrough(self, bridge):
        from core.sampling import get_preset
        browser = FakeBrowser(bridge.port, [
            {"type": "token", "text": "ok"},
            {"type": "done", "stats": {"numTokens": 1, "tokensPerSecond": 5.0,
                                       "stopReason": "max_length"}},
        ])
        browser.start()
        engine = make_engine(bridge)
        engine.generate_streaming(
            [{"role": "user", "content": "hi"}], max_tokens=16,
            temperature=0.9, sampling=get_preset("agent"))
        browser.join(timeout=10)
        samp = browser.served[0]["sampling"]
        assert samp["temperature"] == 0.6         # preset wins
        assert samp["topK"] == 20 and samp["topP"] == 0.95
        assert engine._last_gen_stats["finish_reason"] == "length"


# ── factory/config wiring ──────────────────────────────────────────────────

class TestWiring:
    def test_factory_returns_webgpu_engine(self, monkeypatch):
        import core.engine_factory as factory
        monkeypatch.setattr(factory, "get_active_backend", lambda: "webgpu")
        engine = factory.create_engine()
        assert isinstance(engine, WebGpuEngine)

    def test_backend_whitelisted(self):
        from core.config import set_active_backend, get_active_backend
        prev = get_active_backend()
        try:
            assert set_active_backend("webgpu")
            assert get_active_backend() == "webgpu"
            from core.config import get_model_names
            assert get_model_names() == ["(browser session)"]
        finally:
            set_active_backend(prev)
