"""Regression tests for the model-serving core review (2026-11-09).

Each test pins one defect that review found in the engine / GPU pool /
queue / config layer.  Nothing here starts a server, loads a model or
probes a GPU: HTTP, subprocesses, psutil and the pool are all doubles.
"""

import asyncio
import io
import json
import logging
import os
import subprocess
import sys
import threading
import time
import types
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

import core.engine_llama_cpp as lc
from core.engine_llama_cpp import LlamaCppEngine

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── helpers ────────────────────────────────────────────────────────────

def _sse(*texts, reasoning=()):
    lines = []
    for t in reasoning:
        lines.append(("data: " + json.dumps(
            {"choices": [{"delta": {"reasoning_content": t}}]}) + "\n").encode())
    for t in texts:
        lines.append(("data: " + json.dumps(
            {"choices": [{"delta": {"content": t}}]}) + "\n").encode())
    lines.append(b"data: [DONE]\n")
    return lines


class FakeResponse:
    """urlopen() result: a context manager iterating SSE lines.

    `fail_after` raises that exception once N lines have been read, the way
    a dropped connection surfaces in the read loop.
    """

    def __init__(self, lines, fail_after=None, error=None):
        self.lines, self.fail_after, self.error = lines, fail_after, error
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False

    def __iter__(self):
        for i, line in enumerate(self.lines):
            if self.fail_after is not None and i >= self.fail_after:
                raise self.error
            yield line


def _ready_engine(**cfg):
    """An engine that believes its server is up (no load, no health HTTP)."""
    e = LlamaCppEngine("m", dict({"path": "model.gguf", "num_ctx": 32768}, **cfg))
    e.load = lambda *a, **k: None
    e._is_server_healthy = lambda: True
    e._num_ctx = 32768
    return e


def _generate(engine, responses, **kw):
    posts = []

    def fake_urlopen(req, timeout=None):
        posts.append(req)
        return responses[len(posts) - 1]

    with patch.object(lc.urllib.request, "urlopen", fake_urlopen), \
         patch.object(lc.time, "sleep", lambda s: None):
        try:
            return engine.generate_streaming(
                [{"role": "user", "content": "hi"}], 64, 0.1, **kw), None, posts
        except BaseException as err:   # noqa: BLE001 - the test inspects it
            return None, err, posts


# ── #2: what on_token raises is the caller's, not a lost connection ────

class StopPressed(Exception):
    """Stand-in for any caller-defined abort (not an OSError)."""


@pytest.mark.parametrize("abort", [InterruptedError("Generation cancelled"),
                                   StopPressed("stop"),
                                   BrokenPipeError("client went away")])
def test_on_token_exception_propagates_untouched_and_is_not_retried(abort):
    seen = []

    def on_token(t):
        seen.append(t)
        if len(seen) == 2:
            raise abort

    first = FakeResponse(_sse("Hel", "lo", " world"))
    _, err, posts = _generate(_ready_engine(), [first, FakeResponse(_sse("x"))],
                              on_token=on_token)
    assert err is abort                    # the very object, not a wrapper
    assert len(posts) == 1                 # the prompt was NOT sent again
    assert seen == ["Hel", "lo"]           # nothing replayed into the callback
    assert first.closed                    # closing it is what stops llama-server


def test_drop_before_any_token_is_still_retried():
    dropped = FakeResponse(_sse("Hel"), fail_after=0,
                           error=ConnectionResetError("reset"))
    seen = []
    text, err, posts = _generate(
        _ready_engine(), [dropped, FakeResponse(_sse("Hello"))],
        on_token=seen.append, raw_output=True)
    assert err is None and text == "Hello"
    assert len(posts) == 2 and seen == ["Hello"]


def test_drop_after_tokens_were_delivered_is_not_retried():
    """A re-sent request restarts the answer: the caller would get the
    opening twice (and a second <think>)."""
    dropped = FakeResponse(_sse("Hel", "lo", reasoning=("hm",)), fail_after=2,
                           error=ConnectionResetError("reset"))
    seen = []
    _, err, posts = _generate(
        _ready_engine(), [dropped, FakeResponse(_sse("Hello"))],
        on_token=seen.append)
    assert isinstance(err, ConnectionError)
    assert len(posts) == 1
    assert seen.count("<think>") == 1


# ── #13: no stop-string cut on llama.cpp answers ───────────────────────

TRANSCRIPT = "Here is a transcript:\nBot: hi\nUser: hello there\nBot: bye\nThe end."


def test_llama_cpp_answer_with_a_user_line_is_not_truncated():
    text, err, _ = _generate(_ready_engine(), [FakeResponse(_sse(TRANSCRIPT))])
    assert err is None
    assert text == TRANSCRIPT


def test_clean_response_stop_strings_switch():
    from core.inference import _clean_response
    assert _clean_response(TRANSCRIPT) == "Here is a transcript:\nBot: hi"
    assert _clean_response(TRANSCRIPT, stop_strings=False) == TRANSCRIPT
    # thinking is stripped either way
    assert _clean_response("<think>x</think>ok", stop_strings=False) == "ok"


# ── #15: the repetition scan must not be quadratic ─────────────────────

def test_truncate_repetition_is_fast_on_long_clean_text():
    import random
    from core.inference import _truncate_repetition
    rnd = random.Random(3)
    text = " ".join("".join(rnd.choice("abcdefghij") for _ in range(rnd.randint(2, 9)))
                    for _ in range(16000))[:80000]
    start = time.perf_counter()
    assert _truncate_repetition(text) == text
    # Was ~10 s for 80K chars (whole-text search per start position);
    # the bounded search takes well under a second.
    assert time.perf_counter() - start < 4.0


def test_truncate_repetition_verdicts_unchanged():
    from core.inference import _truncate_repetition
    phrase = "The quick brown fox jumps over the lazy dog near the quiet river bank today. "
    assert len(phrase) < 80
    unit = phrase + "And again it runs. "          # > 80 chars
    looped = "Intro. " + unit * 5
    cut = _truncate_repetition(looped)
    assert looped.startswith(cut) and len(cut) < len("Intro. ") + 2 * len(unit)
    # the same boilerplate with real data between the copies is kept
    spaced = "".join(
        unit + f"item {i}: " + " ".join(str(i * j * 7919) for j in range(1, 15)) + ". "
        for i in range(1, 6))
    assert _truncate_repetition(spaced) == spaced


# ── #5: a server we spawn is ours, whatever the engine adopted before ──

class FakeProc:
    def __init__(self, alive=True):
        self.alive, self.terminated, self.returncode = alive, False, 0

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated, self.alive = True, False

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.alive = False


def _patched_load(engine, *, healthy, spawn_result=None, pool=None, gate=None,
                  served=(None, None), kill_listener=True, reclaim=0):
    """Run engine.load() with every outside effect replaced.

    Returns (error_or_None, mocks) where mocks has spawn / gate / claimed.
    """
    pool = pool or MagicMock()
    spawn = MagicMock(side_effect=FileNotFoundError) if spawn_result is None \
        else MagicMock(return_value=spawn_result)
    gate = gate or MagicMock()
    claimed = MagicMock()
    health = MagicMock(side_effect=healthy) if isinstance(healthy, list) \
        else MagicMock(return_value=healthy)
    err = None
    with patch("core.engine_llama_cpp.os.path.isfile", return_value=True), \
         patch("core.engine_llama_cpp.os.makedirs"), \
         patch("core.engine_llama_cpp.open", MagicMock(), create=True), \
         patch.object(engine, "_is_server_healthy", health), \
         patch.object(engine, "_served_props", return_value=served), \
         patch.object(engine, "_kill_listener_on_port", return_value=kill_listener), \
         patch.object(engine, "_reclaim_orphan_servers", return_value=reclaim), \
         patch.object(engine, "_resolve_gpu_index", return_value=0), \
         patch.object(engine, "_vram_gate", gate), \
         patch("core.gpu_pool.get_pool", return_value=pool), \
         patch("core.services.orpheus_tts.gpu_claimed", claimed), \
         patch("core.services.orpheus_tts.gpu_released"), \
         patch("core.engine_llama_cpp.spawn", spawn), \
         patch.object(lc.time, "sleep", lambda s: None):
        try:
            engine.load()
        except Exception as e:   # noqa: BLE001
            err = e
    return err, types.SimpleNamespace(spawn=spawn, gate=gate, claimed=claimed,
                                      health=health)


def test_spawned_server_is_terminated_on_unload_even_after_an_adoption():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768})
    # State a GUI/CLI engine is in once the server it adopted has died.
    e._adopted, e._loaded, e._process = True, True, None
    proc = FakeProc()
    err, _ = _patched_load(e, healthy=[False, False, True], spawn_result=proc)
    assert err is None and e._loaded and e._process is proc
    assert e._adopted is False
    with patch("core.services.orpheus_tts.gpu_released"):
        e.unload()
    assert proc.terminated, "our own child was 'released' instead of stopped"


# ── #6: an explicit reload stays exact through the VRAM gate ───────────

def test_exact_reload_refuses_instead_of_stepping_down_when_a_server_was_running():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 73728})
    e.set_target_tier(64000, exact=True)
    gate = MagicMock(side_effect=RuntimeError("VRAM gate refused ctx=64000"))
    # A healthy server of OUR model at another ctx is on the port (API
    # restart, pre-warm): the adoption block runs first.
    err, m = _patched_load(e, healthy=True, served=("model.gguf", 32000), gate=gate)
    assert isinstance(err, RuntimeError) and "refused" in str(err)
    assert gate.call_count == 1, "the gate walked down the tiers on an exact reload"
    m.spawn.assert_not_called()
    assert e._target_exact is False        # one-shot


def test_implicit_load_still_steps_down():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 73728})
    e.set_target_tier(64000)
    ctxs = []

    def gate(pool, idx, wait_timeout):
        ctxs.append(e._num_ctx)
        if e._num_ctx > 32000:
            raise RuntimeError("refused")

    err, _ = _patched_load(e, healthy=False, gate=gate)
    assert isinstance(err, FileNotFoundError)      # reached the launch
    assert ctxs == [64000, 32000]


# ── #11: never launch next to our own child or an unreclaimable port ───

def test_load_is_a_no_op_while_our_child_runs_even_if_health_is_slow():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768})
    e._loaded, e._process = True, FakeProc()
    err, m = _patched_load(e, healthy=False)       # /health timed out
    assert err is None
    m.claimed.assert_not_called()                  # never entered the load path
    m.gate.assert_not_called()
    m.spawn.assert_not_called()


def test_load_relaunches_when_our_child_has_died():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768})
    e._loaded, e._process = True, FakeProc(alive=False)
    err, m = _patched_load(e, healthy=False)
    assert isinstance(err, FileNotFoundError)      # went on to spawn
    m.spawn.assert_called_once()


def test_load_refuses_when_the_port_cannot_be_reclaimed():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768, "port": 8123})
    # Healthy, serves another model, and no llama-server process to stop.
    err, m = _patched_load(e, healthy=True, served=("other.gguf", 32768),
                           kill_listener=False)
    assert isinstance(err, RuntimeError)
    assert "8123" in str(err) and "port" in str(err).lower()
    m.spawn.assert_not_called()
    assert not e._loaded


def test_port_reclaimed_normally_proceeds_to_launch():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768})
    err, m = _patched_load(e, healthy=[True, False], served=("other.gguf", 32768),
                           kill_listener=True)
    assert isinstance(err, FileNotFoundError)
    m.spawn.assert_called_once()


# ── #8: no 30 s wait for VRAM that is known to be stable ───────────────

@pytest.mark.parametrize("settled,expected", [
    (True, lc.SETTLED_GATE_TIMEOUT),
    (False, None),          # still reclaiming: the long wait stays
])
def test_first_gate_pass_waits_long_only_while_vram_is_still_moving(settled, expected):
    from core.gpu_pool import DEFAULT_VRAM_WAIT_TIMEOUT
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 32768})
    pool = MagicMock()
    pool.wait_for_vram_settled.return_value = settled
    timeouts = []
    gate = MagicMock(side_effect=lambda p, i, wait_timeout: timeouts.append(wait_timeout))
    _patched_load(e, healthy=False, pool=pool, gate=gate)
    assert timeouts == [expected if expected is not None else DEFAULT_VRAM_WAIT_TIMEOUT]
    assert lc.SETTLED_GATE_TIMEOUT < DEFAULT_VRAM_WAIT_TIMEOUT


# ── #16: one health round trip per request, one enumeration per load ───

def test_is_loaded_caches_a_positive_health_answer_for_adopted_servers():
    e = LlamaCppEngine("m", {"path": "model.gguf"})
    e._loaded, e._process = True, None             # adopted / taken over
    calls = []
    e._is_server_healthy = lambda: calls.append(1) or True
    assert all(e.is_loaded() for _ in range(6))
    assert len(calls) == 1
    with patch("core.services.orpheus_tts.gpu_released"), \
         patch.object(e, "_kill_process"):
        e.unload()
    e._loaded = True
    e.is_loaded()
    assert len(calls) == 2, "unload must drop the cached answer"


def test_is_loaded_never_caches_a_negative_answer():
    e = LlamaCppEngine("m", {"path": "model.gguf"})
    e._loaded, e._process = True, None
    answers = [False, True]
    e._is_server_healthy = lambda: answers.pop(0)
    assert e.is_loaded() is False
    assert e.is_loaded() is True


class CountingPool:
    """Two cards; the big tier does not fit, the small one does."""

    def __init__(self):
        self.enumerations = 0

    def get_device_status(self):
        self.enumerations += 1
        return [{"index": 0, "name": "small", "memory_total_mb": 8000,
                 "memory_free_mb": 7000},
                {"index": 1, "name": "big", "memory_total_mb": 12000,
                 "memory_free_mb": 11000}]

    def refresh_device(self, device_index=0):
        d = self.get_device_status_quiet()[device_index]
        return types.SimpleNamespace(name=d["name"], memory_free_mb=d["memory_free_mb"],
                                     memory_total_mb=d["memory_total_mb"],
                                     memory_used_mb=1000)

    def get_device_status_quiet(self):
        self.enumerations -= 1
        return self.get_device_status()

    def wait_for_vram_settled(self, *a, **k):
        return True

    def wait_for_vram(self, needed_mb, device_index=0, timeout=0, poll_interval=0):
        return needed_mb <= self.get_device_status_quiet()[device_index]["memory_free_mb"]

    def estimate_allocation_mb(self, model_path, num_ctx, kv_quant="f16",
                               extra_flags=None, device_index=0,
                               split_fraction_override=None,
                               compute_buffer_override_mb=None):
        frac = split_fraction_override if split_fraction_override is not None else 1.0
        return {"model_weight_mb": 14000 * frac, "kv_cache_mb": (num_ctx / 16) * frac,
                "compute_buffer_mb": compute_buffer_override_mb or 1000,
                "system_reserve_mb": 2048, "total_mb": 0}


def test_split_gate_enumerates_devices_once_per_load():
    e = LlamaCppEngine("m", {"path": "model.gguf", "num_ctx": 128000,
                             "extra_flags": ["-sm", "layer", "-ts", "5,11"]})
    e.set_target_tier(128000)
    pool = CountingPool()
    with patch("core.engine_llama_cpp.os.path.isfile", return_value=True), \
         patch("core.engine_llama_cpp.os.makedirs"), \
         patch("core.engine_llama_cpp.open", MagicMock(), create=True), \
         patch.object(e, "_is_server_healthy", return_value=False), \
         patch.object(e, "_reclaim_orphan_servers", return_value=0), \
         patch.object(e, "_resolve_gpu_index", return_value=0), \
         patch("core.gpu_pool.get_pool", return_value=pool), \
         patch("core.services.orpheus_tts.gpu_claimed"), \
         patch("core.engine_llama_cpp.spawn", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            e.load()
    assert e._num_ctx < 128000, "fixture should force at least one step-down"
    assert pool.enumerations == 1
    assert e._device_snapshot is None      # not kept past the load


def test_split_assignment_outside_a_load_always_asks_the_pool():
    e = LlamaCppEngine("m", {"path": "model.gguf", "extra_flags": ["-ts", "5,11"]})
    pool = CountingPool()
    e._split_assignment(pool, [5.0, 11.0])
    e._split_assignment(pool, [5.0, 11.0])
    assert pool.enumerations == 2


# ── #21: orphan scan uses the config's default_port ────────────────────

def test_orphan_scan_resolves_missing_ports_to_default_port():
    e = LlamaCppEngine("m", {"path": "model.gguf", "port": 8081})
    models = {"a": {}, "b": {"port": 8086}}
    with patch("core.config.get_llama_cpp_models", return_value=models), \
         patch("core.config._load_llama_cpp_config",
               return_value={"default_port": 9001, "models": models}), \
         patch("core.proc_tree.orphan_listeners", return_value=[]) as scan:
        e._reclaim_orphan_servers()
    assert scan.call_args[0][0] == {9001, 8086}


# ── #10: gate maths (estimate_allocation_mb, mocked sizes) ─────────────

def _bare_pool(devices=()):
    """A GPUPool that never probed anything (the real one is a singleton
    that runs nvidia-smi in __init__)."""
    from core.gpu_pool import GPUPool
    pool = object.__new__(GPUPool)
    pool._lock = threading.Lock()
    pool._devices = list(devices)
    pool._nvidia_smi_available = None
    pool._baseline_used_mb = {0: 2048.0, 1: 2048.0}
    pool._initialized = True
    return pool


MB = 1024 * 1024


def _make_shards(tmp_path, sizes_mb):
    n = len(sizes_mb)
    paths = []
    for i, size in enumerate(sizes_mb, 1):
        p = tmp_path / f"Model-Q4-{i:05d}-of-{n:05d}.gguf"
        with open(p, "wb") as f:
            f.truncate(size * MB)            # sparse: no real bytes written
        paths.append(str(p))
    return paths


def test_gguf_total_bytes_sums_every_shard(tmp_path):
    from core.gpu_pool import gguf_total_bytes
    shards = _make_shards(tmp_path, [40, 40, 25])
    assert gguf_total_bytes(shards[0]) == 105 * MB
    single = tmp_path / "Single.gguf"
    with open(single, "wb") as f:
        f.truncate(7 * MB)
    assert gguf_total_bytes(str(single)) == 7 * MB
    with pytest.raises(OSError):
        gguf_total_bytes(str(tmp_path / "missing.gguf"))


def test_gguf_total_bytes_tolerates_a_missing_later_shard(tmp_path):
    from core.gpu_pool import gguf_total_bytes
    shards = _make_shards(tmp_path, [40, 40, 25])
    os.remove(shards[1])
    assert gguf_total_bytes(shards[0]) == 65 * MB


def test_estimate_counts_all_shards_of_a_split_gguf(tmp_path):
    """The gate used to size a 3-shard model from shard 1 alone."""
    shards = _make_shards(tmp_path, [40, 40, 25])
    pool = _bare_pool()
    with patch("core.gpu_pool.read_gguf_kv_params", return_value=None):
        whole = pool.estimate_allocation_mb(shards[0], num_ctx=8192)
        half = pool.estimate_allocation_mb(shards[0], num_ctx=8192,
                                           split_fraction_override=0.5)
    assert whole["model_weight_mb"] == pytest.approx(105.0)
    assert half["model_weight_mb"] == pytest.approx(52.5)
    assert whole["estimation_method"] == "degraded_no_gguf"


ARCH = {"head_count": 16, "head_count_kv": 8, "key_dim": 128, "val_dim": 128,
        "block_count": 32, "full_attention_interval": 1, "attn_layer_count": 32}


def _kv_mb(bpe_k, bpe_v, ctx):
    return 32 * 8 * (128 * bpe_k + 128 * bpe_v) * ctx / MB


def test_estimate_sizes_values_by_ctv_not_by_ctk(tmp_path):
    from core.gpu_pool import KV_QUANT_BPE
    model = tmp_path / "m.gguf"
    with open(model, "wb") as f:
        f.truncate(10 * MB)
    pool = _bare_pool()
    q8, q4, f16 = KV_QUANT_BPE["q8_0"], KV_QUANT_BPE["q4_0"], KV_QUANT_BPE["f16"]
    with patch("core.gpu_pool.read_gguf_kv_params", return_value=dict(ARCH)):
        same = pool.estimate_allocation_mb(str(model), 65536, kv_quant="q8_0",
                                           extra_flags=["-ctk", "q8_0", "-ctv", "q8_0"])
        mixed = pool.estimate_allocation_mb(str(model), 65536, kv_quant="q8_0",
                                            extra_flags=["-ctk", "q8_0", "-ctv", "q4_0"])
        longf = pool.estimate_allocation_mb(str(model), 65536, kv_quant="q4_0",
                                            extra_flags=["--cache-type-v", "f16"])
        none = pool.estimate_allocation_mb(str(model), 65536, kv_quant="q8_0",
                                           extra_flags=["-fa", "on"])
    assert same["kv_cache_mb"] == pytest.approx(_kv_mb(q8, q8, 65536))
    assert mixed["kv_cache_mb"] == pytest.approx(_kv_mb(q8, q4, 65536))
    # the case that was UNDER-counted: big values, small keys
    assert longf["kv_cache_mb"] == pytest.approx(_kv_mb(q4, f16, 65536))
    assert longf["kv_cache_mb"] > _kv_mb(q4, q4, 65536)
    # no -ctv flag: unchanged behaviour (values sized like keys)
    assert none["kv_cache_mb"] == pytest.approx(_kv_mb(q8, q8, 65536))


def test_engine_reads_the_long_cache_type_k_flag():
    short = LlamaCppEngine("m", {"path": "x.gguf", "extra_flags": ["-ctk", "q8_0"]})
    long_ = LlamaCppEngine("m", {"path": "x.gguf",
                                 "extra_flags": ["--cache-type-k", "q4_0"]})
    assert short._get_kv_quant_str() == "q8_0"
    assert long_._get_kv_quant_str() == "q4_0"     # was assumed f16
    assert LlamaCppEngine("m", {"path": "x.gguf"})._get_kv_quant_str() == "f16"


def test_mmproj_offload_mb(tmp_path):
    from core.gpu_pool import mmproj_offload_mb
    proj = tmp_path / "mmproj.gguf"
    with open(proj, "wb") as f:
        f.truncate(9 * MB)
    assert mmproj_offload_mb(["--mmproj", str(proj)]) == pytest.approx(9.0)
    assert mmproj_offload_mb(["--mmproj", str(proj), "--no-mmproj-offload"]) == 0.0
    assert mmproj_offload_mb(["--mmproj", str(tmp_path / "gone.gguf")]) == 0.0
    assert mmproj_offload_mb(["-fa", "on"]) == 0.0
    assert mmproj_offload_mb(None) == 0.0


# ── #17: a silent nvidia-smi must not renumber the pool ────────────────

def test_transient_nvidia_smi_failure_keeps_cuda_devices_and_indices():
    pool = _bare_pool()
    answers = ["0, RTX Card, 8192, 7000, 1192, 8.9", None]
    adapters = [
        {"vendor_id": 0x1002, "description": "Radeon Card", "dedicated_mb": 12288.0,
         "free_mb": 11000.0, "usage_mb": 1288.0, "luid": "amd"},
        {"vendor_id": 0x10DE, "description": "RTX Card", "dedicated_mb": 8192.0,
         "free_mb": 7000.0, "usage_mb": 1192.0, "luid": "nv"},
    ]

    def smi(query):
        out = answers.pop(0)
        if out is not None:
            pool._nvidia_smi_available = True
        return out

    with patch.object(pool, "_run_nvidia_smi", side_effect=smi), \
         patch("core.dxgi_vram.probe_adapters", return_value=adapters):
        pool._enumerate_devices()
        before = [(d.index, d.name, d.source) for d in pool._devices]
        pool._enumerate_devices()          # nvidia-smi does not answer
        after = [(d.index, d.name, d.source) for d in pool._devices]
    assert before == [(0, "RTX Card", "cuda"), (1, "Radeon Card", "dxgi")]
    assert after == before
    # find_best_device feeds CUDA_VISIBLE_DEVICES: it must still be a CUDA index
    with patch.object(pool, "_enumerate_devices"):
        assert pool.find_best_device(0) == 0


def test_without_nvidia_smi_dxgi_still_covers_nvidia():
    pool = _bare_pool()
    pool._nvidia_smi_available = False     # the binary is not installed
    adapters = [{"vendor_id": 0x10DE, "description": "RTX Card", "dedicated_mb": 8192.0,
                 "free_mb": 7000.0, "usage_mb": 1192.0, "luid": "nv"}]
    with patch.object(pool, "_run_nvidia_smi", return_value=None), \
         patch("core.dxgi_vram.probe_adapters", return_value=adapters):
        pool._enumerate_devices()
    assert [(d.index, d.source) for d in pool._devices] == [(0, "dxgi")]


# ── #4: a media job stops OUR GPU model servers, nothing else ──────────

def _conn(port, pid, listening=True):
    import psutil
    return types.SimpleNamespace(
        laddr=types.SimpleNamespace(port=port), pid=pid,
        status=psutil.CONN_LISTEN if listening else "ESTABLISHED")


def _run_stop(models, conns, names, cfg_extra=None):
    import core.gpu_pool as gp
    cfg = dict({"default_port": 8081, "models": models}, **(cfg_extra or {}))

    def model_config(name):
        entry = dict(models[name])
        entry.setdefault("port", cfg["default_port"])
        return entry

    def process(pid):
        return types.SimpleNamespace(name=lambda: names[pid], exe=lambda: names[pid])

    with patch("core.config._load_llama_cpp_config", return_value=cfg), \
         patch("core.config.get_llama_cpp_models", return_value=models), \
         patch("core.config.get_llama_cpp_model_config", side_effect=model_config), \
         patch("psutil.net_connections", return_value=conns), \
         patch("psutil.Process", side_effect=process), \
         patch("core.proc_tree.stop_port_tree", return_value=len(conns)) as stop, \
         patch.object(gp.time, "sleep", lambda s: None):
        gp.stop_gpu_model_servers()
    return stop


def test_media_job_stops_only_configured_gpu_model_servers():
    models = {
        "main": {"path": "a.gguf"},                               # default port 8081
        "cpu-only": {"path": "b.gguf", "port": 8090, "num_gpu_layers": 0},
        "strata": {"external_launcher": ["launch"], "port": 8085},
        "shares-helper-port": {"path": "c.gguf", "port": 8095},
    }
    conns = [
        _conn(8081, 11),            # our llama-server                 -> stop
        _conn(8085, 12),            # external launcher's own server   -> stop
        _conn(8090, 13),            # CPU-only entry                   -> leave
        _conn(8095, 14),            # the vision helper                -> leave
        _conn(9000, 15),            # another project's llama-server   -> leave
        _conn(8081, 16, listening=False),   # a client socket, not a listener
    ]
    names = {11: "llama-server.exe", 12: "python.exe", 13: "llama-server.exe",
             14: "llama-server.exe", 15: "llama-server.exe", 16: "llama-server.exe"}
    stop = _run_stop(models, conns, names,
                     {"vision_helper": {"path": "v.gguf", "port": 8095}})
    stop.assert_called_once_with(None, [11, 12])


def test_a_stranger_on_a_llama_server_port_is_left_alone():
    stop = _run_stop({"main": {"path": "a.gguf"}}, [_conn(8081, 21)],
                     {21: "some-other-service.exe"})
    stop.assert_not_called()


def test_exclusive_gpu_no_longer_kills_every_llama_server():
    from core.model_queue import ModelQueue

    async def run():
        q = ModelQueue()
        q.register_transformers_unload(lambda: None)
        with patch("core.gpu_pool.stop_gpu_model_servers") as scoped, \
             patch("core.gpu_pool.GPUPool._kill_stale_servers") as machine_wide, \
             patch("core.model_queue.asyncio.sleep", new=_no_sleep):
            async with q.exclusive_gpu("image"):
                pass
        return scoped, machine_wide

    scoped, machine_wide = asyncio.run(run())
    scoped.assert_called_once()
    machine_wide.assert_not_called()


_real_sleep = asyncio.sleep


async def _no_sleep(_seconds):
    await _real_sleep(0)


# ── #1: the idle countdown runs from the END of the work ───────────────

def test_idle_shrink_does_not_unload_right_after_a_long_stream():
    import core.model_queue as mq

    async def run():
        q = mq.ModelQueue()
        unloads = []
        q.register_transformers_unload(lambda: unloads.append(1))
        q._current_model, q._current_backend = "m", "llama_cpp"
        q._current_ctx_tier = 32000
        with patch.object(mq, "IDLE_SHRINK_SEC", 600), \
             patch.object(mq, "IDLE_SHRINK_CHECK_INTERVAL", 0):
            # A stream that started 20 minutes ago still holds the queue.
            q._last_request_at = time.time() - 1200
            await q._lock.acquire()
            task = asyncio.ensure_future(q._idle_shrink_loop())
            for _ in range(20):
                await _real_sleep(0)
            assert not unloads
            q._lock.release()               # ... and now it finishes
            for _ in range(20):
                await _real_sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return unloads, q

    unloads, q = asyncio.run(run())
    assert not unloads, "engine was unloaded the moment the stream ended"
    assert q._current_model == "m"
    assert time.time() - q._last_request_at < 60


def test_idle_shrink_still_unloads_a_genuinely_idle_engine():
    import core.model_queue as mq

    async def run():
        q = mq.ModelQueue()
        unloads = []
        q.register_transformers_unload(lambda: unloads.append(1))
        q._current_model, q._current_backend = "m", "llama_cpp"
        q._last_request_at = time.time() - 1200
        with patch.object(mq, "IDLE_SHRINK_SEC", 600), \
             patch.object(mq, "IDLE_SHRINK_CHECK_INTERVAL", 0), \
             patch("core.model_queue.asyncio.sleep", new=_no_sleep):
            task = asyncio.ensure_future(q._idle_shrink_loop())
            for _ in range(200):
                await _real_sleep(0.001)
                if unloads and q._current_model is None:
                    break
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return unloads, q

    unloads, q = asyncio.run(run())
    assert unloads == [1] and q._current_model is None


def test_note_activity_restarts_the_countdown_only_when_loaded():
    from core.model_queue import ModelQueue
    q = ModelQueue()
    q.note_activity()
    assert q._last_request_at is None       # nothing loaded: stays off
    q._last_request_at = time.time() - 500
    q.note_activity()
    assert time.time() - q._last_request_at < 5


# ── #12: a broken config file is reported and does not empty the list ──

@pytest.fixture
def llama_cfg(tmp_path, monkeypatch):
    import core.config as config
    path = tmp_path / "llama_cpp_config.json"
    monkeypatch.setattr(config, "LLAMA_CPP_CONFIG_PATH", str(path))
    monkeypatch.setattr(config, "_llama_cpp_config", {})
    monkeypatch.setattr(config, "_llama_cpp_config_mtime", 0.0)
    return path


def _write(path, text, stamp):
    path.write_text(text, encoding="utf-8")
    os.utime(path, (stamp, stamp))          # distinct mtimes without sleeping


def test_json_error_keeps_the_last_good_config_and_logs_why(llama_cfg, caplog):
    import core.config as config
    _write(llama_cfg, '{"models": {"good": {"path": "a.gguf"}}}', 1_700_000_000)
    assert list(config.get_llama_cpp_models()) == ["good"]

    _write(llama_cfg, '{"models": {"good": {"path": "a.gguf"},}}', 1_700_000_100)
    with caplog.at_level(logging.ERROR, logger="core.config"):
        for _ in range(5):
            assert list(config.get_llama_cpp_models()) == ["good"]
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1, "reported once, not on every call"
    msg = errors[0].getMessage()
    assert "llama_cpp_config.json" in msg and "line 1" in msg

    _write(llama_cfg, '{"models": {"fixed": {"path": "b.gguf"}}}', 1_700_000_200)
    assert list(config.get_llama_cpp_models()) == ["fixed"]


def test_json_error_on_first_load_is_logged_too(llama_cfg, caplog):
    import core.config as config
    _write(llama_cfg, '{"models": ', 1_700_000_000)
    with caplog.at_level(logging.ERROR, logger="core.config"):
        assert config.get_llama_cpp_models() == {}
    assert any("llama_cpp_config.json" in r.getMessage() for r in caplog.records)


def test_config_cache_follows_the_path_not_only_the_mtime(tmp_path, monkeypatch):
    import core.config as config
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, '{"models": {"from-a": {}}}', 1_700_000_000)
    _write(b, '{"models": {"from-b": {}}}', 1_700_000_000)   # same timestamp
    monkeypatch.setattr(config, "_llama_cpp_config", {})
    monkeypatch.setattr(config, "_llama_cpp_config_mtime", 0.0)
    monkeypatch.setattr(config, "LLAMA_CPP_CONFIG_PATH", str(a))
    assert list(config.get_llama_cpp_models()) == ["from-a"]
    monkeypatch.setattr(config, "LLAMA_CPP_CONFIG_PATH", str(b))
    assert list(config.get_llama_cpp_models()) == ["from-b"]


# ── GPU tier: importing core.config must not import torch ──────────────

def test_importing_config_does_not_import_torch():
    code = ("import sys, core.config as c; "
            "assert 'torch' not in sys.modules, 'torch imported at config import'; "
            "from core.config import get_gpu_tier; "
            "print('ok')")
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONDONTWRITEBYTECODE="1")
    env.pop("ARTIFEX_BACKEND", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip() == "ok"


def test_lazy_tier_attributes_follow_the_detected_tier(monkeypatch):
    import core.config as config
    monkeypatch.setattr(config, "_gpu_tier_value", "ABUNDANT")
    monkeypatch.delitem(vars(config), "MODES", raising=False)
    monkeypatch.setattr(config, "_active_profile", None)
    from core.config import GPU_TIER, MODES            # the import form users rely on
    assert GPU_TIER == "ABUNDANT" and config.GPU_TIER == "ABUNDANT"
    assert MODES["ASSISTANT"].max_tokens == 4096
    assert MODES["CODE"].max_tokens == 8192
    assert config.MODES is MODES                        # built once
    assert config._L == {"mt": 4096, "cw": 15}
    assert config.get_context_profile_name() == "HIGH"
    assert config.get_context_profile() is config.CONTEXT_PROFILES["HIGH"]
    assert config.set_context_profile("STANDARD") is True
    assert config.get_context_profile_name() == "STANDARD"
    with pytest.raises(AttributeError):
        config.NO_SUCH_SETTING
    monkeypatch.delitem(vars(config), "MODES", raising=False)


def test_gpu_tier_is_detected_once(monkeypatch):
    import core.config as config
    monkeypatch.setattr(config, "_gpu_tier_value", None)
    calls = []
    monkeypatch.setattr(config, "_detect_gpu_tier",
                        lambda: calls.append(1) or "COMFORTABLE")
    assert [config.get_gpu_tier(), config.GPU_TIER, config.get_gpu_tier()] == \
        ["COMFORTABLE"] * 3
    assert len(calls) == 1


# ── #22: launcher, downloader, Ollama retry ────────────────────────────

def _snapshot_stub(monkeypatch, behaviour):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = behaviour
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)


def test_failed_download_restores_the_patched_config(tmp_path, monkeypatch):
    import download_model
    out = tmp_path / "model"
    out.mkdir()
    (out / "config.json").write_text("PATCHED", encoding="utf-8")

    def interrupted(repo_id, local_dir):
        (out / "config.json").write_text("FRESH", encoding="utf-8")
        raise ConnectionError("network dropped")

    _snapshot_stub(monkeypatch, interrupted)
    with pytest.raises(ConnectionError):
        download_model.download_hf_model("org/repo", str(out))
    assert (out / "config.json").read_text(encoding="utf-8") == "PATCHED"
    assert not (out / "config.json.dl_backup").exists()


def test_leftover_backup_is_not_overwritten_by_the_unpatched_config(tmp_path, monkeypatch):
    """State after a download that was killed: backup = patched config,
    config.json = the freshly downloaded one."""
    import download_model
    out = tmp_path / "model"
    out.mkdir()
    (out / "config.json").write_text("FRESH", encoding="utf-8")
    (out / "config.json.dl_backup").write_text("PATCHED", encoding="utf-8")
    _snapshot_stub(monkeypatch, lambda repo_id, local_dir: None)
    with patch("torch.cuda.is_available", return_value=False):
        download_model.download_hf_model("org/repo", str(out))
    assert (out / "config.json").read_text(encoding="utf-8") == "PATCHED"
    assert not (out / "config.json.dl_backup").exists()


def _run_main_api(monkeypatch, argv):
    pytest.importorskip("uvicorn")
    import main_api
    for var in ("ARTIFEX_API_KEY", "ARTIFEX_API_KEYS_FILE", "ARTIFEX_BACKEND", "ARTIFEX_MODEL",
                "ARTIFEX_PHONE_APP"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sys, "argv", ["main_api.py"] + argv)
    started = MagicMock()
    monkeypatch.setattr(main_api, "_run_server", started, raising=False)
    monkeypatch.setattr(main_api, "setup_logging", lambda: None)
    main_api.main()
    return started


@pytest.mark.parametrize("host", ["localhost", "::1", "127.0.0.1"])
def test_loopback_hosts_need_no_api_key(monkeypatch, host):
    assert _run_main_api(monkeypatch, ["--host", host]).called


def test_network_bind_without_a_key_is_still_refused(monkeypatch, capsys):
    assert not _run_main_api(monkeypatch, ["--host", "0.0.0.0"]).called
    assert "ARTIFEX_API_KEY" in capsys.readouterr().out


def test_reload_hands_uvicorn_an_import_string(monkeypatch):
    import main_api
    monkeypatch.delenv("ARTIFEX_BACKEND", raising=False)
    monkeypatch.delenv("WEB_GATEWAY_URL", raising=False)
    uv = MagicMock()
    args = types.SimpleNamespace(reload=True, host="127.0.0.1", port=8000,
                                 backend="llama_cpp", gateway=None, model=None)
    main_api._run_server(uv, args)
    (target,), kwargs = uv.run.call_args
    assert target == "api.server:create_app"
    assert kwargs["factory"] is True and kwargs["reload"] is True
    # the worker process rebuilds the app: the backend choice travels by env
    assert os.environ["ARTIFEX_BACKEND"] == "llama_cpp"


def test_model_not_found_warning_lists_the_active_backends_models(monkeypatch):
    import main_api
    monkeypatch.setattr("core.config.get_model_names", lambda: ["qwen-27b", "lfm3"])
    assert main_api._available_model_names() == ["qwen-27b", "lfm3"]


def test_ollama_thinking_retry_keeps_the_callers_arguments():
    from core.engine_ollama import OllamaEngine
    e = OllamaEngine("m")
    e._loaded = True
    unsupported = urllib.error.HTTPError(
        "http://x", 400, "Bad Request", {},
        io.BytesIO(b'{"error":"m does not support thinking"}'))
    body = json.dumps({"message": {"content": TRANSCRIPT}, "done": True}).encode()
    bodies = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        if len(bodies) == 1:
            raise unsupported
        return FakeResponse([body + b"\n"])

    with patch("core.engine_ollama.urllib.request.urlopen", fake_urlopen), \
         patch("core.ollama_ctx.compute_safe_ctx", return_value=4096), \
         patch("core.ollama_ctx.should_disable_mmap", return_value=False), \
         patch("core.ollama_ctx.estimate_prompt_tokens", return_value=10):
        text = e.generate_streaming([{"role": "user", "content": "hi"}], 64, 0.7,
                                    raw_output=True, sampling={"top_k": 7})
    assert len(bodies) == 2 and bodies[1]["think"] is False
    assert text == TRANSCRIPT                      # raw_output survived the retry
    assert bodies[1]["options"]["top_k"] == 7      # ... and so did the sampler
