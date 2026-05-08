"""Tests for ctx-tier launch logic and live VRAM baseline measurement.

Covers:
  - core.engine_llama_cpp.pick_ctx_tier — bucket selection with headroom and cap
  - LlamaCppEngine.set_target_tier / _compute_num_ctx — priority order
  - core.model_queue.ModelQueue — tier change triggers relaunch
  - core.gpu_pool.GPUPool.measure_baseline — floor + caching + force refresh
"""

import asyncio
from unittest.mock import patch, MagicMock

import pytest


# ── pick_ctx_tier ──────────────────────────────────────────────────────


class TestPickCtxTier:
    """Tier picker chooses smallest bucket fitting need + headroom, capped."""

    def test_small_request_lands_in_smallest_tier(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # 4K need + 8K headroom = 12K → 32K is smallest tier ≥ 12K
        assert pick_ctx_tier(4_000) == 32_000

    def test_request_near_tier_boundary_snaps_up(self):
        from core.engine_llama_cpp import pick_ctx_tier, TIER_HEADROOM_TOK
        # Need exactly at boundary minus headroom should still land in next tier
        assert pick_ctx_tier(32_000 - TIER_HEADROOM_TOK + 1) == 64_000

    def test_request_well_within_tier_stays(self):
        from core.engine_llama_cpp import pick_ctx_tier, TIER_HEADROOM_TOK
        # 36K need + 8K headroom = 44K → 64K bucket
        assert pick_ctx_tier(36_000) == 64_000

    def test_medium_request_picks_64k(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # 50K + 8K = 58K → 64K
        assert pick_ctx_tier(50_000) == 64_000

    def test_large_request_picks_128k(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # 100K + 8K = 108K → 128K
        assert pick_ctx_tier(100_000) == 128_000

    def test_very_large_request_picks_256k(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # 200K + 8K = 208K → 256K
        assert pick_ctx_tier(200_000) == 256_000

    def test_overflow_returns_largest_tier(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # Anything bigger than 256K cannot fit any tier — return largest
        assert pick_ctx_tier(500_000) == 256_000

    def test_cap_clamps_selection(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # 100K need would normally pick 128K, but capped at 64K → 64K
        assert pick_ctx_tier(100_000, max_cap=64_000) == 64_000

    def test_cap_below_smallest_tier_returns_smallest(self):
        from core.engine_llama_cpp import pick_ctx_tier
        # Degenerate cap below the smallest tier — picker still returns
        # something usable (the smallest tier) rather than failing.
        assert pick_ctx_tier(10_000, max_cap=20_000) == 32_000

    def test_zero_or_negative_input_clamped(self):
        from core.engine_llama_cpp import pick_ctx_tier
        assert pick_ctx_tier(0) == 32_000
        assert pick_ctx_tier(-5) == 32_000


# ── set_target_tier / _compute_num_ctx ─────────────────────────────────


class TestEngineTargetTier:
    """set_target_tier sets the launch ctx; _compute_num_ctx prefers it."""

    def _make_engine(self, num_ctx=None):
        from core.engine_llama_cpp import LlamaCppEngine
        cfg = {"path": "/fake/model.gguf"}
        if num_ctx is not None:
            cfg["num_ctx"] = num_ctx
        return LlamaCppEngine("test", cfg)

    def test_set_target_tier_stores_value(self):
        engine = self._make_engine(num_ctx=262_144)
        engine.set_target_tier(64_000)
        assert engine._target_ctx == 64_000

    def test_set_target_tier_caps_at_configured(self):
        engine = self._make_engine(num_ctx=128_000)
        engine.set_target_tier(256_000)
        assert engine._target_ctx == 128_000

    def test_set_target_tier_no_cap_when_unconfigured(self):
        engine = self._make_engine(num_ctx=None)
        engine.set_target_tier(256_000)
        assert engine._target_ctx == 256_000

    def test_set_target_tier_ignores_invalid(self):
        engine = self._make_engine(num_ctx=262_144)
        engine.set_target_tier(0)
        assert engine._target_ctx is None
        engine.set_target_tier(-100)
        assert engine._target_ctx is None
        engine.set_target_tier("64000")  # type: ignore[arg-type]
        assert engine._target_ctx is None

    def test_compute_num_ctx_prefers_target(self):
        engine = self._make_engine(num_ctx=262_144)
        engine.set_target_tier(64_000)
        assert engine._compute_num_ctx() == 64_000

    def test_compute_num_ctx_falls_back_to_configured(self):
        engine = self._make_engine(num_ctx=131_072)
        # No target set
        assert engine._compute_num_ctx() == 131_072

    def test_current_tier_zero_when_unloaded(self):
        engine = self._make_engine(num_ctx=262_144)
        assert engine.current_tier() == 0

    def test_current_tier_reports_loaded_ctx(self):
        engine = self._make_engine(num_ctx=262_144)
        engine._num_ctx = 64_000  # simulate post-load state
        assert engine.current_tier() == 64_000


# ── ModelQueue tier-change detection ───────────────────────────────────


class TestModelQueueCtxTier:
    """Queue treats a ctx_tier change as a relaunch trigger."""

    def _fresh_queue(self):
        # Don't use the singleton — we want isolated state per test.
        from core.model_queue import ModelQueue
        return ModelQueue()

    def test_first_call_records_tier(self):
        q = self._fresh_queue()
        unload_calls = []
        q._engine_unload_fn = lambda: unload_calls.append("unload")

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"):
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=64_000)

        asyncio.run(run())
        assert q.current_ctx_tier == 64_000
        assert unload_calls == []  # nothing to unload yet

    def test_same_model_same_tier_no_unload(self):
        q = self._fresh_queue()
        unload_calls = []
        q._engine_unload_fn = lambda: unload_calls.append("unload")

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"):
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=64_000)
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=64_000)

        asyncio.run(run())
        assert unload_calls == []

    def test_tier_change_triggers_unload(self):
        q = self._fresh_queue()
        unload_calls = []
        q._engine_unload_fn = lambda: unload_calls.append("unload")

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"):
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=64_000)
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=128_000)

        asyncio.run(run())
        # Only the second call should unload — the first was a fresh load.
        assert unload_calls == ["unload"]
        assert q.current_ctx_tier == 128_000
        # Both calls count as state transitions: None→m1, then tier 64→128.
        assert q._stats["model_switches"] == 2

    def test_model_change_triggers_unload_regardless_of_tier(self):
        q = self._fresh_queue()
        unload_calls = []
        q._engine_unload_fn = lambda: unload_calls.append("unload")

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"):
                await q.switch_if_needed("m1", "llama_cpp", ctx_tier=64_000)
                await q.switch_if_needed("m2", "llama_cpp", ctx_tier=64_000)

        asyncio.run(run())
        assert unload_calls == ["unload"]

    def test_ollama_ignores_tier_changes(self):
        q = self._fresh_queue()
        ollama_unload_calls = []

        async def fake_ollama_unload(model):
            ollama_unload_calls.append(model)

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"), \
                 patch.object(q, "_unload_ollama", new=fake_ollama_unload):
                # Ollama doesn't have a fixed launch ctx — a ctx_tier change
                # for the same Ollama model should NOT trigger any unload.
                await q.switch_if_needed("m1", "ollama", ctx_tier=64_000)
                await q.switch_if_needed("m1", "ollama", ctx_tier=128_000)

        asyncio.run(run())
        assert ollama_unload_calls == []

    def test_cross_backend_swap_clears_engine_cache(self):
        """Switching from ollama to llama_cpp must call _unload_engine
        in addition to the ollama HTTP unload.  Without this, the api
        layer's _engine global stays as the OllamaEngine instance and
        the next request silently runs against Ollama instead of the
        new backend."""
        q = self._fresh_queue()
        ollama_unload_calls = []
        engine_unload_calls = []
        q._engine_unload_fn = lambda: engine_unload_calls.append("engine")

        async def fake_ollama_unload(model):
            ollama_unload_calls.append(model)

        async def run():
            with patch("core.config.set_active_backend"), \
                 patch("core.config.set_active_model"), \
                 patch.object(q, "_unload_ollama", new=fake_ollama_unload):
                await q.switch_if_needed("m_olla", "ollama", ctx_tier=None)
                await q.switch_if_needed("m_lcpp", "llama_cpp", ctx_tier=64_000)

        asyncio.run(run())
        assert ollama_unload_calls == ["m_olla"]
        assert engine_unload_calls == ["engine"]


# ── GPU baseline measurement ───────────────────────────────────────────


class TestVramBaseline:
    """measure_baseline floors at VRAM_BASELINE_FLOOR_MB and caches."""

    def _fresh_pool(self):
        # Reset the singleton so each test starts clean.
        from core.gpu_pool import GPUPool
        GPUPool._instance = None
        pool = GPUPool()
        return pool

    def test_floor_when_measured_is_low(self):
        from core.gpu_pool import VRAM_BASELINE_FLOOR_MB
        pool = self._fresh_pool()
        fake_dev = MagicMock(memory_used_mb=200.0)
        with patch.object(pool, "refresh_device", return_value=fake_dev):
            baseline = pool.measure_baseline(force=True)
        assert baseline == float(VRAM_BASELINE_FLOOR_MB)

    def test_uses_measured_when_above_floor(self):
        pool = self._fresh_pool()
        fake_dev = MagicMock(memory_used_mb=3376.0)
        with patch.object(pool, "refresh_device", return_value=fake_dev):
            baseline = pool.measure_baseline(force=True)
        assert baseline == 3376.0

    def test_falls_back_to_floor_when_smi_unavailable(self):
        from core.gpu_pool import VRAM_BASELINE_FLOOR_MB
        pool = self._fresh_pool()
        with patch.object(pool, "refresh_device", return_value=None):
            baseline = pool.measure_baseline(force=True)
        assert baseline == float(VRAM_BASELINE_FLOOR_MB)

    def test_cached_after_first_call(self):
        pool = self._fresh_pool()
        fake_dev = MagicMock(memory_used_mb=2500.0)
        call_count = [0]

        def counting_refresh(_):
            call_count[0] += 1
            return fake_dev

        with patch.object(pool, "refresh_device", side_effect=counting_refresh):
            pool.measure_baseline()
            pool.measure_baseline()
            pool.measure_baseline()
        assert call_count[0] == 1

    def test_force_bypasses_cache(self):
        pool = self._fresh_pool()
        first_dev = MagicMock(memory_used_mb=2000.0)
        second_dev = MagicMock(memory_used_mb=4000.0)

        with patch.object(pool, "refresh_device", return_value=first_dev):
            assert pool.measure_baseline(force=True) == 2000.0
        with patch.object(pool, "refresh_device", return_value=second_dev):
            assert pool.measure_baseline(force=True) == 4000.0

    def test_estimate_uses_baseline_for_reserve(self):
        """estimate_allocation_mb should pick max(SYSTEM_RESERVE_MB, baseline)."""
        from core.gpu_pool import SYSTEM_RESERVE_MB
        pool = self._fresh_pool()
        # Baseline well above the static reserve
        high_baseline = float(SYSTEM_RESERVE_MB) + 1500.0
        fake_dev = MagicMock(memory_used_mb=high_baseline)
        with patch.object(pool, "refresh_device", return_value=fake_dev), \
             patch("core.gpu_pool.read_gguf_kv_params", return_value=None), \
             patch("os.path.getsize", return_value=1024 * 1024 * 100):  # 100 MB
            result = pool.estimate_allocation_mb("/fake.gguf", num_ctx=8192)
        assert result["system_reserve_mb"] == high_baseline


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
