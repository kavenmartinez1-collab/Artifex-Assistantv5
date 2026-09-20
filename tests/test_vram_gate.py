"""Tests for the multi-GPU VRAM gate.

The failure these guard against: on a -ts split, the gate only checked
the nvidia-smi-visible card's share, so the other card's (larger) share
loaded unguarded and OOM'd — e.g. an 11/16 share on a busy card.
"""

import sys

import pytest

from core.engine_llama_cpp import LlamaCppEngine, CTX_TIERS

# Stand-in model: roughly the 27B Q4_K_S that every split number here was
# measured against, so the fixtures stay inside the real cards' totals.
FAKE_WEIGHT_MB = 15400


class FakePool:
    """GPU pool double: fixed devices, deterministic free VRAM."""

    def __init__(self, devices):
        # devices: [(index, name, total_mb, free_mb)]
        self._devices = devices
        self._used = {}

    def get_device_status(self):
        return [{"index": i, "name": n, "memory_total_mb": t,
                 "memory_free_mb": f} for i, n, t, f in self._devices]

    def refresh_device(self, device_index):
        for i, n, t, f in self._devices:
            if i == device_index:
                class D:
                    pass
                d = D()
                d.name, d.memory_free_mb, d.memory_total_mb = n, f, t
                d.memory_used_mb = self._used.get(i, t - f)
                return d
        return None

    def set_used(self, device_index, used_mb):
        """Stand in for a completed load: usage after the server started."""
        self._used[device_index] = used_mb

    def wait_for_vram(self, needed_mb, device_index=0, timeout=0,
                      poll_interval=0):
        for i, _, _, free in self._devices:
            if i == device_index:
                return free >= needed_mb
        return True

    def estimate_allocation_mb(self, model_path, num_ctx, kv_quant="f16",
                               extra_flags=None, device_index=0,
                               split_fraction_override=None,
                               compute_buffer_override_mb=None):
        # Weight model: FAKE_WEIGHT_MB total; KV: 1 MB per 64 tokens.
        frac = (split_fraction_override
                if split_fraction_override is not None else 1.0)
        comp = (compute_buffer_override_mb
                if compute_buffer_override_mb is not None else 1000)
        return {
            "model_weight_mb": FAKE_WEIGHT_MB * frac,
            "kv_cache_mb": (num_ctx / 64) * frac,
            "compute_buffer_mb": comp,
            "system_reserve_mb": 2048,
            "total_mb": 0,
        }


def _engine(extra_flags, split_gpu_indices=None, tmp_path=None):
    cfg = {
        "path": str(tmp_path / "model.gguf") if tmp_path else "missing.gguf",
        "port": 9999,
        "extra_flags": extra_flags,
        "num_ctx": 64000,
    }
    if split_gpu_indices is not None:
        cfg["split_gpu_indices"] = split_gpu_indices
    return LlamaCppEngine("gate-test", cfg)


class TestParseTsShares:
    def test_ts_flag(self):
        e = _engine(["-sm", "layer", "-ts", "5,11"])
        assert e._parse_ts_shares() == [5.0, 11.0]

    def test_long_flag_and_slash(self):
        e = _engine(["--tensor-split", "1/3"])
        assert e._parse_ts_shares() == [1.0, 3.0]

    def test_absent(self):
        assert _engine(["-fa", "on"])._parse_ts_shares() == []

    def test_garbage(self):
        assert _engine(["-ts", "a,b"])._parse_ts_shares() == []


class TestSplitAssignment:
    def test_greedy_biggest_share_to_biggest_card(self):
        # device 0 = small card, device 1 = big card
        pool = FakePool([(0, "small", 8000, 7000), (1, "big", 12000, 11000)])
        e = _engine(["-ts", "5,11"])
        assignment = e._split_assignment(pool, [5.0, 11.0])
        # share pos 0 (5/16) → small card 0; share pos 1 (11/16) → big card 1
        assert assignment == [(0, 0, 5 / 16), (1, 1, 11 / 16)]

    def test_greedy_reversed_device_order(self):
        # big card enumerated FIRST — biggest share must still land on it
        pool = FakePool([(0, "big", 12000, 11000), (1, "small", 8000, 7000)])
        e = _engine(["-ts", "5,11"])
        assignment = e._split_assignment(pool, [5.0, 11.0])
        assert assignment == [(0, 1, 5 / 16), (1, 0, 11 / 16)]

    def test_config_override_wins(self):
        pool = FakePool([(0, "a", 12000, 11000), (1, "b", 8000, 7000)])
        e = _engine(["-ts", "5,11"], split_gpu_indices=[0, 1])
        assignment = e._split_assignment(pool, [5.0, 11.0])
        assert assignment == [(0, 0, 5 / 16), (1, 1, 11 / 16)]

    def test_too_few_devices_returns_none(self):
        pool = FakePool([(0, "only", 8000, 7000)])
        e = _engine(["-ts", "5,11"])
        assert e._split_assignment(pool, [5.0, 11.0]) is None


class TestVramGate:
    def test_split_passes_when_both_fit(self):
        # small card needs 15400*5/16 + kv + 750 ≈ 5900; big needs ≈ 12000
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)   # must not raise
        assert len(e._gate_requirements) == 2

    def test_split_charges_every_device_half_a_buffer_plus_headroom(self):
        # The policy, pinned: on a split no device holds all the layers, so
        # none is charged the full single-device compute buffer — and each
        # must keep SPLIT_PREFILL_HEADROOM_MB free on top for prefill's
        # batch-sized buffers.  Same number on both cards: 500 + 250.
        from core.gpu_pool import COMPUTE_BUFFER_MB, SPLIT_PREFILL_HEADROOM_MB
        per_device = COMPUTE_BUFFER_MB // 2 + SPLIT_PREFILL_HEADROOM_MB
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)
        need = dict(e._gate_requirements)
        for dev_idx, share in ((0, 5 / 16), (1, 11 / 16)):
            assert need[dev_idx] == pytest.approx(
                (FAKE_WEIGHT_MB + 64000 / 64) * share + per_device)

    def test_split_primary_not_charged_full_single_device_buffer(self):
        # Regression: charging the primary the whole COMPUTE_BUFFER_MB put a
        # measured-good config 71 MB over an 8 GB card and stepped every load
        # down a tier.  This card fits half a buffer plus headroom (5875) but
        # not the old flat 1000 (6125).
        pool = FakePool([(0, "small", 8151, 6000), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)   # must not raise

    def test_split_refused_when_only_the_prefill_headroom_is_missing(self):
        # Fits weights+KV+graph with 100 MB to spare — which is exactly the
        # state that loads fully resident and then prefills out of shared
        # system memory.  Refuse it so the caller can step down instead.
        from core.gpu_pool import SPLIT_PREFILL_HEADROOM_MB
        fits_without_headroom = (FAKE_WEIGHT_MB + 64000 / 64) * 11 / 16 + 500
        pool = FakePool([
            (0, "small", 8151, 7100),
            (1, "big", 12242, fits_without_headroom + 100),
        ])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        with pytest.raises(RuntimeError) as exc:
            e._vram_gate(pool, 0, wait_timeout=1)
        assert "GPU 1" in str(exc.value)
        assert SPLIT_PREFILL_HEADROOM_MB > 100   # the refusal is the headroom

    def test_single_device_gets_no_headroom(self):
        # The headroom is a split-only policy: it was measured on a -ts
        # split, and adding it to single-device configs would silently
        # tighten every other entry in the config.
        pool = FakePool([(0, "only", 24000, 20000)])
        e = _engine(["-fa", "on"])
        e._num_ctx = 32000
        e._vram_gate(pool, 0, wait_timeout=1)
        assert e._gate_requirements[0][1] == pytest.approx(
            FAKE_WEIGHT_MB + 32000 / 64 + 1000)

    def test_gate_records_resident_floor_and_baseline(self):
        # The floor is weights+KV WITHOUT the compute buffer: those are the
        # allocations that have to be in VRAM for the model to run at
        # speed, and they are what the post-load spill check compares
        # against.  Compute buffers are deliberately excluded — that term
        # is a flat guess (and a deliberately low one on secondaries), so
        # folding it in would make the check fire on estimate error.
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)
        assert e._resident_floors[1] == pytest.approx(
            FAKE_WEIGHT_MB * 11 / 16 + (64000 / 64) * 11 / 16)
        assert e._pre_launch_used_mb[1] == 12242 - 12200

    def test_split_refused_when_big_card_busy(self):
        # A browser holding ~2 GB of the big card dooms the 11/16 share.
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 10200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        with pytest.raises(RuntimeError) as exc:
            e._vram_gate(pool, 0, wait_timeout=1)
        assert "GPU 1" in str(exc.value)
        assert "big" in str(exc.value)

    def test_smaller_tier_fits_where_big_refused(self):
        # Same busy card: 64k refused above, 32k must pass — this is the
        # tier the load() downshift loop lands on.
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 11900)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        with pytest.raises(RuntimeError):
            e._vram_gate(pool, 0, wait_timeout=1)
        lower = max(t for t in CTX_TIERS if t < 64000)
        e._num_ctx = lower
        e._vram_gate(pool, 0, wait_timeout=1)   # must not raise

    def test_spill_warns_when_residency_lands_below_the_floor(self, caplog):
        # The real failure: WDDM accepted the allocation, demoted ~950 MB
        # of it to shared system memory, and llama-server came up healthy
        # and slow.  Residency is the only local signal — the card ends up
        # holding LESS than weights+KV.
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)
        floor = e._resident_floors[1]
        pool.set_used(1, (12242 - 12200) + floor - 950)
        with caplog.at_level("WARNING"):
            e._check_resident_after_load(pool)
        assert "VRAM SPILL on GPU 1" in caplog.text
        assert "big" in caplog.text

    def test_no_spill_warning_when_fully_resident(self, caplog):
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)
        for dev_idx, floor in e._resident_floors.items():
            total = next(t for i, _, t, _ in pool._devices if i == dev_idx)
            free = next(f for i, _, _, f in pool._devices if i == dev_idx)
            pool.set_used(dev_idx, (total - free) + floor + 300)
        with caplog.at_level("WARNING"):
            e._check_resident_after_load(pool)
        assert "SPILL" not in caplog.text

    def test_small_shortfall_is_tolerated(self, caplog):
        # The weight term comes from FILE size, which counts tensors
        # llama.cpp may discard (an unused MTP/nextn block is ~300 MB on a
        # 27B).  A shortfall inside SPILL_TOLERANCE_MB is that, not a spill.
        from core.engine_llama_cpp import SPILL_TOLERANCE_MB
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)
        for dev_idx, floor in e._resident_floors.items():
            total = next(t for i, _, t, _ in pool._devices if i == dev_idx)
            free = next(f for i, _, _, f in pool._devices if i == dev_idx)
            pool.set_used(dev_idx,
                          (total - free) + floor - (SPILL_TOLERANCE_MB - 1))
        with caplog.at_level("WARNING"):
            e._check_resident_after_load(pool)
        assert "SPILL" not in caplog.text

    def test_single_device_path_without_split(self):
        pool = FakePool([(0, "only", 24000, 20000)])
        e = _engine(["-fa", "on"])
        e._num_ctx = 32000
        e._vram_gate(pool, 0, wait_timeout=1)
        assert e._gate_requirements[0][0] == 0

    def test_single_device_refusal(self):
        pool = FakePool([(0, "only", 24000, 9000)])
        e = _engine(["-fa", "on"])
        e._num_ctx = 32000
        with pytest.raises(RuntimeError):
            e._vram_gate(pool, 0, wait_timeout=1)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only APIs")
class TestLiveProbes:
    def test_dxgi_enumerates_hardware(self):
        from core.dxgi_vram import probe_adapters
        adapters = probe_adapters()
        # On any Windows box with a GPU this should find at least one
        # hardware adapter with sane totals.
        for a in adapters:
            assert a["dedicated_mb"] > 0
            assert 0 <= a["usage_mb"] <= a["dedicated_mb"] * 1.05
            assert ":" in a["luid"]

    def test_pdh_luids_match_dxgi(self):
        from core.dxgi_vram import probe_adapters, pdh_dedicated_usage_mb
        adapters = probe_adapters()
        usage = pdh_dedicated_usage_mb()
        if not adapters or not usage:
            pytest.skip("no adapters/counters on this machine")
        # Every DXGI adapter should have a PDH counter instance.
        matched = [a for a in adapters if a["luid"] in usage]
        assert matched, f"no LUID overlap: {[a['luid'] for a in adapters]} vs {list(usage)}"


class TestPoolOverrides:
    def test_split_fraction_override_skips_ts_inference(self, tmp_path):
        from core.gpu_pool import GPUPool
        model = tmp_path / "m.gguf"
        model.write_bytes(b"\x00" * (100 * 1024 * 1024))  # 100 MB, no header
        pool = GPUPool()
        est = pool.estimate_allocation_mb(
            str(model), 8192, extra_flags=["-ts", "1,3"],
            split_fraction_override=0.25,
            compute_buffer_override_mb=500)
        assert est["model_weight_mb"] == pytest.approx(25.0, rel=0.01)
        assert est["compute_buffer_mb"] == 500


class TestWaitForVramSettled:
    """Reclaim settling: free MB says "there is room", not "the driver is
    done".  Launching into a half-drained card is what puts part of a model
    in shared system memory (measured: 35 → 12 tok/s on a model switch).
    """

    def _pool(self, free_sequence):
        """Scripted stand-in for GPUPool.

        Not a subclass: GPUPool.__new__ hands back a process-wide
        singleton, so subclassing it silently returns the real pool.
        Bind the real method to a stub instead.
        """
        from core.gpu_pool import GPUPool

        class ScriptedPool:
            wait_for_vram_settled = GPUPool.wait_for_vram_settled

            def __init__(self):
                self.polls = 0
                self._seq = list(free_sequence)

            def refresh_device(self, device_index=0):
                class D:
                    pass
                d = D()
                d.name = "card"
                d.memory_free_mb = self._seq[min(self.polls,
                                                 len(self._seq) - 1)]
                d.memory_total_mb = 12242
                d.memory_used_mb = 12242 - d.memory_free_mb
                self.polls += 1
                return d

        return ScriptedPool()

    def test_returns_once_free_stops_moving(self):
        # Reclaim in progress for three polls, then flat.
        pool = self._pool([4000, 8000, 11000, 12100, 12100, 12100])
        assert pool.wait_for_vram_settled([0], timeout=10, poll_interval=0)
        # Settles on the second of the two flat polls, not on the first
        # flat reading — one flat sample can land mid-drain.
        assert pool.polls == 6

    def test_already_idle_settles_immediately(self):
        pool = self._pool([12100])
        assert pool.wait_for_vram_settled([0], timeout=10, poll_interval=0)
        assert pool.polls == 3

    def test_timeout_returns_false_and_does_not_raise(self):
        # Never settles — caller must still be allowed to launch.
        pool = self._pool([4000 + 100 * i for i in range(200)])
        assert pool.wait_for_vram_settled([0], timeout=0, poll_interval=0) is False

    def test_no_devices_is_not_an_error(self):
        pool = self._pool([12100])
        assert pool.wait_for_vram_settled([], timeout=10, poll_interval=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
