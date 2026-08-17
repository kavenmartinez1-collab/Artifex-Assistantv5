"""Tests for the multi-GPU VRAM gate.

The failure these guard against: on a -ts split, the gate only checked
the nvidia-smi-visible card's share, so the other card's (larger) share
loaded unguarded and OOM'd — e.g. an 11/16 share on a busy card.
"""

import sys

import pytest

from core.engine_llama_cpp import LlamaCppEngine, CTX_TIERS


class FakePool:
    """GPU pool double: fixed devices, deterministic free VRAM."""

    def __init__(self, devices):
        # devices: [(index, name, total_mb, free_mb)]
        self._devices = devices

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
                return d
        return None

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
        # Weight model: 16000 MB total; KV: 1 MB per 64 tokens.
        frac = (split_fraction_override
                if split_fraction_override is not None else 1.0)
        comp = (compute_buffer_override_mb
                if compute_buffer_override_mb is not None else 1000)
        return {
            "model_weight_mb": 16000 * frac,
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
        # small card needs 16000*5/16 + kv + 1000 ≈ 6300; big needs ≈ 11800
        pool = FakePool([(0, "small", 8151, 7100), (1, "big", 12242, 12200)])
        e = _engine(["-ts", "5,11"])
        e._num_ctx = 64000
        e._vram_gate(pool, 0, wait_timeout=1)   # must not raise
        assert len(e._gate_requirements) == 2

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
