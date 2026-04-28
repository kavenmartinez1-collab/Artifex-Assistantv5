"""Tests for WebGPU engine constants — verify host-side limits match shader limits.

Full kernel-level tests require Playwright + headless Chrome (Task 12/P2).
This file runs in CI without a GPU and catches constant drift.
"""

import re
import os
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WGSL_PATH = os.path.join(REPO_ROOT, "webgpu", "src", "shaders", "attention.wgsl")
TS_PATH = os.path.join(REPO_ROOT, "webgpu", "src", "engine", "forward-pass.ts")


@pytest.mark.skipif(
    not os.path.exists(WGSL_PATH), reason="WebGPU source not present"
)
class TestAttentionCacheLenLimit:
    """The workgroup scores array size in attention.wgsl must match MAX_ATTN_CACHE in forward-pass.ts."""

    def _read_wgsl_scores_size(self):
        with open(WGSL_PATH, "r") as f:
            text = f.read()
        match = re.search(r'var<workgroup>\s+scores:\s+array<f32,\s*(\d+)>', text)
        assert match, "Could not find scores array declaration in attention.wgsl"
        return int(match.group(1))

    def _read_ts_max_attn_cache(self):
        with open(TS_PATH, "r") as f:
            text = f.read()
        match = re.search(r'const\s+MAX_ATTN_CACHE\s*=\s*(\d+)', text)
        assert match, "Could not find MAX_ATTN_CACHE constant in forward-pass.ts"
        return int(match.group(1))

    def test_constants_match(self):
        wgsl_size = self._read_wgsl_scores_size()
        ts_limit = self._read_ts_max_attn_cache()
        assert wgsl_size == ts_limit, (
            f"WGSL scores array ({wgsl_size}) != TS MAX_ATTN_CACHE ({ts_limit}). "
            f"Update forward-pass.ts to match the shader."
        )

    def test_host_clamp_present(self):
        """Verify that dispatchAttention contains the bounds check."""
        with open(TS_PATH, "r") as f:
            text = f.read()
        assert "cacheLen > MAX_ATTN_CACHE" in text, (
            "Missing cacheLen bounds check in forward-pass.ts dispatchAttention"
        )

    def test_host_clamp_present_tq(self):
        """Verify that dispatchAttentionTQ also contains the bounds check."""
        with open(TS_PATH, "r") as f:
            text = f.read()
        # Both dispatch functions should have the clamp
        occurrences = text.count("cacheLen > MAX_ATTN_CACHE")
        assert occurrences >= 2, (
            f"Expected cacheLen clamp in both dispatchAttention and dispatchAttentionTQ, "
            f"found {occurrences} occurrence(s)"
        )
