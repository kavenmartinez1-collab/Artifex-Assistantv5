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
        # The literal lives in the exported MAX_ATTN_SEQ_LEN (KV sessions clamp
        # to it); the dispatch guards keep MAX_ATTN_CACHE as a local alias.
        match = re.search(r'const\s+MAX_ATTN_SEQ_LEN\s*=\s*(\d+)', text)
        assert match, "Could not find MAX_ATTN_SEQ_LEN constant in forward-pass.ts"
        return int(match.group(1))

    def test_constants_match(self):
        wgsl_size = self._read_wgsl_scores_size()
        ts_limit = self._read_ts_max_attn_cache()
        assert wgsl_size == ts_limit, (
            f"WGSL scores array ({wgsl_size}) != TS MAX_ATTN_CACHE ({ts_limit}). "
            f"Update forward-pass.ts to match the shader."
        )

    def test_bounds_check_present(self):
        """Verify both dispatch functions still contain the cacheLen bounds check."""
        with open(TS_PATH, "r") as f:
            text = f.read()
        occurrences = text.count("cacheLen > MAX_ATTN_CACHE")
        assert occurrences >= 2, (
            f"Expected cacheLen bounds check in both dispatchAttention and dispatchAttentionTQ, "
            f"found {occurrences} occurrence(s)"
        )

    def test_overflow_throws_not_clamps(self):
        """Bounds check must throw, not silently clamp.

        Silent clamping discards tokens and produces wrong output the caller
        has no way to detect. A thrown error surfaces the limit at the call
        site so the caller can shorten the prompt or rebuild the shader.
        """
        with open(TS_PATH, "r") as f:
            text = f.read()
        # The old behavior used `cacheLen = MAX_ATTN_CACHE;` to clamp.
        # That assignment must be gone.
        assert "cacheLen = MAX_ATTN_CACHE" not in text, (
            "Found `cacheLen = MAX_ATTN_CACHE` clamp — bounds violations must throw, not clamp."
        )
        # Confirm the throw is present in the dispatch path.
        attn_block_start = text.find("function dispatchAttention(")
        assert attn_block_start >= 0, "dispatchAttention not found"
        # Search a slice large enough to cover both dispatch functions.
        block = text[attn_block_start:attn_block_start + 6000]
        assert "throw new Error" in block, (
            "dispatchAttention should throw when cacheLen > MAX_ATTN_CACHE."
        )
