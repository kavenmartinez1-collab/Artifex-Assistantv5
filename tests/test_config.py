"""Tests for core/config.py — GPU tiers, backend switching, context profiles."""

from unittest.mock import patch, MagicMock
import pytest


def test_gpu_tier_tight():
    """TIGHT tier for <= 12 GB VRAM."""
    props = MagicMock()
    props.total_memory = 12 * 1024 ** 3
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=props):
        from core.config import _detect_gpu_tier
        assert _detect_gpu_tier() == "TIGHT"


def test_gpu_tier_comfortable():
    """COMFORTABLE tier for 13-20 GB VRAM."""
    props = MagicMock()
    props.total_memory = 16 * 1024 ** 3
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=props):
        from core.config import _detect_gpu_tier
        assert _detect_gpu_tier() == "COMFORTABLE"


def test_gpu_tier_abundant():
    """ABUNDANT tier for > 20 GB VRAM."""
    props = MagicMock()
    props.total_memory = 24 * 1024 ** 3
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=props):
        from core.config import _detect_gpu_tier
        assert _detect_gpu_tier() == "ABUNDANT"


def test_gpu_tier_no_cuda():
    """Falls back to TIGHT when CUDA unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        from core.config import _detect_gpu_tier
        assert _detect_gpu_tier() == "TIGHT"


def test_backend_switching():
    """Backend can be switched between transformers and ollama."""
    from core.config import get_active_backend, set_active_backend

    original = get_active_backend()
    try:
        assert set_active_backend("ollama") is True
        assert get_active_backend() == "ollama"

        assert set_active_backend("transformers") is True
        assert get_active_backend() == "transformers"

        assert set_active_backend("invalid") is False
    finally:
        set_active_backend(original)


def test_context_profile_switching():
    """Context profiles can be switched between STANDARD and HIGH."""
    from core.config import (
        get_context_profile, set_context_profile,
        get_context_profile_name, CONTEXT_PROFILES,
    )

    original = get_context_profile_name()
    try:
        assert set_context_profile("STANDARD") is True
        p = get_context_profile()
        assert p.max_total_input_tokens == 10000
        assert p.max_output_tokens == 1536

        assert set_context_profile("HIGH") is True
        p = get_context_profile()
        assert p.max_total_input_tokens == 14000
        assert p.max_output_tokens == 2048

        assert set_context_profile("NONEXISTENT") is False
    finally:
        set_context_profile(original)


def test_modes_exist():
    """ASSISTANT mode is configured."""
    from core.config import MODES
    assert "ASSISTANT" in MODES
    cfg = MODES["ASSISTANT"]
    assert cfg.max_tokens > 0
    assert 0.0 <= cfg.temperature <= 2.0
    assert cfg.context_window > 0


def test_safety_patterns():
    """Dangerous patterns are defined and non-empty."""
    from core.config import DANGEROUS_PATTERNS, ASSISTANT_DANGEROUS_PATTERNS
    assert len(DANGEROUS_PATTERNS) > 0
    assert len(ASSISTANT_DANGEROUS_PATTERNS) > len(DANGEROUS_PATTERNS)
    assert "rm -rf /" in DANGEROUS_PATTERNS
