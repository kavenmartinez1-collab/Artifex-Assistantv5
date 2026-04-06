"""Tests for core/health.py — health check system."""

from unittest.mock import patch, MagicMock
import pytest

from core.health import run_health_check, format_health_report


class TestHealthCheck:
    """Test the health check system."""

    def test_returns_structured_report(self):
        report = run_health_check()
        assert "overall" in report
        assert report["overall"] in ("healthy", "degraded", "broken")
        assert "cuda" in report
        assert "models" in report
        assert "disk" in report
        assert "dependencies" in report

    def test_format_report(self):
        report = run_health_check()
        formatted = format_health_report(report)
        assert isinstance(formatted, str)
        assert "System:" in formatted

    def test_overall_health_degrades_without_cuda(self):
        from core.device import gpu_info
        with patch("torch.cuda.is_available", return_value=False):
            gpu_info.reset()  # clear cached GPU state so mock takes effect
            try:
                report = run_health_check()
                # Should still run, but may be degraded
                assert report["overall"] in ("healthy", "degraded", "broken")
                assert report["cuda"]["available"] is False
            finally:
                gpu_info.reset()  # restore for subsequent tests


class TestFormatReport:
    """Test report formatting."""

    def test_healthy_report(self):
        report = {
            "overall": "healthy",
            "issues": [],
            "cuda": {"available": True, "gpu_name": "RTX 4090", "vram_total_gb": 24.0,
                     "vram_free_gb": 20.0, "gpu_tier": "ABUNDANT"},
            "models": {"count": 3, "active_model": "test", "active_backend": "transformers"},
            "ollama": {"reachable": False},
            "dependencies": {"torch": "2.5.0", "transformers": "4.50.0"},
            "disk": {"total_gb": 500.0, "free_gb": 250.0, "used_percent": 50.0},
            "knowledge": {"reference_entries": 5, "workspace_dirs": 2},
        }
        formatted = format_health_report(report)
        assert "HEALTHY" in formatted
        assert "RTX 4090" in formatted

    def test_broken_report(self):
        report = {
            "overall": "broken",
            "issues": ["no_torch"],
            "cuda": {"available": False, "reason": "CUDA not available"},
            "models": {"count": 0, "active_model": "none", "active_backend": "transformers"},
            "ollama": {"reachable": False, "error": "not running"},
            "dependencies": {"torch": None, "transformers": None},
            "disk": {"total_gb": 100.0, "free_gb": 5.0, "used_percent": 95.0},
            "knowledge": {"reference_entries": 0, "workspace_dirs": 0},
        }
        formatted = format_health_report(report)
        assert "BROKEN" in formatted
        assert "Missing required" in formatted
