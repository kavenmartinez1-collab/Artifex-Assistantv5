"""Regression tests for backend-aware vision routing and context_window
exposure. These cover the bug where Artifex auto-resolved a multimodal
request to a model name from the WRONG backend's registry, causing
set_active_model to silently fail and the server to keep running the
previously-loaded model.
"""

import json
import os
import unittest
from unittest.mock import patch


class TestClassifyCapabilitiesVision(unittest.TestCase):
    """_classify_capabilities must trust --mmproj as the authoritative
    vision signal for llama_cpp models, not just regex on the model name.
    Many GGUF deployments have generic names (e.g. qwen3.6-27b-...) but
    are vision-capable because the launch wires in --mmproj <projector>."""

    def test_mmproj_in_extra_flags_marks_vision(self):
        from core.model_discovery import _classify_capabilities
        cfg = {"extra_flags": ["-fa", "on", "--mmproj", "/path/to/mmproj.gguf"]}
        caps = _classify_capabilities("plain-name-no-vl-token", llama_cpp_config=cfg)
        self.assertIn("vision", caps)

    def test_no_mmproj_falls_back_to_name_pattern(self):
        from core.model_discovery import _classify_capabilities
        cfg = {"extra_flags": ["-fa", "on"]}
        # Name has 'vl' so pattern matches.
        caps = _classify_capabilities("qwen3-vl-4b", llama_cpp_config=cfg)
        self.assertIn("vision", caps)

    def test_no_mmproj_no_pattern_means_text_only(self):
        from core.model_discovery import _classify_capabilities
        cfg = {"extra_flags": ["-fa", "on"]}
        caps = _classify_capabilities("plain-name", llama_cpp_config=cfg)
        self.assertNotIn("vision", caps)


class TestContextWindowFields(unittest.TestCase):
    """Each backend must populate context_window so /v1/models can expose
    it for client-side batching decisions."""

    def test_llama_cpp_context_from_num_ctx(self):
        from core.model_discovery import _discover_llama_cpp
        fake_cfg = {
            "models": {
                "test-model": {
                    "path": "/fake/m.gguf",
                    "num_ctx": 32768,
                    "extra_flags": [],
                },
            },
        }
        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(
                 read_data=json.dumps(fake_cfg))):
            models = _discover_llama_cpp()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["context_window"], 32768)

    def test_llama_cpp_missing_num_ctx_is_zero(self):
        from core.model_discovery import _discover_llama_cpp
        fake_cfg = {
            "models": {
                "test-model": {"path": "/fake/m.gguf", "extra_flags": []},
            },
        }
        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(
                 read_data=json.dumps(fake_cfg))):
            models = _discover_llama_cpp()
        self.assertEqual(models[0]["context_window"], 0)

    def test_transformers_context_from_config_json(self):
        from core.model_discovery import _get_transformers_context_window
        fake_config = json.dumps({"max_position_embeddings": 131072})
        with patch("builtins.open", unittest.mock.mock_open(read_data=fake_config)):
            ctx = _get_transformers_context_window("/fake/model")
        self.assertEqual(ctx, 131072)

    def test_transformers_missing_config_returns_zero(self):
        from core.model_discovery import _get_transformers_context_window
        with patch("builtins.open", side_effect=OSError):
            ctx = _get_transformers_context_window("/missing/model")
        self.assertEqual(ctx, 0)

    def test_transformers_malformed_config_returns_zero(self):
        from core.model_discovery import _get_transformers_context_window
        with patch("builtins.open", unittest.mock.mock_open(read_data="not json")):
            ctx = _get_transformers_context_window("/bad/model")
        self.assertEqual(ctx, 0)


class TestPickVisionModelBackendAware(unittest.TestCase):
    """_pick_vision_model must look at the requested backend's own model
    registry. Returning a name from a different backend's registry was
    the root cause of the empty-output bug — set_active_model silently
    failed and the server kept running the previously-loaded model."""

    def test_llama_cpp_prefers_mmproj_entries(self):
        from api.server import _pick_vision_model
        with patch("core.config.refresh_ollama_models"), \
             patch("core.config.get_llama_cpp_models", return_value={
                 "no-vision": {"extra_flags": ["-fa", "on"]},
                 "qwen3.6-27b-mmproj": {
                     "extra_flags": ["--mmproj", "/path/proj.gguf"],
                 },
             }):
            picked = _pick_vision_model("llama_cpp")
        self.assertEqual(picked, "qwen3.6-27b-mmproj")

    def test_llama_cpp_falls_back_to_name_pattern(self):
        from api.server import _pick_vision_model
        with patch("core.config.refresh_ollama_models"), \
             patch("core.config.get_llama_cpp_models", return_value={
                 "qwen3-vl-2b": {"extra_flags": []},
             }):
            picked = _pick_vision_model("llama_cpp")
        self.assertEqual(picked, "qwen3-vl-2b")

    def test_llama_cpp_returns_none_when_no_vision(self):
        from api.server import _pick_vision_model
        with patch("core.config.get_llama_cpp_models", return_value={
                "text-only": {"extra_flags": []},
             }):
            picked = _pick_vision_model("llama_cpp")
        self.assertIsNone(picked)

    def test_llama_cpp_does_not_return_transformers_name(self):
        """Regression: the buggy version used MODELS (transformers) for
        any non-ollama backend, returning names that didn't exist in
        llama_cpp_config.json."""
        from api.server import _pick_vision_model
        with patch("core.config.MODELS", {"qwen3-vl-4b-instruct": "/path"}), \
             patch("core.config.get_llama_cpp_models", return_value={
                 "non-vision-llama": {"extra_flags": []},
             }):
            picked = _pick_vision_model("llama_cpp")
        # Must NOT be the transformers name.
        self.assertNotEqual(picked, "qwen3-vl-4b-instruct")


class TestSetActiveModelLogging(unittest.TestCase):
    """set_active_model must log when it can't switch — silent False
    return was the trap that hid the wrong-backend routing bug."""

    def test_unknown_llama_cpp_model_logs_warning(self):
        import core.config as cfg
        original_backend = cfg._active_backend
        try:
            cfg._active_backend = "llama_cpp"
            with patch("core.config.get_llama_cpp_models", return_value={"real-model": {}}), \
                 self.assertLogs("core.config", level="WARNING") as logs:
                ok = cfg.set_active_model("nonexistent-model")
            self.assertFalse(ok)
            self.assertTrue(any("nonexistent-model" in m for m in logs.output))
        finally:
            cfg._active_backend = original_backend


if __name__ == "__main__":
    unittest.main()
