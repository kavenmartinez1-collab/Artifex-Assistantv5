"""Regression tests for backend-aware vision routing and the model-info
fields (context_length, recommended_max_completion,
image_token_cost_per_megapixel) consumed by upstream-client's dynamic budget
planner. Also covers the bug where Artifex auto-resolved a multimodal
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


class TestContextLengthFields(unittest.TestCase):
    """Each backend must populate context_length so /v1/models can expose
    it for client-side batching decisions.  Field name matches GGUF
    metadata convention (general.context_length)."""

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
        self.assertEqual(models[0]["context_length"], 32768)

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
        self.assertEqual(models[0]["context_length"], 0)

    def test_transformers_context_from_config_json(self):
        from core.model_discovery import _get_transformers_context_length
        fake_config = json.dumps({"max_position_embeddings": 131072})
        with patch("builtins.open", unittest.mock.mock_open(read_data=fake_config)):
            ctx = _get_transformers_context_length("/fake/model")
        self.assertEqual(ctx, 131072)

    def test_transformers_missing_config_returns_zero(self):
        from core.model_discovery import _get_transformers_context_length
        with patch("builtins.open", side_effect=OSError):
            ctx = _get_transformers_context_length("/missing/model")
        self.assertEqual(ctx, 0)

    def test_transformers_malformed_config_returns_zero(self):
        from core.model_discovery import _get_transformers_context_length
        with patch("builtins.open", unittest.mock.mock_open(read_data="not json")):
            ctx = _get_transformers_context_length("/bad/model")
        self.assertEqual(ctx, 0)


class TestRecommendedMaxCompletion(unittest.TestCase):
    """recommended_max_completion = min(context_length // 2, 16384).
    Lets clients cap their max_tokens against the model's actual ctx
    instead of echoing huge defaults like 32768 against an 8K-ctx model.
    """

    def test_half_of_small_context(self):
        from core.model_discovery import _compute_recommended_max_completion
        self.assertEqual(_compute_recommended_max_completion(8192), 4096)

    def test_capped_at_16384_for_huge_context(self):
        from core.model_discovery import _compute_recommended_max_completion
        # 256K ctx → would be 128K, capped at 16384.
        self.assertEqual(_compute_recommended_max_completion(262144), 16384)

    def test_zero_context_returns_zero(self):
        """Unknown context (e.g. Ollama unreachable) returns 0 so the
        client falls back to its own default rather than a fabricated
        cap."""
        from core.model_discovery import _compute_recommended_max_completion
        self.assertEqual(_compute_recommended_max_completion(0), 0)

    def test_negative_context_returns_zero(self):
        from core.model_discovery import _compute_recommended_max_completion
        self.assertEqual(_compute_recommended_max_completion(-1), 0)

    def test_llama_cpp_discovery_includes_recommended(self):
        from core.model_discovery import _discover_llama_cpp
        fake_cfg = {
            "models": {
                "small": {"path": "/m.gguf", "num_ctx": 8192, "extra_flags": []},
                "large": {"path": "/m.gguf", "num_ctx": 262144, "extra_flags": []},
            },
        }
        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(
                 read_data=json.dumps(fake_cfg))):
            models = _discover_llama_cpp()
        by_id = {m["id"]: m for m in models}
        self.assertEqual(by_id["small"]["recommended_max_completion"], 4096)
        self.assertEqual(by_id["large"]["recommended_max_completion"], 16384)


class TestImageTokenCostField(unittest.TestCase):
    """image_token_cost_per_megapixel: optional per-model declaration in
    llama_cpp_config.json. Null when not set so consumers can pick their
    own per-family heuristic."""

    def test_llama_cpp_reads_from_config(self):
        from core.model_discovery import _discover_llama_cpp
        fake_cfg = {
            "models": {
                "vlm": {
                    "path": "/m.gguf",
                    "num_ctx": 8192,
                    "extra_flags": ["--mmproj", "/p"],
                    "image_token_cost_per_megapixel": 256,
                },
            },
        }
        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(
                 read_data=json.dumps(fake_cfg))):
            models = _discover_llama_cpp()
        self.assertEqual(models[0]["image_token_cost_per_megapixel"], 256)

    def test_llama_cpp_default_null_when_unset(self):
        from core.model_discovery import _discover_llama_cpp
        fake_cfg = {
            "models": {
                "vlm": {"path": "/m.gguf", "num_ctx": 8192, "extra_flags": []},
            },
        }
        with patch("os.path.isfile", return_value=True), \
             patch("builtins.open", unittest.mock.mock_open(
                 read_data=json.dumps(fake_cfg))):
            models = _discover_llama_cpp()
        self.assertIsNone(models[0]["image_token_cost_per_megapixel"])


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
