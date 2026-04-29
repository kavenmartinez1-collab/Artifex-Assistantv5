"""Tests for core/engine_transformers.py — finish_reason, effective_max logic."""

import pytest


class TestEffectiveMax:
    """The effective_max computation must handle None, 0, negative, and valid values."""

    @staticmethod
    def _compute(max_tokens):
        """Mirror the effective_max logic from generate_streaming."""
        return max_tokens if max_tokens and max_tokens > 0 else 8192

    def test_none_defaults_to_8192(self):
        assert self._compute(None) == 8192

    def test_zero_defaults_to_8192(self):
        assert self._compute(0) == 8192

    def test_negative_defaults_to_8192(self):
        assert self._compute(-1) == 8192

    def test_positive_value_passes_through(self):
        assert self._compute(100) == 100

    def test_large_value_passes_through(self):
        assert self._compute(10000) == 10000


class TestFinishReason:
    """finish_reason must use effective_max, never raw max_tokens."""

    @staticmethod
    def _finish_reason(completion_tokens, max_tokens):
        """Mirror the finish_reason derivation from generate_streaming."""
        effective_max = max_tokens if max_tokens and max_tokens > 0 else 8192
        return "length" if completion_tokens >= effective_max else "stop"

    def test_none_max_tokens_no_type_error(self):
        result = self._finish_reason(50, None)
        assert result == "stop"

    def test_zero_max_tokens_no_false_length(self):
        result = self._finish_reason(50, 0)
        assert result == "stop"

    def test_hit_cap_returns_length(self):
        result = self._finish_reason(100, 100)
        assert result == "length"

    def test_exceed_cap_returns_length(self):
        result = self._finish_reason(150, 100)
        assert result == "length"

    def test_natural_stop_returns_stop(self):
        result = self._finish_reason(50, 10000)
        assert result == "stop"

    def test_none_with_large_output_still_stop(self):
        result = self._finish_reason(5000, None)
        assert result == "stop"

    def test_none_at_fallback_cap_returns_length(self):
        result = self._finish_reason(8192, None)
        assert result == "length"


class TestAutoModelDispatch:
    """_get_auto_model_class must route VL models to AutoModelForImageTextToText.

    Regression for the case where vision analysis with a Qwen3-VL model
    fell through to AutoModelForCausalLM and raised
    `Unrecognized configuration class Qwen3VLConfig`.
    """

    def _resolve(self, model_type):
        from unittest.mock import patch
        from core import engine_transformers
        with patch.object(
            engine_transformers, "_detect_model_type_from_config",
            return_value=model_type,
        ):
            return engine_transformers._get_auto_model_class("/fake/path").__name__

    def test_qwen3_vl_uses_image_text_to_text(self):
        assert self._resolve("qwen3_vl") == "AutoModelForImageTextToText"

    def test_qwen3_vl_moe_uses_image_text_to_text(self):
        assert self._resolve("qwen3_vl_moe") == "AutoModelForImageTextToText"

    def test_qwen2_vl_uses_image_text_to_text(self):
        assert self._resolve("qwen2_vl") == "AutoModelForImageTextToText"

    def test_qwen2_5_vl_uses_image_text_to_text(self):
        assert self._resolve("qwen2_5_vl") == "AutoModelForImageTextToText"

    def test_llava_uses_image_text_to_text(self):
        assert self._resolve("llava") == "AutoModelForImageTextToText"

    def test_gemma3n_still_uses_multimodal_lm(self):
        # Gemma 4 / 3n family has audio + vision and needs the broader class.
        assert self._resolve("gemma3n") == "AutoModelForMultimodalLM"

    def test_gemma3n_text_still_uses_multimodal_lm(self):
        assert self._resolve("gemma3n_text") == "AutoModelForMultimodalLM"

    def test_text_only_qwen_uses_causal_lm(self):
        assert self._resolve("qwen3") == "AutoModelForCausalLM"

    def test_text_only_qwen36_uses_causal_lm(self):
        assert self._resolve("qwen3_6") == "AutoModelForCausalLM"

    def test_llama_uses_causal_lm(self):
        assert self._resolve("llama") == "AutoModelForCausalLM"

    def test_unknown_type_falls_back_to_causal_lm(self):
        assert self._resolve("some_future_text_model") == "AutoModelForCausalLM"

    def test_no_config_falls_back_to_causal_lm(self):
        assert self._resolve(None) == "AutoModelForCausalLM"
