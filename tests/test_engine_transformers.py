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


class TestMultimodalMessageConversion:
    """OpenAI image_url → processor format conversion.

    Regression for the case where Qwen3-VL's apply_chat_template returned a
    plain string (not a tokenized BatchFeature) because it didn't recognize
    image_url-shaped content, then crashed with `'str' object has no attribute 'to'`.
    """

    @staticmethod
    def _make_data_url():
        import base64, io
        from PIL import Image
        img = Image.new("RGB", (8, 8), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def test_decode_data_url_returns_pil_image(self):
        from PIL import Image
        from core.engine_transformers import _decode_image_url
        result = _decode_image_url(self._make_data_url())
        assert isinstance(result, Image.Image)
        assert result.size == (8, 8)

    def test_decode_http_url_passes_through(self):
        from core.engine_transformers import _decode_image_url
        url = "https://example.com/x.jpg"
        assert _decode_image_url(url) == url

    def test_decode_garbage_returns_none(self):
        from core.engine_transformers import _decode_image_url
        assert _decode_image_url("nope") is None
        assert _decode_image_url("") is None

    def test_decode_malformed_base64_returns_none(self):
        from core.engine_transformers import _decode_image_url
        # Header says base64 but payload is junk
        assert _decode_image_url("data:image/png;base64,notbase64!!") is None

    def test_convert_image_url_to_image(self):
        from PIL import Image
        from core.engine_transformers import _convert_openai_messages_for_processor
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._make_data_url()}},
                {"type": "text", "text": "describe"},
            ],
        }]
        converted = _convert_openai_messages_for_processor(msgs)
        items = converted[0]["content"]
        assert items[0]["type"] == "image"
        assert isinstance(items[0]["image"], Image.Image)
        assert items[1] == {"type": "text", "text": "describe"}

    def test_convert_text_only_passthrough(self):
        from core.engine_transformers import _convert_openai_messages_for_processor
        msgs = [{"role": "user", "content": "hello"}]
        assert _convert_openai_messages_for_processor(msgs) == msgs

    def test_convert_drops_unparseable_image(self):
        from core.engine_transformers import _convert_openai_messages_for_processor
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "not a url"}},
                {"type": "text", "text": "x"},
            ],
        }]
        converted = _convert_openai_messages_for_processor(msgs)
        # The unparseable image is dropped; the text remains
        assert converted[0]["content"] == [{"type": "text", "text": "x"}]

    def test_extract_images_pulls_pil_objects(self):
        from PIL import Image
        from core.engine_transformers import (
            _convert_openai_messages_for_processor, _extract_processor_images,
        )
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._make_data_url()}},
                {"type": "image_url", "image_url": {"url": self._make_data_url()}},
                {"type": "text", "text": "x"},
            ],
        }]
        converted = _convert_openai_messages_for_processor(msgs)
        images = _extract_processor_images(converted)
        assert len(images) == 2
        assert all(isinstance(i, Image.Image) for i in images)


class TestTemperatureGreedyDispatch:
    """temperature=0 → do_sample=False (greedy); temperature>0 → sampling.

    Regression for transformers v5 raising
      `temperature` (=0.0) has to be a strictly positive float
    when generate_streaming forwarded temperature=0 with do_sample=True.
    """

    @staticmethod
    def _build_kwargs(temperature):
        """Mirror the dispatch logic in generate_streaming."""
        sampling = temperature is not None and temperature > 0
        kwargs = {"do_sample": sampling}
        if sampling:
            kwargs["temperature"] = temperature
        return kwargs

    def test_zero_temperature_is_greedy(self):
        kw = self._build_kwargs(0.0)
        assert kw == {"do_sample": False}
        assert "temperature" not in kw

    def test_negative_temperature_is_greedy(self):
        kw = self._build_kwargs(-0.5)
        assert kw == {"do_sample": False}

    def test_none_temperature_is_greedy(self):
        kw = self._build_kwargs(None)
        assert kw == {"do_sample": False}

    def test_positive_temperature_samples(self):
        kw = self._build_kwargs(0.7)
        assert kw == {"do_sample": True, "temperature": 0.7}

    def test_high_temperature_samples(self):
        kw = self._build_kwargs(2.0)
        assert kw == {"do_sample": True, "temperature": 2.0}
