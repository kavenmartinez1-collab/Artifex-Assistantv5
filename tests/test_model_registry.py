"""Tests for core/model_registry.py — model type detection, VRAM estimation."""

import json
import os
import tempfile
import shutil
import pytest

from core.model_registry import (
    detect_model_type, PIPELINE_TYPES, _MODEL_TYPE_MAP,
    PIPELINE_BACKEND_MAP, _estimate_model_size,
)


@pytest.fixture
def model_dir():
    d = tempfile.mkdtemp(prefix="model_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _write_config(model_dir, config):
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)


class TestDetectModelType:
    """Test model type detection from config.json."""

    def test_detect_qwen2(self, model_dir):
        _write_config(model_dir, {"model_type": "qwen2"})
        result = detect_model_type(model_dir)
        assert result == "text-generation"

    def test_detect_llama(self, model_dir):
        _write_config(model_dir, {"model_type": "llama"})
        result = detect_model_type(model_dir)
        assert result == "text-generation"

    def test_detect_whisper(self, model_dir):
        _write_config(model_dir, {"model_type": "whisper"})
        result = detect_model_type(model_dir)
        assert result == "automatic-speech-recognition"

    def test_detect_bark(self, model_dir):
        _write_config(model_dir, {"model_type": "bark"})
        result = detect_model_type(model_dir)
        assert result == "text-to-audio"

    def test_detect_bert(self, model_dir):
        _write_config(model_dir, {"model_type": "bert"})
        result = detect_model_type(model_dir)
        assert result == "feature-extraction"

    def test_detect_llava(self, model_dir):
        _write_config(model_dir, {"model_type": "llava"})
        result = detect_model_type(model_dir)
        assert result == "image-text-to-text"

    def test_detect_unknown(self, model_dir):
        _write_config(model_dir, {"model_type": "totally_unknown_model"})
        result = detect_model_type(model_dir)
        # Should return None or a default for unknown types
        assert result is None or isinstance(result, str)

    def test_no_config(self, model_dir):
        # No config.json
        result = detect_model_type(model_dir)
        assert result is None or isinstance(result, str)


class TestModelTypeMap:
    """Test the model type mapping is comprehensive."""

    def test_text_gen_models_mapped(self):
        text_models = ["qwen2", "llama", "mistral", "gemma", "phi", "falcon"]
        for m in text_models:
            assert m in _MODEL_TYPE_MAP
            assert _MODEL_TYPE_MAP[m] == "text-generation"

    def test_audio_models_mapped(self):
        assert _MODEL_TYPE_MAP["bark"] == "text-to-audio"
        assert _MODEL_TYPE_MAP["whisper"] == "automatic-speech-recognition"

    def test_vision_models_mapped(self):
        assert _MODEL_TYPE_MAP["llava"] == "image-text-to-text"
        assert _MODEL_TYPE_MAP["qwen2_vl"] == "image-text-to-text"


class TestPipelineTypes:
    """Test pipeline type definitions."""

    def test_all_types_have_display_names(self):
        for ptype, display in PIPELINE_TYPES.items():
            assert display != "", f"{ptype} has no display name"

    def test_core_types_present(self):
        assert "text-generation" in PIPELINE_TYPES
        assert "text-to-image" in PIPELINE_TYPES
        assert "text-to-audio" in PIPELINE_TYPES
        assert "automatic-speech-recognition" in PIPELINE_TYPES


class TestVRAMEstimation:
    """Test VRAM size estimation."""

    def test_estimate_from_config(self, model_dir):
        # A model with known hidden_size and num_layers
        _write_config(model_dir, {
            "model_type": "qwen2",
            "hidden_size": 3584,
            "num_hidden_layers": 28,
            "vocab_size": 152064,
        })
        # Create a fake weight file so estimation has something to count
        weight_path = os.path.join(model_dir, "model.safetensors")
        with open(weight_path, "wb") as f:
            f.write(b"\x00" * (1024 * 1024 * 100))  # 100 MB
        estimate = _estimate_model_size(model_dir)
        assert isinstance(estimate, (int, float))
        assert estimate > 0

    def test_estimate_no_config(self, model_dir):
        """Empty dir returns zero — no weight files to measure."""
        estimate = _estimate_model_size(model_dir)
        assert isinstance(estimate, (int, float))
        assert estimate == 0.0
