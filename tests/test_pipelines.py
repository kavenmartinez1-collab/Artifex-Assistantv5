"""Tests for pipeline architecture — BasePipeline contract, registry."""

import pytest
from core.pipelines.base import BasePipeline, PipelineResult


class TestPipelineResult:
    """Test the PipelineResult dataclass."""

    def test_success_result(self):
        r = PipelineResult(success=True, output_type="text", content="Hello")
        assert r.success is True
        assert r.output_type == "text"
        assert r.content == "Hello"
        assert r.error is None

    def test_error_result(self):
        r = PipelineResult(success=False, output_type="text", error="Something broke")
        assert r.success is False
        assert r.error == "Something broke"

    def test_metadata(self):
        r = PipelineResult(success=True, output_type="image", metadata={"width": 512})
        assert r.metadata["width"] == 512


class TestBasePipelineContract:
    """Verify that all pipeline subclasses implement the required interface."""

    def test_text_gen_implements_contract(self):
        from core.pipelines.text_gen import TextGenerationPipeline
        p = TextGenerationPipeline()
        assert hasattr(p, "load")
        assert hasattr(p, "unload")
        assert hasattr(p, "is_loaded")
        assert hasattr(p, "run")
        assert hasattr(p, "get_vram_estimate")
        assert p.pipeline_type == "text-generation"

    def test_image_gen_implements_contract(self):
        try:
            from core.pipelines.image_gen import ImageGenerationPipeline
        except ImportError:
            pytest.skip("diffusers not installed")
        p = ImageGenerationPipeline()
        assert p.pipeline_type == "text-to-image"
        assert p.output_type == "image"

    def test_shape3d_implements_contract(self):
        try:
            from core.pipelines.shape_3d import Shape3DPipeline
        except ImportError:
            pytest.skip("diffusers not installed")
        p = Shape3DPipeline()
        assert p.pipeline_type == "shap-e"
        assert p.output_type == "mesh"

    def test_vision_implements_contract(self):
        from core.pipelines.vision import VisionPipeline
        p = VisionPipeline()
        assert p.pipeline_type == "image-text-to-text"
        assert p.output_type == "text"

    def test_audio_implements_contract(self):
        from core.pipelines.audio import AudioPipeline
        p = AudioPipeline()
        assert p.pipeline_type == "text-to-audio"
        assert p.output_type == "audio"


class TestPipelineRegistry:
    """Test the pipeline registry and factory."""

    def test_ensure_registered(self):
        from core.pipelines.registry import _ensure_registered, _PIPELINE_REGISTRY
        _ensure_registered()
        assert "text-generation" in _PIPELINE_REGISTRY

    def test_create_pipeline(self):
        from core.pipelines.registry import create_pipeline
        p = create_pipeline("text-generation")
        assert isinstance(p, BasePipeline)
        assert p.pipeline_type == "text-generation"

    def test_create_pipeline_invalid(self):
        from core.pipelines.registry import create_pipeline
        with pytest.raises(ValueError, match="Unsupported pipeline type"):
            create_pipeline("nonexistent-pipeline")

    def test_get_available_pipelines(self):
        from core.pipelines.registry import get_available_pipelines
        available = get_available_pipelines()
        assert isinstance(available, dict)
        assert "text-generation" in available
        # Display name should be non-empty
        assert available["text-generation"] != ""

    def test_get_capabilities(self):
        from core.pipelines.registry import create_pipeline
        p = create_pipeline("text-generation")
        caps = p.get_capabilities()
        assert "pipeline_type" in caps
        assert "display_name" in caps
        assert "output_type" in caps
