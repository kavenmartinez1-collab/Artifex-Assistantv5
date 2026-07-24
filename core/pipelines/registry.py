"""
Artifex Assistant V5 — Pipeline registry and factory.
Creates the right pipeline for a given model type.
"""

from core.pipelines.base import BasePipeline


# Registry of pipeline type -> pipeline class
_PIPELINE_REGISTRY = {}


def _register_pipeline(pipeline_type: str, pipeline_class):
    """Register a pipeline class for a given type."""
    _PIPELINE_REGISTRY[pipeline_type] = pipeline_class


def _ensure_registered():
    """Lazy-register all pipeline classes on first use."""
    if _PIPELINE_REGISTRY:
        return

    from core.pipelines.text_gen import TextGenerationPipeline
    _register_pipeline("text-generation", TextGenerationPipeline)

    try:
        from core.pipelines.image_gen import ImageGenerationPipeline
        _register_pipeline("text-to-image", ImageGenerationPipeline)
    except ImportError:
        pass  # diffusers not installed

    try:
        from core.pipelines.shape_3d import Shape3DPipeline
        _register_pipeline("shap-e", Shape3DPipeline)
    except ImportError:
        pass  # diffusers not installed

    try:
        from core.pipelines.vision import VisionPipeline
        _register_pipeline("image-text-to-text", VisionPipeline)
    except ImportError:
        pass

    try:
        from core.pipelines.audio import AudioPipeline
        _register_pipeline("text-to-audio", AudioPipeline)
        _register_pipeline("automatic-speech-recognition", AudioPipeline)
    except ImportError:
        pass

    # --- V5 new pipelines ---

    try:
        from core.pipelines.embedding import EmbeddingPipeline
        _register_pipeline("feature-extraction", EmbeddingPipeline)
    except ImportError:
        pass

    try:
        from core.pipelines.image_edit import ImageEditPipeline
        _register_pipeline("image-to-image", ImageEditPipeline)
    except ImportError:
        pass  # diffusers not installed

    try:
        from core.pipelines.photo_restore import PhotoRestorePipeline
        _register_pipeline("photo-restoration", PhotoRestorePipeline)
    except ImportError:
        pass  # opencv not installed

    try:
        from core.pipelines.music import MusicPipeline
        _register_pipeline("text-to-music", MusicPipeline)
    except ImportError:
        pass

    try:
        from core.pipelines.video_gen import VideoGenerationPipeline
        _register_pipeline("text-to-video", VideoGenerationPipeline)
    except ImportError:
        pass  # diffusers not installed

    try:
        from core.pipelines.voice_assistant import VoiceAssistantPipeline
        _register_pipeline("voice-assistant", VoiceAssistantPipeline)
    except ImportError:
        pass  # faster-whisper not installed


def create_pipeline(pipeline_type: str) -> BasePipeline:
    """Create a pipeline instance for the given type.

    Args:
        pipeline_type: One of the supported pipeline types
            (text-generation, text-to-image, shap-e, etc.)

    Returns:
        BasePipeline instance

    Raises:
        ValueError: If the pipeline type is not supported
    """
    _ensure_registered()

    if pipeline_type not in _PIPELINE_REGISTRY:
        available = ", ".join(sorted(_PIPELINE_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported pipeline type: {pipeline_type}. "
            f"Available: {available}"
        )

    return _PIPELINE_REGISTRY[pipeline_type]()


def get_available_pipelines() -> dict:
    """Get all available pipeline types and their display names.

    Returns:
        Dict of pipeline_type -> display_name for all installed pipelines
    """
    _ensure_registered()

    result = {}
    for ptype, pclass in _PIPELINE_REGISTRY.items():
        instance = pclass()
        result[ptype] = instance.display_name

    return result
