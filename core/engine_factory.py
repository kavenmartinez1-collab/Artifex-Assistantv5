"""
Artifex Assistant V5 — Engine factory.
Returns the right engine based on the active backend setting.
"""

from core.config import get_active_backend, get_active_ollama_model


def create_engine():
    """Create an engine instance for the currently active backend.

    Returns:
        BaseEngine — either TransformersEngine or OllamaEngine.
    """
    backend = get_active_backend()

    if backend == "ollama":
        from core.engine_ollama import OllamaEngine
        model_name = get_active_ollama_model()
        return OllamaEngine(model_name)
    else:
        from core.engine_transformers import TransformersEngine
        return TransformersEngine()
