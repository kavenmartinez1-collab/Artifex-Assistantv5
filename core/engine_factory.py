"""
Artifex Assistant V5 — Engine factory.
Returns the right engine based on the active backend setting.
"""

from core.config import (
    get_active_backend, get_active_ollama_model,
    get_active_llama_cpp_model, get_llama_cpp_model_config,
    get_active_claude_cli_model, get_claude_cli_model_config,
)


def create_engine():
    """Create an engine instance for the currently active backend.

    Returns:
        BaseEngine — OllamaEngine, LlamaCppEngine, ClaudeCliEngine,
        WebGpuEngine, or TransformersEngine.
    """
    backend = get_active_backend()

    if backend == "webgpu":
        from core.engine_webgpu import WebGpuEngine
        return WebGpuEngine()

    if backend == "ollama":
        from core.engine_ollama import OllamaEngine
        model_name = get_active_ollama_model()
        return OllamaEngine(model_name)

    if backend == "llama_cpp":
        from core.engine_llama_cpp import LlamaCppEngine
        model_name = get_active_llama_cpp_model()
        if not model_name:
            raise RuntimeError(
                "No llama.cpp models configured. "
                "Add models to llama_cpp_config.json."
            )
        model_config = get_llama_cpp_model_config(model_name)
        return LlamaCppEngine(model_name, model_config)

    if backend == "claude_cli":
        from core.engine_claude_cli import ClaudeCliEngine
        model_name = get_active_claude_cli_model()
        if not model_name:
            raise RuntimeError(
                "No claude CLI models configured. "
                "Add models to claude_cli_config.json."
            )
        model_config = get_claude_cli_model_config(model_name)
        return ClaudeCliEngine(model_name, model_config)

    from core.engine_transformers import TransformersEngine
    return TransformersEngine()
