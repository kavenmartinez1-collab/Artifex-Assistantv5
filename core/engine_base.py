"""
Artifex Assistant V5 — Base engine interface.
All backends (transformers, Ollama, future) implement this contract.
"""

from abc import ABC, abstractmethod


class BaseEngine(ABC):
    """Unified engine interface for model loading and streaming generation."""

    @abstractmethod
    def load(self, status_callback=None):
        """Load the model / verify backend is ready."""

    @abstractmethod
    def unload(self, status_callback=None):
        """Unload the model and free resources."""

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if the model is ready for inference."""

    @abstractmethod
    def needs_reload(self) -> bool:
        """Return True if the active model differs from the loaded one."""

    @abstractmethod
    def periodic_cleanup(self):
        """Run lightweight cleanup between generations (e.g. VRAM GC)."""

    def get_context_size(self) -> int:
        """Max context window in tokens, or 0 if unknown."""
        return 0

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string. Used for context budgeting."""

    @abstractmethod
    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, on_complete=None,
                           enable_thinking=True,
                           grammar=None, response_format=None,
                           web_tools=False) -> str:
        """Run streaming inference.

        Args:
            messages: list of {role, content} dicts
            max_tokens: max new tokens to generate
            temperature: sampling temperature
            on_token: callback(str) for each new chunk (raw, includes thinking)
            on_complete: callback(str) with final cleaned response
            enable_thinking: if True, model generates <think> reasoning blocks
            grammar: GBNF grammar string for constrained output (llama.cpp)
            response_format: dict like {"type": "json_object"} or
                {"type": "json_schema", "json_schema": {...}}
            web_tools: if True, the engine should ask the underlying
                model to use any natively-supported web search /
                fetch tools.  Backends without native tool support
                (llama.cpp, ollama, transformers — they get web tools
                via the @search/@web_read post-processor wrapper in
                api/server.py instead) should accept and ignore the
                flag.  claude_cli maps it to `--allowedTools
                WebSearch,WebFetch`.

        Returns:
            The final response string (thinking stripped).
        """
