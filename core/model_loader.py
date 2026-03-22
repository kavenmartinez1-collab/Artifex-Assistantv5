"""
Artifex Assistant V5 — Backwards compatibility shim.
Redirects to engine_transformers.py. Use engine_factory.create_engine() for new code.
"""

# This module exists for backwards compatibility only.
# New code should use:
#   from core.engine_factory import create_engine
#   engine = create_engine()

from core.engine_transformers import TransformersEngine


class ModelManager:
    """Legacy wrapper — delegates to TransformersEngine.

    Deprecated: Use create_engine() from core.engine_factory instead.
    """

    def __init__(self):
        self._engine = TransformersEngine()

    @property
    def model(self):
        return self._engine.model

    @property
    def tokenizer(self):
        return self._engine.tokenizer

    @property
    def gpu_profile(self):
        gpu = self._engine._detect_gpu_tier()
        return gpu if gpu else None

    def is_loaded(self):
        return self._engine.is_loaded()

    def needs_reload(self):
        return self._engine.needs_reload()

    def load(self, status_callback=None):
        self._engine.load(status_callback=status_callback)

    def unload(self, status_callback=None):
        self._engine.unload(status_callback=status_callback)

    def periodic_cleanup(self):
        self._engine.periodic_cleanup()
