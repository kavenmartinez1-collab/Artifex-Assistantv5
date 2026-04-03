"""
Artifex Assistant V5 — Shared service layer.
Provides a singleton MultimodalService and FileManager accessible
from the GUI, CLI, and API.
"""

import threading
from core.services.file_manager import FileManager
from core.services.multimodal_service import MultimodalService

_service: MultimodalService = None
_lock = threading.Lock()


def get_service() -> MultimodalService:
    """Get the singleton MultimodalService instance."""
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                _service = MultimodalService()
    return _service


def get_file_manager() -> FileManager:
    """Shortcut to get the FileManager from the service singleton."""
    return get_service().file_manager
