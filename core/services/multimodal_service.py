"""
Artifex Assistant V5 — Multimodal service layer.
Sits between the UI/CLI/API interfaces and the backend pipelines.
Provides pipeline caching, VRAM-aware loading, progress reporting,
and cancellation support.
"""

import os
import threading
import time
from typing import Callable, Optional

from core.pipelines.base import BasePipeline, PipelineResult
from core.pipelines.registry import create_pipeline, get_available_pipelines
from core.services.file_manager import FileManager


# Type alias for progress callbacks: (current, total, message)
ProgressCallback = Callable[[int, int, str], None]


class _CachedPipeline:
    """Wraps a pipeline instance with usage tracking for LRU eviction."""
    __slots__ = ("pipeline", "pipeline_type", "model_path", "last_used")

    def __init__(self, pipeline: BasePipeline, pipeline_type: str,
                 model_path: str):
        self.pipeline = pipeline
        self.pipeline_type = pipeline_type
        self.model_path = model_path
        self.last_used = time.time()

    def touch(self):
        self.last_used = time.time()


class MultimodalService:
    """Unified service for running any pipeline type.

    Features:
        - Pipeline caching: loaded pipelines are reused across calls
        - VRAM management: auto-unloads LRU pipeline when VRAM is tight
        - Progress reporting: uniform callback interface for all pipelines
        - Cancellation: threading.Event checked between pipeline stages
        - File management: generated outputs stored via FileManager
    """

    def __init__(self, file_manager: FileManager = None):
        self._file_manager = file_manager or FileManager()
        self._cache: dict[str, _CachedPipeline] = {}
        self._lock = threading.Lock()

    @property
    def file_manager(self) -> FileManager:
        return self._file_manager

    # --- Pipeline lifecycle ---

    def load_pipeline(self, pipeline_type: str, model_path: str = "",
                      progress_callback: ProgressCallback = None,
                      **load_kwargs) -> BasePipeline:
        """Load a pipeline, reusing cache if possible.

        Args:
            pipeline_type: Registry key (e.g. "text-to-image")
            model_path: Model directory or HF repo ID
            progress_callback: Optional (current, total, message) callback
            **load_kwargs: Passed to pipeline.load()

        Returns:
            The loaded BasePipeline instance
        """
        def _status(msg):
            if progress_callback:
                progress_callback(0, 0, msg)

        with self._lock:
            cached = self._cache.get(pipeline_type)
            if cached and cached.pipeline.is_loaded():
                if cached.model_path == model_path or not model_path:
                    cached.touch()
                    _status(f"{cached.pipeline.display_name} already loaded")
                    return cached.pipeline
                # Different model requested — unload old one
                _status(f"Unloading {cached.pipeline.display_name}...")
                cached.pipeline.unload()
                del self._cache[pipeline_type]

        # Check VRAM before loading
        pipeline = create_pipeline(pipeline_type)
        self._maybe_free_vram(pipeline, model_path, _status)

        _status(f"Loading {pipeline.display_name}...")
        pipeline.load(model_path, status_callback=lambda m: _status(m),
                      **load_kwargs)

        with self._lock:
            self._cache[pipeline_type] = _CachedPipeline(
                pipeline, pipeline_type, model_path
            )

        return pipeline

    def unload_pipeline(self, pipeline_type: str,
                        progress_callback: ProgressCallback = None):
        """Unload a specific pipeline and free its resources."""
        with self._lock:
            cached = self._cache.pop(pipeline_type, None)
        if cached and cached.pipeline.is_loaded():
            if progress_callback:
                progress_callback(0, 0,
                                  f"Unloading {cached.pipeline.display_name}...")
            cached.pipeline.unload()

    def unload_all(self, progress_callback: ProgressCallback = None):
        """Unload all cached pipelines."""
        with self._lock:
            entries = list(self._cache.items())
            self._cache.clear()
        for ptype, cached in entries:
            if cached.pipeline.is_loaded():
                if progress_callback:
                    progress_callback(0, 0,
                                      f"Unloading {cached.pipeline.display_name}...")
                cached.pipeline.unload()

    def get_loaded_pipelines(self) -> dict[str, str]:
        """Return dict of pipeline_type -> display_name for loaded pipelines."""
        with self._lock:
            return {
                ptype: c.pipeline.display_name
                for ptype, c in self._cache.items()
                if c.pipeline.is_loaded()
            }

    def get_available_pipelines(self) -> dict[str, str]:
        """Return dict of all available pipeline types -> display names."""
        return get_available_pipelines()

    # --- Pipeline execution ---

    def run_pipeline(self, pipeline_type: str, model_path: str = "",
                     kwargs: dict = None,
                     progress_callback: ProgressCallback = None,
                     cancel_event: threading.Event = None,
                     store_output: bool = True) -> PipelineResult:
        """Run a pipeline with full lifecycle management.

        Args:
            pipeline_type: Registry key (e.g. "text-to-image")
            model_path: Model path or "" for default
            kwargs: Arguments passed to pipeline.run()
            progress_callback: Optional (current, total, message) callback
            cancel_event: Optional Event to check for cancellation
            store_output: If True, store generated files via FileManager

        Returns:
            PipelineResult from the pipeline
        """
        kwargs = kwargs or {}

        def _status(msg):
            if progress_callback:
                progress_callback(0, 0, msg)

        # Check cancellation before starting
        if cancel_event and cancel_event.is_set():
            return PipelineResult(
                success=False, output_type="text",
                error="Cancelled before start"
            )

        # Load pipeline
        pipeline = self.load_pipeline(pipeline_type, model_path,
                                      progress_callback)

        # Check cancellation after load
        if cancel_event and cancel_event.is_set():
            return PipelineResult(
                success=False, output_type=pipeline.output_type,
                error="Cancelled after model load"
            )

        # Run inference
        _status(f"Running {pipeline.display_name}...")
        result = pipeline.run(**kwargs)

        # Store generated output if successful
        if result.success and store_output and result.content is not None:
            result = self._store_result(result, pipeline, _status)

        return result

    # --- VRAM management ---

    def _maybe_free_vram(self, pipeline: BasePipeline, model_path: str,
                         status_fn):
        """Check if we need to free VRAM before loading a new pipeline."""
        try:
            import torch
            if not torch.cuda.is_available():
                return

            needed_gb = pipeline.get_vram_estimate(model_path)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated_gb = torch.cuda.memory_allocated() / (1024**3)
            free_gb = total_gb - allocated_gb

            if free_gb >= needed_gb:
                return

            # Need to free VRAM — evict LRU cached pipelines
            status_fn(f"Need {needed_gb:.1f}GB, {free_gb:.1f}GB free — "
                       "evicting cached pipelines...")

            # Snapshot cache inside lock to avoid race with concurrent threads
            with self._lock:
                by_age = sorted(self._cache.items(),
                                key=lambda x: x[1].last_used)
                # Copy to list so we iterate a stable snapshot
                by_age = list(by_age)

            for ptype, cached in by_age:
                if cached.pipeline.is_loaded():
                    status_fn(f"Unloading {cached.pipeline.display_name}...")
                    cached.pipeline.unload()
                    with self._lock:
                        self._cache.pop(ptype, None)  # pop() won't raise if already removed

                    # Recalculate free VRAM
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                    free_gb = total_gb - allocated_gb
                    if free_gb >= needed_gb:
                        break

        except ImportError:
            pass  # no torch, skip VRAM management

    def _store_result(self, result: PipelineResult,
                      pipeline: BasePipeline,
                      status_fn) -> PipelineResult:
        """Store pipeline output via FileManager and update result metadata."""
        output_type = pipeline.output_type

        # Determine what to store based on output type
        if output_type in ("image", "audio", "video", "mesh"):
            content = result.content
            # Check if result already has a saved_to path
            saved_to = result.metadata.get("saved_to")
            if saved_to and os.path.isfile(saved_to):
                content = saved_to

            try:
                record = self._file_manager.store_generated(
                    content=content,
                    output_type=output_type,
                    metadata=result.metadata,
                )
                # Add file_id to result metadata
                result.metadata["file_id"] = record.file_id
                result.metadata["stored_path"] = record.stored_path
                status_fn(f"Saved as {record.original_name} "
                          f"(ID: {record.file_id})")
            except Exception as e:
                status_fn(f"Warning: could not store output: {e}")

        return result
