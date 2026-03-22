"""
Artifex Assistant V5 — Pipeline package.
Each pipeline handles a specific model type (text, image, 3D, audio, etc.).
"""

from core.pipelines.base import BasePipeline, PipelineResult
from core.pipelines.registry import create_pipeline, get_available_pipelines

__all__ = ["BasePipeline", "PipelineResult", "create_pipeline", "get_available_pipelines"]
