"""
Artifex Assistant V5 — Universal Local AI Hosting Platform.
Run any AI model locally: text generation, image generation, 3D, vision, audio.
Supports Transformers and Ollama backends with automatic VRAM management.

Usage:
    python main.py
    Double-click launch.bat
"""

from core.logging_config import setup_logging
setup_logging()

from ui.cli_assistant import run_assistant


if __name__ == "__main__":
    run_assistant()
