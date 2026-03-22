"""
Artifex Assistant V5 — Universal Local AI Hosting Platform.
Run any AI model locally: text generation, image generation, 3D, vision, audio.
Supports Transformers and Ollama backends with automatic VRAM management.

Usage:
    python main.py
    Double-click launch.bat
"""

from ui.cli_assistant import run_assistant


if __name__ == "__main__":
    run_assistant()
