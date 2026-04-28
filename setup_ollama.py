"""
Artifex Assistant V5 — Ollama setup helper.
Checks Ollama installation, starts the server if needed, and pulls models.

Usage:
    python setup_ollama.py qwen3.6:27b         # Pull a specific model
    python setup_ollama.py --list               # List installed models
    python setup_ollama.py --status             # Check server status
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("ARTIFEX_OLLAMA_URL", "http://localhost:11434").rstrip("/")


def find_ollama_binary():
    """Find the ollama binary on this system."""
    path = shutil.which("ollama")
    if path:
        return path

    if platform.system() == "Windows":
        appdata = os.environ.get("LOCALAPPDATA", "")
        candidate = os.path.join(appdata, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate

    return None


def is_server_running():
    """Check if the Ollama server is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        urllib.request.urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def start_server(ollama_bin):
    """Start ollama serve in the background."""
    kwargs = {}
    if platform.system() == "Windows":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        kwargs["startupinfo"] = si

    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )

    print("Waiting for Ollama server...", end="", flush=True)
    for _ in range(30):
        time.sleep(0.5)
        if is_server_running():
            print(" ready.")
            return True
        print(".", end="", flush=True)

    print(" timed out.")
    return False


def list_models():
    """List installed Ollama models."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        models = data.get("models", [])
        if not models:
            print("No models installed.")
            return
        print(f"{'NAME':<30} {'SIZE':<12} {'MODIFIED'}")
        print("-" * 60)
        for m in models:
            name = m.get("name", "?")
            size_gb = m.get("size", 0) / (1024 ** 3)
            modified = m.get("modified_at", "?")[:10]
            print(f"{name:<30} {size_gb:.1f} GB      {modified}")
    except Exception as e:
        print(f"Error listing models: {e}")


def pull_model(ollama_bin, model_name):
    """Pull a model using the ollama CLI (shows progress)."""
    print(f"Pulling {model_name}...")
    try:
        subprocess.run([ollama_bin, "pull", model_name], check=True)
        print(f"\n{model_name} is ready.")
    except subprocess.CalledProcessError as e:
        print(f"\nFailed to pull {model_name}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Artifex Assistant V5 — Ollama setup helper"
    )
    parser.add_argument(
        "model", nargs="?", default=None,
        help="Model to pull (e.g. qwen3.6:27b). Required for first install.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List installed Ollama models",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check Ollama server status",
    )
    args = parser.parse_args()

    ollama_bin = find_ollama_binary()
    if not ollama_bin:
        print("Ollama is not installed.")
        print("Install it with:  winget install Ollama.Ollama")
        print("Or download from: https://ollama.com")
        sys.exit(1)
    print(f"Ollama binary: {ollama_bin}")

    if is_server_running():
        print("Ollama server: running")
    else:
        print("Ollama server: not running — starting...")
        if not start_server(ollama_bin):
            print("Failed to start Ollama server.")
            sys.exit(1)

    if args.status:
        list_models()
        return

    if args.list:
        list_models()
        return

    if not args.model:
        print("No model specified. Usage: python setup_ollama.py <model>")
        print("Examples: qwen3.6:27b, qwen3.5:9b, llama3:8b")
        print("Run --list to see installed models.")
        sys.exit(1)

    pull_model(ollama_bin, args.model)


if __name__ == "__main__":
    main()
