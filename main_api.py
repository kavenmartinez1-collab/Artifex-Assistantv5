"""
Artifex Assistant V5 — API server launcher.
Starts an OpenAI-compatible REST API for local AI inference.

Usage:
    python main_api.py
    python main_api.py --port 8080
    python main_api.py --host 0.0.0.0 --port 8000
"""

import argparse
import os

from core.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Artifex Assistant V5 API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--backend", default=os.environ.get("ARTIFEX_BACKEND"),
                        choices=["transformers", "ollama"],
                        help="Backend to use (default: auto-detect, or set ARTIFEX_BACKEND env var)")
    parser.add_argument("--model", default=os.environ.get("ARTIFEX_MODEL"),
                        help="Model name (or set ARTIFEX_MODEL env var)")
    parser.add_argument("--gateway", default=None,
                        help="Web gateway URL for search tools (e.g. http://localhost:8080)")
    args = parser.parse_args()

    setup_logging()

    if args.backend:
        from core.config import set_active_backend
        set_active_backend(args.backend)

    if args.model:
        from core.config import set_active_model, MODELS, OLLAMA_MODELS
        if not set_active_model(args.model):
            available = list(MODELS.keys()) or list(OLLAMA_MODELS.keys()) or ["(none found)"]
            print(f"WARNING: Model '{args.model}' not found in registry.")
            print(f"Available models: {', '.join(available)}")
            print(f"Using default model instead.")

    if args.gateway:
        import api.web_tools as wt
        wt.WEB_GATEWAY_URL = args.gateway

    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required for the API server.")
        print("Install with: pip install uvicorn fastapi")
        return

    from api.server import create_app
    app = create_app()

    print(f"Artifex Assistant V5 API Server")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
