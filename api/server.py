"""
Artifex Assistant V5 — OpenAI-compatible REST API server.
Exposes chat completions, image generation, embeddings, and model listing.

Usage:
    python main_api.py --port 8000
    curl http://localhost:8000/v1/models
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[...]}'
"""

import json
import os
import threading
import time
import uuid
import urllib.request
import urllib.error
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from core.config import (
    MODES, get_active_backend, get_active_model_name, get_model_names,
)
from core.engine_factory import create_engine
from core.inference import strip_think_blocks
from core.health import run_health_check, format_health_report
from core.logging_config import get_logger

_log = get_logger(__name__)

# Global engine — initialized on server start, protected by lock
_engine = None
_engine_lock = threading.Lock()
_api_key = os.environ.get("ARTIFEX_API_KEY", "")
_inference_busy = threading.Event()  # set() = GPU busy, clear() = free

if not _api_key:
    _log.warning("ARTIFEX_API_KEY not set — API authentication is DISABLED")


def _get_engine():
    global _engine
    with _engine_lock:
        if _engine is None or not _engine.is_loaded():
            # Ensure Ollama models are discovered before engine init
            from core.config import get_active_backend, refresh_ollama_models
            if get_active_backend() == "ollama":
                refresh_ollama_models()
            _engine = create_engine()
            _engine.load(status_callback=lambda msg: _log.info(msg))
        return _engine


def _check_auth(request):
    """Validate API key if configured. Uses constant-time comparison."""
    if not _api_key:
        return True  # No auth required
    import hmac
    auth_header = request.headers.get("Authorization", "")
    key_header = request.headers.get("X-API-Key", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else key_header
    if not token:
        return False
    return hmac.compare_digest(token, _api_key)


# ── Request models for Swagger docs ─────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    content: Union[str, List[dict]] = Field(
        ..., example="What is the capital of France?"
    )

class ChatCompletionRequest(BaseModel):
    model: str = Field("qwen3.5:27b", example="qwen3.5:27b")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(None, ge=1, le=65536)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    options: Optional[dict] = None

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., example="A sunset over mountains")
    negative_prompt: Optional[str] = ""
    width: Optional[int] = Field(512, ge=64, le=2048)
    height: Optional[int] = Field(512, ge=64, le=2048)
    steps: Optional[int] = Field(30, ge=1, le=100)

class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., example=["Hello world"])


# ── Ollama proxy helpers ───────────────────────────────────────────────

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"


def _convert_messages_for_ollama(messages):
    """Convert OpenAI-format messages to Ollama /api/chat format.

    Handles both plain text (content is str) and multimodal
    (content is list of {type: text/image_url} dicts).
    """
    ollama_msgs = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            ollama_msgs.append({"role": role, "content": content})
        elif isinstance(content, list):
            text_parts = []
            images = []
            for item in content:
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        # Extract raw base64 from data URL
                        b64 = url.split(",", 1)[1] if "," in url else ""
                        if b64:
                            images.append(b64)
            entry = {"role": role, "content": "\n".join(text_parts)}
            if images:
                entry["images"] = images
            ollama_msgs.append(entry)
        else:
            ollama_msgs.append({"role": role, "content": str(content)})

    return ollama_msgs


def _messages_have_images(messages):
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    return True
    return False


def _proxy_ollama_chat(ollama_messages, model, temperature, max_tokens, options, timeout):
    """Forward a request to Ollama's /api/chat and return an OpenAI-format response."""
    ollama_options = {}
    if options:
        ollama_options.update(options)
    if temperature is not None:
        ollama_options["temperature"] = temperature
    if max_tokens is not None:
        ollama_options["num_predict"] = max_tokens

    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": ollama_options,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama error ({e.code}): {error_body}")
    except urllib.error.URLError as e:
        raise ConnectionError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e}")

    content = data.get("message", {}).get("content", "")
    content = strip_think_blocks(content) if content else ""

    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def create_app():
    """Create and configure the FastAPI application."""
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError(
            "FastAPI is required for the API server.\n"
            "Install with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="Artifex Assistant V5 API",
        version="5.0.0",
        description="OpenAI-compatible local AI API",
    )

    # CORS: only allow localhost origins since we bind to 127.0.0.1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:5173",    # WebGPU dev server
            "http://localhost:3001",    # Metrics server
            "http://localhost:5000",    # Common dev server port
            "http://localhost:8080",    # Common dev server port
            "http://127.0.0.1",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:5000",
            "http://127.0.0.1:8080",
        ],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

    # ─── Global error handler — never leak stack traces ────────────────
    @app.exception_handler(Exception)
    async def _global_error_handler(request: Request, exc: Exception):
        _log.error("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"},
        )

    # ─── Health ───────────────────────────────────────────────────────────

    @app.get("/health")
    async def health(request: Request):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")
        report = run_health_check()
        return report

    # ─── Models ───────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models(request: Request):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")
        models = get_model_names()
        data = [
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
            for name in models
        ]
        return {"object": "list", "data": data}

    # ─── Chat Completions ─────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        messages = [m.model_dump() for m in body.messages]
        max_tokens = body.max_tokens
        temperature = body.temperature
        stream = body.stream
        model = body.model or get_active_model_name()

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # ── Ollama proxy path ──────────────────────────────────────────
        if get_active_backend() == "ollama":
            # Concurrency gate — reject if GPU is already busy
            if _inference_busy.is_set():
                return JSONResponse(
                    status_code=503,
                    content={"error": "GPU busy — retry later"},
                    headers={"Retry-After": "30"},
                )

            _inference_busy.set()
            try:
                # Ensure Ollama engine is initialized (model verified, num_gpu set)
                _get_engine()

                has_images = _messages_have_images(messages)
                timeout = 600 if has_images else 300
                ollama_msgs = _convert_messages_for_ollama(messages)

                _log.info(
                    "Ollama proxy: model=%s images=%s msgs=%d",
                    model, has_images, len(messages),
                )

                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: _proxy_ollama_chat(
                        ollama_msgs, model, temperature,
                        max_tokens, body.options, timeout,
                    ),
                )
                return result
            except (RuntimeError, ConnectionError) as e:
                _log.error("Ollama proxy error: %s", e)
                raise HTTPException(status_code=502, detail=str(e))
            finally:
                _inference_busy.clear()

        # ── Transformers engine path (existing) ────────────────────────
        engine = _get_engine()

        if stream:
            return StreamingResponse(
                _stream_chat(engine, messages, max_tokens, temperature, model),
                media_type="text/event-stream",
            )

        # Non-streaming — run in thread to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: engine.generate_streaming(
                messages, max_tokens=max_tokens or 4096,
                temperature=temperature,
            ),
        )
        response = strip_think_blocks(response) if response else ""

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
                "completion_tokens": len(response) // 4,
                "total_tokens": (sum(len(m.get("content", "")) // 4 for m in messages)
                                + len(response) // 4),
            },
        }

    async def _stream_chat(engine, messages, max_tokens, temperature, model):
        """Stream SSE events for chat completions."""
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        chunks = []

        def on_token(text):
            chunks.append(text)

        # Run generation in a thread to not block the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: engine.generate_streaming(
                messages, max_tokens=max_tokens, temperature=temperature,
                on_token=on_token,
            ),
        )

        # Emit accumulated chunks as SSE
        for chunk_text in chunks:
            clean = strip_think_blocks(chunk_text) if "</think>" in chunk_text else chunk_text
            if not clean:
                continue
            event = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": clean},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(event)}\n\n"

        # Final event
        final = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    # ─── Image Generation ────────────────────────────────────────────────

    @app.post("/v1/images/generations")
    async def image_generation(request: Request, body: ImageGenerationRequest):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        try:
            from core.pipelines.registry import create_pipeline
            pipe = create_pipeline("text-to-image")
            if not pipe.is_loaded():
                raise HTTPException(
                    status_code=503,
                    detail="Image generation model not loaded. Load a model first.",
                )
            result = pipe.run(
                prompt=body.prompt[:2000],
                negative_prompt=(body.negative_prompt or "")[:500],
                width=body.width,
                height=body.height,
                num_steps=body.steps,
            )
            if result.success:
                return {
                    "created": int(time.time()),
                    "data": [{"url": f"file://{os.path.basename(result.content)}"}],
                }
            raise HTTPException(status_code=500, detail=result.error)
        except ValueError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # ─── Embeddings ──────────────────────────────────────────────────────

    @app.post("/v1/embeddings")
    async def embeddings(request: Request, body: EmbeddingRequest):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        input_texts = body.input
        if not input_texts:
            raise HTTPException(status_code=400, detail="input is required")

        try:
            from core.pipelines.registry import create_pipeline
            pipe = create_pipeline("feature-extraction")
            if not pipe.is_loaded():
                pipe.load("")
            result = pipe.run(texts=input_texts)
            if result.success:
                data = []
                for i, emb in enumerate(result.content):
                    data.append({
                        "object": "embedding",
                        "index": i,
                        "embedding": emb.tolist(),
                    })
                return {
                    "object": "list",
                    "data": data,
                    "model": "all-MiniLM-L6-v2",
                    "usage": {"prompt_tokens": sum(len(t) // 4 for t in input_texts)},
                }
            raise HTTPException(status_code=500, detail=result.error)
        except (ValueError, ImportError) as e:
            raise HTTPException(status_code=503, detail=str(e))

    return app
