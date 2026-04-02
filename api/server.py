"""
Artifex Assistant V5 — OpenAI-compatible REST API server.
Exposes chat completions, image generation, embeddings, and model listing.
Supports streaming for both Ollama and Transformers backends with optional
web search tool execution via the Docker web-gateway.

Usage:
    python main_api.py --port 8000
    curl http://localhost:8000/v1/models
    curl http://localhost:8000/v1/chat/completions -d '{"messages":[...]}'
"""

import asyncio
import json
import os
import queue
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
from core.inference import strip_think_blocks, ThinkFilter
from core.model_queue import get_model_queue
from core.health import run_health_check, format_health_report
from core.logging_config import get_logger
from api.web_tools import (
    extract_web_tools, execute_web_tools, tool_status_labels,
    gateway_available, MAX_TOOL_ROUNDS,
)

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
    web_tools: Optional[bool] = False

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., example="A sunset over mountains")
    negative_prompt: Optional[str] = ""
    width: Optional[int] = Field(512, ge=64, le=2048)
    height: Optional[int] = Field(512, ge=64, le=2048)
    steps: Optional[int] = Field(30, ge=1, le=100)

class EmbeddingRequest(BaseModel):
    input: List[str] = Field(..., example=["Hello world"])


# ── SSE helpers ──────────────────────────────────────────────────────────

_SENTINEL = object()  # Marks end of generation in queue


def _make_sse_chunk(chat_id: str, model: str, delta: dict,
                    finish_reason=None, extra: dict | None = None) -> str:
    """Build an SSE-formatted OpenAI chunk with optional extended fields."""
    event = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }
    if extra:
        event.update(extra)
    return f"data: {json.dumps(event)}\n\n"


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


# ── Ollama streaming ────────────────────────────────────────────────────

def _stream_ollama_raw(ollama_messages, model, temperature, max_tokens,
                       options, timeout, out_queue: queue.Queue):
    """Stream from Ollama into a queue. Runs in a background thread.

    Pushes (type, data) tuples:
        ("thinking", text)  — thinking token
        ("content", text)   — response token
        ("usage", dict)     — final usage stats
        _SENTINEL           — marks end of stream
    """
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
        "stream": True,
        "think": True,
        "options": ollama_options,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        out_queue.put(("error", str(e)))
        out_queue.put(_SENTINEL)
        return

    try:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg = chunk.get("message", {})

            thinking = msg.get("thinking", "")
            if thinking:
                out_queue.put(("thinking", thinking))

            content = msg.get("content", "")
            if content:
                out_queue.put(("content", content))

            if chunk.get("done", False):
                out_queue.put(("usage", {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0),
                    "completion_tokens": chunk.get("eval_count", 0),
                }))
                break
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        resp.close()
        out_queue.put(_SENTINEL)


# ── Transformers streaming ──────────────────────────────────────────────

def _stream_transformers_raw(engine, messages, max_tokens, temperature,
                             out_queue: queue.Queue):
    """Stream from Transformers engine into a queue. Runs in a background thread.

    Uses ThinkFilter to separate thinking from response content.
    Pushes same tuple format as _stream_ollama_raw.
    """
    tf = ThinkFilter(
        on_response=lambda t: out_queue.put(("content", t)),
        on_thinking=lambda t: out_queue.put(("thinking", t)),
    )

    full_text = ""
    def on_token(text):
        nonlocal full_text
        full_text += text
        tf.feed(text)

    try:
        engine.generate_streaming(
            messages, max_tokens=max_tokens or 4096,
            temperature=temperature, on_token=on_token,
        )
        tf.flush()

        # Approximate token counts
        out_queue.put(("usage", {
            "prompt_tokens": sum(len(m.get("content", "")) // 4 for m in messages),
            "completion_tokens": len(full_text) // 4,
        }))
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        out_queue.put(_SENTINEL)


# ── Unified streaming with optional tool execution ──────────────────────

async def _stream_with_tools(messages: list, model: str, max_tokens: int,
                             temperature: float, options: dict,
                             use_web_tools: bool, backend: str):
    """Full streaming generator with optional tool execution loop.

    Yields SSE events. Handles both Ollama and Transformers backends.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    search_cache = []  # Request-scoped cache for @web_read(N)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    current_messages = list(messages)
    round_count = 0

    while True:
        # Start generation in a background thread
        q = queue.Queue(maxsize=256)

        if backend == "ollama":
            has_images = _messages_have_images(current_messages)
            timeout = 600 if has_images else 300
            ollama_msgs = _convert_messages_for_ollama(current_messages)
            thread = threading.Thread(
                target=_stream_ollama_raw,
                args=(ollama_msgs, model, temperature, max_tokens,
                      options, timeout, q),
                daemon=True,
            )
        else:
            engine = _get_engine()
            thread = threading.Thread(
                target=_stream_transformers_raw,
                args=(engine, current_messages, max_tokens, temperature, q),
                daemon=True,
            )

        thread.start()

        # Consume queue and yield SSE events, collecting full response
        full_response = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        loop = asyncio.get_event_loop()

        while True:
            item = await loop.run_in_executor(None, q.get)
            if item is _SENTINEL:
                break

            kind, data = item
            if kind == "thinking":
                yield _make_sse_chunk(chat_id, model, {"x_thinking": data})
            elif kind == "content":
                full_response += data
                yield _make_sse_chunk(chat_id, model, {"content": data})
            elif kind == "usage":
                usage = data
            elif kind == "error":
                yield f"data: {json.dumps({'error': data})}\n\n"

        thread.join(timeout=10)

        # Accumulate usage across rounds
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)

        _log.info("Generation complete: %d chars, web_tools=%s, response='%s'",
                   len(full_response), use_web_tools, full_response[:120])

        # ── Tool execution check ──────────────────────────────────────
        if not use_web_tools:
            _log.info("web_tools disabled, skipping tool check")
            break

        tools = extract_web_tools(full_response)
        has_gw = gateway_available() if tools else False
        _log.info("Tool check: %d tools found, gateway=%s", len(tools), has_gw)

        if not tools or not has_gw:
            if tools and not has_gw:
                _log.warning("Model requested tools but web gateway unavailable")
            break

        round_count += 1
        if round_count > MAX_TOOL_ROUNDS:
            _log.warning("Max tool rounds (%d) reached", MAX_TOOL_ROUNDS)
            break

        _log.info("Tool round %d: %d tools detected", round_count, len(tools))

        # Notify client that tools are executing
        labels = tool_status_labels(tools)
        yield f"data: {json.dumps({'x_tool_status': labels})}\n\n"

        # Execute tools (blocking, but SSE connection stays open)
        try:
            tool_output = await loop.run_in_executor(
                None, lambda: execute_web_tools(tools, search_cache)
            )
            _log.info("Tool execution complete: %d chars of results", len(tool_output))
        except Exception as e:
            _log.error("Tool execution error: %s", e)
            tool_output = f"Tool execution failed: {e}"

        # Signal client to reset for follow-up response
        yield f"data: {json.dumps({'x_tool_done': True})}\n\n"
        _log.info("Starting follow-up generation with tool results")

        # Build follow-up messages with tool results
        current_messages.append({"role": "assistant", "content": full_response})
        current_messages.append({
            "role": "user",
            "content": (
                f"[Tool results — use this data to answer the original question]\n\n"
                f"{tool_output}"
            ),
        })
        # Loop continues with new generation round

    # ── Final events ──────────────────────────────────────────────────
    yield _make_sse_chunk(chat_id, model, {}, finish_reason="stop",
                          extra={"x_usage": total_usage})
    yield "data: [DONE]\n\n"


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
        description="OpenAI-compatible local AI API with web search tools",
    )

    # CORS: allow localhost origins for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:5173",    # WebGPU dev server
            "http://localhost:3001",    # Metrics server
            "http://localhost:5000",    # Common dev server port
            "http://localhost:8080",    # Common dev server port
            "http://localhost:8114",    # Kbot Web Suite
            "http://127.0.0.1",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:5000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:8114",
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
        # Include web gateway status and backend info
        report["web_gateway"] = gateway_available()
        report["backend"] = get_active_backend()
        return report

    @app.get("/v1/queue")
    async def queue_status(request: Request):
        """Model queue status — shows current model, pending requests, switch count."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")
        return get_model_queue().stats

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
        use_web_tools = body.web_tools or False
        backend = get_active_backend()

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # ── Streaming path (both backends) ────────────────────────────
        # Model queue serializes requests and handles model/backend switching
        # for both Ollama and Transformers. No more 503 rejections — requests
        # queue and wait their turn.
        if stream:
            async def guarded_stream():
                """Wrap streaming with model queue lock (both backends)."""
                mq = get_model_queue()
                try:
                    async with mq._lock:
                        await mq.switch_if_needed(model, backend)

                        if backend == "ollama":
                            _get_engine()  # Verify Ollama is reachable
                        else:
                            _get_engine()  # Load/reload transformers model

                        async for chunk in _stream_with_tools(
                            messages, model, max_tokens or 4096, temperature,
                            body.options, use_web_tools, backend,
                        ):
                            yield chunk
                except Exception as e:
                    _log.error("Streaming error: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return StreamingResponse(
                guarded_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Non-streaming path (both backends via model queue) ────────
        mq = get_model_queue()
        async with mq._lock:
            await mq.switch_if_needed(model, backend)

            if backend == "ollama":
                try:
                    _get_engine()
                    has_images = _messages_have_images(messages)
                    timeout = 600 if has_images else 300
                    ollama_msgs = _convert_messages_for_ollama(messages)

                    _log.info(
                        "Ollama proxy: model=%s images=%s msgs=%d",
                        model, has_images, len(messages),
                    )

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

            # Transformers non-streaming
            engine = _get_engine()
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
