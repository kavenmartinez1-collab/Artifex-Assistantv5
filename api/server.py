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
import re
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
# Rate limiting for failed auth attempts
_auth_failures: dict[str, list[float]] = {}  # ip -> list of timestamps
_AUTH_FAIL_WINDOW = 60.0    # seconds
_AUTH_FAIL_MAX = 10         # max failures per window

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
    """Validate API key if configured. Uses constant-time comparison with rate limiting."""
    if not _api_key:
        return True  # No auth required

    # Rate limit check
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    fails = _auth_failures.get(client_ip, [])
    # Prune old failures outside the window
    fails = [t for t in fails if now - t < _AUTH_FAIL_WINDOW]
    _auth_failures[client_ip] = fails

    if len(fails) >= _AUTH_FAIL_MAX:
        _log.warning("Rate limited auth from %s (%d failures)", client_ip, len(fails))
        return False

    import hmac
    auth_header = request.headers.get("Authorization", "")
    key_header = request.headers.get("X-API-Key", "")
    token = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else key_header
    if not token:
        fails.append(now)
        _auth_failures[client_ip] = fails
        return False
    if not hmac.compare_digest(token, _api_key):
        fails.append(now)
        _auth_failures[client_ip] = fails
        return False
    return True


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

class ImageEditRequest(BaseModel):
    file_id: str = Field(..., example="a1b2c3d4e5f6")
    prompt: str = Field(..., example="Make the sky more dramatic")
    mode: Optional[str] = Field("img2img", example="img2img")
    strength: Optional[float] = Field(0.75, ge=0.0, le=1.0)
    num_steps: Optional[int] = Field(30, ge=1, le=100)

class VisionAnalyzeRequest(BaseModel):
    file_id: Optional[str] = None
    image_base64: Optional[str] = None
    prompt: str = Field("Describe this image in detail.", example="What is in this image?")
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    # Optional model override. For Transformers backend this is a directory
    # name in models/ or an HF repo id; for Ollama it's a tag like
    # "qwen3-vl:8b-instruct". When omitted, the active model is used.
    model: Optional[str] = None

class AudioSpeechRequest(BaseModel):
    text: str = Field(..., example="Hello, how are you?")
    model: Optional[str] = None

class AudioTranscriptionRequest(BaseModel):
    file_id: str = Field(..., example="a1b2c3d4e5f6")

class MusicGenerationRequest(BaseModel):
    prompt: str = Field(..., example="An upbeat jazz piano melody")
    duration_seconds: Optional[float] = Field(10.0, ge=1.0, le=30.0)

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., example="A cat playing with a ball")
    num_frames: Optional[int] = Field(16, ge=4, le=32)
    fps: Optional[int] = Field(8, ge=1, le=30)

class Shape3DGenerationRequest(BaseModel):
    prompt: str = Field(..., example="A small red chair")
    num_steps: Optional[int] = Field(64, ge=16, le=128)

class VoiceAssistantRequest(BaseModel):
    text: str = Field(..., example="What's the weather like?")
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    play_audio: Optional[bool] = Field(False)

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


# ── Model resolution ────────────────────────────────────────────────────
#
# Clients shouldn't need to know which models are installed on this server.
# When a request arrives with no model, an empty/sentinel model name, or a
# request type that needs a specific kind of model (e.g. multimodal input
# requires a vision-capable model), the server resolves a sensible default
# from what's actually available on the active backend.
#
# Sentinels: any of these as the request's `model` field means "you pick".
_AUTO_MODEL_SENTINELS = {"", "default", "auto", "none"}

# Vision-capable model name patterns. Order = priority (first match wins).
# Patterns are case-insensitive and matched against either Ollama tag names
# or Transformers directory names.
_VISION_MODEL_PATTERNS = [
    re.compile(r"qwen3.*vl", re.I),       # Qwen3-VL — best for GUI/OCR
    re.compile(r"qwen2\.?5.*vl", re.I),   # Qwen2.5-VL
    re.compile(r"qwen2.*vl", re.I),       # Qwen2-VL
    re.compile(r"llava", re.I),
    re.compile(r"minicpm.*v", re.I),
    re.compile(r"intern.*vl", re.I),
    re.compile(r"gemma.*vision", re.I),
    re.compile(r"pixtral", re.I),
]


def _is_auto_sentinel(name) -> bool:
    """True if `name` is empty/None or one of the auto-resolve keywords."""
    if name is None:
        return True
    return str(name).strip().lower() in _AUTO_MODEL_SENTINELS


def _pick_vision_model(backend: str) -> str | None:
    """Find a vision-capable model installed on `backend`, by name pattern.

    For Ollama we refresh the model list first so a freshly-pulled model
    becomes available without restarting the server. Returns None if no
    matching model is installed.
    """
    from core.config import MODELS, OLLAMA_MODELS, refresh_ollama_models

    if backend == "ollama":
        refresh_ollama_models()
        names = list(OLLAMA_MODELS.keys())
    else:
        names = list(MODELS.keys())

    for pattern in _VISION_MODEL_PATTERNS:
        matches = [n for n in names if pattern.search(n)]
        if not matches:
            continue
        # Prefer -instruct variants (faster, no thinking-token tax)
        instruct = sorted(n for n in matches if "instruct" in n.lower())
        if instruct:
            return instruct[0]
        return sorted(matches)[0]

    return None


def _resolve_model_for_request(requested, has_images: bool, backend: str) -> str:
    """Pick a real installed model for this request.

    Behaviour:
      • Sentinel names ("", "default", "auto", None) → server picks:
          - multimodal request → a vision-capable model from `backend`,
            falling back to the active model if none matches
          - text-only request → the active model
      • Real installed name → returned unchanged
      • Unknown explicit name → 400 with the list of available models,
        so typos surface immediately instead of being forwarded to the
        backend (where they become opaque 404s)

    Raises HTTPException on unresolvable cases.
    """
    # FastAPI is imported lazily inside create_app(), so HTTPException is
    # not available at module scope. Import it locally to keep this helper
    # callable from anywhere in the module.
    from fastapi import HTTPException
    from core.config import MODELS, OLLAMA_MODELS, get_active_model_name, refresh_ollama_models

    if _is_auto_sentinel(requested):
        if has_images:
            picked = _pick_vision_model(backend)
            if picked:
                _log.info("Auto-resolved vision model: %s (%s)", picked, backend)
                return picked
            # Fall through to active model — better than 503 if the user
            # has manually loaded a multimodal model that doesn't match
            # any of our name patterns.
            _log.warning(
                "No vision-capable model auto-detected on %s backend; "
                "using active model. Install one with e.g. "
                "`ollama pull qwen3-vl:8b-instruct`.",
                backend,
            )
        active = get_active_model_name()
        if not active:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"No model loaded on the {backend} backend. "
                    f"Install one and retry."
                ),
            )
        return active

    # Explicit name — validate it actually exists on this backend
    if backend == "ollama":
        if requested in OLLAMA_MODELS:
            return requested
        # Refresh once in case it was just pulled
        refresh_ollama_models()
        if requested in OLLAMA_MODELS:
            return requested
        available = sorted(OLLAMA_MODELS.keys())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{requested}' not installed on Ollama. "
                f"Available: {', '.join(available) if available else '(none — pull one first)'}. "
                f"Pull with: `ollama pull {requested}`"
            ),
        )

    # Transformers
    if requested in MODELS:
        return requested
    available = sorted(MODELS.keys())
    raise HTTPException(
        status_code=400,
        detail=(
            f"Model '{requested}' not in Transformers registry. "
            f"Available: {', '.join(available) if available else '(none in models/)'}."
        ),
    )


def _proxy_ollama_chat(ollama_messages, model, temperature, max_tokens, options, timeout):
    """Forward a request to Ollama's /api/chat and return an OpenAI-format response."""
    from core.config import get_ollama_model_config

    # Start with per-model config (num_ctx, etc.) — only if explicitly set
    model_config = get_ollama_model_config(model)
    ollama_options = {}
    if "num_ctx" in model_config:
        ollama_options["num_ctx"] = model_config["num_ctx"]

    # Merge caller-provided options (can override num_ctx if explicit)
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
    # NOTE: no strip_think_blocks here. Ollama already separates
    # message.thinking from message.content at the protocol level when
    # thinking mode is active, so `content` is already the visible
    # response with thinking excluded. Running strip_think_blocks on top
    # of that was a legacy safety net from before Ollama had native
    # thinking support — and the greedy regex (^.*?</think>) is actively
    # dangerous: if any model hallucinates a stray </think> tag inside
    # its content (or if the resolver picks a thinking-variant model),
    # the regex silently wipes everything before the tag, truncating
    # real output like JSON analysis from vision models.

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
    from core.config import get_ollama_model_config

    # Start with per-model config (num_ctx, etc.) — only if explicitly set
    model_config = get_ollama_model_config(model)
    ollama_options = {}
    if "num_ctx" in model_config:
        ollama_options["num_ctx"] = model_config["num_ctx"]

    # Merge caller-provided options (can override num_ctx if explicit)
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
            _log.warning("Max tool rounds (%d) reached — forcing answer", MAX_TOOL_ROUNDS)
            # Force the model to synthesize an answer with what it has
            current_messages.append({
                "role": "system",
                "content": (
                    "You have used all available research rounds. STOP making tool calls. "
                    "Answer the user's question NOW using the information you have already "
                    "gathered. Do NOT output any @search() or @web_read() calls."
                ),
            })
            # One final generation pass with tools disabled
            q_final = queue.Queue(maxsize=256)
            if backend == "ollama":
                ollama_final = _convert_messages_for_ollama(current_messages)
                t = threading.Thread(
                    target=_stream_ollama_raw,
                    args=(ollama_final, model, temperature, max_tokens, options, 300, q_final),
                    daemon=True,
                )
            else:
                engine = _get_engine()
                t = threading.Thread(
                    target=_stream_transformers_raw,
                    args=(engine, current_messages, max_tokens, temperature, q_final),
                    daemon=True,
                )
            t.start()
            while True:
                item = await loop.run_in_executor(None, q_final.get)
                if item is _SENTINEL:
                    break
                kind, data = item
                if kind == "thinking":
                    yield _make_sse_chunk(chat_id, model, {"x_thinking": data})
                elif kind == "content":
                    yield _make_sse_chunk(chat_id, model, {"content": data})
            t.join(timeout=10)
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

    # CORS: allow localhost and LAN origins for local dev + upstream-client Web Suite
    cors_origins = os.environ.get("ARTIFEX_CORS_ORIGINS", "").split(",")
    cors_origins = [o.strip() for o in cors_origins if o.strip()]
    cors_origins += [
        "http://localhost",
        "http://localhost:5173",    # WebGPU dev server
        "http://localhost:3001",    # Metrics server
        "http://localhost:5000",    # Common dev server port
        "http://localhost:8080",    # Common dev server port
        "http://localhost:8114",    # upstream client suite
        "http://127.0.0.1",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8114",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_origin_regex=r"http://192\.168\.\d+\.\d+:\d+",  # LAN access
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
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

    # ─── Register model queue callback ────────────────────────────────────
    # Provides model_queue a way to unload the transformers engine without
    # importing api.server directly (keeps core/ decoupled from api/).
    def _unload_transformers_engine():
        global _engine
        with _engine_lock:
            if _engine is not None:
                _engine.unload()
                _engine = None

    get_model_queue().register_transformers_unload(_unload_transformers_engine)

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
        use_web_tools = body.web_tools or False
        backend = get_active_backend()

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # Resolve the model up front. Sentinel names ("default", "auto", "")
        # and image-bearing requests are auto-routed to a sensible installed
        # model. Unknown names raise 400 here so they don't reach Ollama as
        # opaque 404s. See _resolve_model_for_request for full behaviour.
        has_images = _messages_have_images(messages)
        model = _resolve_model_for_request(body.model, has_images, backend)

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

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "text-to-image",
                kwargs={
                    "prompt": body.prompt[:2000],
                    "negative_prompt": (body.negative_prompt or "")[:500],
                    "width": body.width,
                    "height": body.height,
                    "num_steps": body.steps,
                },
                store_output=True,
            ))
            if result.success:
                return {
                    "created": int(time.time()),
                    "data": [{
                        "file_id": result.metadata.get("file_id"),
                        "url": f"/v1/files/{result.metadata.get('file_id', '')}",
                    }],
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

    # ─── File Management ────────────────────────────────────────────────

    @app.post("/v1/files")
    async def upload_file(request: Request):
        """Upload a file (image, audio, video, document). Multipart form data."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        try:
            from fastapi import UploadFile, File as FastAPIFile
        except ImportError:
            raise HTTPException(status_code=500, detail="FastAPI file support unavailable")

        form = await request.form()
        upload = form.get("file")
        if upload is None:
            raise HTTPException(status_code=400, detail="No file provided. Use multipart form with 'file' field.")

        purpose = form.get("purpose", "upload")
        file_bytes = await upload.read()
        filename = upload.filename or "unknown"

        from core.services.file_manager import _detect_file_type
        file_type = _detect_file_type(filename)
        if file_type == "unknown":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {filename}. "
                       "Accepted: images, audio, video, documents, meshes.",
            )

        from core.services import get_file_manager
        fm = get_file_manager()
        record = fm.store_upload(file_bytes, filename, upload.content_type)

        return {
            "id": record.file_id,
            "object": "file",
            "filename": record.original_name,
            "purpose": purpose,
            "bytes": record.size_bytes,
            "file_type": record.file_type,
            "created_at": int(record.created_at),
        }

    @app.get("/v1/files")
    async def list_files(request: Request, purpose: Optional[str] = None,
                         file_type: Optional[str] = None):
        """List uploaded and generated files."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_file_manager
        fm = get_file_manager()
        records = fm.list_files(file_type=file_type, purpose=purpose)

        return {
            "object": "list",
            "data": [
                {
                    "id": r.file_id,
                    "object": "file",
                    "filename": r.original_name,
                    "purpose": r.purpose,
                    "bytes": r.size_bytes,
                    "file_type": r.file_type,
                    "created_at": int(r.created_at),
                }
                for r in records
            ],
        }

    @app.get("/v1/files/{file_id}")
    async def download_file(request: Request, file_id: str):
        """Download a file by its ID."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from fastapi.responses import FileResponse
        from core.services import get_file_manager
        fm = get_file_manager()

        record = fm.get_file(file_id)
        if record is None:
            raise HTTPException(status_code=404, detail="File not found")

        path = fm.get_file_path(file_id)
        if path is None:
            raise HTTPException(status_code=404, detail="File data missing from disk")

        return FileResponse(
            path=path,
            media_type=record.content_type,
            filename=record.original_name,
        )

    @app.delete("/v1/files/{file_id}")
    async def delete_file(request: Request, file_id: str):
        """Delete a file by its ID."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_file_manager
        fm = get_file_manager()

        if not fm.delete_file(file_id):
            raise HTTPException(status_code=404, detail="File not found")

        return {"id": file_id, "object": "file", "deleted": True}

    # ─── Vision (Image Understanding) ────────────────────────────────────

    @app.post("/v1/vision/analyze")
    async def vision_analyze(request: Request, body: VisionAnalyzeRequest):
        """Analyze an image using a vision-capable model.

        Routing:
          • Ollama backend → forwards to /api/chat with the image attached.
            Works with any multimodal Ollama model (qwen3-vl, llava, etc.).
          • Transformers backend → uses the cached vision pipeline if a
            multimodal model is already loaded, or loads `body.model` /
            registry-resolved path. Returns 503 with a clear message if no
            vision model can be resolved.
        """
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        import base64 as _b64
        import tempfile
        from core.config import (
            MODELS, OLLAMA_MODELS, get_active_backend, get_active_model_name,
        )
        from core.services import get_service

        svc = get_service()

        # ── Resolve image bytes (need both raw bytes for Ollama and a
        # filesystem path for the Transformers pipeline). ────────────────
        if body.file_id:
            image_path = svc.file_manager.get_file_path(body.file_id)
            if image_path is None:
                raise HTTPException(status_code=404, detail="File not found")
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            img_b64 = _b64.b64encode(img_bytes).decode("ascii")
        elif body.image_base64:
            try:
                img_bytes = _b64.b64decode(body.image_base64)
            except Exception:
                raise HTTPException(status_code=400, detail="image_base64 is not valid base64")
            img_b64 = body.image_base64
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(img_bytes)
            tmp.close()
            image_path = tmp.name
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either file_id or image_base64",
            )

        backend = get_active_backend()
        # Vision endpoint always implies a multimodal request, so the
        # resolver picks a vision-capable model regardless of body.model.
        requested_model = _resolve_model_for_request(body.model, has_images=True, backend=backend)

        # ── Ollama path: proxy as a multimodal chat completion ──────────
        if backend == "ollama":
            ollama_messages = [{
                "role": "user",
                "content": body.prompt,
                "images": [img_b64],
            }]
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: _proxy_ollama_chat(
                        ollama_messages, requested_model, 0.1,
                        body.max_tokens, None, 600,
                    ),
                )
                content = result["choices"][0]["message"]["content"]
                return {
                    "object": "vision.analysis",
                    "analysis": content,
                    "model": requested_model,
                }
            except (RuntimeError, ConnectionError) as e:
                _log.error("Ollama vision proxy failed: %s", e)
                raise HTTPException(status_code=502, detail=str(e))

        # ── Transformers path: resolve a real model_path before loading ─
        # Priority: explicit body.model → already-cached vision pipeline →
        # registry lookup. We refuse to call pipeline.load("") because that
        # surfaces as a confusing "Repo id ''" error from HuggingFace.
        loaded = svc.get_loaded_pipelines()
        already_cached = "image-text-to-text" in loaded

        model_path = ""
        if body.model:
            model_path = MODELS.get(body.model, body.model)
        elif not already_cached:
            # Nothing is cached and the caller didn't tell us what to load —
            # try the active model name as a registry lookup.
            active = get_active_model_name()
            if active and active in MODELS:
                model_path = MODELS[active]

        if not already_cached and not model_path:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No vision model loaded. Either (a) load a multimodal model "
                    "via the Control Center on the vision pipeline, (b) include "
                    "`model` in the request body (directory name in models/ or "
                    "HF repo id), or (c) switch the API server to the Ollama "
                    "backend with a multimodal Ollama model installed."
                ),
            )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "image-text-to-text",
                model_path=model_path,
                kwargs={
                    "image_path": image_path,
                    "prompt": body.prompt,
                    "max_tokens": body.max_tokens,
                },
                store_output=False,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "vision.analysis",
                "analysis": result.content,
                "model": result.metadata.get("model", body.model or "vision"),
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("Vision pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"Vision pipeline failed: {e}")

    # ─── Audio TTS ───────────────────────────────────────────────────────

    @app.post("/v1/audio/speech")
    async def audio_speech(request: Request, body: AudioSpeechRequest):
        """Generate speech from text (TTS)."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "text-to-audio",
                model_path=body.model or "",
                kwargs={"text": body.text[:5000]},
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            file_id = result.metadata.get("file_id")
            return {
                "object": "audio.speech",
                "file_id": file_id,
                "content_type": "audio/wav",
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("TTS pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"TTS pipeline failed: {e}")

    # ─── Audio STT ───────────────────────────────────────────────────────

    @app.post("/v1/audio/transcriptions")
    async def audio_transcriptions(request: Request,
                                   body: AudioTranscriptionRequest = None):
        """Transcribe audio to text (STT). Accepts file_id."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        audio_path = svc.file_manager.get_file_path(body.file_id)
        if audio_path is None:
            raise HTTPException(status_code=404, detail="Audio file not found")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "automatic-speech-recognition",
                kwargs={"audio_path": audio_path},
                store_output=False,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {"object": "transcription", "text": result.content}
        except HTTPException:
            raise
        except Exception as e:
            _log.error("STT pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"STT pipeline failed: {e}")

    # ─── Music Generation ────────────────────────────────────────────────

    @app.post("/v1/audio/music")
    async def music_generation(request: Request, body: MusicGenerationRequest):
        """Generate music from a text description."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "text-to-music",
                kwargs={
                    "prompt": body.prompt[:1000],
                    "duration_seconds": body.duration_seconds,
                },
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "audio.music",
                "file_id": result.metadata.get("file_id"),
                "duration_seconds": body.duration_seconds,
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("Music pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"Music pipeline failed: {e}")

    # ─── Image Editing ───────────────────────────────────────────────────

    @app.post("/v1/images/edits")
    async def image_edits(request: Request, body: ImageEditRequest):
        """Edit an image (img2img, inpaint, upscale)."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        image_path = svc.file_manager.get_file_path(body.file_id)
        if image_path is None:
            raise HTTPException(status_code=404, detail="Source image not found")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "image-to-image",
                kwargs={
                    "image_path": image_path,
                    "prompt": body.prompt[:2000],
                    "strength": body.strength,
                    "num_steps": body.num_steps,
                },
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "image.edit",
                "file_id": result.metadata.get("file_id"),
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("Image edit pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"Image edit pipeline failed: {e}")

    # ─── Video Generation ────────────────────────────────────────────────

    @app.post("/v1/video/generations")
    async def video_generations(request: Request, body: VideoGenerationRequest):
        """Generate a video from a text prompt."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "text-to-video",
                kwargs={
                    "prompt": body.prompt[:2000],
                    "num_frames": body.num_frames,
                    "fps": body.fps,
                },
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "video.generation",
                "file_id": result.metadata.get("file_id"),
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("Video pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"Video pipeline failed: {e}")

    # ─── 3D Generation ───────────────────────────────────────────────────

    @app.post("/v1/3d/generations")
    async def shape_3d_generations(request: Request,
                                   body: Shape3DGenerationRequest):
        """Generate a 3D mesh from a text prompt."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "shap-e",
                kwargs={
                    "prompt": body.prompt[:1000],
                    "num_steps": body.num_steps,
                },
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "3d.generation",
                "file_id": result.metadata.get("file_id"),
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("3D pipeline error: %s", e)
            raise HTTPException(status_code=500, detail=f"3D pipeline failed: {e}")

    @app.post("/v1/voice/chat")
    async def voice_chat(request: Request,
                         body: VoiceAssistantRequest):
        """Send text to Artifex Voice Assistant and get a spoken response."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        from core.services import get_service
        svc = get_service()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: svc.run_pipeline(
                "voice-assistant",
                kwargs={
                    "text": body.text[:2000],
                    "max_tokens": body.max_tokens,
                    "temperature": body.temperature,
                    "play_audio": body.play_audio,
                },
                store_output=True,
            ))

            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)

            return {
                "object": "voice.chat",
                "response_text": result.metadata.get("response_text", ""),
                "user_text": result.metadata.get("user_text", body.text),
                "audio_file_id": result.metadata.get("file_id"),
                "audio_chunks": result.metadata.get("audio_chunks", 0),
            }
        except HTTPException:
            raise
        except Exception as e:
            _log.error("Voice assistant error: %s", e)
            raise HTTPException(status_code=500, detail=f"Voice assistant failed: {e}")

    return app
