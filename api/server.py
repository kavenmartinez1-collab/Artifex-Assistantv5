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
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field

from core.config import (
    MODES, get_active_backend, get_active_model_name, get_model_names,
)
from core.engine_factory import create_engine
from core.inference import strip_think_blocks, ThinkFilter, trim_messages_to_context
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


def _get_engine(ctx_tier: int | None = None):
    """Get or create the active engine.

    Args:
        ctx_tier: For llama_cpp only — sets the launch ctx for the next
            load() so the server starts with a tier sized for the actual
            request rather than the model's max ctx.  Ignored for engines
            that don't expose set_target_tier (Ollama, Transformers).
    """
    global _engine
    with _engine_lock:
        if _engine is None or not _engine.is_loaded():
            # Ensure Ollama models are discovered before engine init
            from core.config import get_active_backend, refresh_ollama_models
            if get_active_backend() == "ollama":
                refresh_ollama_models()
            _engine = create_engine()
            if ctx_tier and hasattr(_engine, "set_target_tier"):
                _engine.set_target_tier(ctx_tier)
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
    model: Optional[str] = Field(None, description="Model name, or omit for auto-selection")
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: Optional[bool] = False
    options: Optional[dict] = None
    web_tools: Optional[bool] = False
    grammar: Optional[str] = Field(None, description="GBNF grammar string for constrained generation (llama.cpp only)")
    response_format: Optional[dict] = Field(None, description="Response format constraint, e.g. {\"type\": \"json_object\"} or {\"type\": \"json_schema\", \"json_schema\": {...}}")
    backend: Optional[Literal["ollama", "transformers", "llama_cpp"]] = Field(
        None,
        description=(
            "Backend override for this single request. When set, the model "
            "queue switches to this backend (unloading the previous one) "
            "before serving. When omitted, the backend is inferred from "
            "`model` if possible, else falls back to the server-wide active "
            "backend. This enables per-stage model switching within one "
            "task without restarting the server."
        ),
    )

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
    # Per-request backend override. `llama_cpp` is rejected here because
    # llama.cpp has no vision pipeline in this codebase — vision goes
    # through Ollama (image VLMs) or transformers (full VL family + video).
    backend: Optional[Literal["ollama", "transformers"]] = Field(
        None,
        description=(
            "Backend override. Omit to infer from `model` or fall back to "
            "the active backend. llama_cpp is not supported on this endpoint."
        ),
    )

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

class TokenizeRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw text to tokenize")
    messages: Optional[List[ChatMessage]] = Field(None, description="Messages to tokenize (rendered via chat template)")


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

from core.engine_ollama import OLLAMA_BASE_URL as _OLLAMA_BASE
OLLAMA_CHAT_URL = f"{_OLLAMA_BASE}/api/chat"
_ollama_no_think: set = set()


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


def _describe_content_shape(messages) -> str:
    """Compact per-message content shape for logging.

    Format: `role(part:size,part:size) ...` — e.g.
        system(text:500) user(text:17000,image_url:application/pdf:2048KB)

    Purpose: when requests fail or return wrong output, the log shows what
    shape the client actually sent — in particular whether attachments came
    through as image_url data URLs and what mime type (image/png vs
    application/pdf vs etc.), since PDF sent as image_url is a common
    silently-dropped case with text-only models.
    """
    out = []
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content")
        # Ollama post-conversion: content is a string plus optional `images`
        # (list of base64 strings). OpenAI multimodal: content is a list of
        # {type, text/image_url} dicts. Handle both so callers can use the
        # same helper regardless of where they sit in the pipeline.
        if isinstance(content, str):
            parts = [f"text:{len(content)}"]
            images = msg.get("images")
            if isinstance(images, list) and images:
                total_kb = sum(
                    len(img) * 3 // 4 // 1024
                    for img in images if isinstance(img, str)
                )
                parts.append(f"images[{len(images)}]:~{total_kb}KB")
            out.append(f"{role}({','.join(parts)})")
            continue
        if not isinstance(content, list):
            out.append(f"{role}(other:{type(content).__name__})")
            continue
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type", "?")
            if t == "text":
                parts.append(f"text:{len(item.get('text', ''))}")
            elif t == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    header_end = url.find(",")
                    header = url[5:header_end] if header_end > 0 else url[5:]
                    mime = header.split(";", 1)[0] or "unknown"
                    b64_len = len(url) - header_end - 1 if header_end > 0 else 0
                    size_kb = b64_len * 3 // 4 // 1024
                    parts.append(f"image_url:{mime}:{size_kb}KB")
                else:
                    parts.append("image_url:url")
            else:
                parts.append(t)
        out.append(f"{role}({','.join(parts) if parts else 'empty'})")
    return " ".join(out)


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


def _infer_backend_from_model(name) -> str | None:
    """Look up which backend hosts a given model name.

    Returns the backend id ("ollama", "llama_cpp", "transformers") if the
    name matches an installed/configured model, else None. Sentinel names
    return None — let the caller fall back to the active backend.

    Probe order is "cheapest first, most-likely first": Ollama's tag list
    is an in-memory dict, llama.cpp is a small JSON config, and the
    transformers MODELS map is also in memory. If a name appears under
    multiple backends, the first hit wins (Ollama → llama.cpp → transformers).
    Callers wanting a specific backend should set request.backend explicitly.
    """
    if _is_auto_sentinel(name):
        return None

    from core.config import (
        MODELS, OLLAMA_MODELS, get_llama_cpp_models, refresh_ollama_models,
    )

    if name in OLLAMA_MODELS:
        return "ollama"
    # Refresh once in case it was just pulled — same pattern as
    # _resolve_model_for_request uses for explicit names.
    refresh_ollama_models()
    if name in OLLAMA_MODELS:
        return "ollama"

    if name in get_llama_cpp_models():
        return "llama_cpp"

    if name in MODELS:
        return "transformers"

    return None


def _pick_vision_model(backend: str) -> str | None:
    """Find a vision-capable model installed on `backend`.

    Looks at the backend's own model registry — never returns a name from
    a different backend's registry, since the caller routes the request
    via `backend` and an unknown name would silently fall through
    set_active_model and stay on the previously-loaded model.

    Per backend:
      - ollama: OLLAMA_MODELS, name-pattern match. Refreshes the tag list
        first so a freshly-pulled model is visible without restart.
      - llama_cpp: llama_cpp_config.json models. Prefers entries with
        '--mmproj' in extra_flags (the explicit signal a llama-server
        launch can do vision), then falls back to name patterns.
      - transformers: MODELS dict, name-pattern match.

    Returns None if no matching model is installed on `backend`.
    """
    from core.config import (
        MODELS, OLLAMA_MODELS, refresh_ollama_models, get_llama_cpp_models,
    )

    if backend == "ollama":
        refresh_ollama_models()
        names = list(OLLAMA_MODELS.keys())
    elif backend == "llama_cpp":
        lcpp = get_llama_cpp_models()
        # --mmproj is the explicit "this launch can do vision" signal in
        # llama_cpp_config.json. Pick from that set first; only fall back
        # to name patterns if no entry advertises mmproj.
        mmproj_names = [
            n for n, cfg in lcpp.items()
            if "--mmproj" in (cfg.get("extra_flags") or [])
        ]
        if mmproj_names:
            instruct = sorted(n for n in mmproj_names if "instruct" in n.lower())
            return instruct[0] if instruct else sorted(mmproj_names)[0]
        names = list(lcpp.keys())
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
        # Model not found — fall back to smallest available model
        if not OLLAMA_MODELS:
            from core.model_discovery import get_model_names_for_backend
            available = get_model_names_for_backend("ollama")
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{requested}' not found and no Ollama models available. "
                    f"Install one with: ollama pull <model>"
                ),
            )
        # For text requests, skip VLM models (they're optimized for vision)
        candidates = list(OLLAMA_MODELS.items())
        if not has_images:
            text_only = [(n, i) for n, i in candidates if "vl" not in n.lower() and "vision" not in n.lower()]
            if text_only:
                candidates = text_only
        smallest = min(candidates, key=lambda x: x[1].get("size", float("inf")))
        _log.warning("Model '%s' not found, falling back to %s (smallest)", requested, smallest[0])
        return smallest[0]

    if backend == "llama_cpp":
        from core.config import get_llama_cpp_models
        lcpp_models = get_llama_cpp_models()
        if requested in lcpp_models:
            return requested
        available = sorted(lcpp_models.keys())
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{requested}' not in llama.cpp config. "
                f"Available: {', '.join(available) if available else '(none — add models to llama_cpp_config.json)'}."
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
    from core.ollama_ctx import compute_safe_ctx, estimate_prompt_tokens, should_disable_mmap

    model_config = get_ollama_model_config(model)
    ollama_options = {}

    # Auto-size num_ctx from prompt + VRAM budget. Per-model config can pin
    # num_ctx outright or set a max_ctx ceiling (see core/ollama_ctx.py).
    est_tokens = estimate_prompt_tokens(ollama_messages)
    safe_ctx = compute_safe_ctx(model, est_tokens, model_config)
    ollama_options["num_ctx"] = safe_ctx

    if options:
        ollama_options.update(options)
        if "num_ctx" in options and int(options["num_ctx"]) > safe_ctx:
            _log.warning(
                "Client requested num_ctx=%s for %s, capping to VRAM-safe %d",
                options["num_ctx"], model, safe_ctx,
            )
            ollama_options["num_ctx"] = safe_ctx
    if temperature is not None:
        ollama_options["temperature"] = temperature
    ollama_options["num_predict"] = max_tokens if max_tokens and max_tokens > 0 else -1

    if should_disable_mmap(model, model_config):
        ollama_options["use_mmap"] = False

    # Scale socket timeout with num_ctx. A 300s text default works for 8B
    # models on small prompts, but 27B 2-bit doing 17K-token prompt + 5K-gen
    # legitimately runs 6-7 min. num_ctx is a reasonable proxy for worst-case
    # work, and caps at 1800s (30 min) to still catch genuine hangs.
    timeout = max(timeout, min(1800, ollama_options["num_ctx"] // 30))

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

    # Ollama's done_reason is "stop" (EOS), "length" (hit num_predict),
    # "load" (just loaded the model), etc. Map to OpenAI finish_reason:
    # only "length" and "stop" are meaningful to OpenAI clients — pass
    # "length" through verbatim, treat everything else as a natural stop.
    done_reason = data.get("done_reason", "stop")
    finish_reason = "length" if done_reason == "length" else "stop"

    preview = content[:160].replace("\n", "\\n").replace("\r", "")
    _log.info(
        "Ollama proxy complete: model=%s num_ctx=%d prompt=%d completion=%d content=%d finish=%s preview=%r",
        model, ollama_options.get("num_ctx", 0),
        prompt_tokens, completion_tokens, len(content), finish_reason, preview,
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "x_backend": "ollama",
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
    from core.ollama_ctx import compute_safe_ctx, estimate_prompt_tokens, should_disable_mmap

    model_config = get_ollama_model_config(model)
    ollama_options = {}

    # Auto-size num_ctx from prompt + VRAM budget.
    est_tokens = estimate_prompt_tokens(ollama_messages)
    safe_ctx = compute_safe_ctx(model, est_tokens, model_config)
    ollama_options["num_ctx"] = safe_ctx

    if options:
        ollama_options.update(options)
        if "num_ctx" in options and int(options["num_ctx"]) > safe_ctx:
            _log.warning(
                "Client requested num_ctx=%s for %s, capping to VRAM-safe %d",
                options["num_ctx"], model, safe_ctx,
            )
            ollama_options["num_ctx"] = safe_ctx
    if temperature is not None:
        ollama_options["temperature"] = temperature
    ollama_options["num_predict"] = max_tokens if max_tokens and max_tokens > 0 else -1

    if should_disable_mmap(model, model_config):
        ollama_options["use_mmap"] = False

    # See _proxy_ollama_chat for rationale — same num_ctx-scaled timeout.
    timeout = max(timeout, min(1800, ollama_options["num_ctx"] // 30))

    _log.info(
        "Ollama stream start: model=%s num_ctx=%d est_prompt_tokens=%d shape=[%s]",
        model, ollama_options["num_ctx"], est_tokens,
        _describe_content_shape(ollama_messages),
    )

    payload = {
        "model": model,
        "messages": ollama_messages,
        "stream": True,
        "options": ollama_options,
    }
    if model not in _ollama_no_think:
        payload["think"] = True

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_CHAT_URL, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 400 and b"does not support thinking" in (e.read()):
            _ollama_no_think.add(model)
            _log.info("Model %s: think not supported, disabled for future requests", model)
            payload.pop("think", None)
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                OLLAMA_CHAT_URL, data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
            except Exception as e2:
                out_queue.put(("error", str(e2)))
                out_queue.put(_SENTINEL)
                return
        else:
            out_queue.put(("error", str(e)))
            out_queue.put(_SENTINEL)
            return
    except Exception as e:
        out_queue.put(("error", str(e)))
        out_queue.put(_SENTINEL)
        return

    tf = ThinkFilter(
        on_response=lambda t: out_queue.put(("content", t)),
        on_thinking=lambda t: out_queue.put(("thinking", t)),
        starts_in_think=False,
    )

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
                if thinking:
                    out_queue.put(("content", content))
                else:
                    tf.feed(content)

            if chunk.get("done", False):
                # Pass Ollama's done_reason through so the caller can render
                # an honest finish_reason. "length" = hit num_predict cap,
                # anything else collapses to a natural "stop".
                out_queue.put(("usage", {
                    "prompt_tokens": chunk.get("prompt_eval_count", 0),
                    "completion_tokens": chunk.get("eval_count", 0),
                    "done_reason": chunk.get("done_reason", "stop"),
                }))
                break
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        tf.flush()
        resp.close()
        out_queue.put(_SENTINEL)


# ── Transformers streaming ──────────────────────────────────────────────

def _stream_transformers_raw(engine, messages, max_tokens, temperature,
                             out_queue: queue.Queue,
                             grammar=None, response_format=None,
                             enable_thinking=True):
    """Stream from Transformers/llama.cpp engine into a queue. Runs in a background thread.

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
            messages, max_tokens=max_tokens if max_tokens and max_tokens > 0 else -1,
            temperature=temperature, on_token=on_token,
            grammar=grammar, response_format=response_format,
            raw_output=True,
            enable_thinking=enable_thinking,
        )
        tf.flush()

        # Use real token counts from engine._last_gen_stats (populated by
        # TransformersEngine.generate_streaming) instead of char//4 heuristic.
        stats = getattr(engine, "_last_gen_stats", None) or {}
        out_queue.put(("usage", {
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "completion_tokens": stats.get("completion_tokens", 0),
            "finish_reason": stats.get("finish_reason", "stop"),
        }))
    except Exception as e:
        out_queue.put(("error", str(e)))
    finally:
        out_queue.put(_SENTINEL)


_WEB_TOOL_SYSTEM_PROMPT = (
    "You have access to web search tools. When answering questions that "
    "would benefit from current or external information, use them:\n"
    "- @search(\"query\") — web search. Returns numbered results.\n"
    "- @web_read(N) — read full page for search result N.\n"
    "- @web_read(\"url\") — read a specific URL.\n"
    "Write tool calls on their own line. After using @search(), review "
    "the results and use @web_read(N) to get details from the most "
    "relevant pages. Then answer the question using what you found."
)


def _inject_web_tool_prompt(messages: list):
    """Prepend web tool instructions to the conversation's system prompt.

    If the first message is a system message, appends the tool instructions
    to it. Otherwise inserts a new system message at the front.
    """
    if messages and messages[0].get("role") == "system":
        messages[0] = dict(messages[0])
        messages[0]["content"] = (
            messages[0]["content"] + "\n\n" + _WEB_TOOL_SYSTEM_PROMPT
        )
    else:
        messages.insert(0, {"role": "system", "content": _WEB_TOOL_SYSTEM_PROMPT})


def _get_context_budget(backend: str) -> int:
    """Return input token budget for the current engine, or 0 if unknown."""
    if backend == "ollama":
        return 0
    try:
        engine = _get_engine()
        ctx = engine.get_context_size()
        if ctx > 0:
            return int(ctx * 0.85)
    except Exception as e:
        _log.debug("Could not query engine context size: %s", e)
    return 0


# ── Unified streaming with optional tool execution ──────────────────────

async def _stream_with_tools(messages: list, model: str, max_tokens: int,
                             temperature: float, options: dict,
                             use_web_tools: bool, backend: str,
                             grammar=None, response_format=None,
                             enable_thinking=True):
    """Full streaming generator with optional tool execution loop.

    Yields SSE events. Handles both Ollama and Transformers backends.
    """
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    search_cache = []  # Request-scoped cache for @web_read(N)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    last_finish_reason = "stop"  # Updated from engine usage events

    current_messages = list(messages)
    msg_tokens_est = sum(len(m.get("content", "")) for m in current_messages) // 4

    _log.info("[%s] web_tools=%s, backend=%s, %d messages (~%d tok), max_tokens=%s",
              chat_id, use_web_tools, backend, len(current_messages), msg_tokens_est, max_tokens)

    if use_web_tools:
        _inject_web_tool_prompt(current_messages)
        _log.info("[%s] Injected web tool system prompt", chat_id)

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
            # Both transformers and llama_cpp use BaseEngine.generate_streaming
            engine = _get_engine()
            thread = threading.Thread(
                target=_stream_transformers_raw,
                args=(engine, current_messages, max_tokens, temperature, q,
                      grammar, response_format, enable_thinking),
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
        # Derive finish_reason: Ollama passes done_reason, Transformers
        # passes finish_reason directly. Map Ollama's "length" through;
        # everything else collapses to "stop".
        done_reason = usage.get("done_reason")
        if done_reason is not None:
            last_finish_reason = "length" if done_reason == "length" else "stop"
        else:
            last_finish_reason = usage.get("finish_reason", "stop")

        _log.info("[%s] Round %d complete: %d chars, response='%s'",
                   chat_id, round_count, len(full_response), full_response[:200])

        # ── Tool execution check ──────────────────────────────────────
        if not use_web_tools:
            _log.info("[%s] web_tools disabled — returning final response", chat_id)
            break

        tools = extract_web_tools(full_response)
        has_gw = gateway_available() if tools else False
        _log.info("[%s] Tool extraction: found %d tool calls, gateway_up=%s",
                  chat_id, len(tools), has_gw)
        if tools:
            for t in tools:
                _log.info("[%s]   → %s: %s", chat_id, t.get("type"), t.get("query") or t.get("ref"))

        if not tools or not has_gw:
            if tools and not has_gw:
                _log.warning("[%s] Model emitted tool calls but web gateway unavailable — "
                             "returning raw response", chat_id)
            elif not tools and use_web_tools:
                _log.info("[%s] Model did NOT emit any @search/@web_read calls despite "
                          "web_tools=true — check system prompt compliance", chat_id)
            break

        round_count += 1
        if round_count > MAX_TOOL_ROUNDS:
            _log.warning("Max tool rounds (%d) reached — forcing answer", MAX_TOOL_ROUNDS)
            budget = _get_context_budget(backend)
            if budget > 0:
                current_messages = trim_messages_to_context(current_messages, budget)
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
                    args=(engine, current_messages, max_tokens, temperature, q_final,
                          grammar, response_format, enable_thinking),
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

        _log.info("[%s] Tool round %d: executing %d tools", chat_id, round_count, len(tools))

        # Notify client that tools are executing
        labels = tool_status_labels(tools)
        yield f"data: {json.dumps({'x_tool_status': labels})}\n\n"

        # Execute tools (blocking, but SSE connection stays open)
        try:
            tool_output = await loop.run_in_executor(
                None, lambda: execute_web_tools(tools, search_cache)
            )
            _log.info("[%s] Tool results: %d chars, preview='%s'",
                      chat_id, len(tool_output), tool_output[:200])
        except Exception as e:
            _log.error("[%s] Tool execution error: %s", chat_id, e)
            tool_output = f"Tool execution failed: {e}"

        # Signal client to reset for follow-up response
        yield f"data: {json.dumps({'x_tool_done': True})}\n\n"

        # Build follow-up messages with tool results
        msg_count_before = len(current_messages)
        current_messages.append({"role": "assistant", "content": full_response})
        current_messages.append({
            "role": "user",
            "content": (
                f"[Tool results — use this data to answer the original question]\n\n"
                f"{tool_output}"
            ),
        })

        # Compact then trim to fit context window
        from core.inference import auto_compact_if_needed
        budget = _get_context_budget(backend)
        est_tokens = sum(len(m.get("content", "")) for m in current_messages) // 4
        _log.info("[%s] Pre-trim: %d messages (~%d tok), ctx_budget=%d",
                  chat_id, len(current_messages), est_tokens, budget)
        if budget > 0:
            current_messages, compacted = auto_compact_if_needed(
                current_messages, budget, context_window=15
            )
            if compacted:
                _log.info("[%s] Auto-compacted to %d messages", chat_id, len(current_messages))
            current_messages = trim_messages_to_context(current_messages, budget)
            new_est = sum(len(m.get("content", "")) for m in current_messages) // 4
            if new_est < est_tokens:
                _log.info("[%s] Trimmed: %d messages (~%d tok)", chat_id, len(current_messages), new_est)
        _log.info("[%s] Starting follow-up round %d", chat_id, round_count + 1)
        # Loop continues with new generation round

    # ── Final events ──────────────────────────────────────────────────
    yield _make_sse_chunk(chat_id, model, {}, finish_reason=last_finish_reason,
                          extra={"x_usage": total_usage, "x_backend": backend})
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

    from core.sandbox import install_all_hooks
    install_all_hooks()

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
    async def list_models(request: Request, backend: str = None):
        """List available models. Returns all backends by default, or filter with ?backend=ollama."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")
        from core.model_discovery import discover_all, discover_by_backend
        from core.config import get_active_model_name

        if backend:
            models = discover_by_backend(backend)
        else:
            models = discover_all()

        active_model = get_active_model_name()
        active_backend = get_active_backend()

        data = [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
                "x_backend": m["backend"],
                "x_capabilities": m.get("capabilities", ["text"]),
                "x_size": m.get("size", 0),
                # x_context_window: model's max input+output token capacity.
                # 0 = unknown (Ollama unreachable, missing config.json, etc.)
                # Clients can use this to decide whether to chunk long jobs
                # or pick a different model.
                "x_context_window": int(m.get("context_window") or 0),
                "x_vision": "vision" in m.get("capabilities", []),
                "x_active": (m["id"] == active_model and m["backend"] == active_backend),
            }
            for m in models
        ]
        return {"object": "list", "data": data}

    # ─── Tokenize ────────────────────────────────────────────────────────

    @app.post("/v1/tokenize")
    async def tokenize(request: Request, body: TokenizeRequest):
        """Count tokens in text or messages. Uses the active engine's tokenizer."""
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not body.text and not body.messages:
            raise HTTPException(status_code=400, detail="Provide 'text' or 'messages'")

        if body.messages:
            text = "\n".join(
                f"{m.role}: {m.content}" if isinstance(m.content, str)
                else f"{m.role}: [multimodal]"
                for m in body.messages
            )
        else:
            text = body.text

        engine = _get_engine()
        count = engine.count_tokens(text)
        return {"token_count": count, "text_length": len(text)}

    # ─── Chat Completions ─────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, body: ChatCompletionRequest):
        if not _check_auth(request):
            raise HTTPException(status_code=401, detail="Invalid API key")

        _DEFAULT_MAX_TOKENS = 32768

        messages = [m.model_dump() for m in body.messages]
        max_tokens = body.max_tokens or _DEFAULT_MAX_TOKENS
        temperature = body.temperature
        stream = body.stream
        use_web_tools = body.web_tools or False
        options = body.options or {}
        enable_thinking = options.get("think", True)
        backend = (
            body.backend
            or _infer_backend_from_model(body.model)
            or get_active_backend()
        )
        _log.info("POST /v1/chat/completions: model=%s, web_tools=%s, stream=%s, "
                  "backend=%s, %d messages",
                  body.model, use_web_tools, stream, backend, len(messages))

        if not messages:
            raise HTTPException(status_code=400, detail="messages is required")

        # Resolve the model up front. Sentinel names ("default", "auto", "")
        # and image-bearing requests are auto-routed to a sensible installed
        # model. Unknown names raise 400 here so they don't reach Ollama as
        # opaque 404s. See _resolve_model_for_request for full behaviour.
        has_images = _messages_have_images(messages)
        model = _resolve_model_for_request(body.model, has_images, backend)

        # ── Request resource estimation (pre-routing diagnostics) ─────
        from core.request_estimator import estimate_request_requirements
        model_config = None
        if backend == "llama_cpp":
            from core.config import get_llama_cpp_model_config
            model_config = get_llama_cpp_model_config(model)
        req_estimate = estimate_request_requirements(
            messages, max_tokens,
            web_tools=use_web_tools,
            enable_thinking=enable_thinking,
            model_config=model_config,
        )
        if model_config and model_config.get("num_ctx"):
            configured_ctx = model_config["num_ctx"]
            if req_estimate.total_context_needed > configured_ctx:
                _log.warning(
                    "Request may exceed context: estimated %d tokens needed, "
                    "server configured for %d (model=%s)",
                    req_estimate.total_context_needed, configured_ctx, model,
                )

        # ── Pick ctx tier for llama_cpp launch ────────────────────────
        # The tier is a small bucket sized for the actual request, much
        # smaller than the model's max ctx — saves several GB of KV cache
        # on every request that doesn't need a 256K window.  None for other
        # backends; the queue and engine ignore it there.
        ctx_tier: int | None = None
        if backend == "llama_cpp" and model_config:
            from core.engine_llama_cpp import pick_ctx_tier
            ctx_tier = pick_ctx_tier(
                req_estimate.total_context_needed,
                max_cap=model_config.get("num_ctx"),
            )
            _log.info(
                "Picked ctx tier %d for %s (need=%d, cap=%s)",
                ctx_tier, model, req_estimate.total_context_needed,
                model_config.get("num_ctx"),
            )

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
                        await mq.switch_if_needed(model, backend, ctx_tier=ctx_tier)

                        if backend == "ollama":
                            _get_engine()  # Verify Ollama is reachable
                        else:
                            _get_engine(ctx_tier=ctx_tier)  # Load engine at tier

                        async for chunk in _stream_with_tools(
                            messages, model, max_tokens, temperature,
                            options, use_web_tools, backend,
                            grammar=body.grammar,
                            response_format=body.response_format,
                            enable_thinking=enable_thinking,
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
            await mq.switch_if_needed(model, backend, ctx_tier=ctx_tier)

            if backend == "ollama":
                try:
                    _get_engine()
                    timeout = 600 if has_images else 300
                    ollama_msgs = _convert_messages_for_ollama(messages)

                    _log.info(
                        "Ollama proxy: model=%s images=%s msgs=%d shape=[%s]",
                        model, has_images, len(messages),
                        _describe_content_shape(messages),
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

            # Transformers / llama_cpp non-streaming
            engine = _get_engine(ctx_tier=ctx_tier)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: engine.generate_streaming(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    grammar=body.grammar,
                    response_format=body.response_format,
                    raw_output=True,
                    enable_thinking=enable_thinking,
                ),
            )
            response = response or ""

            # Real token counts + finish_reason from engine stats
            stats = getattr(engine, "_last_gen_stats", None) or {}
            prompt_tokens = stats.get("prompt_tokens", 0)
            completion_tokens = stats.get("completion_tokens", 0)
            finish_reason = stats.get("finish_reason", "stop")

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "x_backend": "transformers",
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
                    "x_backend": result.backend or "transformers",
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

        backend = (
            body.backend
            or _infer_backend_from_model(body.model)
            or get_active_backend()
        )
        if backend == "llama_cpp":
            raise HTTPException(
                status_code=400,
                detail=(
                    "llama_cpp backend has no vision pipeline. Use 'ollama' "
                    "or 'transformers' for /v1/vision/analyze."
                ),
            )
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
                    "x_backend": "ollama",
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
                "x_backend": result.backend or "transformers",
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
