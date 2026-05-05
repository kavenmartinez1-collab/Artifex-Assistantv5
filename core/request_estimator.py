"""
Artifex Assistant V5 — Request resource estimator.

Estimates context window and VRAM needs for an inference request BEFORE
routing it to a backend.  Pure computation (no side effects) except optional
GGUF header reading for precise KV cache sizing.

Called on every request by the router; must stay fast.
"""

import logging
import os
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_CTX_BUCKETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Chars-per-token heuristic (matches api/server.py usage)
CHARS_PER_TOKEN = 4

# Image token cost (conservative — covers typical 512×512 patch encoding)
TOKENS_PER_IMAGE = 1000

# Safety margin multiplier when choosing context bucket
SAFETY_MARGIN = 1.2

# Web tools budget constants (derived from api/web_tools.py behavior)
_TOOL_CALL_TOKENS = 200       # model generates @search/@web_read per round
_SEARCH_RESULT_TOKENS = 960   # 8 results × ~120 tokens each
_WEB_READ_TOKENS = 1000       # 4000 chars truncated → ~1000 tokens
_EXPECTED_ROUNDS = 3          # typical multi-round tool usage
WEB_TOOLS_BUFFER_DEFAULT = (
    (_TOOL_CALL_TOKENS + _SEARCH_RESULT_TOKENS + _WEB_READ_TOKENS)
    * _EXPECTED_ROUNDS
)  # ~6480, rounded concept is ~8000 — we add a flat pad below

# Flat pad on top of per-round estimate to cover variance
_WEB_TOOLS_FLAT_PAD = 1500
WEB_TOOLS_BUFFER = WEB_TOOLS_BUFFER_DEFAULT + _WEB_TOOLS_FLAT_PAD  # ~7980

# KV quant bytes-per-element table (mirrors engine_llama_cpp._KV_QUANT_BPE)
_KV_QUANT_BPE = {
    "f16": 2.0, "f32": 4.0,
    "q8_0": 1.0625, "q4_0": 0.5625, "q4_1": 0.625,
    "q5_0": 0.6875, "q5_1": 0.75,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InferenceRequirements:
    """Estimated resource needs for an inference request."""
    estimated_prompt_tokens: int
    max_completion_tokens: int
    web_tools_buffer: int           # extra tokens reserved for tool result injection
    thinking_buffer: int            # extra tokens if thinking/reasoning is enabled
    total_context_needed: int       # sum: prompt + completion + buffers
    recommended_num_ctx: int        # rounded up to standard bucket
    estimated_kv_mb: float          # KV cache VRAM at recommended_num_ctx
    notes: list[str] = field(default_factory=list)




# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------

def _count_message_tokens(messages: list[dict]) -> tuple[int, int]:
    """Estimate prompt tokens from messages using chars//4 heuristic.

    Returns (token_count, image_count).
    """
    total_chars = 0
    image_count = 0

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            # Multimodal content array (OpenAI vision format)
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        total_chars += len(part.get("text", ""))
                    elif part.get("type") in ("image_url", "image"):
                        image_count += 1
                elif isinstance(part, str):
                    total_chars += len(part)

    token_count = total_chars // CHARS_PER_TOKEN
    return token_count, image_count


def _pick_bucket(total_tokens: int) -> int:
    """Pick the smallest standard bucket that fits total_tokens with safety margin."""
    target = int(total_tokens * SAFETY_MARGIN)
    for bucket in STANDARD_CTX_BUCKETS:
        if bucket >= target:
            return bucket
    # Exceeds largest bucket — return max
    return STANDARD_CTX_BUCKETS[-1]


def _get_kv_quant_bpe(extra_flags: list[str]) -> tuple[float, float]:
    """Extract (bpe_k, bpe_v) from extra_flags list. Defaults to f16 (2.0)."""
    bpe_k = bpe_v = 2.0
    for i, flag in enumerate(extra_flags):
        if flag == "-ctk" and i + 1 < len(extra_flags):
            bpe_k = _KV_QUANT_BPE.get(extra_flags[i + 1], 2.0)
        elif flag == "-ctv" and i + 1 < len(extra_flags):
            bpe_v = _KV_QUANT_BPE.get(extra_flags[i + 1], 2.0)
    return bpe_k, bpe_v


def _estimate_kv_cache_mb(
    num_ctx: int,
    gguf_path: str,
    extra_flags: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Estimate KV cache VRAM in MiB for a given context size.

    Uses gpu_pool's hybrid-aware GGUF reader (accounts for
    full_attention_interval in models like Qwen3.6).

    Returns (mb, notes_list).
    """
    notes = []
    if not gguf_path or not os.path.isfile(gguf_path):
        return 0.0, ["No GGUF path available for KV estimation"]

    from core.gpu_pool import read_gguf_kv_params
    params = read_gguf_kv_params(gguf_path)
    if params is None:
        return 0.0, ["Could not parse GGUF header for KV params"]

    bpe_k, bpe_v = _get_kv_quant_bpe(extra_flags or [])

    attn_layers = params["attn_layer_count"]
    kv_bytes_per_token = (
        attn_layers
        * params["head_count_kv"]
        * (params["key_dim"] * bpe_k + params["val_dim"] * bpe_v)
    )
    kv_per_token_mb = kv_bytes_per_token / (1024 ** 2)
    total_kv_mb = kv_per_token_mb * num_ctx

    notes.append(
        f"KV: {attn_layers} attn_layers (of {params['block_count']}, interval={params['full_attention_interval']}) "
        f"x {params['head_count_kv']} kv_heads, "
        f"dims={params['key_dim']:.0f}k/{params['val_dim']:.0f}v, "
        f"quant=({bpe_k:.3f}/{bpe_v:.3f} bpe) -> "
        f"{kv_per_token_mb:.4f} MB/tok, {total_kv_mb:.1f} MB @ {num_ctx} ctx"
    )
    return total_kv_mb, notes


# ---------------------------------------------------------------------------
# Main estimation function
# ---------------------------------------------------------------------------

def estimate_request_requirements(
    messages: list[dict],
    max_tokens: int,
    web_tools: bool = False,
    enable_thinking: bool = True,
    model_config: dict | None = None,
) -> InferenceRequirements:
    """Estimate resource needs for an inference request.

    Args:
        messages: Chat messages in OpenAI format.
        max_tokens: Maximum completion tokens requested.
        web_tools: Whether @search/@web_read tools are enabled.
        enable_thinking: Whether thinking/reasoning is enabled.
        model_config: Model entry from llama_cpp_config.json (has 'path',
                      'extra_flags', 'num_ctx', etc.). None for Ollama backend.

    Returns:
        InferenceRequirements with all estimates and diagnostic notes.
    """
    notes = []

    # --- 1. Prompt token estimation ---
    text_tokens, image_count = _count_message_tokens(messages)
    image_tokens = image_count * TOKENS_PER_IMAGE
    estimated_prompt_tokens = text_tokens + image_tokens

    if image_count > 0:
        notes.append(f"Multimodal: {image_count} image(s) adding ~{image_tokens} tokens")

    notes.append(f"Prompt estimate: {text_tokens} text + {image_tokens} image = {estimated_prompt_tokens} tokens")

    # --- 2. Web tools buffer ---
    web_tools_buffer = 0
    if web_tools:
        web_tools_buffer = WEB_TOOLS_BUFFER
        notes.append(
            f"Web tools enabled: reserving {web_tools_buffer} tokens "
            f"({_EXPECTED_ROUNDS} rounds x search+read + pad)"
        )

    # --- 3. Thinking buffer ---
    thinking_buffer = 0
    if enable_thinking:
        # Thinking consumes completion budget, not additional context.
        # The model uses up to ~80% of max_tokens for reasoning internally.
        notes.append(
            f"Thinking enabled: model may use up to ~{int(max_tokens * 0.8)} of "
            f"{max_tokens} completion tokens for reasoning (no extra context needed, "
            f"but effective answer budget is reduced)"
        )

    # --- 4. Total context needed ---
    total_context_needed = estimated_prompt_tokens + max_tokens + web_tools_buffer + thinking_buffer

    notes.append(
        f"Total context: {estimated_prompt_tokens} prompt + {max_tokens} completion "
        f"+ {web_tools_buffer} web_tools + {thinking_buffer} thinking = {total_context_needed}"
    )

    # --- 5. Recommended num_ctx bucket ---
    recommended_num_ctx = _pick_bucket(total_context_needed)

    # If model_config specifies a configured num_ctx, don't recommend above it
    # (the server won't allocate more than configured anyway)
    configured_ctx = None
    if model_config:
        configured_ctx = model_config.get("num_ctx")
        if configured_ctx and recommended_num_ctx > configured_ctx:
            notes.append(
                f"WARNING: Estimated need ({recommended_num_ctx}) exceeds configured "
                f"num_ctx ({configured_ctx}) — request may be trimmed or fail"
            )
        elif configured_ctx and recommended_num_ctx < configured_ctx:
            notes.append(
                f"Bucket {recommended_num_ctx} fits within configured num_ctx {configured_ctx}"
            )

    notes.append(f"Recommended bucket: {recommended_num_ctx} (from {total_context_needed} * {SAFETY_MARGIN})")

    # --- 6. KV VRAM estimation ---
    estimated_kv_mb = 0.0
    if model_config:
        gguf_path = model_config.get("path")
        extra_flags = model_config.get("extra_flags", [])
        ctx_for_kv = recommended_num_ctx
        # Use configured ctx if it's larger (that's what actually gets allocated)
        if configured_ctx and configured_ctx > ctx_for_kv:
            ctx_for_kv = configured_ctx
            notes.append(f"KV estimated at configured ctx ({ctx_for_kv}), not bucket")

        estimated_kv_mb, kv_notes = _estimate_kv_cache_mb(ctx_for_kv, gguf_path, extra_flags)
        notes.extend(kv_notes)
    else:
        notes.append("No model config provided — KV VRAM estimation skipped")

    result = InferenceRequirements(
        estimated_prompt_tokens=estimated_prompt_tokens,
        max_completion_tokens=max_tokens,
        web_tools_buffer=web_tools_buffer,
        thinking_buffer=thinking_buffer,
        total_context_needed=total_context_needed,
        recommended_num_ctx=recommended_num_ctx,
        estimated_kv_mb=estimated_kv_mb,
        notes=notes,
    )

    _log.info(
        "Request estimate: ~%d prompt tok, %d max_completion, web=%s, think=%s -> "
        "ctx_needed=%d, bucket=%d, kv=%.1f MB",
        estimated_prompt_tokens, max_tokens, web_tools, enable_thinking,
        total_context_needed, recommended_num_ctx, estimated_kv_mb,
    )

    return result
