"""
Artifex Assistant V5 — Error recovery and retry utilities.
Handles transient failures (OOM, Ollama disconnects) with smart retry logic.
"""

import functools
import gc
import time

from core.logging_config import get_logger

_log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Exception classification
# ─────────────────────────────────────────────────────────────────────────────

_TRANSIENT_PATTERNS = [
    "out of memory",
    "cuda out of memory",
    "oom",
    "connection refused",
    "connection reset",
    "timed out",
    "timeout",
    "connection aborted",
    "broken pipe",
    "503",
    "429",
    "temporarily unavailable",
]


def is_recoverable(exc):
    """Classify whether an exception is transient (worth retrying).

    Returns True for: CUDA OOM, network errors, Ollama connection issues.
    Returns False for: model not found, CUDA unavailable, config errors.
    """
    msg = str(exc).lower()
    return any(p in msg for p in _TRANSIENT_PATTERNS)


def is_oom(exc):
    """Check if an exception is specifically a CUDA out-of-memory error."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "oom" in msg


# ─────────────────────────────────────────────────────────────────────────────
# Retry decorator
# ─────────────────────────────────────────────────────────────────────────────

def retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0,
                       recoverable_check=None):
    """Decorator that retries a function on transient failures.

    Args:
        max_retries: Maximum retry attempts (total calls = max_retries + 1)
        initial_delay: Seconds to wait before first retry
        backoff_factor: Multiply delay by this each retry
        recoverable_check: callable(exc) -> bool, defaults to is_recoverable
    """
    check = recoverable_check or is_recoverable

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt >= max_retries or not check(exc):
                        raise
                    _log.warning(
                        "Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1, max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise last_exc  # shouldn't reach here, but safety
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Engine recovery context manager
# ─────────────────────────────────────────────────────────────────────────────

class engine_recovery:
    """Context manager for engine generation with OOM recovery.

    Usage:
        with engine_recovery(engine) as ctx:
            response = engine.generate_streaming(messages, ...)
            ctx.response = response

    On OOM: frees VRAM, compresses history, and sets ctx.should_retry = True.
    The caller can then retry with the compressed history.
    """

    def __init__(self, engine):
        self.engine = engine
        self.response = None
        self.should_retry = False
        self.error = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            return False

        if is_oom(exc_val):
            _log.warning("CUDA OOM detected — attempting VRAM recovery")
            self._free_vram()
            self.should_retry = True
            self.error = str(exc_val)
            return True  # suppress the exception

        if is_recoverable(exc_val):
            _log.warning("Transient error: %s — flagging for retry", exc_val)
            self.should_retry = True
            self.error = str(exc_val)
            return True

        # Non-recoverable — let it propagate
        _log.error("Non-recoverable error: %s", exc_val)
        return False

    def _free_vram(self):
        """Emergency VRAM cleanup."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                _log.info("VRAM after cleanup: %.2f GB allocated", allocated)
        except ImportError:
            pass


def generate_with_recovery(engine, messages, max_tokens, temperature,
                           on_token=None, compress_fn=None, context_window=6):
    """Generate a streaming response with automatic OOM recovery.

    On OOM: frees VRAM, optionally compresses history, retries once.

    Args:
        engine: BaseEngine instance
        messages: list of {role, content} dicts
        max_tokens: max tokens to generate
        temperature: sampling temperature
        on_token: streaming callback
        compress_fn: callable(messages, context_window) -> compressed messages
        context_window: passed to compress_fn

    Returns:
        response string

    Raises:
        Exception if recovery fails or error is non-recoverable
    """
    with engine_recovery(engine) as ctx:
        response = engine.generate_streaming(
            messages, max_tokens=max_tokens, temperature=temperature,
            on_token=on_token,
        )
        ctx.response = response
        return response

    # If we get here, recovery was triggered
    if not ctx.should_retry:
        raise RuntimeError(f"Generation failed: {ctx.error}")

    _log.info("Retrying generation after recovery...")

    # Compress history if possible
    if compress_fn and len(messages) > 3:
        messages = compress_fn(messages, context_window)
        _log.info("Compressed history to %d messages for retry", len(messages))

    # Reduce max_tokens on retry
    retry_tokens = max(max_tokens // 2, 256)
    _log.info("Reduced max_tokens from %d to %d for retry", max_tokens, retry_tokens)

    # Single retry — don't loop forever
    return engine.generate_streaming(
        messages, max_tokens=retry_tokens, temperature=temperature,
        on_token=on_token,
    )
