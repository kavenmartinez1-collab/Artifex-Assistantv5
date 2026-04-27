"""
Artifex Assistant V5 — Shared inference utilities.
Backend-agnostic helpers used by all engines: token counting, history
compression, thinking block handling, VRAM pressure management.
"""

import gc
import re

import torch


# Strings that indicate the model is faking a new conversational turn.
# When any of these appear in the output, generation halts immediately.
STOP_STRINGS = [
    # Colon-separated format (Llama, Mistral, etc.)
    "\nUser:", "\nUSER:", "\nYOU:", "\nuser:", "\nHuman:",
    # ChatML format (Qwen) — role names without colons
    "\nuser\n", "\nUser\n", "\nassistant\n", "\nassistant ",
    # Gemma turn markers
    "\n<end_of_turn>", "<end_of_turn>",
    # Tool output markers
    "\n[TOOL OUTPUT", "\n[tool output",
]


class ThinkFilter:
    """
    Splits streaming model output into thinking and response content.

    Transformers models (Qwen3.5 with enable_thinking=True) start generation
    inside a <think> block — use starts_in_think=True. Ollama wraps thinking
    in explicit <think>...</think> tags in the stream — use starts_in_think=False.

    Usage:
        tf = ThinkFilter(on_response=print, on_thinking=debug_print,
                         starts_in_think=False)  # for Ollama
        engine.generate_streaming(..., on_token=tf.feed)
        tf.flush()  # after streaming completes
    """

    def __init__(self, on_response, on_thinking=None, starts_in_think=True):
        self.on_response = on_response
        self.on_thinking = on_thinking
        self.in_think = starts_in_think
        self.buffer = ""

    def feed(self, text):
        """Process a new chunk of streamed text."""
        self.buffer += text
        self._process()

    def _process(self):
        while True:
            if self.in_think:
                # Check for end-of-thinking: </think> (Qwen) or <channel|> (Gemma 4)
                idx = self.buffer.find("</think>")
                end_len = 8
                if idx == -1:
                    idx = self.buffer.find("<channel|>")
                    end_len = 10
                if idx != -1:
                    if self.on_thinking and idx > 0:
                        self.on_thinking(self.buffer[:idx])
                    self.in_think = False
                    self.buffer = self.buffer[idx + end_len:]
                    continue
                else:
                    if self.on_thinking and len(self.buffer) > 10:
                        self.on_thinking(self.buffer[:-10])
                        self.buffer = self.buffer[-10:]
                    break
            else:
                idx = self.buffer.find("<think>")
                if idx != -1:
                    if idx > 0:
                        self.on_response(self.buffer[:idx])
                    self.in_think = True
                    self.buffer = self.buffer[idx + 7:]
                    continue
                else:
                    if len(self.buffer) > 7:
                        self.on_response(self.buffer[:-7])
                        self.buffer = self.buffer[-7:]
                    break

    def flush(self):
        """Flush remaining buffer. Call after streaming completes."""
        if not self.in_think and self.buffer:
            self.on_response(self.buffer)
        elif self.in_think and self.on_thinking and self.buffer:
            self.on_thinking(self.buffer)
        self.buffer = ""


def strip_think_blocks(text):
    """Remove thinking content from model output.

    Handles two cases:
      1. Matched <think>...</think> pairs — removed completely.
      2. "Started inside a think block" — some models (Qwen3.5) emit the
         opening <think> inside the prompt, so the output begins with raw
         thinking content followed by </think>. We strip up to the FIRST
         </think> ONLY if it's unpaired (appears before any <think>),
         proving we're genuinely in the "started mid-thinking" case.

    The old approach used `re.sub(r"^.*?</think>", ...)` which was
    dangerously broad: if a model hallucinated a stray </think> ANYWHERE
    in its output (e.g. a vision model describing HTML, or a code model
    quoting template syntax), everything before it was silently wiped —
    including real JSON analysis from the vision pipeline.
    """
    # Step 1: strip matched pairs
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Step 2: handle the unpaired-closer case (started mid-thinking).
    # Only strip if the first </think> comes before any <think>, meaning
    # there's no opening tag to pair with — we're definitely mid-think.
    first_close = text.find("</think>")
    if first_close != -1:
        first_open = text.find("<think>")
        if first_open == -1 or first_close < first_open:
            text = text[first_close + 8:]
    return text.strip()


def _truncate_repetition(text, min_phrase_len=80):
    """Detect and truncate when the model is stuck in a verbatim loop.

    Requires the same phrase (80+ chars) to appear at least 3 times.
    Two occurrences is normal in structured output (JSON arrays, markdown
    tables); three consecutive matches is a genuine repetition loop.
    """
    if len(text) < min_phrase_len * 4:
        return text

    for phrase_len in range(min_phrase_len, min(500, len(text) // 3) + 1, 20):
        for start in range(0, len(text) - phrase_len * 3):
            phrase = text[start:start + phrase_len]
            second = text.find(phrase, start + phrase_len)
            if second == -1:
                continue
            third = text.find(phrase, second + phrase_len)
            if third != -1:
                return text[:second].rstrip()

    return text


def _clean_response(text):
    """Strip thinking blocks, trailing fake user turns, and repetition loops."""
    clean = strip_think_blocks(text)

    for marker in STOP_STRINGS:
        idx = clean.find(marker)
        if idx != -1:
            clean = clean[:idx]

    clean = _truncate_repetition(clean)

    return clean.rstrip()


def _extract_key_point(msg):
    """Pull the essential content from a message for compression."""
    content = msg["content"]

    if content.startswith("[TOOL OUTPUT") or content.startswith("[TOOL RESULT"):
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        if any("architecture" in l.lower() for l in lines[:3]):
            return "Ran: @architecture() — full project map generated"

        for line in lines:
            if line.startswith("Function:"):
                return f"Read: {line[:120]}"

        for line in lines:
            if "SKELETON VIEW" in line:
                fname = next((l for l in lines if l.startswith("File:")), "")
                return f"Read skeleton: {fname[:100]}" if fname else "Read: skeleton view"

        for line in lines:
            if line.startswith("File:"):
                return f"Read: {line[:120]}"

        for line in lines:
            if line.startswith("Found "):
                return line[:120]

        for line in lines:
            if line.startswith("$") or line.startswith("`"):
                return f"Ran: {line[:80]}"
        return lines[0][:80] if lines else "(tool output)"

    if msg["role"] == "user":
        return content.split("\n")[0].strip()[:100]

    sentences = re.split(r'[.!?\n]', content)
    for s in sentences:
        s = s.strip().lstrip("#*- ")
        if len(s) > 15:
            return s[:100]
    return content[:100]


# Module-level tokenizer reference — set by TransformersEngine.generate_streaming()
# after model load. Enables accurate token counting without changing callers.
_tokenizer = None


def _count_tokens(messages, tokenizer=None):
    """Count tokens in messages. Uses actual tokenizer if available, else char/4 fallback."""
    tok = tokenizer or _tokenizer
    if tok is not None:
        text = "\n".join(m.get("content", "") for m in messages)
        return len(tok.encode(text, add_special_tokens=False))
    return sum(len(m.get("content", "")) for m in messages) // 4


def build_active_messages(history, context_window, max_history_tokens=None):
    """Build the active message list for the next generation call.

    Uses max_total_input_tokens as a HARD CAP on everything sent to the model.

    Args:
        history: full history list (system prompt at index 0)
        context_window: max number of recent messages to consider
        max_history_tokens: override for history token cap

    Returns:
        (updated_history, active_messages)
    """
    from core.config import get_context_profile
    profile = get_context_profile()

    system_tokens = _count_tokens([history[0]])

    total_cap = profile.max_total_input_tokens
    history_budget = max_history_tokens or max(total_cap - system_tokens, 200)

    if max_history_tokens is None:
        history_budget = min(history_budget, profile.max_history_tokens)

    active = [history[0]] + history[1:][-context_window:]

    history_msgs = active[1:]
    if _count_tokens(history_msgs) <= history_budget:
        return list(history), active

    compressed = compress_history(history, context_window)
    for shrink in (context_window // 2, context_window // 4, 2):
        active = [compressed[0]] + compressed[1:][-max(shrink, 2):]
        if _count_tokens(active[1:]) <= history_budget:
            return compressed, active

    active = [compressed[0]] + compressed[1:][-2:]
    return compressed, active


def compress_history(history, context_window):
    """Compress old messages into key-point summaries.

    Keeps the system prompt (history[0]) and recent messages intact.
    PINS the first user message — the original task request.
    """
    convo = history[1:]
    if len(convo) <= context_window:
        return list(history)

    pinned_msg = convo[0] if convo and convo[0]["role"] == "user" else None
    compressible = convo[1:] if pinned_msg else convo

    keep_count = max(context_window - 1, 1)
    if len(compressible) <= keep_count:
        return list(history)

    old_messages = compressible[:-keep_count]
    recent_messages = compressible[-keep_count:]

    points = []
    for msg in old_messages:
        role = "Q" if msg["role"] == "user" else "A"
        point = _extract_key_point(msg)
        points.append(f"{role}: {point}")

    summary_text = (
        "[EARLIER CONVERSATION — key points]\n"
        + "\n".join(points)
    )

    result = [history[0]]
    if pinned_msg:
        result.append(pinned_msg)
    result.append({"role": "user", "content": summary_text})
    result.extend(recent_messages)
    return result


def trim_messages_to_context(messages, max_input_tokens):
    """Drop oldest middle messages so total tokens fit within a hard budget.

    Keeps: system prompt (first message) + most recent 2 messages.
    Falls back to truncating the last message content if it alone exceeds budget.
    """
    if max_input_tokens <= 0 or len(messages) <= 1:
        return messages

    est = lambda msgs: _count_tokens(msgs)

    if est(messages) <= max_input_tokens:
        return messages

    system = messages[:1]
    middle = list(messages[1:-2]) if len(messages) > 3 else []
    tail = list(messages[-2:]) if len(messages) >= 2 else []

    while middle and est(system + middle + tail) > max_input_tokens:
        middle.pop(0)

    result = system + middle + tail

    if est(result) > max_input_tokens and len(result) >= 2:
        last = dict(result[-1])
        content = last.get("content", "")
        overshoot = est(result) - max_input_tokens
        chars_to_cut = overshoot * 4 + 200
        if chars_to_cut < len(content):
            last["content"] = content[:len(content) - chars_to_cut] + "\n[...trimmed...]"
            result[-1] = last

    return result


# =============================================================================
# VRAM PRESSURE MANAGEMENT
# =============================================================================

def check_vram_pressure(threshold=0.85):
    """Check if VRAM usage exceeds threshold (0.0-1.0).

    Returns:
        (is_pressured, usage_fraction, allocated_gb, total_gb)
    """
    from core.device import gpu_info
    if not gpu_info.is_available:
        return False, 0.0, 0.0, 0.0
    allocated = gpu_info.allocated_gb
    total = gpu_info.total_gb
    fraction = gpu_info.usage_fraction
    return fraction >= threshold, fraction, allocated, total


def vram_pressure_relief(engine, history, context_window, threshold=0.85):
    """If VRAM is above threshold, unload engine, compress history, and GC.

    The model will lazy-reload on the next generate_streaming() call with
    a smaller context (because history was compressed).

    Args:
        engine: BaseEngine instance (or any object with is_loaded/unload)
        history: conversation history list (modified in-place)
        context_window: max messages for compress_history
        threshold: VRAM fraction that triggers relief (default 0.85 = 85%)

    Returns:
        (did_relieve, message) — message describes what happened
    """
    pressured, fraction, alloc_gb, total_gb = check_vram_pressure(threshold)
    if not pressured:
        return False, ""

    if engine.is_loaded():
        engine.unload()

    old_len = len(history)
    compressed = compress_history(history, context_window)
    history.clear()
    history.extend(compressed)
    new_len = len(history)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    _, post_frac, post_gb, _ = check_vram_pressure(threshold=1.0)

    msg = (
        f"VRAM pressure relief: {alloc_gb:.1f}/{total_gb:.1f} GB ({fraction:.0%}) -> "
        f"{post_gb:.1f} GB ({post_frac:.0%}). "
        f"History: {old_len} -> {new_len} messages. Model will reload on next message."
    )
    return True, msg
