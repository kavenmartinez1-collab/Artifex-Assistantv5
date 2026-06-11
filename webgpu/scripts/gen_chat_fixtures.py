"""Generate golden fixtures for the TS chat-module ports.

Runs the REAL core/inference.py functions on deterministic inputs and dumps
inputs+outputs to test-fixtures/chat-modules-golden.json. The TS side
(scripts/test-chat-modules.mts) replays the same inputs through
src/chat/{compression,context}.ts and diffs byte-for-byte.

Run from the repo root with the project venv:
    ./venv/Scripts/python.exe webgpu/scripts/gen_chat_fixtures.py

Note: build_active_messages cases use engine_ctx>0 or an explicit
max_history_tokens override so results don't depend on this machine's
GPU-tier context profile.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.inference import (  # noqa: E402
    _extract_key_point,
    compress_history,
    build_active_messages,
    trim_messages_to_context,
)


def msg(role, content):
    return {"role": role, "content": content}


# ── Inputs ────────────────────────────────────────────────────────────────

KEY_POINT_CASES = [
    msg("user", "How do I fix the CUDA device ordering?\nIt keeps flipping."),
    msg("user", "short"),
    msg("assistant", "The fix is to set CUDA_DEVICE_ORDER=PCI_BUS_ID. That aligns indices with nvidia-smi."),
    msg("assistant", "## Header\n* bullet point that is long enough to extract\nMore text follows here."),
    msg("assistant", "ok. no! hm?"),
    msg("assistant", "x" * 250),
    msg("user", "[TOOL OUTPUT]\nFunction: parse_config(path) -> dict\nbody text"),
    msg("user", "[TOOL OUTPUT]\nThe ARCHITECTURE overview follows\nstuff"),
    msg("user", "[TOOL RESULT]\nSKELETON VIEW\nFile: core/engine.py\nmore"),
    msg("user", "[TOOL OUTPUT]\nFile: webgpu/src/main.ts\ncontents here"),
    msg("user", "[TOOL OUTPUT]\nFound 17 matches in 4 files\n..."),
    msg("user", "[TOOL OUTPUT]\n$ git status --short\nM core/engine.py"),
    msg("user", "[TOOL OUTPUT]\njust some plain first line that is quite long and should be cut to eighty characters exactly here\nsecond"),
    msg("user", "[TOOL OUTPUT]\n"),
]


def build_convo(turns, prefix="turn"):
    """Deterministic alternating conversation. history[0] is the system msg."""
    out = [msg("system", "You are Artifex, a helpful AI assistant.")]
    for i in range(turns):
        role = "user" if i % 2 == 0 else "assistant"
        out.append(msg(role, f"{prefix} {i}: " + ("lorem ipsum dolor sit amet " * (3 + i % 5)).strip()))
    return out


LONG_CONVO = build_convo(24)
SHORT_CONVO = build_convo(4)
ASSISTANT_FIRST = [msg("system", "sys"), msg("assistant", "I begin."), *build_convo(6)[1:]]
BIG_CONVO = build_convo(40, prefix="big")

COMPRESS_CASES = [
    {"history": LONG_CONVO, "context_window": 6},
    {"history": LONG_CONVO, "context_window": 10},
    {"history": SHORT_CONVO, "context_window": 10},   # no-op (fits)
    {"history": ASSISTANT_FIRST, "context_window": 2}, # no pinned first user msg
]

ACTIVE_CASES = [
    {"history": LONG_CONVO, "context_window": 10, "engine_ctx": 8192},
    {"history": BIG_CONVO, "context_window": 10, "engine_ctx": 4096},
    {"history": BIG_CONVO, "context_window": 6, "engine_ctx": 0, "max_history_tokens": 300},
    {"history": SHORT_CONVO, "context_window": 10, "engine_ctx": 8192},
]

TRIM_CASES = [
    {"messages": LONG_CONVO, "max_input_tokens": 200},
    {"messages": LONG_CONVO, "max_input_tokens": 50},    # forces last-message truncation
    {"messages": SHORT_CONVO, "max_input_tokens": 10000}, # no-op
]


def main():
    golden = {
        "extract_key_point": [
            {"input": m, "expected": _extract_key_point(m)} for m in KEY_POINT_CASES
        ],
        "compress_history": [
            {**c, "expected": compress_history(c["history"], c["context_window"])}
            for c in COMPRESS_CASES
        ],
        "build_active_messages": [],
        "trim_messages_to_context": [
            {**c, "expected": trim_messages_to_context(c["messages"], c["max_input_tokens"])}
            for c in TRIM_CASES
        ],
    }
    for c in ACTIVE_CASES:
        updated, active = build_active_messages(
            c["history"], c["context_window"],
            max_history_tokens=c.get("max_history_tokens"),
            engine_ctx=c.get("engine_ctx", 0),
        )
        golden["build_active_messages"].append(
            {**c, "expected_history": updated, "expected_active": active})

    out = os.path.join(os.path.dirname(__file__), "..", "test-fixtures", "chat-modules-golden.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1, ensure_ascii=False)
    n = sum(len(v) for v in golden.values())
    print(f"Wrote {n} golden cases to {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
