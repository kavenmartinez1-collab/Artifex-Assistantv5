"""
Artifex Assistant V5 — Sampler preset contract.

Single source of truth for sampling parameters across backends. The preset
table mirrors the WebGPU engine's dropdown (webgpu/src/main.ts) so both
halves of the platform speak the same contract, plus an "agent" preset tuned
for autonomous tool-use runs (see agent_bench/).

Why this exists: llama-server silently applies its own compiled-in request
defaults (min_p=0.05, top_k=40, ...) to any parameter a request omits — the
exact class of invisible-sampler bug behind the 2026-04 Qwen3.5 coherence
collapse (minP+DRY defaults, see memory). Every dict produced here is FULLY
explicit, and the llama.cpp engine sends the whole thing on every request,
so generation behavior is owned by Artifex, never by whichever llama.cpp
build happens to be running.

Keys use llama-server /v1/chat/completions names verbatim.
"""

# Parameters the llama.cpp engine forwards to llama-server. Anything not in
# this whitelist is dropped before the request is built (typo hygiene, and
# keeps engine-internal keys like a future "preset" tag out of the payload).
SAMPLING_PAYLOAD_KEYS = (
    "temperature",
    "top_k", "top_p", "min_p", "typical_p", "top_n_sigma",
    "repeat_penalty", "repeat_last_n",
    "presence_penalty", "frequency_penalty",
    "dry_multiplier", "dry_base", "dry_allowed_length", "dry_penalty_last_n",
    "xtc_probability", "xtc_threshold",
    "seed", "stop",
)

# Fully-neutral base: every sampler off / identity. Presets are merged over
# this, so a preset only names the knobs it actually turns and the rest are
# still sent explicitly (never left to server defaults).
_NEUTRAL = {
    "top_k": 0,
    "top_p": 1.0,
    "min_p": 0.0,
    "typical_p": 1.0,
    "repeat_penalty": 1.0,
    "repeat_last_n": 64,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "dry_multiplier": 0.0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
}

# The platform preset contract. balanced/deterministic/creative/reference
# mirror webgpu/src/main.ts SAMPLER_PRESETS exactly (temperature included).
#
# "agent" — the Qwen3-series recommended thinking configuration (temp 0.6 /
# top_p 0.95 / top_k 20 / min_p 0), validated by agent_bench on
# Qwen3.6-35B-A3B (2026-08-05): 1.000 on the 6-scenario full suite with
# proper run-the-test-before-done discipline. Measured alternatives:
# greedy also scored 1.000 but repeats failed attempts verbatim on retries;
# presence_penalty=1.0 (best micro discipline, 0.95) proactively breaks
# <tool_call> repetition collapse but once skipped test verification
# (0.967 full) — use it if collapse recurs; the agent loop's format-retry
# already handles collapse reactively. Full data: agent_bench/TUNING_REPORT.md.
PRESETS = {
    "balanced": {
        "temperature": 0.7, "top_k": 40, "top_p": 0.9,
        "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
    },
    "deterministic": {
        "temperature": 0.0, "top_k": 0, "top_p": 1.0,
        "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
    },
    "creative": {
        "temperature": 0.9, "top_k": 50, "top_p": 0.95,
        "min_p": 0.05, "repeat_penalty": 1.0, "dry_multiplier": 0.8,
    },
    "reference": {
        "temperature": 1.0, "top_k": 50, "top_p": 1.0,
        "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
    },
    "agent": {
        "temperature": 0.6, "top_k": 20, "top_p": 0.95,
        "min_p": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
    },
}

# What the llama.cpp engine sends when a caller passes sampling=None.
# Balanced-shaped minus temperature: existing call sites keep owning
# temperature positionally (0.7 assistant / 0.2 code), while the rest of
# the sampler chain becomes explicit and mild instead of build-dependent.
DEFAULT_SAMPLING = {k: v for k, v in {**_NEUTRAL, **PRESETS["balanced"]}.items()
                    if k != "temperature"}


def get_preset(name: str, **overrides) -> dict:
    """Return a fully-explicit sampling dict for a named preset.

    Unknown names fall back to "balanced" (with a KeyError-free contract so
    a stale preset name in a saved config can't break generation). Keyword
    overrides are merged last — e.g. get_preset("agent", seed=42).
    """
    base = PRESETS.get(name) or PRESETS["balanced"]
    out = {**_NEUTRAL, **base}
    out.update(overrides)
    return out


def list_presets() -> tuple:
    """Preset names in UI display order."""
    return ("balanced", "deterministic", "creative", "reference", "agent")
