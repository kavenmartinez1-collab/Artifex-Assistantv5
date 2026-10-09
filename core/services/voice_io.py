"""
Artifex Assistant V5 — Voice in/out for the phone app.

Text-to-speech with Piper and speech-to-text with faster-whisper, both on
the CPU so they never compete with the chat LLM for the GPU. Separate from
the voice-assistant pipeline, which also loads its own LLM engine.
"""

from __future__ import annotations

import io
import os
import re
import threading
import wave

from core.logging_config import get_logger

_log = get_logger(__name__)

# faster-whisper sizes, default first (all three are in the HF cache here)
STT_MODELS = ["small.en", "base.en", "tiny.en"]

_lock = threading.Lock()
_voices: dict = {}
_whisper: dict = {}


def _voices_dir() -> str:
    from core.config import BASE_DIR
    return os.path.join(BASE_DIR, "models", "piper-voices")


def list_voices() -> list[str]:
    d = _voices_dir()
    if not os.path.isdir(d):
        return []
    return sorted(f[:-5] for f in os.listdir(d) if f.endswith(".onnx"))


def speakable(text: str) -> str:
    """Markdown reply -> what should be read aloud."""
    text = re.sub(r"```.*?```", " (code omitted) ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)          # links -> label
    text = re.sub(r"https?://\S+", " link ", text)
    text = re.sub(r"^\s*(#{1,6}|[-*+]|\d+\.|>)\s+", "", text, flags=re.M)
    text = re.sub(r"[*_~|]+", "", text)
    text = re.sub(r"^\s*-{3,}\s*$", " ", text, flags=re.M)
    return re.sub(r"\s+", " ", text).strip()


def tts(text: str, voice: str | None = None) -> bytes:
    """Synthesize `text` with a Piper voice; returns WAV bytes."""
    voices = list_voices()
    if not voices:
        raise RuntimeError(f"No Piper voices in {_voices_dir()}")
    voice = voice if voice in voices else voices[0]
    text = speakable(text)[:6000]
    if not text:
        raise ValueError("Nothing to read aloud.")
    with _lock:
        pv = _voices.get(voice)
        if pv is None:
            from piper.voice import PiperVoice
            pv = _voices[voice] = PiperVoice.load(
                os.path.join(_voices_dir(), voice + ".onnx"))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            pv.synthesize_wav(text, wf)
    return buf.getvalue()


def stt(audio: bytes, model: str | None = None) -> str:
    """Transcribe recorded audio (wav/m4a/webm/ogg/mp3: PyAV decodes it)."""
    model = model if model in STT_MODELS else STT_MODELS[0]
    with _lock:
        wm = _whisper.get(model)
        if wm is None:
            from faster_whisper import WhisperModel
            wm = _whisper[model] = WhisperModel(model, device="cpu",
                                                compute_type="int8")
        segments, _ = wm.transcribe(io.BytesIO(audio), beam_size=5,
                                    language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()
