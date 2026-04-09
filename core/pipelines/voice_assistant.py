"""
Artifex Assistant V5 — Voice Assistant Pipeline.
Full voice conversation loop: STT → LLM → TTS, all local.

STT:  faster-whisper (CTranslate2-optimized Whisper)
LLM:  Existing Artifex engine (Transformers or Ollama)
TTS:  Existing Artifex audio pipeline (Bark / SpeechT5)

The pipeline chains all three stages together so a single .run()
call takes voice audio in and returns spoken audio out, with the
text response available in metadata.
"""

import gc
import os
import re
import threading
import time

import numpy as np
import torch

from core.pipelines.base import BasePipeline, PipelineResult


# ── Artifex Voice personality prompt ─────────────────────────────────

ARTIFEX_VOICE_SYSTEM_PROMPT = """\
You are Artifex, a local AI voice assistant. You are knowledgeable, composed, \
and conversational.

PERSONALITY:
- Calm, clear, and helpful. Speak naturally like a trusted assistant.
- Concise. You are a voice assistant. Keep responses to 1-3 sentences unless the \
user explicitly asks for detail. Brevity is essential for spoken conversation.
- Never use emoji, markdown formatting, bullet points, or code blocks in responses. \
Everything you say will be spoken aloud. Write in natural spoken English.
- When you don't know something, say so directly.

CONSTRAINTS:
- Keep responses SHORT. This is a real-time voice conversation. 1-2 sentences \
for simple queries. Up to 4-5 sentences maximum for complex explanations.
- Do not use asterisks, parenthetical stage directions, or narration. \
Just speak naturally.\
"""

ARTIFEX_GREETING = "Artifex voice assistant is online. How can I help?"


# ── Sentence chunking for low-latency TTS ──────────────────────────

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')


def _chunk_for_tts(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks for TTS.

    Piper handles longer text well, so we use a generous limit.
    Only splits when necessary.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_RE.split(text)
    chunks = []
    current = ""

    for sentence in sentences:
        if not sentence.strip():
            continue
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence

    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# Default Piper voice model path (relative to project root)
_PIPER_VOICE_MODEL = os.path.join("models", "piper-voices", "en_US-joe-medium.onnx")


# ── Listener (STT via faster-whisper) ──────────────────────────────

class _Listener:
    """Speech-to-text using faster-whisper (local, CTranslate2-optimized)."""

    def __init__(self, whisper_model: str = "base.en",
                 device: str = "auto",
                 input_device: int | None = None):
        self._whisper_model = None
        self._model_name = whisper_model
        self._device = device
        self._input_device = input_device
        self._record_sr = 16_000  # Whisper expects 16 kHz

    def load(self, status_callback=None):
        """Load the faster-whisper model."""
        if self._whisper_model is not None:
            return

        from faster_whisper import WhisperModel

        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device

        compute_type = "float16" if device == "cuda" else "int8"

        if status_callback:
            status_callback(f"Loading STT: faster-whisper {self._model_name} ({device})...")

        self._whisper_model = WhisperModel(
            self._model_name,
            device=device,
            compute_type=compute_type,
        )

        if status_callback:
            status_callback(f"STT ready: faster-whisper {self._model_name}")

    def record_audio(self, stop_event: threading.Event) -> np.ndarray | None:
        """Record from microphone until stop_event is set.

        Returns float32 numpy array at 16 kHz, or None if silence/empty.
        """
        import sounddevice as sd

        audio_buffer: list[np.ndarray] = []
        recording = True

        def callback(indata, frames, time_info, status):
            if recording:
                audio_buffer.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=self._record_sr,
                channels=1,
                dtype="float32",
                device=self._input_device,
                callback=callback,
                blocksize=1024,
            ):
                stop_event.wait()
        finally:
            recording = False

        if not audio_buffer:
            return None

        audio = np.concatenate(audio_buffer, axis=0).flatten()

        # Filter out dead-mic recordings (no signal at all)
        if np.max(np.abs(audio)) < 0.0001:
            return None

        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio numpy array to text."""
        if self._whisper_model is None:
            raise RuntimeError("Listener not loaded. Call load() first.")

        segments, _ = self._whisper_model.transcribe(
            audio,
            beam_size=5,
            language="en",
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

    def record_and_transcribe(self, stop_event: threading.Event) -> str | None:
        """Record audio then transcribe. Returns text or None."""
        audio = self.record_audio(stop_event)
        if audio is None:
            return None
        return self.transcribe(audio)

    def unload(self):
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def is_loaded(self) -> bool:
        return self._whisper_model is not None


# ── Speaker (audio playback via sounddevice) ───────────────────────

class _Speaker:
    """Plays WAV audio files through the system audio device."""

    def __init__(self, output_device: int | None = None, muted: bool = False):
        self.muted = muted
        self._output_device = output_device

    def play_file(self, file_path: str, block: bool = True):
        """Play an audio file. Supports WAV (via scipy) and common formats."""
        if self.muted or not file_path or not os.path.exists(file_path):
            return

        import sounddevice as sd

        try:
            # Try scipy for WAV files first (no extra deps)
            import scipy.io.wavfile
            sr, audio = scipy.io.wavfile.read(file_path)

            # Normalize to float32 [-1, 1]
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            elif audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            sd.play(audio, samplerate=sr, device=self._output_device)
            if block:
                sd.wait()
        except Exception:
            pass  # Silently skip playback errors

    def stop(self):
        import sounddevice as sd
        sd.stop()


# ── Voice Assistant Pipeline ───────────────────────────────────────

class VoiceAssistantPipeline(BasePipeline):
    """Voice conversation loop: STT → LLM → TTS, all local.

    This pipeline orchestrates three stages:
    1. STT — faster-whisper transcribes audio to text
    2. LLM — Artifex engine generates a response (streaming)
    3. TTS — Bark/SpeechT5 synthesizes speech from the response

    Each stage can also be used independently via run() kwargs.
    """

    pipeline_type = "voice-assistant"
    display_name = "Voice Assistant"
    output_type = "audio"

    def __init__(self):
        self._listener: _Listener | None = None
        self._speaker: _Speaker | None = None
        self._engine = None
        self._tts_voice = None
        self._tts_sample_rate = 22050

        # Conversation state
        self._messages: list[dict] = []
        self._max_history_turns: int = 20
        self._system_prompt: str = ARTIFEX_VOICE_SYSTEM_PROMPT
        self._tools_enabled: bool = True
        self._loaded = False

    # ── Pipeline interface ──────────────────────────────────────────

    def load(self, model_path: str = "", status_callback=None, **kwargs):
        """Load all three components: STT, LLM engine, and TTS.

        kwargs:
            whisper_model: str — faster-whisper model size (default "base.en")
            whisper_device: str — "auto", "cuda", or "cpu"
            piper_model: str — path to Piper ONNX voice model
            input_device: int | None — sounddevice input device index
            output_device: int | None — sounddevice output device index
            system_prompt: str — override default personality
            backend: str — "transformers" or "ollama"
            model_name: str — LLM model name for Ollama
            enable_tools: bool — enable tool execution (default True)
        """
        cb = status_callback or (lambda msg: None)

        # Tool support
        self._tools_enabled = kwargs.get("enable_tools", True)

        # Custom system prompt
        if "system_prompt" in kwargs:
            self._system_prompt = kwargs["system_prompt"]
        elif self._tools_enabled:
            from core.prompts import build_voice_tool_prompt
            from tools.agent_tools import get_assistant_tools_prompt
            system_info = get_assistant_tools_prompt()
            self._system_prompt = build_voice_tool_prompt(system_info, os.getcwd())

        # ── STT ─────────────────────────────────────────────────────
        whisper_model = kwargs.get("whisper_model", "base.en")
        whisper_device = kwargs.get("whisper_device", "auto")
        input_device = kwargs.get("input_device", None)

        try:
            self._listener = _Listener(
                whisper_model=whisper_model,
                device=whisper_device,
                input_device=input_device,
            )
            self._listener.load(status_callback=cb)
        except Exception as e:
            cb(f"STT load failed (continuing without voice input): {e}")
            self._listener = None

        # ── LLM ─────────────────────────────────────────────────────
        cb("Loading LLM engine...")

        backend = kwargs.get("backend", None)
        if backend:
            from core.config import set_active_backend
            set_active_backend(backend)

        model_name = kwargs.get("model_name", None)
        if model_name:
            from core.config import set_active_model
            set_active_model(model_name)

        from core.engine_factory import create_engine
        self._engine = create_engine()
        self._engine.load(status_callback=cb)

        # ── TTS (Piper — ONNX, CPU, instant) ──────────────────────
        piper_model = kwargs.get("piper_model", _PIPER_VOICE_MODEL)
        cb(f"Loading TTS: Piper ({os.path.basename(piper_model)})...")

        try:
            from piper.voice import PiperVoice
            self._tts_voice = PiperVoice.load(piper_model)
            self._tts_sample_rate = self._tts_voice.config.sample_rate
            cb(f"TTS ready: Piper {os.path.basename(piper_model)} ({self._tts_sample_rate}Hz)")
        except Exception as e:
            cb(f"TTS load failed (continuing without voice output): {e}")
            self._tts_voice = None
            self._tts_sample_rate = 22050

        # ── Speaker ─────────────────────────────────────────────────
        output_device = kwargs.get("output_device", None)
        self._speaker = _Speaker(output_device=output_device)

        self._loaded = True
        cb("Voice Assistant ready. All systems nominal.")

    def unload(self, status_callback=None):
        cb = status_callback or (lambda msg: None)

        if self._listener:
            self._listener.unload()
            self._listener = None

        if self._engine:
            self._engine.unload(status_callback=cb)
            self._engine = None

        if self._tts_voice:
            del self._tts_voice
            self._tts_voice = None

        self._speaker = None
        self._messages.clear()
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cb("Voice Assistant unloaded.")

    def is_loaded(self) -> bool:
        return self._loaded and self._engine is not None

    def run(self, **kwargs) -> PipelineResult:
        """Run the voice assistant pipeline.

        Supports three modes of operation:

        1. Full voice loop (audio_data → text response + spoken audio):
            audio_data: np.ndarray — raw 16kHz float32 audio from mic
            on_token: callback — streaming token callback
            play_audio: bool — auto-play TTS output (default True)

        2. Text-only (text prompt → text response + spoken audio):
            text: str — user text input (skips STT)
            on_token: callback — streaming token callback
            play_audio: bool — auto-play TTS output (default True)

        3. STT-only (audio → text):
            audio_data: np.ndarray + stt_only: True

        4. TTS-only (text → audio file):
            tts_text: str + tts_only: True

        Returns:
            PipelineResult with output_type "audio" (full/tts) or "text" (stt)
            metadata includes: user_text, response_text, audio_path, audio_chunks
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="audio",
                error="Voice Assistant not loaded. Call load() first."
            )

        try:
            # Route to the right sub-pipeline
            if kwargs.get("stt_only"):
                return self._run_stt_only(**kwargs)
            elif kwargs.get("tts_only"):
                return self._run_tts_only(**kwargs)
            else:
                return self._run_conversation(**kwargs)
        except Exception as e:
            return PipelineResult(
                success=False, output_type="audio",
                error=str(e),
            )

    # ── Voice tool execution ───────────────────────────────────────

    MAX_VOICE_TOOL_ROUNDS = 3

    def _run_voice_tools(self, actions, progress_cb=None):
        """Execute tool actions and return concatenated output string."""
        from tools.agent_tools import run_agent_action, get_tool_output_limit

        outputs = []
        for action in actions:
            if progress_cb:
                progress_cb(f"Running {action.type}: {action.display[:40]}...")
            success, output = run_agent_action(action)
            if success:
                result = output if output else "(no output)"
                limit = get_tool_output_limit(action.type)
                if len(result) > limit:
                    result = result[:limit] + "\n[...truncated...]"
                outputs.append(f"[{action.type} output] `{action.display}`:\n{result}")
            else:
                outputs.append(
                    f"[{action.type} ERROR] `{action.display}`:\nERROR: {output}"
                )
        return "\n\n".join(outputs) if outputs else None

    def _execute_tool_loop(self, response, progress_cb=None, **kwargs):
        """Extract and execute tools from a response, looping up to MAX rounds.

        Returns the final response text (with tool results incorporated).
        If no tools were requested, returns the original response immediately.
        """
        from tools.agent_tools import extract_agent_actions
        from core.inference import strip_think_blocks

        actions = extract_agent_actions(response)
        if not actions:
            return response  # Fast path — no tools, just a spoken answer

        for _round in range(self.MAX_VOICE_TOOL_ROUNDS):
            if progress_cb:
                progress_cb(f"Tool round {_round + 1}/{self.MAX_VOICE_TOOL_ROUNDS}...")

            # Store the tool-invoking assistant text in history
            self._messages.append({"role": "assistant", "content": response})

            # Execute the tools
            tool_output = self._run_voice_tools(actions, progress_cb)
            if not tool_output:
                break

            # Feed results back for the model to summarize
            feedback = (
                "[TOOL OUTPUT]\n\n" + tool_output + "\n\n"
                "Based on the tool output above, give the user a brief spoken answer."
            )
            self._messages.append({"role": "user", "content": feedback})

            messages = [
                {"role": "system", "content": self._system_prompt},
                *self._messages,
            ]

            if progress_cb:
                progress_cb("Thinking...")

            # Generate silently (no on_token — intermediate round)
            response = self._engine.generate_streaming(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                on_token=None,
                enable_thinking=False,
            )
            response = strip_think_blocks(response)

            # Check if the new response requests more tools
            actions = extract_agent_actions(response)
            if not actions:
                break  # Final spoken answer — exit loop

        # Strip any leftover tool markers that would be spoken aloud
        response = re.sub(r'@\w+\([^)]*\)', '', response)
        response = re.sub(r'```(?:bash|sh|shell|python|py|edit)\s*\n.*?```',
                          '', response, flags=re.DOTALL)
        return response.strip()

    # ── Core conversation loop ──────────────────────────────────────

    def _run_conversation(self, **kwargs) -> PipelineResult:
        """Full STT → LLM → TTS conversation turn."""
        on_token = kwargs.get("on_token", None)
        play_audio = kwargs.get("play_audio", True)
        progress_callback = kwargs.get("progress_callback", None)

        def _progress(msg):
            if progress_callback:
                progress_callback(0, 0, msg)

        # ── Stage 1: Get user text ──────────────────────────────────
        user_text = kwargs.get("text", None)

        if user_text is None:
            # STT from audio data
            audio_data = kwargs.get("audio_data", None)
            if audio_data is None:
                return PipelineResult(
                    success=False, output_type="audio",
                    error="Provide 'text' or 'audio_data' for conversation."
                )

            if self._listener is None or not self._listener.is_loaded():
                return PipelineResult(
                    success=False, output_type="audio",
                    error="STT not available — faster-whisper not loaded."
                )

            _progress("Transcribing speech...")
            user_text = self._listener.transcribe(audio_data)

            if not user_text or not user_text.strip():
                return PipelineResult(
                    success=True, output_type="text",
                    content="",
                    metadata={"user_text": "", "reason": "silence"},
                )

        # ── Stage 2: LLM response ──────────────────────────────────
        _progress("Thinking...")

        self._messages.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._messages,
        ]

        from core.inference import strip_think_blocks

        if self._tools_enabled:
            # Generate silently first — tool loop may produce multiple rounds
            initial_response = self._engine.generate_streaming(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                on_token=None,
                enable_thinking=False,
            )
            initial_response = strip_think_blocks(initial_response)

            # Run tool loop (returns immediately if no tools requested)
            full_response = self._execute_tool_loop(
                initial_response, progress_cb=_progress, **kwargs
            )

            # Deliver final response to streaming callback
            if on_token and full_response:
                on_token(full_response)
        else:
            # Original path — stream directly, no tool support
            full_response = ""

            def collect_token(token):
                nonlocal full_response
                full_response += token
                if on_token:
                    on_token(token)

            self._engine.generate_streaming(
                messages=messages,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                on_token=collect_token,
                enable_thinking=False,
            )
            full_response = strip_think_blocks(full_response)

        self._messages.append({"role": "assistant", "content": full_response})

        # ── Stage 3: TTS synthesis (Piper — CPU, instant) ──────────
        _progress("Speaking...")

        output_dir = os.path.join("output", "voice")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "artifex_response.wav")

        if self._tts_voice and full_response.strip():
            import io
            import wave

            chunks = _chunk_for_tts(full_response)
            all_audio = b""

            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    _progress(f"Synthesizing speech ({i + 1}/{len(chunks)})...")
                try:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        self._tts_voice.synthesize_wav(chunk, wf)
                    # Extract raw PCM frames (skip WAV header)
                    buf.seek(0)
                    with wave.open(buf, "rb") as rf:
                        all_audio += rf.readframes(rf.getnframes())
                except Exception:
                    continue

            if all_audio:
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self._tts_sample_rate)
                    wf.writeframes(all_audio)

                if play_audio and self._speaker:
                    _progress("Playing response...")
                    self._speaker.play_file(out_path, block=True)

        _progress("Ready")

        return PipelineResult(
            success=True,
            output_type="audio",
            content=out_path if os.path.isfile(out_path) else None,
            metadata={
                "user_text": user_text,
                "response_text": full_response,
                "audio_path": out_path if os.path.isfile(out_path) else None,
            },
        )

    # ── STT-only ────────────────────────────────────────────────────

    def _run_stt_only(self, **kwargs) -> PipelineResult:
        """Transcribe audio data to text."""
        audio_data = kwargs.get("audio_data", None)
        if audio_data is None:
            return PipelineResult(
                success=False, output_type="text",
                error="audio_data required for STT."
            )

        if self._listener is None or not self._listener.is_loaded():
            return PipelineResult(
                success=False, output_type="text",
                error="STT not available."
            )

        text = self._listener.transcribe(audio_data)
        return PipelineResult(
            success=True, output_type="text",
            content=text,
            metadata={"source": "faster-whisper"},
        )

    # ── TTS-only ────────────────────────────────────────────────────

    def _run_tts_only(self, **kwargs) -> PipelineResult:
        """Synthesize text to audio file."""
        text = kwargs.get("tts_text", "")
        if not text:
            return PipelineResult(
                success=False, output_type="audio",
                error="tts_text required for TTS."
            )

        if self._tts_voice is None:
            return PipelineResult(
                success=False, output_type="audio",
                error="TTS not available."
            )

        play_audio = kwargs.get("play_audio", True)
        import io
        import wave

        output_dir = os.path.join("output", "voice")
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "voice_tts.wav")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self._tts_voice.synthesize_wav(text, wf)
        buf.seek(0)
        with open(out_path, "wb") as f:
            f.write(buf.getvalue())

        if play_audio and self._speaker:
            self._speaker.play_file(out_path, block=True)

        return PipelineResult(
            success=True, output_type="audio",
            content=out_path,
            metadata={"text": text, "sampling_rate": self._tts_sample_rate},
        )

    # ── Conversation management ─────────────────────────────────────

    def _trim_history(self):
        max_messages = self._max_history_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def clear_history(self):
        """Clear conversation history."""
        self._messages.clear()

    def get_history(self) -> list[dict]:
        """Get current conversation messages."""
        return list(self._messages)

    # ── Push-to-talk helper ─────────────────────────────────────────

    def record_push_to_talk(self, on_listening=None, on_processing=None) -> str | None:
        """Record via push-to-talk (hold space) and return transcribed text.

        This is a blocking call — use from a background thread.

        Args:
            on_listening: callback when recording starts
            on_processing: callback when transcription starts

        Returns:
            Transcribed text, or None if silence/no mic
        """
        if self._listener is None or not self._listener.is_loaded():
            return None

        from pynput import keyboard

        stop_event = threading.Event()
        space_held = threading.Event()
        audio_result = [None]
        record_thread = None

        def record_worker():
            audio_result[0] = self._listener.record_audio(stop_event)

        def on_press(key):
            nonlocal record_thread
            if key == keyboard.Key.space and not space_held.is_set():
                space_held.set()
                if on_listening:
                    on_listening()
                record_thread = threading.Thread(target=record_worker, daemon=True)
                record_thread.start()

        def on_release(key):
            if key == keyboard.Key.space and space_held.is_set():
                stop_event.set()
                return False

        with keyboard.Listener(on_press=on_press, on_release=on_release) as kb:
            kb.join()

        if record_thread:
            record_thread.join(timeout=5)

        if audio_result[0] is None:
            return None

        if on_processing:
            on_processing()

        return self._listener.transcribe(audio_result[0])

    # ── Pipeline metadata ───────────────────────────────────────────

    def get_vram_estimate(self, model_path: str = "") -> float:
        """Voice assistant VRAM: LLM + STT only (TTS runs on CPU).

        faster-whisper base.en: ~0.3 GB
        Piper TTS: 0 GB (ONNX, CPU only)
        LLM: varies (estimated separately)
        """
        return 2.5  # LLM + STT, no GPU TTS

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["modes"] = ["conversation", "stt_only", "tts_only"]
        caps["stt_engine"] = "faster-whisper"
        caps["tts_engine"] = "Piper (ONNX, CPU)"
        caps["supports_streaming"] = True
        caps["supports_push_to_talk"] = True
        caps["personality"] = "Artifex"
        caps["components"] = {
            "stt_loaded": self._listener.is_loaded() if self._listener else False,
            "llm_loaded": self._engine is not None,
            "tts_loaded": self._tts_voice is not None,
            "speaker_ready": self._speaker is not None,
        }
        return caps
