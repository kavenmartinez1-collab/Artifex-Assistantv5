"""
Artifex Assistant V5 — Audio Pipeline.
Text-to-speech (TTS) and speech-to-text (STT) using transformers.
"""

import gc
import os

import torch

from core.pipelines.base import BasePipeline, PipelineResult


class AudioPipeline(BasePipeline):
    """Audio generation (TTS) and speech recognition (STT)."""

    pipeline_type = "text-to-audio"
    display_name = "Audio (TTS / STT)"
    output_type = "audio"

    def __init__(self):
        self.pipe = None
        self._model_path = None
        self._mode = None  # "tts" or "stt"

    def load(self, model_path: str, status_callback=None, **kwargs):
        """Load an audio model.

        For TTS models (e.g. bark, speecht5):
            Uses text-to-audio pipeline

        For STT models (e.g. whisper):
            Uses automatic-speech-recognition pipeline

        Args:
            model_path: Local path or HuggingFace repo ID
            status_callback: Progress callback
            mode: "tts" or "stt" (auto-detected if not specified)
        """
        from transformers import pipeline as hf_pipeline

        # Check audio dependencies early — missing these causes crashes
        try:
            import scipy.io.wavfile  # noqa: F401 — needed for TTS WAV export
        except ImportError:
            raise ImportError(
                "scipy is required for audio WAV export.\n"
                "Install with: pip install scipy"
            )

        if status_callback:
            status_callback(f"Loading audio model: {os.path.basename(model_path)}...")

        mode = kwargs.get("mode", None)

        # Auto-detect mode from model config
        if mode is None:
            mode = self._detect_mode(model_path)

        dtype = kwargs.get("dtype", torch.float16)

        if mode == "stt":
            # STT needs librosa or soundfile for audio loading
            try:
                import soundfile  # noqa: F401
            except ImportError:
                raise ImportError(
                    "soundfile is required for audio file loading (STT).\n"
                    "Install with: pip install soundfile librosa"
                )
            self.pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=model_path,
                torch_dtype=dtype,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            self.pipeline_type = "automatic-speech-recognition"
            self.display_name = "Speech Recognition (STT)"
        else:
            # TTS
            try:
                self.pipe = hf_pipeline(
                    "text-to-audio",
                    model=model_path,
                    torch_dtype=dtype,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception:
                # Some models use text-to-speech task
                self.pipe = hf_pipeline(
                    "text-to-speech",
                    model=model_path,
                    torch_dtype=dtype,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

        self._mode = mode
        self._model_path = model_path

        if status_callback:
            label = "TTS" if mode == "tts" else "STT"
            status_callback(f"Audio model loaded ({label}): {os.path.basename(model_path)}")

    def _detect_mode(self, model_path: str) -> str:
        """Auto-detect whether this is a TTS or STT model."""
        import json
        config_path = os.path.join(model_path, "config.json")
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                model_type = cfg.get("model_type", "").lower()
                if "whisper" in model_type or "wav2vec" in model_type:
                    return "stt"
            except Exception:
                pass

        # Check model name heuristics
        name_lower = os.path.basename(model_path).lower()
        if "whisper" in name_lower or "wav2vec" in name_lower or "stt" in name_lower:
            return "stt"

        return "tts"

    def unload(self, status_callback=None):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self._model_path = None
            self._mode = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if status_callback:
                status_callback("Audio model unloaded.")

    def is_loaded(self) -> bool:
        return self.pipe is not None

    def run(self, **kwargs) -> PipelineResult:
        """Run audio generation or recognition.

        For TTS:
            text: Text to speak
            output_path: Path to save audio file

        For STT:
            audio_path: Path to audio file to transcribe

        Returns:
            PipelineResult with audio path (TTS) or text (STT)
        """
        if not self.is_loaded():
            return PipelineResult(
                success=False, output_type="audio",
                error="No model loaded."
            )

        try:
            if self._mode == "stt":
                return self._run_stt(**kwargs)
            else:
                return self._run_tts(**kwargs)
        except Exception as e:
            return PipelineResult(
                success=False,
                output_type="audio" if self._mode == "tts" else "text",
                error=str(e),
            )

    def _run_tts(self, **kwargs) -> PipelineResult:
        """Text-to-speech generation."""
        text = kwargs.get("text", "")
        output_path = kwargs.get("output_path", None)

        if not text:
            return PipelineResult(
                success=False, output_type="audio",
                error="Text is required for TTS."
            )

        result = self.pipe(text)

        # Save audio
        if output_path is None:
            os.makedirs("output", exist_ok=True)
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in text)[:30]
            output_path = os.path.join("output", f"{safe_name}.wav")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        import scipy.io.wavfile
        audio = result["audio"]
        sampling_rate = result["sampling_rate"]
        scipy.io.wavfile.write(output_path, sampling_rate, audio)

        return PipelineResult(
            success=True,
            output_type="audio",
            content=output_path,
            metadata={"text": text, "sampling_rate": sampling_rate},
        )

    def _run_stt(self, **kwargs) -> PipelineResult:
        """Speech-to-text recognition."""
        audio_path = kwargs.get("audio_path", "")

        if not audio_path or not os.path.isfile(audio_path):
            return PipelineResult(
                success=False, output_type="text",
                error=f"Audio file not found: {audio_path}"
            )

        result = self.pipe(audio_path)
        text = result.get("text", "")

        return PipelineResult(
            success=True,
            output_type="text",
            content=text,
            metadata={"audio_path": audio_path},
        )

    def get_vram_estimate(self, model_path: str) -> float:
        """Audio models are typically small: 1-4 GB."""
        from core.model_registry import _estimate_model_size
        size_gb = _estimate_model_size(model_path)
        return round(max(size_gb * 1.2, 2.0), 1)

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        caps["modes"] = ["tts", "stt"]
        caps["audio_formats"] = ["wav", "mp3", "flac"]
        return caps
