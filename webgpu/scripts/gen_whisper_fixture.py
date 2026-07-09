"""
Whisper-base.en parity fixture generator (STT port ground truth).

HF transformers is the oracle. For a deterministic synthetic 16 kHz clip (and
any real .wav dropped into scripts/whisper_fixture/clips/) it dumps, per stage:

  W1 mel     : input_features [80, 3000]  (WhisperFeatureExtractor, Slaney LUT,
               log10 + global-max clamp + (x+4)/4 normalize)
  W2 encoder : last_hidden_state [1500, 512]  (conv stem /2 + 6 transformer layers)
  W3 decoder : logits [T, vocab] for a forced token prefix (cross-attn to encoder)

Plus the raw f32 samples (so the TS side reads the identical array) and a full
generate() transcription (sanity only, not a parity target).

Run: PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe webgpu/scripts/gen_whisper_fixture.py
Out: webgpu/scripts/whisper_fixture/  (gitignored)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Oracle loads the pristine HF repo (cached by convert-whisper-hf.py), not the
# engine-format webgpu/models/whisper-base-en (whose config marker + missing
# preprocessor_config.json are for the TS runtime, not HF loaders).
HF_REPO = "openai/whisper-base.en"
OUT_DIR = HERE / "whisper_fixture"
CLIPS_DIR = OUT_DIR / "clips"


def synth_clip(seconds: float = 5.0, sr: int = 16000) -> np.ndarray:
    """Deterministic, frontend-exercising signal: two tones + seeded noise."""
    rng = np.random.default_rng(1234)
    t = np.arange(int(seconds * sr), dtype=np.float64) / sr
    x = (0.4 * np.sin(2 * np.pi * 220.0 * t)
         + 0.25 * np.sin(2 * np.pi * 660.0 * t)
         + 0.1 * rng.standard_normal(t.shape))
    # a gentle amplitude envelope so it isn't stationary
    env = 0.5 * (1 - np.cos(2 * np.pi * t / seconds))
    return (x * env).astype(np.float32)


def load_clips() -> dict[str, np.ndarray]:
    clips = {"synth": synth_clip()}
    if CLIPS_DIR.exists():
        import soundfile as sf
        for wav in sorted(CLIPS_DIR.glob("*.wav")):
            audio, sr = sf.read(str(wav), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                # linear resample to 16 kHz (parity oracle uses the same array)
                n = int(round(len(audio) * 16000 / sr))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, n),
                    np.arange(len(audio)), audio).astype(np.float32)
            clips[wav.stem] = audio
    return clips


def main():
    import torch
    from transformers import (WhisperFeatureExtractor,
                              WhisperForConditionalGeneration, WhisperTokenizer)

    OUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    fe = WhisperFeatureExtractor.from_pretrained(HF_REPO)
    tok = WhisperTokenizer.from_pretrained(HF_REPO)
    model = WhisperForConditionalGeneration.from_pretrained(
        HF_REPO, torch_dtype=torch.float32)
    model.eval()

    manifest = {"meta": {"variant": "base.en", "sample_rate": 16000},
                "tensors": {}}

    def dump(name: str, arr: np.ndarray):
        f = name.replace("/", "_").replace(".", "_") + ".bin"
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        (OUT_DIR / f).write_bytes(arr.tobytes())
        manifest["tensors"][name] = {"file": f, "shape": list(arr.shape), "dtype": "f32"}

    clips = load_clips()
    for cid, audio in clips.items():
        dump(f"{cid}.audio", audio)

        feat = fe(audio, sampling_rate=16000, return_tensors="pt")
        mel = feat.input_features  # [1, 80, 3000]
        dump(f"{cid}.mel", mel[0].numpy())

        with torch.no_grad():
            enc = model.model.encoder(mel).last_hidden_state  # [1, 1500, 512]
        dump(f"{cid}.encoder", enc[0].numpy())

        # forced-prefix decoder logits (isolates decoder math from search)
        prefix = [model.config.decoder_start_token_id]  # <|startoftranscript|>
        dec_in = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            out = model.model.decoder(input_ids=dec_in, encoder_hidden_states=enc)
            logits = model.proj_out(out.last_hidden_state)  # [1, T, vocab]
        dump(f"{cid}.dec_logits", logits[0].numpy())
        manifest["meta"][f"{cid}.dec_prefix"] = prefix

        # full greedy generate() — the W4 gate: exact token-sequence match
        with torch.no_grad():
            gen = model.generate(mel, max_new_tokens=64, num_beams=1, do_sample=False)
        gen_tokens = gen[0].tolist()
        text = tok.decode(gen[0], skip_special_tokens=True)
        manifest["meta"][f"{cid}.gen_tokens"] = gen_tokens
        manifest["meta"][f"{cid}.transcription"] = text
        print(f"{cid}: {len(audio)} samples, enc {tuple(enc.shape)}, "
              f"tokens={gen_tokens} transcription={text!r}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {OUT_DIR}  ({len(clips)} clips)")


if __name__ == "__main__":
    main()
