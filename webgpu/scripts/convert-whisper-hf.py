"""
Fetch openai/whisper-base.en (HF safetensors) -> engine format for the WebGPU
STT port. base.en is English-only (~72M params): a good, standard port/parity
source (unlike the local faster-whisper CT2 model.bin, which is a custom packed
serialization, or whisper-large-v3 at 3 GB).

Downloads via huggingface_hub (~145 MB) into the HF cache, then re-emits into
  webgpu/models/whisper-base-en/:
    - model.safetensors   (all weights, HF tensor names kept, cast to f32)
    - config.json         (model_type marker + dims + special token ids)
    - mel_filters.bin     (80x201 f32 Slaney mel filterbank — the frontend LUT)
    - tokenizer.json      (GPT2-BPE + whisper specials, copied verbatim)

The HF names are already clean and hierarchical
(model.encoder.layers.0.self_attn.q_proj.weight, ...) so no graph-traversal
naming is needed here — unlike the Piper ONNX export which fused weight-norm and
dropped names. The TS runtime references these names directly.

Run:  ./venv/Scripts/python.exe webgpu/scripts/convert-whisper-hf.py
Gate: asserts param count == 72,593,920 (whisper-base.en) and mel LUT shape.
"""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "webgpu/models/whisper-base-en"
REPO = "openai/whisper-base.en"
EXPECT_PARAMS = 72_593_408  # whisper-base.en unique params (decoder embed<->proj_out tied)


def main():
    from huggingface_hub import snapshot_download
    from transformers import WhisperFeatureExtractor
    from safetensors.torch import load_file as st_load
    import torch

    print(f"downloading {REPO} (~145 MB) into the HF cache ...")
    local = Path(snapshot_download(
        REPO,
        allow_patterns=[
            "model.safetensors", "config.json", "generation_config.json",
            "tokenizer.json", "tokenizer_config.json", "vocab.json",
            "merges.txt", "added_tokens.json", "special_tokens_map.json",
            "normalizer.json", "preprocessor_config.json",
        ],
    ))
    print(f"snapshot at {local}")

    hf_cfg = json.loads((local / "config.json").read_text(encoding="utf-8"))

    # ---- weights: cast every tensor to f32, keep HF names -------------------
    src = st_load(str(local / "model.safetensors"))
    tensors = {}
    n_params = 0
    for name, t in src.items():
        # HF ties decoder input embed <-> lm_head; the proj_out weight is the
        # same storage as decoder.embed_tokens.weight. Keep both names so the
        # runtime can look up either; count params once (dedupe by data_ptr).
        arr = t.to(torch.float32).contiguous().cpu().numpy()
        tensors[name] = arr
    seen_ptrs = set()
    for name, t in src.items():
        ptr = t.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        n_params += int(np.prod(t.shape))
    print(f"tensors: {len(tensors)}   unique params: {n_params:,}")
    assert n_params == EXPECT_PARAMS, f"param count {n_params:,} != {EXPECT_PARAMS:,}"

    # ---- mel filterbank LUT (Slaney) — freeze the exact HF frontend LUT -----
    fe = WhisperFeatureExtractor.from_pretrained(str(local))
    mel = np.asarray(fe.mel_filters, dtype=np.float32)  # [n_freq=201, n_mels=80]
    # store as [n_mels, n_freq] row-major so the TS matmul is (mel @ power)
    mel_t = np.ascontiguousarray(mel.T)  # [80, 201]
    assert mel_t.shape == (80, 201), f"mel LUT shape {mel_t.shape} != (80, 201)"

    # ---- config: dims + special tokens the runtime needs -------------------
    config = {
        "model_type": "whisper-stt",
        "variant": "base.en",
        "d_model": hf_cfg["d_model"],
        "encoder_layers": hf_cfg["encoder_layers"],
        "encoder_attention_heads": hf_cfg["encoder_attention_heads"],
        "encoder_ffn_dim": hf_cfg["encoder_ffn_dim"],
        "decoder_layers": hf_cfg["decoder_layers"],
        "decoder_attention_heads": hf_cfg["decoder_attention_heads"],
        "decoder_ffn_dim": hf_cfg["decoder_ffn_dim"],
        "vocab_size": hf_cfg["vocab_size"],
        "num_mel_bins": hf_cfg["num_mel_bins"],
        "max_source_positions": hf_cfg["max_source_positions"],
        "max_target_positions": hf_cfg["max_target_positions"],
        "bos_token_id": hf_cfg["bos_token_id"],
        "eos_token_id": hf_cfg["eos_token_id"],
        "pad_token_id": hf_cfg["pad_token_id"],
        "decoder_start_token_id": hf_cfg["decoder_start_token_id"],
        "activation_function": hf_cfg["activation_function"],
        "frontend": {  # mel frontend params (preprocessor_config.json)
            "sampling_rate": 16000,
            "n_fft": 400,
            "hop_length": 160,
            "n_samples": 480000,
            "nb_max_frames": 3000,
            "mel_scale": "slaney",
            "mel_norm": "slaney",
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file
    save_file(tensors, str(OUT_DIR / "model.safetensors"))
    (OUT_DIR / "mel_filters.bin").write_bytes(mel_t.tobytes())
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json",
               "merges.txt", "added_tokens.json", "special_tokens_map.json",
               "normalizer.json", "generation_config.json"):
        srcf = local / fn
        if srcf.exists():
            shutil.copy2(srcf, OUT_DIR / fn)

    size = (OUT_DIR / "model.safetensors").stat().st_size
    print(f"wrote {OUT_DIR}")
    print(f"  model.safetensors {size/1e6:.1f} MB, {len(tensors)} tensors")
    print(f"  mel_filters.bin   {mel_t.nbytes} bytes ({mel_t.shape})")


if __name__ == "__main__":
    main()
