"""
Make the 4-bit Qwen-Image 2.1 folder Artifex's image pipelines load.

Downloads the official bf16 weights (~32 GB, or uses a local copy), then
saves the transformer and the Qwen3-VL text encoder as bitsandbytes NF4:
about 11 GB in total, and the folder loads straight into
core/pipelines/image_gen.py and image_edit.py with no re-quantizing.
Measured on an 8 GB RTX 5060 Ti: 1024px generation 24-29 s, edit 45-75 s.

The bf16 source can be deleted afterwards (--cleanup does it when this
script downloaded it).

    python scripts/quantize_qwen_image.py                      # Turbo (8 steps)
    python scripts/quantize_qwen_image.py --repo Qwen/Qwen-Image-2.1 \
        --out models/qwen-image-2.1                            # base (40 steps)

Needs diffusers>=0.41 and bitsandbytes, and a CUDA GPU (bitsandbytes
quantizes on the GPU; peak ~4 GB for the transformer).
"""

import argparse
import gc
import os
import shutil
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIG = {"transformer", "text_encoder"}


def gb(path):
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(path) for f in fs) / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repo", default="Qwen/Qwen-Image-2.1-Turbo",
                    help="Hugging Face repo id, or a local bf16 folder")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "qwen-image-2.1-turbo"),
                    help="output folder (keep 'qwen-image' in the name: the GUIs "
                         "use it to pick 1024px defaults)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the downloaded bf16 source when done")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("A CUDA GPU is required (bitsandbytes quantizes on the GPU).")

    if os.path.isdir(a.repo):
        src, downloaded = a.repo, False
    else:
        from huggingface_hub import snapshot_download
        src = os.path.join(REPO_ROOT, "models", ".download-" + a.repo.replace("/", "--"))
        print(f"Downloading {a.repo} (~32 GB) to {src} ...", flush=True)
        snapshot_download(a.repo, local_dir=src, max_workers=8)
        downloaded = True

    os.makedirs(a.out, exist_ok=True)
    for name in os.listdir(src):
        if name.startswith(".") or name in BIG:
            continue
        s, d = os.path.join(src, name), os.path.join(a.out, name)
        shutil.copytree(s, d, dirs_exist_ok=True) if os.path.isdir(s) else shutil.copy2(s, d)

    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import QwenImage21Transformer2DModel
    t = time.time()
    m = QwenImage21Transformer2DModel.from_pretrained(
        src, subfolder="transformer", torch_dtype=torch.bfloat16,
        quantization_config=DiffusersBnb(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16))
    m.save_pretrained(os.path.join(a.out, "transformer"))
    print(f"transformer: {gb(os.path.join(a.out, 'transformer')):.2f} GB "
          f"({time.time() - t:.0f} s)", flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import BitsAndBytesConfig as TransformersBnb
    from transformers import Qwen3VLForConditionalGeneration
    t = time.time()
    m = Qwen3VLForConditionalGeneration.from_pretrained(
        src, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersBnb(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            # vision tower stays bf16: small, and only used when editing
            llm_int8_skip_modules=["visual", "lm_head"]))
    m.save_pretrained(os.path.join(a.out, "text_encoder"))
    print(f"text encoder: {gb(os.path.join(a.out, 'text_encoder')):.2f} GB "
          f"({time.time() - t:.0f} s)", flush=True)

    if a.cleanup and downloaded:
        shutil.rmtree(src, ignore_errors=True)
    print(f"Done: {gb(a.out):.1f} GB at {a.out}")


if __name__ == "__main__":
    main()
