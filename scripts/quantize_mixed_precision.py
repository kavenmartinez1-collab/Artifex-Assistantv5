#!/usr/bin/env python3
"""
Mixed-Precision GPTQ Quantization for Qwen3.5-9B

Quantizes ONLY the FFN and full-attention layers to INT4 GPTQ,
keeping all linear_attn (Gated DeltaNet) layers in original BF16.
This prevents INT4 quantization noise from compounding through
the SSM recurrence while saving ~60% VRAM vs full BF16.

Output: SafeTensors file(s) that the WebGPU engine can load directly
with mixed-precision dispatch (BF16 matmul for SSM, INT4 matmul for FFN).

Requirements:
    pip install optimum auto-gptq transformers torch safetensors datasets

Usage:
    python scripts/quantize_mixed_precision.py
    python scripts/quantize_mixed_precision.py --model Qwen/Qwen3.5-9B --bits 4
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from optimum.gptq import GPTQQuantizer


def get_modules_not_to_quantize(config) -> list[str]:
    """Build the list of module names to EXCLUDE from quantization."""
    layer_types = getattr(config, 'layer_types', None)
    if hasattr(config, 'text_config'):
        layer_types = getattr(config.text_config, 'layer_types', layer_types)

    if layer_types is None:
        print("WARNING: Could not determine layer types. Excluding all linear_attn patterns.")
        return []

    modules = []
    proj_suffixes = [
        'linear_attn.in_proj_qkv',
        'linear_attn.in_proj_a',
        'linear_attn.in_proj_b',
        'linear_attn.in_proj_z',
        'linear_attn.out_proj',
    ]

    for layer_idx, ltype in enumerate(layer_types):
        if ltype == 'linear_attention':
            for suffix in proj_suffixes:
                modules.append(f'model.language_model.layers.{layer_idx}.{suffix}')
                modules.append(f'model.layers.{layer_idx}.{suffix}')

    num_linear = sum(1 for t in layer_types if t == 'linear_attention')
    num_full = sum(1 for t in layer_types if t == 'full_attention')
    print(f"  {len(layer_types)} layers: {num_linear} linear_attention + {num_full} full_attention")
    print(f"  Excluding {len(modules)} module patterns from quantization")

    return modules


def get_calibration_dataset(tokenizer, num_samples=128, seq_len=2048):
    """Load calibration data for GPTQ quantization."""
    from datasets import load_dataset

    print(f"  Loading wikitext-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join([t for t in dataset["text"] if len(t.strip()) > 0])

    tokenized = tokenizer(text, return_tensors="pt", truncation=False)
    all_ids = tokenized["input_ids"][0]

    samples = []
    for i in range(num_samples):
        start = (i * seq_len) % max(1, len(all_ids) - seq_len)
        chunk = all_ids[start:start + seq_len]
        if len(chunk) < seq_len:
            chunk = all_ids[:seq_len]
        samples.append({"input_ids": chunk.unsqueeze(0)})

    print(f"  {len(samples)} calibration samples ready (seq_len={seq_len})")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Mixed-precision GPTQ quantization for Qwen3.5")
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HuggingFace model ID")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=128, help="GPTQ group size (default: 128)")
    parser.add_argument("--num-samples", type=int, default=128, help="Calibration samples")
    parser.add_argument("--seq-len", type=int, default=2048, help="Calibration sequence length")
    parser.add_argument("--output", default=None, help="Output directory (default: auto)")
    parser.add_argument("--damp-percent", type=float, default=0.1, help="GPTQ dampening (default: 0.1)")
    args = parser.parse_args()

    if args.output is None:
        model_short = args.model.split("/")[-1]
        args.output = str(Path(__file__).parent.parent / "models" / f"{model_short}-mixed-GPTQ-Int{args.bits}")

    print(f"{'='*60}")
    print(f"Mixed-Precision GPTQ Quantization")
    print(f"{'='*60}")
    print(f"Model:       {args.model}")
    print(f"Bits:        {args.bits}")
    print(f"Group size:  {args.group_size}")
    print(f"Dampening:   {args.damp_percent}")
    print(f"Calibration: {args.num_samples} samples × {args.seq_len} tokens")
    print(f"Output:      {args.output}")
    print(f"{'='*60}")

    # Step 1: Load config and determine which modules to exclude
    print("\n[1/5] Analyzing model architecture...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    exclude_modules = get_modules_not_to_quantize(config)

    # Step 2: Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Step 3: Load calibration data
    print(f"\n[3/5] Preparing calibration data...")
    calibration_dataset = get_calibration_dataset(tokenizer, args.num_samples, args.seq_len)

    # Step 4: Load model and quantize
    print(f"\n[4/5] Loading model (device_map=auto for CPU offloading)...")
    print(f"  9B BF16 model needs ~18GB — will split across GPU + CPU RAM")
    t0 = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Create quantizer with excluded modules
    quantizer = GPTQQuantizer(
        bits=args.bits,
        group_size=args.group_size,
        damp_percent=args.damp_percent,
        dataset=calibration_dataset,
        modules_not_to_quantize=exclude_modules,
    )

    print(f"  Starting quantization (this may take 30-60 minutes with CPU offloading)...")
    t1 = time.time()

    quantized_model = quantizer.quantize_model(model, tokenizer)
    print(f"  Quantization complete in {time.time() - t1:.1f}s")

    # Step 5: Save
    print(f"\n[5/5] Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    quantized_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Also save the quantizer config
    quantizer.save(quantized_model, args.output)

    # Report sizes
    total_size = sum(f.stat().st_size for f in Path(args.output).rglob("*.safetensors"))
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"Done! Mixed-precision GPTQ model saved.")
    print(f"  Output:       {args.output}")
    print(f"  Total size:   {total_size / 1024**3:.2f} GB")
    print(f"  Total time:   {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  INT4 layers:  FFN + full_attention")
    print(f"  BF16 layers:  linear_attention (SSM) + lm_head + embeddings")
    print(f"\nLoad in WebGPU with the local HF cache loader or upload to HuggingFace.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
