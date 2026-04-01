#!/usr/bin/env python3
"""Quick perplexity test for quantized models via HuggingFace.

Usage:
  python scripts/test_ppl.py --model ./models/qwen3.5-9b-HailMary
  python scripts/test_ppl.py --model ./models/qwen3.5-9b-GPTQv2-noact
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn.functional as F


def log(msg: str):
    print(msg, flush=True)


def test_perplexity(model_path: str, device: str = "cuda",
                    num_samples: int = 8, seq_length: int = 512):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load wikitext eval
    log(f"Loading evaluation data...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    except ImportError:
        log("WARNING: datasets not installed, using random tokens")
        text = "The quick brown fox " * 10000

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    all_ids = encoded["input_ids"][0]

    samples = []
    for i in range(0, len(all_ids) - seq_length, seq_length):
        chunk = all_ids[i:i + seq_length].unsqueeze(0)
        samples.append(chunk)
        if len(samples) >= num_samples:
            break

    log(f"  {len(samples)} eval samples, {seq_length} tokens each")

    # Run inference
    gpu = torch.device(device)
    total_ce = 0
    total_tokens = 0

    log(f"Computing perplexity...")
    t_start = time.time()

    # Process layer-by-layer to fit in VRAM
    # For simplicity, just move the whole model to GPU if it fits
    try:
        model.to(gpu)
        log(f"  Model on GPU")
    except RuntimeError:
        log(f"  Model too large for GPU, using CPU (slow)")
        gpu = torch.device("cpu")

    with torch.no_grad():
        for i, sample in enumerate(samples):
            outputs = model(sample.to(gpu), labels=sample.to(gpu))
            total_ce += outputs.loss.item() * (sample.shape[1] - 1)
            total_tokens += sample.shape[1] - 1
            if (i + 1) % 4 == 0:
                avg_ce = total_ce / total_tokens
                log(f"  [{i+1}/{len(samples)}] CE={avg_ce:.4f} PPL={math.exp(min(avg_ce, 100)):.2f}")

    avg_ce = total_ce / total_tokens
    ppl = math.exp(min(avg_ce, 100))
    dt = time.time() - t_start

    log(f"\n{'='*60}")
    log(f"Model: {model_path}")
    log(f"Cross-entropy: {avg_ce:.4f}")
    log(f"Perplexity:    {ppl:.2f}")
    log(f"Samples:       {len(samples)} x {seq_length} tokens")
    log(f"Time:          {dt:.1f}s")
    log(f"{'='*60}")

    # Also test generation quality
    log(f"\nGeneration test:")
    prompt = "What is Newton's law of gravity?"
    inputs = tokenizer(prompt, return_tensors="pt").to(gpu)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False,
                             temperature=1.0, top_p=1.0)
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    log(f"Prompt: {prompt}")
    log(f"Response: {response[:500]}")

    return avg_ce, ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=512)
    args = parser.parse_args()

    test_perplexity(args.model, args.device, args.num_samples, args.seq_length)
