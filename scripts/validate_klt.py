#!/usr/bin/env python3
"""
KLT Rotation Validator — Traces rotation fidelity through the forward pass.

Runs the original BF16 model and KLT-rotated model side-by-side on the same
input, comparing hidden states at every layer boundary. Reports WHERE the
rotation first diverges beyond tolerance.

Usage:
  python scripts/validate_klt.py --model ./models/qwen3.5-9b --device cuda
"""
import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def log(msg: str):
    print(msg, flush=True)


def validate_klt_rotation(model_path: str, device: str = "cuda"):
    """Compare original vs KLT-rotated model layer by layer."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from scripts.quantize_gptq import (
        compute_klt_rotation, find_inner_model,
        absorb_norm_scales, fuse_klt_into_model,
    )

    gpu = torch.device(device)

    # ── Load original model ──
    log("[1/5] Loading original model...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model_orig.eval()

    # ── Load KLT model (copy + rotate) ──
    log("[2/5] Loading KLT model (will absorb norms + rotate)...")
    model_klt = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model_klt.eval()

    # Apply KLT rotation
    log("[3/5] Absorbing norms + computing KLT rotation...")
    absorb_norm_scales(model_klt, gpu)

    inner_klt = find_inner_model(model_klt)

    # Compute R from original model's embeddings (same as quantizer does)
    inner_orig = find_inner_model(model_orig)

    # Create test input
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    test_text = "What is the equation of Newton's law of gravity?"
    tokens = tokenizer(test_text, return_tensors="pt")["input_ids"][:, :32]
    log(f"  Test input: {tokens.shape[1]} tokens")

    # Compute R from embeddings
    inner_orig.embed_tokens.to(gpu)
    with torch.no_grad():
        embed_out = inner_orig.embed_tokens(tokens.to(gpu))
    inner_orig.embed_tokens.to('cpu')

    R = compute_klt_rotation([embed_out.cpu()], gpu)
    log(f"  R orthogonality error: {(R @ R.T - torch.eye(R.shape[0], device=R.device, dtype=R.dtype)).abs().max().item():.2e}")

    # Fuse R into KLT model
    fuse_klt_into_model(model_klt, R, gpu)
    R_cpu = R.float().cpu()

    if gpu.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Run both models and compare at each layer ──
    log(f"\n[4/5] Running side-by-side comparison...")
    log(f"{'Layer':>8} {'Type':>5} {'CosSim(Rh,h_klt)':>18} {'MaxAbsDiff':>14} {'RelDiff%':>10} {'Status':>8}")
    log(f"{'-'*70}")

    inner_orig = find_inner_model(model_orig)
    inner_klt = find_inner_model(model_klt)
    layers_orig = inner_orig.layers
    layers_klt = inner_klt.layers

    hf_config = config.to_dict()
    if hasattr(config, 'text_config'):
        hf_config.update(config.text_config.to_dict() if hasattr(config.text_config, 'to_dict') else {})
    layer_types = hf_config.get('layer_types', [])

    # Step 1: Embed
    inner_orig.embed_tokens.to(gpu)
    inner_klt.embed_tokens.to(gpu)
    with torch.no_grad():
        h_orig = inner_orig.embed_tokens(tokens.to(gpu)).float()
        h_klt = inner_klt.embed_tokens(tokens.to(gpu)).float()
    inner_orig.embed_tokens.to('cpu')
    inner_klt.embed_tokens.to('cpu')

    # Check: h_klt should be R @ h_orig (rotated embedding)
    h_orig_rotated = h_orig @ R_cpu.to(gpu).T
    cos = torch.nn.functional.cosine_similarity(
        h_orig_rotated.reshape(-1, h_orig.shape[-1]),
        h_klt.reshape(-1, h_klt.shape[-1]), dim=1
    ).mean().item()
    diff = (h_orig_rotated - h_klt).abs().max().item()
    rel = diff / (h_orig.abs().max().item() + 1e-10) * 100
    status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "FAIL"
    log(f"{'embed':>8} {'---':>5} {cos:>18.6f} {diff:>14.6f} {rel:>9.2f}% {status:>8}")

    # Compute rotary embeddings
    rotary_emb = getattr(inner_orig, 'rotary_emb', None)
    pos_embeds_orig = None
    pos_embeds_klt = None
    if rotary_emb is not None:
        rotary_emb.to(gpu)
        position_ids = torch.arange(tokens.shape[1], device=gpu).unsqueeze(0)
        with torch.no_grad():
            pos_embeds_orig = rotary_emb(h_orig.to(torch.bfloat16), position_ids)
        pos_embeds_orig = tuple(p for p in pos_embeds_orig)
        pos_embeds_klt = pos_embeds_orig  # Same position embeddings
        rotary_emb.to('cpu')

    first_fail_layer = None

    # Step 2: Layer by layer
    for li in range(len(layers_orig)):
        ltype = layer_types[li] if li < len(layer_types) else 'unknown'
        ltype_short = 'SSM' if ltype == 'linear_attention' else 'ATN'

        layers_orig[li].to(gpu)
        layers_klt[li].to(gpu)

        kw = {}
        if pos_embeds_orig is not None:
            kw['position_embeddings'] = pos_embeds_orig

        with torch.no_grad():
            out_orig = layers_orig[li](h_orig.to(torch.bfloat16), **kw)
            out_klt = layers_klt[li](h_klt.to(torch.bfloat16), **kw)

        if isinstance(out_orig, tuple):
            out_orig = out_orig[0]
        if isinstance(out_klt, tuple):
            out_klt = out_klt[0]

        h_orig = out_orig.float()
        h_klt = out_klt.float()

        layers_orig[li].to('cpu')
        layers_klt[li].to('cpu')

        # Compare: R @ h_orig should approximately equal h_klt
        h_orig_rotated = h_orig @ R_cpu.to(gpu).T
        cos = torch.nn.functional.cosine_similarity(
            h_orig_rotated.reshape(-1, h_orig.shape[-1]),
            h_klt.reshape(-1, h_klt.shape[-1]), dim=1
        ).mean().item()
        diff = (h_orig_rotated - h_klt).abs().max().item()
        rel = diff / (h_orig.abs().max().item() + 1e-10) * 100
        status = "OK" if cos > 0.999 else "WARN" if cos > 0.99 else "FAIL"

        log(f"{'L'+str(li):>8} {ltype_short:>5} {cos:>18.6f} {diff:>14.6f} {rel:>9.2f}% {status:>8}")

        if status == "FAIL" and first_fail_layer is None:
            first_fail_layer = li
            log(f"  >>> FIRST FAILURE at layer {li} ({ltype_short})")
            log(f"  >>> h_orig norm: {h_orig.norm():.4f}")
            log(f"  >>> h_klt norm:  {h_klt.norm():.4f}")
            log(f"  >>> h_orig[0,0,:8]: {h_orig[0,0,:8].tolist()}")
            log(f"  >>> h_klt[0,0,:8]:  {h_klt[0,0,:8].tolist()}")
            log(f"  >>> R@h_orig[0,0,:8]: {h_orig_rotated[0,0,:8].tolist()}")

        if gpu.type == 'cuda':
            torch.cuda.empty_cache()

    # ── Summary ──
    log(f"\n[5/5] Summary")
    log(f"{'='*70}")
    if first_fail_layer is not None:
        log(f"ROTATION BREAKS at layer {first_fail_layer} ({layer_types[first_fail_layer] if first_fail_layer < len(layer_types) else 'unknown'})")
        log(f"All layers before L{first_fail_layer} maintain rotation fidelity.")
        log(f"Investigate: what operation in L{first_fail_layer} doesn't commute with R?")
    else:
        log(f"ALL LAYERS PASS - rotation fidelity maintained throughout.")
        log(f"Issue may be in quantization, not rotation.")

    del model_orig, model_klt
    gc.collect()
    if gpu.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate KLT rotation fidelity")
    parser.add_argument("--model", required=True, help="Path to base BF16 model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    validate_klt_rotation(args.model, args.device)
