#!/usr/bin/env python3
"""
Build a hybrid Qwen3.5-9B model:
  - SSM (linear_attn) weights: original BF16 from base model
  - FFN + attention weights: calibrated GPTQ INT4 from external quantization
  - lm_head + embed_tokens: INT4 from our RTN+ quantization (VRAM-friendly)
  - Norms, biases: original from base model

This combines the best of both worlds:
  - Zero quantization error in SSM recurrence (prevents state corruption)
  - Calibrated GPTQ INT4 for FFN/attention (30-50% less error than RTN+)
  - INT4 embed/lm_head to fit within 8 GB VRAM

Usage:
  python scripts/build_hybrid_model.py \
    --rtn-model models/qwen3.5-9b-mixed-GPTQ-Int4 \
    --gptq-model <path-to-gptq-safetensors> \
    --output models/qwen3.5-9b-hybrid-GPTQ-Int4
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def is_ffn_or_attn_gptq(name: str) -> bool:
    """True if this tensor is a GPTQ triplet for FFN or attention (NOT SSM)."""
    gptq_suffixes = ('.qweight', '.scales', '.qzeros')
    if not any(name.endswith(s) for s in gptq_suffixes):
        return False
    # FFN projections
    ffn_patterns = ['gate_proj', 'up_proj', 'down_proj']
    # Attention projections
    attn_patterns = ['self_attn.q_proj', 'self_attn.k_proj',
                     'self_attn.v_proj', 'self_attn.o_proj']
    for pat in ffn_patterns + attn_patterns:
        if pat in name:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Build hybrid BF16-SSM + GPTQ-INT4 model')
    parser.add_argument('--rtn-model', required=True,
                        help='Path to our RTN+ mixed-precision model directory')
    parser.add_argument('--gptq-model', required=True,
                        help='Path to external GPTQ safetensors file (single file)')
    parser.add_argument('--output', required=True,
                        help='Output directory for hybrid model')
    args = parser.parse_args()

    rtn_path = Path(args.rtn_model)
    gptq_path = Path(args.gptq_model)
    out_path = Path(args.output)
    os.makedirs(out_path, exist_ok=True)

    print(f"{'='*60}")
    print(f"Hybrid Model Builder")
    print(f"{'='*60}")
    print(f"RTN+ model:  {rtn_path}")
    print(f"GPTQ source: {gptq_path}")
    print(f"Output:      {out_path}")
    print(f"{'='*60}")

    # ── Step 1: Load all RTN+ model tensors (multi-shard) ──────────────
    print("\n[1/4] Loading RTN+ model tensors...")
    rtn_shards = sorted(rtn_path.glob("model.safetensors*.safetensors"))
    if not rtn_shards:
        # Single shard
        single = rtn_path / "model.safetensors"
        if single.exists():
            rtn_shards = [single]
        else:
            raise FileNotFoundError(f"No safetensors files in {rtn_path}")

    output_tensors = {}
    for shard in rtn_shards:
        print(f"  Loading {shard.name}...")
        tensors = load_file(str(shard))
        output_tensors.update(tensors)
    print(f"  RTN+ total: {len(output_tensors)} tensors")

    # ── Step 2: Load GPTQ model (single file or multi-shard directory) ─
    print("\n[2/4] Loading GPTQ model tensors...")
    gptq_tensors = {}
    if gptq_path.is_dir():
        gptq_shards = sorted(gptq_path.glob("model.safetensors*.safetensors"))
        if not gptq_shards:
            single = gptq_path / "model.safetensors"
            if single.exists():
                gptq_shards = [single]
            else:
                raise FileNotFoundError(f"No safetensors files in {gptq_path}")
        for shard in gptq_shards:
            print(f"  Loading {shard.name}...")
            gptq_tensors.update(load_file(str(shard)))
    else:
        gptq_tensors = load_file(str(gptq_path))
    print(f"  GPTQ total: {len(gptq_tensors)} tensors")

    # ── Step 3: Replace FFN/attention INT4 weights ─────────────────────
    print("\n[3/4] Replacing FFN + attention weights with GPTQ calibrated...")
    replaced = 0
    skipped_gidx = 0

    # Detect name prefix used by each model
    # RTN+ may use "model.language_model.layers.X" while GPTQ uses "model.layers.X"
    rtn_has_lm = any('language_model' in k for k in output_tensors.keys())
    gptq_has_lm = any('language_model' in k for k in gptq_tensors.keys())

    def to_rtn_name(gptq_name: str) -> str:
        """Convert GPTQ tensor name to match RTN+ naming convention."""
        if rtn_has_lm and not gptq_has_lm:
            return gptq_name.replace('model.layers.', 'model.language_model.layers.')
        if not rtn_has_lm and gptq_has_lm:
            return gptq_name.replace('model.language_model.layers.', 'model.layers.')
        return gptq_name

    for name, tensor in gptq_tensors.items():
        # Skip g_idx tensors (not used by our engine)
        if name.endswith('.g_idx'):
            skipped_gidx += 1
            continue

        if is_ffn_or_attn_gptq(name):
            target_name = to_rtn_name(name)
            if target_name in output_tensors:
                old_shape = output_tensors[target_name].shape
                new_shape = tensor.shape
                if old_shape != new_shape:
                    print(f"  WARNING: shape mismatch for {target_name}: "
                          f"RTN+={old_shape} vs GPTQ={new_shape}")
                    continue
                # Replace with GPTQ version (using RTN+ naming)
                output_tensors[target_name] = tensor
            else:
                # Name not found in RTN+ — add with RTN+ naming
                output_tensors[target_name] = tensor
            replaced += 1

    print(f"  Replaced {replaced} FFN/attention GPTQ tensors")
    print(f"  Skipped {skipped_gidx} g_idx tensors")
    if rtn_has_lm != gptq_has_lm:
        print(f"  Name prefix mapping: GPTQ {'model.layers' if not gptq_has_lm else 'model.language_model.layers'}"
              f" -> RTN+ {'model.language_model.layers' if rtn_has_lm else 'model.layers'}")

    # Verify SSM weights are still BF16 (not GPTQ)
    ssm_bf16_count = 0
    ssm_q4_count = 0
    for name in output_tensors:
        if 'linear_attn' in name:
            if name.endswith('.weight') and output_tensors[name].dtype == torch.bfloat16:
                ssm_bf16_count += 1
            elif name.endswith('.qweight'):
                ssm_q4_count += 1

    print(f"  SSM projection weights: {ssm_bf16_count} BF16, {ssm_q4_count} INT4")
    if ssm_q4_count > 0:
        print("  WARNING: Some SSM projections are still INT4!")

    # ── Step 4: Save hybrid model ──────────────────────────────────────
    print("\n[4/4] Saving hybrid model...")

    # Estimate output size
    total_bytes = sum(t.numel() * t.element_size() for t in output_tensors.values())
    print(f"  Total size: {total_bytes / (1024**3):.2f} GB")
    print(f"  Total tensors: {len(output_tensors)}")

    # Save as safetensors (split into shards if > 4 GB)
    SHARD_LIMIT = 4 * 1024 * 1024 * 1024  # 4 GB per shard

    # Sort tensors by name for deterministic output
    sorted_names = sorted(output_tensors.keys())

    shards = []
    current_shard = {}
    current_size = 0
    shard_idx = 1

    for name in sorted_names:
        tensor = output_tensors[name]
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > SHARD_LIMIT and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
            shard_idx += 1

        current_shard[name] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    total_shards = len(shards)
    weight_map = {}

    for i, shard_tensors in enumerate(shards):
        if total_shards == 1:
            filename = "model.safetensors"
        else:
            filename = f"model.safetensors-{i+1:05d}-of-{total_shards:05d}.safetensors"

        filepath = out_path / filename
        print(f"  Writing {filename} ({len(shard_tensors)} tensors)...")
        save_file(shard_tensors, str(filepath))

        for name in shard_tensors:
            weight_map[name] = filename

    # Write index if multi-shard
    if total_shards > 1:
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        index_path = out_path / "model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
        print(f"  Written {index_path.name}")

    # Copy config, tokenizer, and other metadata from RTN+ model
    for cfg_file in ['tokenizer.json', 'tokenizer_config.json',
                     'chat_template.jinja', 'merges.txt', 'vocab.json',
                     'preprocessor_config.json']:
        src = rtn_path / cfg_file
        if src.exists():
            shutil.copy2(str(src), str(out_path / cfg_file))

    # Build config.json with hybrid quantization metadata
    config_src = rtn_path / 'config.json'
    with open(config_src) as f:
        config = json.load(f)

    config['quantization_config'] = {
        "quant_method": "gptq",
        "bits": 4,
        "group_size": 128,
        "checkpoint_format": "gptq_v2",
        "mixed_precision": True,
        "hybrid_source": "calibrated_gptq",
        "modules_not_quantized": [
            "linear_attn", "norm"
        ],
    }

    with open(out_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Hybrid model saved to {out_path}")
    print(f"  SSM: original BF16 (zero quant error)")
    print(f"  FFN + attention: calibrated GPTQ INT4")
    print(f"  embed/lm_head: RTN+ INT4")
    print(f"  Total: {total_bytes / (1024**3):.2f} GB ({len(output_tensors)} tensors)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
