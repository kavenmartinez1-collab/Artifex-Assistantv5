#!/usr/bin/env python3
"""
Mixed-Precision RTN Quantization for Qwen3.5-9B

Quantizes FFN and full-attention layers to INT4 (RTN round-to-nearest),
keeping all linear_attn (Gated DeltaNet) layers in original BF16.

Output format is GPTQ-compatible SafeTensors:
  - .qweight (I32, packed 8 nibbles per int32)
  - .scales (F16, per group)
  - .qzeros (I32, packed 8 nibbles per int32)

No external GPTQ library needed — just PyTorch + safetensors.

Usage:
    python scripts/quantize_mixed_precision.py
    python scripts/quantize_mixed_precision.py --model ./models/qwen3.5-9b --bits 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def quantize_rtn_int4(weight: torch.Tensor, group_size: int = 128):
    """
    Round-to-nearest INT4 quantization with group-wise scaling.

    Input:  weight [N, K] (out_features, in_features) as float/bf16
    Output: qweight [K//8, N] as int32 (packed nibbles)
            scales  [K//group_size, N] as float16
            qzeros  [K//group_size, N//8] as int32 (packed nibbles)
    """
    N, K = weight.shape
    w = weight.float()

    num_groups = K // group_size
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"

    # Reshape to [N, num_groups, group_size]
    w_grouped = w.reshape(N, num_groups, group_size)

    # Per-group min/max
    w_min = w_grouped.min(dim=2).values  # [N, num_groups]
    w_max = w_grouped.max(dim=2).values  # [N, num_groups]

    # Symmetric quantization around zero (zero_point = 8 for signed-ish 4-bit)
    # Use asymmetric for better range: scale = (max - min) / 15
    scales = (w_max - w_min) / 15.0  # [N, num_groups]
    scales = scales.clamp(min=1e-10)  # avoid division by zero
    zeros = torch.round(-w_min / scales).clamp(0, 15).to(torch.int32)  # [N, num_groups]

    # Quantize: q = round(w / scale + zero), clamp to [0, 15]
    scales_expanded = scales.unsqueeze(2).expand_as(w_grouped)  # [N, num_groups, group_size]
    zeros_expanded = zeros.unsqueeze(2).expand_as(w_grouped).float()

    q = torch.round(w_grouped / scales_expanded + zeros_expanded)
    q = q.clamp(0, 15).to(torch.int32)  # [N, num_groups, group_size]
    q = q.reshape(N, K)  # [N, K]

    # Pack qweight: [K//8, N] — 8 nibbles per int32, column-major packing
    # The WebGPU matmul_bt_q4 shader reads: qweight[k//8 * N + n], nibble at (k%8)*4
    qweight = torch.zeros(K // 8, N, dtype=torch.int32)
    for nibble in range(8):
        k_indices = torch.arange(nibble, K, 8)
        qweight[k_indices // 8] |= (q[:, k_indices].T << (nibble * 4))

    # Scales: [num_groups, N] as float16 (transposed from [N, num_groups])
    scales_packed = scales.T.contiguous().to(torch.float16)  # [num_groups, N]

    # Pack qzeros: [num_groups, N//8] — 8 nibbles per int32
    qzeros = torch.zeros(num_groups, N // 8, dtype=torch.int32)
    zeros_t = zeros.T.contiguous()  # [num_groups, N]
    for nibble in range(8):
        n_indices = torch.arange(nibble, N, 8)
        qzeros[:, n_indices // 8] |= (zeros_t[:, n_indices] << (nibble * 4))

    return qweight, scales_packed, qzeros


def should_quantize(name: str, layer_types: list, num_layers: int) -> bool:
    """Determine if a weight should be quantized (True) or kept BF16 (False)."""
    # Never quantize embeddings, norms, biases. Skip MTP entirely (not needed for inference)
    skip_patterns = ['embed_tokens', 'norm', 'bias', 'A_log', 'dt_bias', 'conv1d']
    # MTP (multi-token prediction) layer — skip entirely
    if 'mtp.' in name:
        return False
    for pat in skip_patterns:
        if pat in name:
            return False

    # Never quantize linear_attn projections (SSM recurrence is sensitive)
    if 'linear_attn' in name and '.weight' in name:
        return False

    # Quantize: FFN projections + full attention projections + lm_head + mtp
    quant_patterns = ['gate_proj', 'up_proj', 'down_proj', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'lm_head']
    for pat in quant_patterns:
        if pat in name and '.weight' in name:
            return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Mixed-precision INT4 quantization for Qwen3.5")
    parser.add_argument("--model", default="./models/qwen3.5-9b", help="Model path or HF repo")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--group-size", type=int, default=128, help="Group size")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    if args.output is None:
        model_short = Path(args.model).name
        args.output = str(Path(__file__).parent.parent / "models" / f"{model_short}-mixed-GPTQ-Int{args.bits}")

    model_path = Path(args.model)

    print(f"{'='*60}")
    print(f"Mixed-Precision INT4 Quantization (RTN)")
    print(f"{'='*60}")
    print(f"Model:      {model_path}")
    print(f"Bits:       {args.bits}")
    print(f"Group size: {args.group_size}")
    print(f"Output:     {args.output}")
    print(f"{'='*60}")

    # Load config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Handle nested text_config
    if 'text_config' in config:
        text_config = config['text_config']
        config = {**config, **text_config}

    layer_types = config.get('layer_types', [])
    num_layers = len(layer_types)
    num_linear = sum(1 for t in layer_types if t == 'linear_attention')
    num_full = sum(1 for t in layer_types if t == 'full_attention')
    print(f"\nArchitecture: {num_layers} layers ({num_linear} linear + {num_full} full attention)")

    # Find all safetensors shards
    shards = sorted(model_path.glob("model*.safetensors"))
    print(f"Shards: {len(shards)}")

    # Process each shard
    os.makedirs(args.output, exist_ok=True)
    total_original = 0
    total_quantized = 0
    quantized_count = 0
    kept_count = 0
    t0 = time.time()

    for shard_idx, shard_path in enumerate(shards):
        print(f"\n[Shard {shard_idx+1}/{len(shards)}] {shard_path.name}")
        tensors = load_file(str(shard_path))

        output_tensors = {}

        for name, tensor in tensors.items():
            original_bytes = tensor.numel() * tensor.element_size()
            total_original += original_bytes

            # Skip MTP (multi-token prediction) layer entirely — not needed for inference
            if 'mtp.' in name:
                print(f"  SKIP MTP: {name}")
                continue

            if should_quantize(name, layer_types, num_layers):
                # Quantize to INT4
                base = name.replace('.weight', '')
                N, K = tensor.shape

                # Check dimensions are compatible
                if K % args.group_size != 0:
                    print(f"  SKIP {name} — K={K} not divisible by {args.group_size}")
                    output_tensors[name] = tensor
                    kept_count += 1
                    total_quantized += original_bytes
                    continue
                if N % 8 != 0:
                    print(f"  SKIP {name} — N={N} not divisible by 8")
                    output_tensors[name] = tensor
                    kept_count += 1
                    total_quantized += original_bytes
                    continue

                qweight, scales, qzeros = quantize_rtn_int4(tensor, args.group_size)

                output_tensors[f"{base}.qweight"] = qweight
                output_tensors[f"{base}.scales"] = scales
                output_tensors[f"{base}.qzeros"] = qzeros

                q_bytes = qweight.numel() * 4 + scales.numel() * 2 + qzeros.numel() * 4
                total_quantized += q_bytes
                ratio = original_bytes / q_bytes
                quantized_count += 1
                print(f"  Q4 {name}: [{N},{K}] {original_bytes/1024/1024:.1f}MB -> {q_bytes/1024/1024:.1f}MB ({ratio:.1f}x)")
            else:
                # Keep as-is (BF16/F32)
                output_tensors[name] = tensor
                total_quantized += original_bytes
                kept_count += 1

        # Save output shard
        out_path = os.path.join(args.output, shard_path.name)
        save_file(output_tensors, out_path)
        print(f"  Saved {out_path} ({len(output_tensors)} tensors)")

    # Copy config files
    for cfg_file in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'merges.txt', 'vocab.json', 'chat_template.jinja',
                     'preprocessor_config.json']:
        src = model_path / cfg_file
        if src.exists():
            import shutil
            shutil.copy2(src, os.path.join(args.output, cfg_file))

    # Write quantization config (so WebGPU engine knows it's GPTQ)
    quant_config = {
        "quantization_config": {
            "quant_method": "gptq",
            "bits": args.bits,
            "group_size": args.group_size,
            "checkpoint_format": "gptq_v2",
            "mixed_precision": True,
            "modules_not_quantized": [
                "linear_attn", "embed_tokens", "lm_head", "norm"
            ],
        }
    }
    # Merge into config
    with open(os.path.join(args.output, "config.json")) as f:
        out_config = json.load(f)
    out_config.update(quant_config)
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(out_config, f, indent=2)

    # Write safetensors index if multi-shard
    if len(shards) > 1:
        index_src = model_path / "model.safetensors.index.json"
        if index_src.exists():
            import shutil
            shutil.copy2(index_src, os.path.join(args.output, "model.safetensors.index.json"))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"  Original:   {total_original / 1024**3:.2f} GB")
    print(f"  Quantized:  {total_quantized / 1024**3:.2f} GB")
    print(f"  Compression: {total_original / total_quantized:.1f}x")
    print(f"  INT4:       {quantized_count} weights (FFN + full attention)")
    print(f"  BF16:       {kept_count} weights (linear_attn + norms + lm_head)")
    print(f"  Time:       {elapsed:.0f}s")
    print(f"  Output:     {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
