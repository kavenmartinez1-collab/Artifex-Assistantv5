"""Compare GPTQ dequantization: our formula vs AutoGPTQ's actual output.

Loads a GPTQ model's L0 QKV weights, dequantizes them using both
our formula and AutoGPTQ's, and reports the difference.
"""
import torch
import struct
import numpy as np
from safetensors import safe_open

MODEL = "RafaDom/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GPTQ-Int4-HQ"

print(f"Loading GPTQ weights from {MODEL}...")

# Try loading from HuggingFace cache
from huggingface_hub import hf_hub_download
path = hf_hub_download(MODEL, "model.safetensors")

with safe_open(path, framework="pt") as f:
    keys = list(f.keys())
    # Find L0 linear attention QKV projection
    qkv_keys = [k for k in keys if "layers.0." in k and "in_proj_qkv" in k]
    print(f"L0 QKV keys: {qkv_keys}")

    qweight = f.get_tensor([k for k in qkv_keys if "qweight" in k][0])
    scales = f.get_tensor([k for k in qkv_keys if "scales" in k][0])
    qzeros = f.get_tensor([k for k in qkv_keys if "qzeros" in k][0])

print(f"qweight: {qweight.shape} {qweight.dtype}")
print(f"scales:  {scales.shape} {scales.dtype}")
print(f"qzeros:  {qzeros.shape} {qzeros.dtype}")

K_packed, N = qweight.shape  # [K/8, N]
K = K_packed * 8
group_size = K // scales.shape[0]
print(f"K={K}, N={N}, group_size={group_size}, num_groups={scales.shape[0]}")

# Our dequant formula (no +1)
def our_dequant(qw, sc, qz, k_idx, n_idx, gs):
    packed = qw[k_idx // 8, n_idx].item()
    nibble = k_idx % 8
    q4_val = (packed >> (nibble * 4)) & 0xF

    group = k_idx // gs
    scale = sc[group, n_idx].item()

    zero_packed = qz[group, n_idx // 8].item()
    zero_nibble = n_idx % 8
    zero = (zero_packed >> (zero_nibble * 4)) & 0xF

    return (q4_val - zero) * scale

# AutoGPTQ dequant (+1 on zero)
def autogptq_dequant(qw, sc, qz, k_idx, n_idx, gs):
    packed = qw[k_idx // 8, n_idx].item()
    nibble = k_idx % 8
    q4_val = (packed >> (nibble * 4)) & 0xF

    group = k_idx // gs
    scale = sc[group, n_idx].item()

    zero_packed = qz[group, n_idx // 8].item()
    zero_nibble = n_idx % 8
    zero = ((zero_packed >> (zero_nibble * 4)) & 0xF) + 1

    return (q4_val - zero) * scale

# Also try: PyTorch's own GPTQ dequant via transformers
try:
    from transformers import AutoModelForCausalLM
    print("\nLoading full GPTQ model via transformers to get reference dequant...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="cpu", torch_dtype=torch.float32)

    # Get the dequantized weight from the model
    layer0 = model.model.language_model.model.layers[0]
    if hasattr(layer0, 'linear_attn'):
        qkv_module = layer0.linear_attn.in_proj_qkv
    elif hasattr(layer0, 'self_attn'):
        qkv_module = layer0.self_attn.q_proj

    # Get the actual dequantized weight
    if hasattr(qkv_module, 'qweight'):
        # Use AutoGPTQ's dequant
        ref_weight = qkv_module.dequantize()
        print(f"Reference dequantized weight shape: {ref_weight.shape}")

        # Compare a few elements
        print("\n=== Comparison (k=0, n=0..7) ===")
        print(f"{'k,n':>8} | {'Our':>12} | {'AutoGPTQ+1':>12} | {'Reference':>12} | {'Our err':>10} | {'+1 err':>10}")
        print("-" * 85)
        for n in range(8):
            ours = our_dequant(qweight, scales, qzeros, 0, n, group_size)
            ag = autogptq_dequant(qweight, scales, qzeros, 0, n, group_size)
            ref = ref_weight[n, 0].item()  # [N, K] layout
            our_err = abs(ours - ref)
            ag_err = abs(ag - ref)
            print(f"  (0,{n})  | {ours:12.6f} | {ag:12.6f} | {ref:12.6f} | {our_err:10.6f} | {ag_err:10.6f}")

        # Compute overall error statistics
        print("\n=== Overall statistics (first 128 weights) ===")
        our_errors = []
        ag_errors = []
        for k in range(min(8, K)):
            for n in range(min(16, N)):
                ours = our_dequant(qweight, scales, qzeros, k, n, group_size)
                ag = autogptq_dequant(qweight, scales, qzeros, k, n, group_size)
                ref = ref_weight[n, k].item()
                our_errors.append(abs(ours - ref))
                ag_errors.append(abs(ag - ref))

        print(f"Our formula:    mean_err={np.mean(our_errors):.6f}, max_err={np.max(our_errors):.6f}")
        print(f"AutoGPTQ (+1):  mean_err={np.mean(ag_errors):.6f}, max_err={np.max(ag_errors):.6f}")

        if np.mean(our_errors) < np.mean(ag_errors):
            print("\n>>> Our formula (no +1) is CLOSER to reference <<<")
        else:
            print("\n>>> AutoGPTQ (+1) is CLOSER to reference <<<")
    else:
        print("Module doesn't have qweight — not GPTQ quantized?")

except Exception as e:
    print(f"\nCouldn't load via transformers: {e}")
    print("\nManual comparison only:")
    print(f"{'k,n':>8} | {'Our (no +1)':>12} | {'With +1':>12} | {'Diff':>10}")
    print("-" * 55)
    for n in range(8):
        ours = our_dequant(qweight, scales, qzeros, 0, n, group_size)
        ag = autogptq_dequant(qweight, scales, qzeros, 0, n, group_size)
        print(f"  (0,{n})  | {ours:12.6f} | {ag:12.6f} | {abs(ours-ag):10.6f}")
