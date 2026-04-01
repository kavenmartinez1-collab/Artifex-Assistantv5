#!/usr/bin/env python3
"""
Quantization Quality Diagnostic — Finds WHERE coherence breaks.

Runs GPTQ quantization layer-by-layer, measuring hidden state divergence
from BF16 reference after each layer. Pinpoints exactly which layer/sublayer
causes the quality cliff.

Two modes:
  - original: Quantize in original basis (baseline)
  - klt:      Apply KLT rotation first, then quantize (identifies rotation impact)

Output: Per-layer table of GPTQ loss, cosine sim, MSE, max diff, plus
        per-sublayer detail for layers where divergence spikes.

Usage:
  python scripts/diagnose_quant_quality.py --model ./models/qwen3.5-9b --mode original
  python scripts/diagnose_quant_quality.py --model ./models/qwen3.5-9b --mode klt
  python scripts/diagnose_quant_quality.py --model ./models/qwen3.5-9b --mode both
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
import torch.nn.functional as F

# Disable TF32 for precise Hessian computation
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def log(msg: str):
    print(msg, flush=True)


def compute_divergence(ref: torch.Tensor, actual: torch.Tensor):
    """Compute divergence metrics between reference and actual hidden states."""
    ref_f = ref.float()
    act_f = actual.float()

    # Cosine similarity (averaged over all token positions)
    cos = F.cosine_similarity(
        ref_f.reshape(-1, ref_f.shape[-1]),
        act_f.reshape(-1, act_f.shape[-1]),
        dim=1,
    ).mean().item()

    # MSE
    mse = ((ref_f - act_f) ** 2).mean().item()

    # Max absolute difference
    max_diff = (ref_f - act_f).abs().max().item()

    # Relative L2 error
    ref_norm = ref_f.norm().item()
    rel_l2 = (ref_f - act_f).norm().item() / (ref_norm + 1e-10)

    # Signal-to-noise ratio (dB)
    snr = 10 * math.log10(ref_norm ** 2 / (mse * ref_f.numel() + 1e-20))

    return {
        'cos_sim': cos,
        'mse': mse,
        'max_diff': max_diff,
        'rel_l2': rel_l2,
        'snr_db': snr,
    }


def run_diagnostic(model_path: str, mode: str, device: str = "cuda",
                   num_cal_samples: int = 32, seq_length: int = 512,
                   bits: int = 4, group_size: int = 128,
                   keep_bf16_patterns=None):
    """Run quantization diagnostic in specified mode."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from scripts.quantize_gptq import (
        compute_global_klt, find_inner_model, find_layer_prefix,
        absorb_norm_scales, fuse_klt_into_model,
        HessianCollector, GPTQQuantizer, group_sublayers,
        should_quantize, load_calibration_data,
        QUANT_PATTERNS, SKIP_PATTERNS,
    )

    if keep_bf16_patterns is None:
        keep_bf16_patterns = ["norm", "embed_tokens", "lm_head"]

    gpu = torch.device(device)
    use_gpu = device != "cpu" and torch.cuda.is_available()

    # ---- Load calibration data ----
    log(f"Loading calibration data ({num_cal_samples} samples, {seq_length} tokens)...")
    calibration_samples = load_calibration_data(
        dataset_name="wikitext",
        num_samples=num_cal_samples,
        seq_length=seq_length,
        tokenizer_path=model_path,
    )
    log(f"  Loaded {len(calibration_samples)} samples")

    # Use a subset for divergence measurement (speed)
    eval_samples = calibration_samples[:4]

    # ---- Load model ----
    log(f"\nLoading model ({mode} mode)...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.eval()

    hf_config = config.to_dict()
    if hasattr(config, 'text_config'):
        hf_config.update(config.text_config.to_dict() if hasattr(config.text_config, 'to_dict') else {})

    layer_types = hf_config.get('layer_types', [])
    num_layers = hf_config.get('num_hidden_layers', 32)
    hidden_size = hf_config.get('hidden_size', 3584)

    inner_model = find_inner_model(model)
    layer_prefix = find_layer_prefix(model)
    layers = inner_model.layers

    log(f"  Layers: {num_layers}, Hidden: {hidden_size}")

    # ---- Apply KLT if requested ----
    klt_applied = False
    if mode == 'klt':
        log(f"\nApplying KLT rotation...")
        absorb_norm_scales(model, gpu)
        R = compute_global_klt(inner_model, calibration_samples, gpu,
                               num_samples=min(32, len(calibration_samples)))
        fuse_klt_into_model(model, R, gpu)
        klt_applied = True
        log(f"  KLT fusion complete")

        # Log eigenvalue spread (tells us about channel variance uniformity)
        # R was computed from embedding covariance eigendecomposition
        inner_model.embed_tokens.to(gpu)
        with torch.no_grad():
            embed_out = inner_model.embed_tokens(calibration_samples[0].to(gpu))
        inner_model.embed_tokens.to('cpu')
        X = embed_out.reshape(-1, embed_out.shape[-1]).float()
        C = (X.T @ X) / X.shape[0]
        eigenvalues = torch.linalg.eigvalsh(C.to(gpu))
        log(f"  Eigenvalue range: {eigenvalues.min().item():.4f} - {eigenvalues.max().item():.4f}")
        log(f"  Eigenvalue ratio (max/min): {(eigenvalues.max() / (eigenvalues.min() + 1e-10)).item():.1f}x")
        log(f"  Top 10 eigenvalues: {eigenvalues[-10:].cpu().tolist()}")
        log(f"  Bottom 10 eigenvalues: {eigenvalues[:10].cpu().tolist()}")
        del X, C, eigenvalues, embed_out
        if use_gpu:
            torch.cuda.empty_cache()

    # ---- Compute initial hidden states ----
    log(f"\nComputing initial hidden states...")
    layer_inputs = []
    inner_model.embed_tokens.to(gpu)
    with torch.no_grad():
        for sample in calibration_samples:
            hidden = inner_model.embed_tokens(sample.to(gpu))
            layer_inputs.append(hidden.to(torch.bfloat16).cpu())
    inner_model.embed_tokens.to('cpu')
    if use_gpu:
        torch.cuda.empty_cache()

    # Reference path: BF16 hidden states for divergence comparison
    ref_inputs = [layer_inputs[i].clone() for i in range(len(eval_samples))]

    # Rotary position embeddings
    pos_embeds = None
    rotary_emb = getattr(inner_model, 'rotary_emb', None)
    if rotary_emb is not None:
        rotary_emb.to(gpu)
        position_ids = torch.arange(seq_length, device=gpu).unsqueeze(0)
        with torch.no_grad():
            pos_embeds = rotary_emb(layer_inputs[0].to(gpu), position_ids)
        pos_embeds = tuple(p.cpu() for p in pos_embeds)
        rotary_emb.to('cpu')
        if use_gpu:
            torch.cuda.empty_cache()

    def layer_kwargs():
        kw = {}
        if pos_embeds is not None:
            kw['position_embeddings'] = tuple(p.to(gpu) for p in pos_embeds)
        return kw

    # ---- GPTQ quantizer config ----
    gptq = GPTQQuantizer(bits=bits, group_size=group_size, percdamp=0.01,
                         blocksize=128, actorder=True, percclip=0.001)

    # ---- Layer-by-layer diagnostic ----
    log(f"\n{'='*100}")
    log(f"PROGRESSIVE QUANTIZATION DIAGNOSTIC ({mode.upper()} mode)")
    log(f"{'='*100}")
    log(f"{'Layer':>6} {'Type':>4} {'Sublayer':<30} {'GPTQLoss':>10} "
        f"{'CosSim':>10} {'MSE':>12} {'MaxDiff':>10} {'RelL2':>10} {'SNR(dB)':>8}")
    log(f"{'-'*100}")

    results = []

    for li in range(len(layers)):
        layer = layers[li]
        ltype = layer_types[li] if li < len(layer_types) else 'unk'
        ltype_short = 'SSM' if ltype == 'linear_attention' else 'ATN'

        layer.to(gpu)

        # Step 1: BF16 reference forward (before quantization)
        ref_outputs = []
        kw = layer_kwargs()
        with torch.no_grad():
            for i in range(len(eval_samples)):
                out = layer(ref_inputs[i].to(gpu), **kw)
                if isinstance(out, tuple):
                    out = out[0]
                ref_outputs.append(out.to(torch.bfloat16).cpu())

        # Step 2: Find and quantize sublayers
        sublayers = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{li}.{name}.weight"
                if should_quantize(full_name, keep_bf16_patterns):
                    sublayers[name] = module

        sublayer_losses = {}

        if sublayers:
            groups = group_sublayers(list(sublayers.keys()))

            for group_names in groups:
                group_modules = {n: sublayers[n] for n in group_names if n in sublayers}
                if not group_modules:
                    continue

                # Collect Hessians
                collectors = {}
                hooks = []
                for name, module in group_modules.items():
                    col = HessianCollector(module, gpu)
                    collectors[name] = col
                    h = module.register_forward_hook(col.hook)
                    hooks.append(h)

                with torch.no_grad():
                    for inp in layer_inputs:
                        try:
                            layer(inp.to(gpu), **kw)
                        except Exception:
                            continue

                for h in hooks:
                    h.remove()

                # Quantize each sublayer
                for name, module in group_modules.items():
                    W = module.weight.data.float().to(gpu)
                    N, K = W.shape
                    H = collectors[name].get_hessian()
                    collectors[name].free()

                    if K % 8 != 0 or N % 8 != 0:
                        sublayer_losses[name] = ('SKIP', 0, 0, 0)
                        del W, H
                        continue

                    # Hessian diagnostic: log diagonal statistics
                    h_diag = torch.diag(H)
                    h_diag_nz = h_diag[h_diag > 0]

                    # GPTQ quantize
                    gs = group_size
                    if K % gs != 0:
                        for alt in [64, 32, K]:
                            if K % alt == 0:
                                gs = alt
                                break

                    Q_int, scales, zeros, g_idx, Q_deq, loss = gptq.quantize_weight(W, H)
                    mse = ((W - Q_deq) ** 2).mean().item()
                    rel = mse / ((W ** 2).mean().item() + 1e-10)
                    w_norm = W.norm().item()

                    sublayer_losses[name] = {
                        'loss': loss,
                        'mse': mse,
                        'rel_mse': rel,
                        'w_norm': w_norm,
                        'shape': (N, K),
                        'h_diag_min': h_diag_nz.min().item() if len(h_diag_nz) > 0 else 0,
                        'h_diag_max': h_diag_nz.max().item() if len(h_diag_nz) > 0 else 0,
                        'h_diag_ratio': (h_diag_nz.max() / (h_diag_nz.min() + 1e-10)).item() if len(h_diag_nz) > 0 else 0,
                        'h_diag_std': h_diag_nz.std().item() if len(h_diag_nz) > 1 else 0,
                        'dead_cols': (h_diag == 0).sum().item(),
                    }

                    # Replace weight with dequantized
                    module.weight.data = Q_deq.to(module.weight.dtype)

                    del W, H, Q_int, scales, zeros, g_idx, Q_deq

                if use_gpu:
                    torch.cuda.empty_cache()

        # Step 3: Forward eval samples through QUANTIZED layer
        quant_outputs = []
        with torch.no_grad():
            for i in range(len(eval_samples)):
                out = layer(ref_inputs[i].to(gpu), **kw)
                if isinstance(out, tuple):
                    out = out[0]
                quant_outputs.append(out.to(torch.bfloat16).cpu())

        # Step 4: Compute divergence (BF16 reference vs quantized)
        all_ref = torch.cat([r.float() for r in ref_outputs], dim=1)
        all_quant = torch.cat([q.float() for q in quant_outputs], dim=1)
        div = compute_divergence(all_ref, all_quant)
        del all_ref, all_quant

        # Step 5: Propagate through quantized layer for next iteration
        # (using ALL calibration samples, not just eval)
        layer_outputs = []
        with torch.no_grad():
            for inp in layer_inputs:
                try:
                    out = layer(inp.to(gpu), **kw)
                    if isinstance(out, tuple):
                        out = out[0]
                    layer_outputs.append(out.to(torch.bfloat16).cpu())
                except Exception:
                    layer_outputs.append(inp)

        # Update reference path: feed BF16 reference through BF16 layer
        # We need to keep the ref path using BF16 layer outputs (computed in step 1)
        ref_inputs = ref_outputs

        layer.to('cpu')
        gc.collect()
        if use_gpu:
            torch.cuda.empty_cache()

        layer_inputs = layer_outputs

        # Log per-sublayer detail
        total_layer_loss = 0
        for name, info in sublayer_losses.items():
            if isinstance(info, tuple):
                log(f"  L{li:2d} {ltype_short:>4} {name:<30} {'SKIP':>10} "
                    f"{div['cos_sim']:>10.6f} {div['mse']:>12.2e} "
                    f"{div['max_diff']:>10.4f} {div['rel_l2']:>10.6f} "
                    f"{div['snr_db']:>8.1f}")
            else:
                total_layer_loss += info['loss']
                log(f"  L{li:2d} {ltype_short:>4} {name:<30} {info['mse']:>10.2e} "
                    f"{div['cos_sim']:>10.6f} {div['mse']:>12.2e} "
                    f"{div['max_diff']:>10.4f} {div['rel_l2']:>10.6f} "
                    f"{div['snr_db']:>8.1f}"
                    f"  H_ratio={info['h_diag_ratio']:.0f}x "
                    f"dead={info['dead_cols']}")

        # Summary line for the layer
        layer_result = {
            'layer': li,
            'type': ltype_short,
            'sublayer_losses': {k: v for k, v in sublayer_losses.items() if isinstance(v, dict)},
            'divergence': div,
        }
        results.append(layer_result)

        # Alert on concerning divergence
        if div['cos_sim'] < 0.999:
            log(f"  >>> WARNING: Layer {li} cos_sim dropped to {div['cos_sim']:.6f}")
        if div['cos_sim'] < 0.99:
            log(f"  >>> CRITICAL: Layer {li} divergence is severe!")

    # ---- Final CE loss ----
    log(f"\n{'='*100}")
    log(f"COMPUTING FINAL CROSS-ENTROPY LOSS")
    log(f"{'='*100}")

    # Forward through final norm + lm_head
    final_norm = getattr(inner_model, 'norm', None)
    lm_head = getattr(model, 'lm_head', None)

    if final_norm is not None and lm_head is not None:
        final_norm.to(gpu)
        lm_head.to(gpu)

        total_ce = 0
        total_tokens = 0

        with torch.no_grad():
            for i, sample in enumerate(eval_samples):
                h = layer_inputs[i].to(gpu)  # These are post-quantization hidden states
                h = final_norm(h)
                logits = lm_head(h)

                # CE loss: predict next token
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = sample[:, 1:sample.shape[1]].to(gpu)  # Truncate to match
                if shift_labels.shape[1] > shift_logits.shape[1]:
                    shift_labels = shift_labels[:, :shift_logits.shape[1]]

                ce = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    reduction='sum',
                )
                total_ce += ce.item()
                total_tokens += shift_labels.numel()

        avg_ce = total_ce / total_tokens
        ppl = math.exp(min(avg_ce, 100))  # cap to avoid overflow

        final_norm.to('cpu')
        lm_head.to('cpu')

        log(f"  Cross-entropy: {avg_ce:.4f}")
        log(f"  Perplexity:    {ppl:.2f}")
    else:
        avg_ce = -1
        ppl = -1
        log(f"  Could not compute CE (missing norm/lm_head)")

    # ---- Summary Report ----
    log(f"\n{'='*100}")
    log(f"SUMMARY ({mode.upper()})")
    log(f"{'='*100}")

    log(f"\n{'Layer':>6} {'Type':>4} {'CosSim':>10} {'RelL2':>10} "
        f"{'SNR(dB)':>8} {'MaxDiff':>10} {'TotalQLoss':>12} {'Status':>8}")
    log(f"{'-'*75}")

    worst_layer = None
    worst_drop = 0

    for i, r in enumerate(results):
        d = r['divergence']
        total_qloss = sum(v['loss'] for v in r['sublayer_losses'].values())

        status = 'OK'
        if d['cos_sim'] < 0.999:
            status = 'WARN'
        if d['cos_sim'] < 0.99:
            status = 'FAIL'

        log(f"  L{r['layer']:2d}  {r['type']:>4} {d['cos_sim']:>10.6f} "
            f"{d['rel_l2']:>10.6f} {d['snr_db']:>8.1f} "
            f"{d['max_diff']:>10.4f} {total_qloss:>12.4f} {status:>8}")

        # Track worst per-layer degradation
        if i > 0:
            prev_cos = results[i-1]['divergence']['cos_sim']
            drop = prev_cos - d['cos_sim']
            if drop > worst_drop:
                worst_drop = drop
                worst_layer = r['layer']

    log(f"\nFinal CE: {avg_ce:.4f}, PPL: {ppl:.2f}")
    if worst_layer is not None:
        log(f"Worst single-layer cos_sim drop: Layer {worst_layer} "
            f"(dropped {worst_drop:.6f})")

    # Identify SSM vs attention layer breakdown
    ssm_losses = []
    attn_losses = []
    for r in results:
        total = sum(v['loss'] for v in r['sublayer_losses'].values())
        if r['type'] == 'SSM':
            ssm_losses.append(total)
        else:
            attn_losses.append(total)

    if ssm_losses:
        log(f"\nSSM layers  ({len(ssm_losses)}): avg GPTQ loss = {sum(ssm_losses)/len(ssm_losses):.4f}")
    if attn_losses:
        log(f"ATN layers  ({len(attn_losses)}): avg GPTQ loss = {sum(attn_losses)/len(attn_losses):.4f}")

    # Identify worst sublayer types across all layers
    sublayer_agg = {}
    for r in results:
        for name, info in r['sublayer_losses'].items():
            proj = name.split('.')[-1]
            if proj not in sublayer_agg:
                sublayer_agg[proj] = {'mse_sum': 0, 'count': 0, 'h_ratio_max': 0}
            sublayer_agg[proj]['mse_sum'] += info['mse']
            sublayer_agg[proj]['count'] += 1
            sublayer_agg[proj]['h_ratio_max'] = max(
                sublayer_agg[proj]['h_ratio_max'], info['h_diag_ratio'])

    log(f"\n{'Projection':<25} {'AvgMSE':>12} {'Count':>6} {'MaxH_ratio':>12}")
    log(f"{'-'*60}")
    for proj, agg in sorted(sublayer_agg.items(),
                             key=lambda x: x[1]['mse_sum']/x[1]['count'],
                             reverse=True):
        avg_mse = agg['mse_sum'] / agg['count']
        log(f"  {proj:<23} {avg_mse:>12.2e} {agg['count']:>6} "
            f"{agg['h_ratio_max']:>12.0f}x")

    # Save results to JSON
    report = {
        'mode': mode,
        'klt_applied': klt_applied,
        'bits': bits,
        'group_size': group_size,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'final_ce': avg_ce,
        'final_ppl': ppl,
        'layers': [],
    }
    for r in results:
        layer_data = {
            'layer': r['layer'],
            'type': r['type'],
            'divergence': r['divergence'],
            'sublayers': {},
        }
        for name, info in r['sublayer_losses'].items():
            layer_data['sublayers'][name] = {
                'mse': info['mse'],
                'rel_mse': info['rel_mse'],
                'loss': info['loss'],
                'shape': list(info['shape']),
                'h_diag_ratio': info['h_diag_ratio'],
                'dead_cols': info['dead_cols'],
            }
        report['layers'].append(layer_data)

    report_path = f"quant-diagnostic-{mode}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    log(f"\nFull report saved to {report_path}")

    del model
    gc.collect()
    if use_gpu:
        torch.cuda.empty_cache()

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantization Quality Diagnostic")
    parser.add_argument("--model", required=True, help="Path to base BF16 model")
    parser.add_argument("--mode", choices=["original", "klt", "both"],
                        default="both",
                        help="Diagnostic mode: original, klt, or both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=32,
                        help="Calibration samples (default: 32)")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length (default: 512, shorter = faster)")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    if args.mode == "both":
        log("Running ORIGINAL basis diagnostic...")
        log(f"{'='*100}\n")
        report_orig = run_diagnostic(
            args.model, "original", args.device,
            args.num_samples, args.seq_length, args.bits, args.group_size)

        log(f"\n\n{'#'*100}")
        log(f"{'#'*100}\n")
        log("Running KLT basis diagnostic...")
        log(f"{'='*100}\n")
        report_klt = run_diagnostic(
            args.model, "klt", args.device,
            args.num_samples, args.seq_length, args.bits, args.group_size)

        # Side-by-side comparison
        log(f"\n\n{'='*100}")
        log(f"SIDE-BY-SIDE COMPARISON: Original vs KLT")
        log(f"{'='*100}")
        log(f"\n{'Layer':>6} {'Type':>4} {'Orig CosSim':>12} {'KLT CosSim':>12} "
            f"{'Orig RelL2':>12} {'KLT RelL2':>12} {'Winner':>8}")
        log(f"{'-'*75}")

        for orig, klt in zip(report_orig['layers'], report_klt['layers']):
            o_cos = orig['divergence']['cos_sim']
            k_cos = klt['divergence']['cos_sim']
            o_rel = orig['divergence']['rel_l2']
            k_rel = klt['divergence']['rel_l2']
            winner = 'ORIG' if o_cos > k_cos else 'KLT' if k_cos > o_cos else 'TIE'
            log(f"  L{orig['layer']:2d}  {orig['type']:>4} "
                f"{o_cos:>12.6f} {k_cos:>12.6f} "
                f"{o_rel:>12.6f} {k_rel:>12.6f} {winner:>8}")

        log(f"\nOriginal: CE={report_orig['final_ce']:.4f}, PPL={report_orig['final_ppl']:.2f}")
        log(f"KLT:      CE={report_klt['final_ce']:.4f}, PPL={report_klt['final_ppl']:.2f}")

    else:
        run_diagnostic(args.model, args.mode, args.device,
                       args.num_samples, args.seq_length,
                       args.bits, args.group_size)
