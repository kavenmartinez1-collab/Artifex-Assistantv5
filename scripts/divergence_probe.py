"""Python reference probe for Qwen3.5-9B — divergence diagnostics.

Loads the non-quantized base `qwen3.5-9b` and runs a fixed prompt, capturing
per-layer hidden states so they can be compared against the WebGPU engine's
dump (__DEBUG_DUMP_RESULT__).

Usage (from repo root, with activated venv or the explicit python path):
    ./venv/Scripts/python.exe scripts/divergence_probe.py \\
        --prompt "Hello Artifex?" --out reference_dump.json

Caveats:
- This loads the BASE qwen3.5-9b, NOT HailMary. HailMary is distilled +
  quantized from the base, so some deep-layer divergence is expected-by-
  design. Early-layer (L0-L5) divergence is still engine-side.
- Runs on CPU in bf16 by default — a 9B model is too large for 8 GB VRAM.
  One forward pass is slow (minutes) but we only do it once per probe.

The dump format matches the engine's __DEBUG_DUMP_RESULT__:
    { seqLen, pos, H, V, layers: [ { label, stats, sample: [last-row floats] } ] }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def tensor_stats(x: torch.Tensor) -> dict:
    """Match the engine's dumpBufStats() output schema."""
    x = x.detach().float().flatten()
    nan_mask = torch.isnan(x)
    inf_mask = torch.isinf(x) & ~nan_mask
    valid = x[~(nan_mask | inf_mask)]
    stats = {
        "size": int(x.numel()),
        "nanCount": int(nan_mask.sum().item()),
        "infCount": int(inf_mask.sum().item()),
    }
    if valid.numel() > 0:
        stats.update(
            min=float(valid.min()),
            max=float(valid.max()),
            mean=float(valid.mean()),
            absMean=float(valid.abs().mean()),
            std=float(valid.std(unbiased=False)),
        )
    else:
        stats.update(min=0.0, max=0.0, mean=0.0, absMean=0.0, std=0.0)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="models/qwen3.5-9b",
                    help="path to the reference model directory")
    ap.add_argument("--prompt", default="Hello Artifex?",
                    help="user prompt to forward (must match engine's test prompt)")
    ap.add_argument("--system", default="You are Artifex, a helpful AI assistant.",
                    help="system prompt (must match engine's test setup)")
    ap.add_argument("--out", default="reference_dump.json")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda — cpu is safer for 8 GB GPUs (9B bf16 = ~18 GB)")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--teacher-tokens",
                    help="JSON array of token IDs the engine generated; we concat them onto the "
                         "prompt and forward once so each requested decode-step has the same "
                         "left-context the engine had at that step (teacher forcing).")
    ap.add_argument("--dump-tags",
                    help="JSON object mapping tag -> 0-based decode step (0 = last prefill). "
                         "E.g. '{\"prefill-end\":0,\"decode-1\":1,\"decode-10\":10}'. If omitted, "
                         "we dump only at the last prompt token (single-tag back-compat).")
    ap.add_argument("--audit-layers",
                    help="JSON array of layer indices to deep-audit. For each listed layer, "
                         "we register hooks on every nn.Linear submodule and dump its output "
                         "with label 'L${i}-${dotted_path}-out'. Generic by design — works "
                         "for any model family; the comparison script maps engine labels to "
                         "these dotted paths via a per-family JSON name-map.")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[probe] ERROR: model path not found: {model_path}", file=sys.stderr)
        return 1

    print(f"[probe] loading tokenizer from {model_path}")
    tok = AutoTokenizer.from_pretrained(str(model_path))

    print(f"[probe] loading model ({args.dtype}) on {args.device} ...")
    print(f"        (9B on CPU takes a few minutes for one forward pass)")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=args.device,
        trust_remote_code=False,
    )
    model.eval()

    # Apply chat template to match the engine's autotest message framing.
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    try:
        ct = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        # Newer transformers returns BatchEncoding; older returns a tensor.
        if hasattr(ct, "input_ids"):
            input_ids = ct.input_ids.to(args.device)
        elif isinstance(ct, dict):
            input_ids = ct["input_ids"].to(args.device)
        else:
            input_ids = ct.to(args.device)
    except Exception as e:
        print(f"[probe] chat template failed ({e}) — falling back to raw prompt")
        text = f"{args.system}\n\n{args.prompt}"
        input_ids = tok(text, return_tensors="pt").input_ids.to(args.device)

    prompt_len = int(input_ids.shape[-1])

    # Teacher-forcing: append engine-generated tokens so a single forward pass
    # supplies the correct left-context for each decode step we want to dump.
    teacher_tokens: list[int] = []
    if args.teacher_tokens:
        try:
            teacher_tokens = json.loads(args.teacher_tokens)
            if not isinstance(teacher_tokens, list) or not all(isinstance(t, int) for t in teacher_tokens):
                raise ValueError("must be a JSON array of ints")
            if teacher_tokens:
                tt = torch.tensor([teacher_tokens], dtype=input_ids.dtype, device=input_ids.device)
                input_ids = torch.cat([input_ids, tt], dim=-1)
                print(f"[probe] teacher-forced with {len(teacher_tokens)} engine tokens "
                      f"(prompt={prompt_len}, total={input_ids.shape[-1]})")
        except Exception as e:
            print(f"[probe] WARNING: --teacher-tokens parse failed ({e}); ignoring")
            teacher_tokens = []

    # Resolve tag -> absolute position (in the concatenated sequence). Position
    # is 0-based step offset from end of prompt: 0 = last prompt token (prefill
    # end), N = the N-th decoded token's input position.
    if args.dump_tags:
        try:
            tag_step_map: dict[str, int] = json.loads(args.dump_tags)
        except Exception as e:
            print(f"[probe] WARNING: --dump-tags parse failed ({e}); using single dump")
            tag_step_map = {"prefill-end": 0}
    else:
        tag_step_map = {"prefill-end": 0}

    total_len = int(input_ids.shape[-1])
    tag_pos_map: dict[str, int] = {}
    for tag, step in tag_step_map.items():
        # Hidden state at decode step N corresponds to the prediction made AFTER
        # consuming (prompt + step) tokens — i.e. the row at index (prompt_len-1+step).
        abs_pos = (prompt_len - 1) + int(step)
        if abs_pos < 0 or abs_pos >= total_len:
            print(f"[probe] WARNING: tag '{tag}' step={step} -> pos {abs_pos} out of range "
                  f"[0, {total_len - 1}]; skipping (need more teacher tokens)")
            continue
        tag_pos_map[tag] = abs_pos
    if not tag_pos_map:
        print("[probe] ERROR: no valid dump positions resolved", file=sys.stderr)
        return 1
    print(f"[probe] dump positions: {tag_pos_map}")

    # Locate decoder layers + embed + final-norm. The wrapper module exposes
    # `model.model` (Qwen3_5TextModel); the text model has .layers / .embed_tokens / .norm.
    text_model = model.model if hasattr(model, "model") else model
    # Some wrappers have double-nesting: model.model.model.layers — walk down until we find .layers.
    while not hasattr(text_model, "layers") and hasattr(text_model, "model"):
        text_model = text_model.model
    if not hasattr(text_model, "layers"):
        print("[probe] ERROR: could not find decoder .layers on model", file=sys.stderr)
        return 1

    layers = text_model.layers
    embed = text_model.embed_tokens
    finalnorm = text_model.norm
    H = embed.weight.shape[1]
    V = embed.weight.shape[0]
    print(f"[probe] found {len(layers)} decoder layers, H={H}, V={V}")

    # Per-tag accumulators: tags -> list of {label, stats, sample}
    per_tag: dict[str, list[dict]] = {tag: [] for tag in tag_pos_map}
    hooks = []

    def make_hook(label: str):
        def fn(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h is None or not isinstance(h, torch.Tensor):
                return
            # Whole-tensor stats once (all positions seen by this layer).
            full_stats = tensor_stats(h)
            # Per-tag: slice out the row at the absolute position for that tag.
            # Handle both [B, S, F] (transformer modules) and [B*S, F] / [S, F]
            # shapes (some Linear layers may flatten before projection).
            for tag, abs_pos in tag_pos_map.items():
                if h.dim() >= 3:
                    if abs_pos >= h.shape[1]:
                        continue
                    row = h[0, abs_pos, :].detach()
                elif h.dim() == 2:
                    if abs_pos >= h.shape[0]:
                        continue
                    row = h[abs_pos, :].detach()
                else:
                    continue
                per_tag[tag].append({
                    "label": label,
                    "stats": full_stats,
                    "sample": row.float().cpu().numpy().tolist(),
                })
        return fn

    hooks.append(embed.register_forward_hook(make_hook("embed-out")))
    for i, L in enumerate(layers):
        hooks.append(L.register_forward_hook(make_hook(f"layer-{i}-out")))
    hooks.append(finalnorm.register_forward_hook(make_hook("final-norm-out")))

    # Per-tensor audit: walk each requested layer and hook every nn.Linear
    # submodule. Label format: "L${i}-${dotted_path}-out" so the comparison
    # script can resolve engine projection labels (e.g., "L21-lin-qkv-out")
    # via a per-family map (compare_label_map.qwen35.json, .gemma4.json ...).
    audit_layer_indices: list[int] = []
    if args.audit_layers:
        try:
            audit_layer_indices = json.loads(args.audit_layers)
            if not isinstance(audit_layer_indices, list) or not all(
                isinstance(x, int) for x in audit_layer_indices
            ):
                raise ValueError("must be a JSON array of ints")
        except Exception as e:
            print(f"[probe] WARNING: --audit-layers parse failed ({e}); ignoring")
            audit_layer_indices = []
    if audit_layer_indices:
        audit_count = 0
        for li in audit_layer_indices:
            if li < 0 or li >= len(layers):
                print(f"[probe] WARNING: audit layer {li} out of range [0, {len(layers) - 1}]")
                continue
            L = layers[li]
            for path, sub in L.named_modules():
                if isinstance(sub, torch.nn.Linear):
                    label = f"L{li}-{path}-out" if path else f"L{li}-self-out"
                    hooks.append(sub.register_forward_hook(make_hook(label)))
                    audit_count += 1
        print(f"[probe] audit: registered {audit_count} nn.Linear hooks "
              f"across layers {audit_layer_indices}")

    print(f"[probe] forward pass on {input_ids.shape[-1]} tokens ...")
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)

    # Logits per tag (full V-vector at the relevant position).
    for tag, abs_pos in tag_pos_map.items():
        if abs_pos >= out.logits.shape[1]:
            continue
        log_row = out.logits[0, abs_pos, :].detach()
        per_tag[tag].append({
            "label": "logits",
            "stats": tensor_stats(log_row),
            "sample": log_row.float().cpu().numpy().tolist(),
        })

    for h in hooks:
        h.remove()

    # Assemble multi-tag result. Mirror the engine's __DEBUG_DUMP_RESULT__ shape:
    # { tag: { tag, seqLen, pos, H, V, layers: [...] }, ... }
    tags_out: dict[str, dict] = {}
    for tag, layers_list in per_tag.items():
        if not layers_list:
            continue
        tags_out[tag] = {
            "tag": tag,
            "seqLen": int(input_ids.shape[-1]),
            "pos": tag_pos_map[tag],
            "H": int(H),
            "V": int(V),
            "layers": layers_list,
        }

    result = {
        "model": str(model_path),
        "prompt": args.prompt,
        "promptLen": prompt_len,
        "teacherTokens": teacher_tokens,
        "tags": tags_out,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result))
    size_mb = out_path.stat().st_size / 1e6
    print(f"[probe] wrote {out_path} ({size_mb:.2f} MB, tags={list(tags_out.keys())})")

    # Quick sanity print per tag.
    for tag, t_dump in tags_out.items():
        layers_list = t_dump["layers"]
        if not layers_list:
            continue
        embed_entry = next((e for e in layers_list if e["label"] == "embed-out"), None)
        last_layer_entry = next(
            (e for e in reversed(layers_list) if e["label"].startswith("layer-")), None
        )
        logits_entry = next((e for e in layers_list if e["label"] == "logits"), None)
        em = embed_entry["stats"]["absMean"] if embed_entry else float("nan")
        ll = last_layer_entry["stats"]["absMean"] if last_layer_entry else float("nan")
        am = (int(torch.tensor(logits_entry["sample"]).argmax().item())
              if logits_entry else -1)
        print(f"[probe] [{tag}] embed_abs={em:.4f} last_layer_abs={ll:.4f} argmax_id={am}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
