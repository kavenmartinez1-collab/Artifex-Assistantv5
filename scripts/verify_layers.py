"""
verify_layers.py — Run the first token of "Hello Artifex?" through all 32 layers
of Qwen3.5-9B and capture intermediate hidden states for comparison against
WebGPU debug output.

Usage:
    python scripts/verify_layers.py

Loads model in float32 on CPU (no quantization).
"""

import sys
import os
import json
import torch
import torch.nn as nn
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3.5-9b")
MODEL_PATH = os.path.normpath(MODEL_PATH)

FULL_ATTN_INDICES = {3, 7, 11, 15, 19, 23, 27, 31}
NUM_PREVIEW = 8  # how many values to print per tensor

MESSAGES = [
    {"role": "system", "content": "You are Artifex, a helpful AI assistant."},
    {"role": "user", "content": "Hello Artifex?"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(t, n=NUM_PREVIEW):
    """Format first n values of a tensor for display."""
    flat = t.detach().float().flatten()
    vals = flat[:n].tolist()
    return "[" + ", ".join(f"{v: .8f}" for v in vals) + "]"


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def print_values(label, tensor, n=NUM_PREVIEW):
    """Print labelled tensor values."""
    print(f"  {label:.<50s} {fmt(tensor, n)}")


# ---------------------------------------------------------------------------
# Hook storage
# ---------------------------------------------------------------------------

class CaptureStore:
    """Stores captured intermediate values from hooks."""

    def __init__(self):
        self.layer_outputs = OrderedDict()         # layer_idx -> hidden_state after layer
        self.layer_inputs = OrderedDict()          # layer_idx -> hidden_state into layer
        self.linear_attn_details = OrderedDict()   # layer_idx -> dict of captures
        self.full_attn_details = OrderedDict()     # layer_idx -> dict of captures

    def clear(self):
        self.layer_outputs.clear()
        self.layer_inputs.clear()
        self.linear_attn_details.clear()
        self.full_attn_details.clear()


store = CaptureStore()


# ---------------------------------------------------------------------------
# Resolve model layers
# ---------------------------------------------------------------------------

def get_layers(model):
    """Find the decoder layers list regardless of model wrapper structure.

    Tries model.model.layers first (Qwen3_5ForCausalLM),
    then model.model.language_model.layers (Qwen3_5ForConditionalGeneration).
    """
    # Preferred path per user spec
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # Fallback for multimodal wrapper
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return lm.model.layers
    # Last resort: walk named_modules
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, nn.ModuleList):
            if len(mod) == 32:
                return mod
    raise RuntimeError(
        "Could not locate decoder layers. Model structure:\n"
        + "\n".join(f"  {n}" for n, _ in model.named_children())
    )


def get_embed_tokens(model):
    """Find the embedding layer."""
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "embed_tokens"):
            return m.embed_tokens
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "embed_tokens"):
                return lm.embed_tokens
            if hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
                return lm.model.embed_tokens
    for name, mod in model.named_modules():
        if "embed_tokens" in name and isinstance(mod, nn.Embedding):
            return mod
    raise RuntimeError("Could not locate embed_tokens layer.")


def get_final_norm(model):
    """Find the final RMS norm before the LM head."""
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "norm"):
            return m.norm
        if hasattr(m, "language_model"):
            lm = m.language_model
            if hasattr(lm, "norm"):
                return lm.norm
            if hasattr(lm, "model") and hasattr(lm.model, "norm"):
                return lm.model.norm
    for name, mod in model.named_modules():
        if name.endswith(".norm") and "layer" not in name:
            return mod
    return None


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------

def install_layer_hooks(layers):
    """Install forward hooks on each decoder layer to capture input/output."""
    handles = []

    for idx, layer in enumerate(layers):
        def make_hook(i):
            def hook(module, args, output):
                # Input hidden state is the first positional arg
                if isinstance(args, tuple) and len(args) > 0:
                    store.layer_inputs[i] = args[0].detach().clone()

                # Output can be a tuple; first element is hidden_state
                if isinstance(output, tuple):
                    store.layer_outputs[i] = output[0].detach().clone()
                else:
                    store.layer_outputs[i] = output.detach().clone()
            return hook

        h = layer.register_forward_hook(make_hook(idx))
        handles.append(h)

    return handles


def install_linear_attn_hooks(layers):
    """Install hooks inside linear attention sub-layers (Mamba-style SSM layers)."""
    handles = []

    for idx, layer in enumerate(layers):
        if idx in FULL_ATTN_INDICES:
            continue

        attn = None
        if hasattr(layer, "linear_attn"):
            attn = layer.linear_attn
        elif hasattr(layer, "self_attn"):
            # Some transformers versions name it self_attn even for linear
            attn = layer.self_attn
        else:
            print(f"  WARNING: Layer {idx} has no linear_attn or self_attn attribute")
            continue

        details = {}
        store.linear_attn_details[idx] = details

        # --- Hook: in_proj_qkv (QKV projection) ---
        if hasattr(attn, "in_proj_qkv"):
            def make_qkv_hook(i, d):
                def hook(module, args, output):
                    d["qkv_proj_output"] = output.detach().clone()
                return hook
            h = attn.in_proj_qkv.register_forward_hook(make_qkv_hook(idx, details))
            handles.append(h)

        # --- Hook: conv1d ---
        if hasattr(attn, "conv1d"):
            def make_conv_hook(i, d):
                def hook(module, args, output):
                    d["conv1d_output"] = output.detach().clone()
                return hook
            h = attn.conv1d.register_forward_hook(make_conv_hook(idx, details))
            handles.append(h)

        # --- Hook: in_proj_a (SSM "A" input projection — dt + B + C) ---
        if hasattr(attn, "in_proj_a"):
            def make_a_hook(i, d):
                def hook(module, args, output):
                    d["in_proj_a_output"] = output.detach().clone()
                return hook
            h = attn.in_proj_a.register_forward_hook(make_a_hook(idx, details))
            handles.append(h)

        # --- Hook: in_proj_b (down-project for SSM) ---
        if hasattr(attn, "in_proj_b"):
            def make_b_hook(i, d):
                def hook(module, args, output):
                    d["in_proj_b_output"] = output.detach().clone()
                return hook
            h = attn.in_proj_b.register_forward_hook(make_b_hook(idx, details))
            handles.append(h)

        # --- Hook: in_proj_z (gate) ---
        if hasattr(attn, "in_proj_z"):
            def make_z_hook(i, d):
                def hook(module, args, output):
                    d["in_proj_z_output"] = output.detach().clone()
                return hook
            h = attn.in_proj_z.register_forward_hook(make_z_hook(idx, details))
            handles.append(h)

        # --- Hook: norm (after SSM, before gate multiply) ---
        if hasattr(attn, "norm"):
            def make_norm_hook(i, d):
                def hook(module, args, output):
                    d["norm_output"] = output.detach().clone()
                return hook
            h = attn.norm.register_forward_hook(make_norm_hook(idx, details))
            handles.append(h)

        # --- Hook: out_proj ---
        if hasattr(attn, "out_proj"):
            def make_out_hook(i, d):
                def hook(module, args, output):
                    d["out_proj_output"] = output.detach().clone()
                return hook
            h = attn.out_proj.register_forward_hook(make_out_hook(idx, details))
            handles.append(h)

    return handles


def install_full_attn_hooks(layers):
    """Install hooks inside full (standard) attention sub-layers."""
    handles = []

    for idx, layer in enumerate(layers):
        if idx not in FULL_ATTN_INDICES:
            continue

        attn = None
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
        else:
            print(f"  WARNING: Layer {idx} has no self_attn attribute")
            continue

        details = {}
        store.full_attn_details[idx] = details

        # --- Q projection ---
        if hasattr(attn, "q_proj"):
            def make_q_hook(i, d):
                def hook(module, args, output):
                    d["q_proj_output"] = output.detach().clone()
                return hook
            h = attn.q_proj.register_forward_hook(make_q_hook(idx, details))
            handles.append(h)

        # --- K projection ---
        if hasattr(attn, "k_proj"):
            def make_k_hook(i, d):
                def hook(module, args, output):
                    d["k_proj_output"] = output.detach().clone()
                return hook
            h = attn.k_proj.register_forward_hook(make_k_hook(idx, details))
            handles.append(h)

        # --- V projection ---
        if hasattr(attn, "v_proj"):
            def make_v_hook(i, d):
                def hook(module, args, output):
                    d["v_proj_output"] = output.detach().clone()
                return hook
            h = attn.v_proj.register_forward_hook(make_v_hook(idx, details))
            handles.append(h)

        # --- O projection (attention output) ---
        if hasattr(attn, "o_proj"):
            def make_o_hook(i, d):
                def hook(module, args, output):
                    d["o_proj_output"] = output.detach().clone()
                return hook
            h = attn.o_proj.register_forward_hook(make_o_hook(idx, details))
            handles.append(h)

        # --- Q norm ---
        if hasattr(attn, "q_norm"):
            def make_qn_hook(i, d):
                def hook(module, args, output):
                    d["q_norm_output"] = output.detach().clone()
                return hook
            h = attn.q_norm.register_forward_hook(make_qn_hook(idx, details))
            handles.append(h)

        # --- K norm ---
        if hasattr(attn, "k_norm"):
            def make_kn_hook(i, d):
                def hook(module, args, output):
                    d["k_norm_output"] = output.detach().clone()
                return hook
            h = attn.k_norm.register_forward_hook(make_kn_hook(idx, details))
            handles.append(h)

    return handles


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Model path: {MODEL_PATH}")
    if not os.path.isdir(MODEL_PATH):
        print(f"ERROR: Model directory not found: {MODEL_PATH}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 1. Load tokenizer and model
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    print("Loading model (float32, CPU) — this will use ~19 GB RAM...")
    # Try AutoModelForCausalLM first (may auto-detect the text sub-model)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print(f"AutoModelForCausalLM failed ({e}), trying AutoModel...")
        model = AutoModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    model.eval()
    print(f"Model type: {type(model).__name__}")

    # Print top-level structure for debugging
    print("\nModel top-level children:")
    for name, child in model.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, "named_children"):
            for name2, child2 in child.named_children():
                ctype = type(child2).__name__
                extra = ""
                if isinstance(child2, nn.ModuleList):
                    extra = f" ({len(child2)} items)"
                print(f"    {name2}: {ctype}{extra}")

    # ------------------------------------------------------------------
    # 2. Resolve model components
    # ------------------------------------------------------------------
    layers = get_layers(model)
    embed = get_embed_tokens(model)
    final_norm = get_final_norm(model)

    print(f"\nFound {len(layers)} decoder layers")
    print(f"Embedding: {type(embed).__name__} (vocab={embed.num_embeddings}, dim={embed.embedding_dim})")
    if final_norm:
        print(f"Final norm: {type(final_norm).__name__}")

    # Print layer types
    for i, layer in enumerate(layers):
        ltype = "FULL_ATTN" if i in FULL_ATTN_INDICES else "LINEAR_ATTN"
        attn_name = "self_attn" if hasattr(layer, "self_attn") else (
            "linear_attn" if hasattr(layer, "linear_attn") else "unknown")
        print(f"  Layer {i:2d}: {ltype:12s} (attr: {attn_name})")

    # ------------------------------------------------------------------
    # 3. Tokenize the prompt (first token only for prefill verification)
    # ------------------------------------------------------------------
    print_section("TOKENIZATION")

    input_text = tokenizer.apply_chat_template(
        MESSAGES,
        add_generation_prompt=True,
        tokenize=False,
    )
    print(f"Formatted prompt:\n{input_text}\n")

    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    seq_len = input_ids.shape[1]
    print(f"Token IDs ({seq_len} tokens): {input_ids[0].tolist()}")

    # Decode individual tokens for reference
    tokens_decoded = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]
    print(f"Decoded tokens: {tokens_decoded}")

    # We process ALL tokens (the full prompt) and capture states for the
    # FIRST token position (index 0) since we want to trace the first token
    # through all 32 layers.
    # Note: For SSM/linear attention layers, "first token" may not be
    # independently separable from the sequence, so we capture position 0
    # of the sequence output.

    # ------------------------------------------------------------------
    # 4. Install hooks
    # ------------------------------------------------------------------
    print_section("INSTALLING HOOKS")
    store.clear()

    handles = []
    handles += install_layer_hooks(layers)
    handles += install_linear_attn_hooks(layers)
    handles += install_full_attn_hooks(layers)
    print(f"Installed {len(handles)} hooks total")

    # ------------------------------------------------------------------
    # 5. Run forward pass
    # ------------------------------------------------------------------
    print_section("RUNNING FORWARD PASS")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)

    print("Forward pass complete.")

    # ------------------------------------------------------------------
    # 6. Print embedding output
    # ------------------------------------------------------------------
    print_section("EMBEDDING OUTPUT (position 0)")
    with torch.no_grad():
        embed_out = embed(input_ids)
    print_values("embed_tokens[0, 0, :8]", embed_out[0, 0, :NUM_PREVIEW])
    print(f"  Embedding shape: {embed_out.shape}")

    # ------------------------------------------------------------------
    # 7. Print per-layer results
    # ------------------------------------------------------------------
    for idx in range(len(layers)):
        ltype = "FULL_ATTENTION" if idx in FULL_ATTN_INDICES else "LINEAR_ATTENTION"
        print_section(f"LAYER {idx} ({ltype})")

        # Layer input (hidden state going in)
        if idx in store.layer_inputs:
            inp = store.layer_inputs[idx]
            print_values(f"layer[{idx}] input[0, 0, :8]", inp[0, 0, :NUM_PREVIEW])
            print(f"  Input shape: {inp.shape}")

        # Layer output (hidden state coming out, with residual)
        if idx in store.layer_outputs:
            out = store.layer_outputs[idx]
            print_values(f"layer[{idx}] output[0, 0, :8]", out[0, 0, :NUM_PREVIEW])
            print(f"  Output shape: {out.shape}")

        # ----- LINEAR ATTENTION details -----
        if idx in store.linear_attn_details:
            details = store.linear_attn_details[idx]
            print(f"\n  --- Linear Attention Internals (layer {idx}) ---")

            if "qkv_proj_output" in details:
                qkv = details["qkv_proj_output"]
                print(f"  QKV projection shape: {qkv.shape}")
                # QKV is concatenated: Q_heads, K_heads, V_heads
                # Q: num_attention_heads * head_dim = 16 * 128 = 2048
                # K: linear_num_key_heads * linear_key_head_dim = 16 * 128 = 2048
                # V: linear_num_value_heads * linear_value_head_dim = 32 * 128 = 4096
                # Total = 8192
                flat = qkv[0, 0].flatten()
                total = flat.shape[0]
                # Try to infer Q/K/V split sizes from config
                config_path = os.path.join(MODEL_PATH, "config.json")
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                tc = cfg.get("text_config", cfg)
                q_size = tc.get("num_attention_heads", 16) * tc.get("linear_key_head_dim", 128)
                k_size = tc.get("linear_num_key_heads", 16) * tc.get("linear_key_head_dim", 128)
                v_size = tc.get("linear_num_value_heads", 32) * tc.get("linear_value_head_dim", 128)
                print(f"  QKV split: Q={q_size}, K={k_size}, V={v_size} (total={q_size+k_size+v_size}, actual={total})")
                print_values(f"  Q section[:8]", flat[:NUM_PREVIEW])
                print_values(f"  K section[:8]", flat[q_size:q_size+NUM_PREVIEW])
                print_values(f"  V section[:8]", flat[q_size+k_size:q_size+k_size+NUM_PREVIEW])

            if "conv1d_output" in details:
                conv = details["conv1d_output"]
                print(f"\n  Conv1d output shape: {conv.shape}")
                flat = conv.flatten()
                # Conv1d output covers the QKV channels
                # Show first 8 values from Q, K, V regions
                print_values(f"  conv1d Q[:8]", flat[:NUM_PREVIEW])
                if flat.shape[0] > q_size:
                    print_values(f"  conv1d K[:8]", flat[q_size:q_size+NUM_PREVIEW])
                if flat.shape[0] > q_size + k_size:
                    print_values(f"  conv1d V[:8]", flat[q_size+k_size:q_size+k_size+NUM_PREVIEW])

            if "in_proj_a_output" in details:
                a_out = details["in_proj_a_output"]
                print(f"\n  in_proj_a (SSM dt+B+C) shape: {a_out.shape}")
                print_values(f"  in_proj_a[:8]", a_out.flatten()[:NUM_PREVIEW])

            if "in_proj_b_output" in details:
                b_out = details["in_proj_b_output"]
                print(f"  in_proj_b (SSM down) shape: {b_out.shape}")
                print_values(f"  in_proj_b[:8]", b_out.flatten()[:NUM_PREVIEW])

            if "norm_output" in details:
                norm = details["norm_output"]
                print(f"\n  Norm (after SSM) shape: {norm.shape}")
                print_values(f"  norm[:8]", norm.flatten()[:NUM_PREVIEW])

            if "in_proj_z_output" in details:
                z_out = details["in_proj_z_output"]
                print(f"\n  in_proj_z (gate) shape: {z_out.shape}")
                print_values(f"  gate[:8]", z_out.flatten()[:NUM_PREVIEW])

            if "out_proj_output" in details:
                o_out = details["out_proj_output"]
                print(f"\n  out_proj shape: {o_out.shape}")
                print_values(f"  SSM out_proj[:8]", o_out.flatten()[:NUM_PREVIEW])

        # ----- FULL ATTENTION details -----
        if idx in store.full_attn_details:
            details = store.full_attn_details[idx]
            print(f"\n  --- Full Attention Internals (layer {idx}) ---")

            if "q_proj_output" in details:
                q = details["q_proj_output"]
                print(f"  Q proj shape: {q.shape}")
                print_values(f"  Q[:8]", q[0, 0].flatten()[:NUM_PREVIEW])

            if "k_proj_output" in details:
                k = details["k_proj_output"]
                print(f"  K proj shape: {k.shape}")
                print_values(f"  K[:8]", k[0, 0].flatten()[:NUM_PREVIEW])

            if "v_proj_output" in details:
                v = details["v_proj_output"]
                print(f"  V proj shape: {v.shape}")
                print_values(f"  V[:8]", v[0, 0].flatten()[:NUM_PREVIEW])

            if "q_norm_output" in details:
                qn = details["q_norm_output"]
                print(f"\n  Q after norm shape: {qn.shape}")
                print_values(f"  Q_norm[:8]", qn.flatten()[:NUM_PREVIEW])

            if "k_norm_output" in details:
                kn = details["k_norm_output"]
                print(f"  K after norm shape: {kn.shape}")
                print_values(f"  K_norm[:8]", kn.flatten()[:NUM_PREVIEW])

            if "o_proj_output" in details:
                o = details["o_proj_output"]
                print(f"\n  Attention output (o_proj) shape: {o.shape}")
                print_values(f"  attn_output[:8]", o[0, 0].flatten()[:NUM_PREVIEW])

    # ------------------------------------------------------------------
    # 8. Final norm + logits
    # ------------------------------------------------------------------
    print_section("FINAL NORM + LOGITS")

    last_hidden = store.layer_outputs[len(layers) - 1]
    print_values("last_hidden[0, 0, :8]", last_hidden[0, 0, :NUM_PREVIEW])

    if final_norm is not None:
        with torch.no_grad():
            normed = final_norm(last_hidden)
        print_values("after_final_norm[0, 0, :8]", normed[0, 0, :NUM_PREVIEW])

    # Logits from model output
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        print(f"\n  Logits shape: {logits.shape}")
        print_values("logits[0, 0, :8]", logits[0, 0, :NUM_PREVIEW])
        # Top-5 predictions for first token position
        top5 = torch.topk(logits[0, 0], 5)
        print(f"\n  Top-5 predictions (position 0):")
        for val, tid in zip(top5.values.tolist(), top5.indices.tolist()):
            tok = tokenizer.decode([tid])
            print(f"    token_id={tid:6d}  logit={val:+.4f}  decoded={repr(tok)}")

    # Also show last position (the generation position)
    if hasattr(outputs, "logits") and logits.shape[1] > 1:
        print(f"\n  --- Last position ({logits.shape[1]-1}) predictions ---")
        print_values(f"logits[0, -1, :8]", logits[0, -1, :NUM_PREVIEW])
        top5_last = torch.topk(logits[0, -1], 5)
        print(f"\n  Top-5 predictions (last position):")
        for val, tid in zip(top5_last.values.tolist(), top5_last.indices.tolist()):
            tok = tokenizer.decode([tid])
            print(f"    token_id={tid:6d}  logit={val:+.4f}  decoded={repr(tok)}")

    # ------------------------------------------------------------------
    # 9. Cleanup
    # ------------------------------------------------------------------
    for h in handles:
        h.remove()

    print_section("DONE")
    print(f"  Captured hidden states for {len(store.layer_outputs)} layers")
    print(f"  Linear attention details for layers: {sorted(store.linear_attn_details.keys())}")
    print(f"  Full attention details for layers: {sorted(store.full_attn_details.keys())}")
    print()


if __name__ == "__main__":
    main()
