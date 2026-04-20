# Gemma 4 E4B — Architecture Intel (2026-04-19)

Source: `google/gemma-4-E4B-it` `config.json`, `transformers==5.5.0.dev0`.
Architecture class: `Gemma4ForConditionalGeneration` → text submodel `Gemma4TextModel`.

## Text model spec

| Field | Value | Notes |
|---|---|---|
| `hidden_size` | 2560 | Main residual stream |
| `num_hidden_layers` | 42 | |
| `num_attention_heads` | 8 | |
| `num_key_value_heads` | 2 | GQA 4:1 |
| `head_dim` | 256 | Sliding-attention heads |
| `global_head_dim` | 512 | **Full-attention heads use 512-dim heads** |
| `intermediate_size` | 10240 | FFN 4× hidden |
| `hidden_activation` | `gelu_pytorch_tanh` | **NOT SwiGLU** |
| `vocab_size` | 262144 | |
| `max_position_embeddings` | 131072 | |
| `sliding_window` | 512 | |
| `rms_norm_eps` | 1e-6 | |
| `tie_word_embeddings` | true | lm_head shares embed weights |
| `final_logit_softcapping` | 30.0 | Clip logits via `tanh(x/30) * 30` |
| `hidden_size_per_layer_input` | 256 | **Per-Layer Embeddings (PLE)** |
| `num_kv_shared_layers` | 18 | **KV cache sharing** |

## Layer pattern

42 layers: sliding/full mix. Full-attention layers at indices **5, 11, 17, 23, 29, 35, 41** (every 6th). Pattern: 5×sliding, 1×full, repeat.

## RoPE — two configs

```json
{
  "full_attention":    { "partial_rotary_factor": 0.25, "rope_theta": 1000000, "rope_type": "proportional" },
  "sliding_attention": { "rope_theta": 10000, "rope_type": "default" }
}
```

- **Full attention**: partial rotary (0.25 of head_dim=512 → 128 rotary dims), θ=1e6. This matches our Qwen3.5 partial rotary path.
- **Sliding attention**: full rotary (no partial), θ=1e4 (standard).
- **`proportional` rope_type** is a new scaling variant we don't currently support.

## Per-Layer Embeddings (PLE) — the novel bit

Each layer receives its main residual input *plus* a per-layer 256-dim vector pulled from a separate embedding table of shape `[vocab_size=262144, num_layers=42 × hidden_size_per_layer_input=256]`.

- PLE table size: 262144 × 42 × 256 = **2.8 B params** (≈5.6 GB at BF16, ≈1.4 GB at INT4)
- At each layer, look up `ple_embedding[token_id][layer_idx]` → 256-dim vector → project into hidden stream

This is THE memory killer on 8 GB. Must quantize or stream.

## KV cache sharing

`num_kv_shared_layers: 18` — of 42 layers, 18 share KV with a neighbor. Reduces KV cache cost by ~43 %. Implementation: a mapping `layer_idx → kv_source_layer` that the attention kernel respects.

## Logit softcapping

Final output layer applies `logits = tanh(logits / 30) * 30` before sampling. Without this the model's logit distribution is off-scale.

## Multimodal encoders (not in scope tonight)

- **Vision**: 16-layer ViT, 768 hidden, patch 16, pooling 3. Output 280 soft tokens.
- **Audio**: 12-layer encoder with chunked attention (chunk=12, context 13 left / 0 right), conv subsampling [128,32,5-kernel].

## What's new vs our current engine

| Feature | Have? | Status |
|---|---|---|
| RMSNorm | ✅ | weight init=w (NOT 1+w for Gemma 4) |
| RoPE partial rotary | ✅ | Qwen3.5 already does 0.25 |
| RoPE proportional scaling | ❌ | New — need to implement |
| Two head_dims per model | ❌ | New — need per-layer kernel dispatch |
| GQA | ✅ | |
| Sliding-window attention | ❌ | Need mask variant |
| GELU-tanh activation | ❌ | Have SiLU; need new activation |
| Gated MLP (Gemma uses up+gate like Llama) | ✅ | |
| Tied embeddings | ✅ | |
| Per-Layer Embeddings | ❌ | Novel — significant new kernel + memory |
| KV cache sharing | ❌ | New — attention dispatch change |
| Final logit softcapping | ❌ | Trivial — single pass before sampling |
| Vocab 262K | ✅ | We already handle 248K |

## Memory budget — E4B text-only on 8 GB

| Item | BF16 | INT4 |
|---|---|---|
| Transformer body (42 layers) | ≈7 GB | ≈1.75 GB |
| Main embeddings (262K × 2560) | 1.3 GB | 0.33 GB |
| PLE table (262K × 42 × 256) | 5.6 GB | 1.4 GB |
| KV cache (4K ctx, with sharing) | 0.4 GB | 0.4 GB |
| Activations/workspace | 0.8 GB | 0.6 GB |
| Browser overhead | 1.0 GB | 1.0 GB |
| **Total** | **≈16 GB** | **≈5.5 GB** |

**BF16 does not fit.** INT4 fits comfortably but requires a quantized release or our own GPTQ pipeline run.

## Open questions before coding

1. Does a pre-quantized INT4 `gemma-4-E4B-it` exist on HF, or do we need to quantize ourselves?
2. For "proportional" RoPE scaling, get the formula from `modeling_gemma4.py`.
3. Exact PLE fusion path — is the 256-dim input added to the residual stream, or concatenated, or projected into attention Q/K/V? (Need modeling code.)
4. KV-shared pair mapping — is it `layer[i] → layer[i-1]` or some other pattern?
