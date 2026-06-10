/**
 * ModelDescriptor — the engine's model-agnostic architecture contract.
 *
 * Extends ModelConfig with an explicit per-layer description so the forward
 * pass never consults family-specific fields (layerTypes, sliding windows,
 * per-layer head dims) directly. Built from either an HF config.json
 * (descriptorFromHFConfig) or — in Phase B — a GGUF header (descriptorFromGGUF).
 *
 * Phase A: descriptorFromHFConfig only; produces layers[] that exactly mirror
 * the legacy config.layerTypes semantics (zero behavior change).
 */

import { parseModelConfig, type ModelConfig } from './model-config';

export type LayerKind = 'full_attention' | 'sliding_attention' | 'linear_attention';

export interface RopeSpec {
  /** RoPE base frequency for this layer. */
  theta: number;
  /** Fraction of head dims that get RoPE (e.g., 0.25 for Qwen3.5/3.6). */
  partialRotaryFactor?: number;
  /** MRoPE section split (Qwen3.6 GGUF `rope.dimension_sections`). */
  dimensionSections?: number[];
}

/** Mixture-of-Experts FFN specification (Phase C). */
export interface MoESpec {
  numExperts: number;
  numExpertsPerToken: number;
  /** Routed expert FFN intermediate dim. */
  expertFFNDim: number;
  /** Shared (always-on) expert FFN dim, if present. */
  sharedExpertFFNDim?: number;
  /** Gating applied to the shared expert output. */
  sharedExpertGate?: 'sigmoid' | 'none';
}

export interface LayerDescriptor {
  kind: LayerKind;
  /** Per-layer head dim override (Gemma 4 dual head_dim). Falls back to config.headDim. */
  headDim?: number;
  /** Per-layer RoPE (Gemma 4 uses different theta on sliding layers). */
  rope?: RopeSpec;
  /** Attention window for kind='sliding_attention'. */
  slidingWindow?: number;
  /** Reuse K/V from this layer instead of computing/caching (Gemma 4 KV sharing). */
  kvSourceLayer?: number;
  /** MoE FFN spec; undefined = dense FFN. */
  moe?: MoESpec;
}

export interface ModelDescriptor extends ModelConfig {
  /** One entry per transformer layer. length === numLayers. */
  layers: LayerDescriptor[];
  /** FFN activation. Derived from hiddenAct ('gelu_pytorch_tanh' → 'gelu_tanh'). */
  activation: 'silu' | 'gelu_tanh';
  /** tanh(x/c)*c applied after lm_head (Gemma). Undefined = no softcap. */
  finalLogitSoftcap?: number;
  /** Where the weights come from. Affects loader routing, not the forward pass. */
  sourceFormat: 'safetensors' | 'gguf';
  /** Model has per-layer embeddings fused into the residual stream (Gemma 4 PLE). */
  perLayerEmbed?: boolean;
}

/** Build per-layer descriptors from a parsed ModelConfig (HF semantics). */
export function buildLayerDescriptors(config: ModelConfig): LayerDescriptor[] {
  const rope: RopeSpec = {
    theta: config.ropeTheta,
    partialRotaryFactor: config.partialRotaryFactor,
  };
  const layers: LayerDescriptor[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    const t = config.layerTypes?.[l];
    const kind: LayerKind =
      t === 'linear_attention' ? 'linear_attention'
      : t === 'sliding_attention' ? 'sliding_attention'
      : 'full_attention';
    layers.push({ kind, rope });
  }
  return layers;
}

function activationOf(hiddenAct: string): 'silu' | 'gelu_tanh' {
  if (hiddenAct === 'gelu_pytorch_tanh' || hiddenAct === 'gelu_tanh') return 'gelu_tanh';
  return 'silu';
}

/**
 * Parse an HF config.json into a ModelDescriptor.
 * Superset of parseModelConfig — all legacy fields remain valid.
 */
export function descriptorFromHFConfig(hfConfig: Record<string, any>): ModelDescriptor {
  const config = parseModelConfig(hfConfig);
  return {
    ...config,
    layers: buildLayerDescriptors(config),
    activation: activationOf(config.hiddenAct),
    finalLogitSoftcap: hfConfig.final_logit_softcapping ?? undefined,
    sourceFormat: 'safetensors',
  };
}

/** Wrap an already-parsed ModelConfig (for callers that parsed separately). */
export function descriptorFromConfig(config: ModelConfig): ModelDescriptor {
  return {
    ...config,
    layers: buildLayerDescriptors(config),
    activation: activationOf(config.hiddenAct),
    sourceFormat: 'safetensors',
  };
}
