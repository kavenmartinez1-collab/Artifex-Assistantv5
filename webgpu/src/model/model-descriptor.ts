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
import { archKV, ggufArchitecture, type GGUFFile } from './gguf';

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

// ── GGUF → ModelDescriptor ─────────────────────────────────────────────

/** Per-arch facts that aren't recoverable from GGUF KV metadata alone. */
interface GGUFArchInfo {
  /** HF model_type equivalent — keeps downstream modelType checks working. */
  modelType: string;
  /** Attention output gate fused into q_proj (Qwen3.5/3.6 gated attention). */
  attnOutputGate?: boolean;
}

const GGUF_ARCHS: Record<string, GGUFArchInfo> = {
  qwen35:    { modelType: 'qwen3_5', attnOutputGate: true },
  qwen35moe: { modelType: 'qwen3_5_moe', attnOutputGate: true },
  qwen3:     { modelType: 'qwen3' },
  qwen3moe:  { modelType: 'qwen3_moe' },
  llama:     { modelType: 'llama' },
};

/**
 * Parse a GGUF header into a ModelDescriptor.
 *
 * Conventions verified against llama.cpp (vendor/llama.cpp):
 *  - layer i is recurrent (linear attention) iff (i+1) % full_attention_interval != 0
 *    (src/models/qwen35moe.cpp load_arch_hparams)
 *  - DeltaNet dims: key/value head dim = ssm.state_size, key heads = ssm.group_count,
 *    value heads = ssm.time_step_rank
 *  - tie_word_embeddings ⇔ no 'output.weight' tensor (output dup'd from token_embd)
 *  - vocab size from token_embd.weight ne[1]
 */
export function descriptorFromGGUF(file: GGUFFile): ModelDescriptor {
  const arch = ggufArchitecture(file);
  const info = GGUF_ARCHS[arch];
  if (!info) {
    throw new Error(`[GGUF] Unsupported architecture "${arch}" (known: ${Object.keys(GGUF_ARCHS).join(', ')})`);
  }

  const embedTensor = file.tensors.get('token_embd.weight');
  if (!embedTensor) throw new Error('[GGUF] Missing token_embd.weight');
  const vocabSize = embedTensor.ne[1];
  const hiddenSize = archKV<number>(file, 'embedding_length');
  const numLayers = archKV<number>(file, 'block_count') - archKV<number>(file, 'nextn_predict_layers', 0);
  const numAttentionHeads = archKV<number>(file, 'attention.head_count');
  const numKVHeads = archKV<number>(file, 'attention.head_count_kv', numAttentionHeads);
  const headDim = archKV<number>(file, 'attention.key_length', hiddenSize / numAttentionHeads);

  // Hybrid (Gated DeltaNet) detection via ssm.* keys
  const convKernel = archKV<number | undefined>(file, 'ssm.conv_kernel', undefined);
  const isHybrid = convKernel !== undefined;
  const fullAttnInterval = archKV<number>(file, 'full_attention_interval', 1);
  let layerTypes: string[] | undefined;
  if (isHybrid) {
    layerTypes = [];
    for (let i = 0; i < numLayers; i++) {
      layerTypes.push((i + 1) % fullAttnInterval !== 0 ? 'linear_attention' : 'full_attention');
    }
  }

  // MoE detection
  const numExperts = archKV<number>(file, 'expert_count', 0);
  const moe: MoESpec | undefined = numExperts > 0 ? {
    numExperts,
    numExpertsPerToken: archKV<number>(file, 'expert_used_count'),
    expertFFNDim: archKV<number>(file, 'expert_feed_forward_length'),
    sharedExpertFFNDim: archKV<number | undefined>(file, 'expert_shared_feed_forward_length', undefined),
    sharedExpertGate: file.tensors.has('blk.0.ffn_gate_inp_shexp.weight') ? 'sigmoid' : 'none',
  } : undefined;

  // Dense FFN dim; MoE archs may not ship feed_forward_length — fall back to expert dim
  const intermediateSize = archKV<number>(file, 'feed_forward_length', moe?.expertFFNDim ?? 0);

  const ropeTheta = archKV<number>(file, 'rope.freq_base', 10000);
  const ropeDims = archKV<number | undefined>(file, 'rope.dimension_count', undefined);
  const partialRotaryFactor = ropeDims !== undefined && ropeDims < headDim ? ropeDims / headDim : undefined;
  const dimensionSections = archKV<number[] | undefined>(file, 'rope.dimension_sections', undefined);

  // Bias presence is encoded by tensor existence in GGUF
  const attentionBias = [...file.tensors.keys()].some(n => n.endsWith('.attn_q.bias'));

  const config: ModelConfig = {
    modelType: info.modelType,
    hiddenSize,
    numLayers,
    numAttentionHeads,
    numKVHeads,
    headDim,
    intermediateSize,
    vocabSize,
    rmsNormEps: archKV<number>(file, 'attention.layer_norm_rms_epsilon', 1e-6),
    ropeTheta,
    maxPositionEmbeddings: archKV<number>(file, 'context_length', 32768),
    hiddenAct: 'silu',
    attentionBias,
    tieWordEmbeddings: !file.tensors.has('output.weight'),
    quantMethod: 'gguf',
    quantBits: 0,
    quantGroupSize: 0,
    layerTypes,
    isHybrid,
    linearKeyHeadDim: isHybrid ? archKV<number>(file, 'ssm.state_size') : undefined,
    linearValueHeadDim: isHybrid ? archKV<number>(file, 'ssm.state_size') : undefined,
    linearNumKeyHeads: isHybrid ? archKV<number>(file, 'ssm.group_count') : undefined,
    linearNumValueHeads: isHybrid ? archKV<number>(file, 'ssm.time_step_rank') : undefined,
    linearConvKernelDim: convKernel,
    partialRotaryFactor,
    attnOutputGate: info.attnOutputGate,
    numQPerKV: numAttentionHeads / numKVHeads,
    isGQA: numKVHeads < numAttentionHeads,
    isQuantized: true,
  };

  const rope: RopeSpec = { theta: ropeTheta, partialRotaryFactor, dimensionSections };
  const layers: LayerDescriptor[] = [];
  for (let l = 0; l < numLayers; l++) {
    const kind: LayerKind = layerTypes?.[l] === 'linear_attention' ? 'linear_attention' : 'full_attention';
    layers.push({ kind, rope, moe });
  }

  return {
    ...config,
    layers,
    activation: activationOf(config.hiddenAct),
    sourceFormat: 'gguf',
  };
}
