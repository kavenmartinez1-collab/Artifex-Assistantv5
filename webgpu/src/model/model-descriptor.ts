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
  /**
   * Gemma 4 "proportional" RoPE: pairing spans the FULL head (pair i with
   * i+headDim/2) but only the first `rotatedPairs` pairs rotate; the rest are
   * identity. Derived from the GGUF `rope_freqs` sentinel tensor
   * (1.0 = rotate, ~1e30 = frozen) via applyRopeFreqFactors().
   * Undefined = all pairs rotate (subject to partialRotaryFactor).
   */
  rotatedPairs?: number;
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
  /** Attention score scale. Undefined = 1/√headDim. Gemma 4 uses 1.0. */
  attnScale?: number;
  /** Where the weights come from. Affects loader routing, not the forward pass. */
  sourceFormat: 'safetensors' | 'gguf';
  /** Recognized GGUF arch that hasn't been verified end-to-end (UI flags it). */
  experimentalArch?: boolean;
  /** Model has per-layer embeddings fused into the residual stream (Gemma 4 PLE). */
  perLayerEmbed?: boolean;
  /** Per-layer embedding dim (Gemma 4 `embedding_length_per_layer_input`, 256). */
  perLayerEmbedDim?: number;
  /** Multiply token embeddings by this after lookup (Gemma: √hiddenSize). */
  embedScale?: number;
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

/** Optional arch KV read — archKV(file, key, undefined) would throw. */
function archKVOpt<T>(file: GGUFFile, key: string): T | undefined {
  return file.kv.get(`${ggufArchitecture(file)}.${key}`) as T | undefined;
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
  /** FFN activation when the arch hardcodes it (GGUF has no hidden_act key). */
  hiddenAct?: string;
  /** Recognized but not yet verified end-to-end — the engine attempts it via
   *  the standard-transformer path (the UI flags it). These archs are
   *  structurally identical to a verified one per llama.cpp, so the worst case
   *  is a clean shape/tensor error, not silent garbage. */
  experimental?: boolean;
}

const GGUF_ARCHS: Record<string, GGUFArchInfo> = {
  // ── Verified ───────────────────────────────────────────────────────
  qwen35:    { modelType: 'qwen3_5', attnOutputGate: true },
  qwen35moe: { modelType: 'qwen3_5_moe', attnOutputGate: true },
  qwen3:     { modelType: 'qwen3' },
  qwen3moe:  { modelType: 'qwen3_moe' },
  llama:     { modelType: 'llama' },
  // llama.cpp hardcodes LLM_FFN_GELU for gemma4 (src/models/gemma4.cpp)
  gemma4:    { modelType: 'gemma4_text', hiddenAct: 'gelu_pytorch_tanh' },
  // ── Experimental: standard transformers, llama.cpp-confirmed structurally
  //    identical to the verified paths above. Dims read generically from GGUF
  //    metadata; bias/MoE handled by tensor presence. Attempted, not yet
  //    end-to-end verified. (Gemma 2/3, Phi3 fused-QKV, DeepSeek MLA, and the
  //    Mamba2 hybrids are intentionally NOT here — they need real compute work,
  //    and a bad mapping would produce wrong output rather than a clean error.)
  qwen2:     { modelType: 'qwen2', experimental: true },     // attn bias, no qk-norm
  qwen2moe:  { modelType: 'qwen2_moe', experimental: true },
  mistral3:  { modelType: 'llama', experimental: true },     // llama-architecture
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
  // Some qwen35 GGUFs (e.g. Ollama's qwen3.5:9b) store head_count_kv PER LAYER
  // — 0 on the DeltaNet linear layers, the real count on full-attention layers
  // — while others (the 27B) store a plain scalar. Reduce an array to the
  // scalar full-attention KV head count (the layer's own kvDim is derived from
  // this; linear layers don't use it). Reading the array as a scalar made every
  // KV size garbage → "wrong sizes, layer 0".
  const kvRaw = archKV<number | number[]>(file, 'attention.head_count_kv', numAttentionHeads);
  let numKVHeads: number;
  if (Array.isArray(kvRaw)) {
    const nonzero = kvRaw.filter((v) => typeof v === 'number' && v > 0);
    numKVHeads = nonzero.length > 0 ? Math.max(...nonzero) : numAttentionHeads;
  } else {
    numKVHeads = kvRaw;
  }
  const headDim = archKV<number>(file, 'attention.key_length', hiddenSize / numAttentionHeads);

  // Hybrid (Gated DeltaNet) detection via ssm.* keys
  const convKernel = archKVOpt<number>(file, 'ssm.conv_kernel');
  const isHybrid = convKernel !== undefined;
  const fullAttnInterval = archKV<number>(file, 'full_attention_interval', 1);
  let layerTypes: string[] | undefined;
  if (isHybrid) {
    layerTypes = [];
    for (let i = 0; i < numLayers; i++) {
      layerTypes.push((i + 1) % fullAttnInterval !== 0 ? 'linear_attention' : 'full_attention');
    }
  }

  // Sliding-window pattern (Gemma 4): one bool per layer, true = sliding.
  const swaPattern = archKVOpt<boolean[]>(file, 'attention.sliding_window_pattern');
  if (swaPattern !== undefined) {
    if (swaPattern.length < numLayers) {
      throw new Error(`[GGUF] sliding_window_pattern has ${swaPattern.length} entries, need ${numLayers}`);
    }
    layerTypes = swaPattern.slice(0, numLayers).map(b => (b ? 'sliding_attention' : 'full_attention'));
  }
  const slidingWindow = archKVOpt<number>(file, 'attention.sliding_window');
  // Gemma 4 dual head_dim: sliding layers use *_swa key/value lengths.
  const headDimSwa = archKV<number>(file, 'attention.key_length_swa', headDim);
  const valueLen = archKV<number>(file, 'attention.value_length', headDim);
  const valueLenSwa = archKV<number>(file, 'attention.value_length_swa', headDimSwa);
  if (valueLen !== headDim || valueLenSwa !== headDimSwa) {
    throw new Error(`[GGUF] key/value head dims differ (k=${headDim}/${headDimSwa}, v=${valueLen}/${valueLenSwa}) — unsupported`);
  }
  // Gemma 4 KV sharing: last `shared_kv_layers` layers reuse earlier KV caches.
  const sharedKvLayers = archKV<number>(file, 'attention.shared_kv_layers', 0);
  const kvFromStart = numLayers - sharedKvLayers;

  // MoE detection
  const numExperts = archKV<number>(file, 'expert_count', 0);
  const moe: MoESpec | undefined = numExperts > 0 ? {
    numExperts,
    numExpertsPerToken: archKV<number>(file, 'expert_used_count'),
    expertFFNDim: archKV<number>(file, 'expert_feed_forward_length'),
    sharedExpertFFNDim: archKVOpt<number>(file, 'expert_shared_feed_forward_length'),
    sharedExpertGate: file.tensors.has('blk.0.ffn_gate_inp_shexp.weight') ? 'sigmoid' : 'none',
  } : undefined;

  // Dense FFN dim; MoE archs may not ship feed_forward_length — fall back to expert dim
  const intermediateSize = archKV<number>(file, 'feed_forward_length', moe?.expertFFNDim ?? 0);

  const ropeTheta = archKV<number>(file, 'rope.freq_base', 10000);
  const ropeThetaSwa = archKV<number>(file, 'rope.freq_base_swa', ropeTheta);
  const ropeDims = archKVOpt<number>(file, 'rope.dimension_count');
  const partialRotaryFactor = ropeDims !== undefined && ropeDims < headDim ? ropeDims / headDim : undefined;
  const dimensionSections = archKVOpt<number[]>(file, 'rope.dimension_sections');

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
    hiddenAct: info.hiddenAct ?? 'silu',
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
    const t = layerTypes?.[l];
    const kind: LayerKind =
      t === 'linear_attention' ? 'linear_attention'
      : t === 'sliding_attention' ? 'sliding_attention'
      : 'full_attention';
    const layer: LayerDescriptor = { kind, rope, moe };
    if (swaPattern !== undefined) {
      const sliding = kind === 'sliding_attention';
      layer.headDim = sliding ? headDimSwa : headDim;
      // Full layers: proportional RoPE — rotatedPairs filled later from the
      // rope_freqs tensor data via applyRopeFreqFactors().
      layer.rope = sliding ? { theta: ropeThetaSwa } : { theta: ropeTheta };
      if (sliding) layer.slidingWindow = slidingWindow;
      if (sharedKvLayers > 0 && l >= kvFromStart) {
        // llama.cpp llama-model.cpp: kv_from_start - (is_swa ? 2 : 1)
        layer.kvSourceLayer = kvFromStart - (sliding ? 2 : 1);
      }
    }
    layers.push(layer);
  }

  // Per-layer embeddings (Gemma 4 PLE)
  const perLayerEmbedDim = archKV<number>(file, 'embedding_length_per_layer_input', 0);

  return {
    ...config,
    layers,
    activation: activationOf(config.hiddenAct),
    finalLogitSoftcap: archKVOpt<number>(file, 'final_logit_softcapping'),
    // gemma4.cpp: hparams.f_attention_scale = 1.0f (no 1/√d pre-attn scaling)
    attnScale: arch === 'gemma4' ? 1.0 : undefined,
    sourceFormat: 'gguf',
    experimentalArch: info.experimental ? true : undefined,
    perLayerEmbed: perLayerEmbedDim > 0 ? true : undefined,
    perLayerEmbedDim: perLayerEmbedDim > 0 ? perLayerEmbedDim : undefined,
    // Gemma scales embeddings by √hiddenSize (gemma4.cpp:155)
    embedScale: arch === 'gemma4' ? Math.sqrt(hiddenSize) : undefined,
  };
}

/**
 * Fill per-layer `rope.rotatedPairs` from the GGUF `rope_freqs` tensor data
 * (Gemma 4 proportional RoPE). The tensor holds headDim/2 frequency factors:
 * 1.0 = the pair rotates normally, ~1e30 sentinel = the pair is identity.
 * Applies to full_attention layers only (sliding layers have no freq factors).
 * Throws if the factor pattern is not a clean [1.0 × N, sentinel × rest] split.
 */
export function applyRopeFreqFactors(desc: ModelDescriptor, factors: Float32Array): void {
  let rotated = 0;
  while (rotated < factors.length && factors[rotated] === 1.0) rotated++;
  for (let i = rotated; i < factors.length; i++) {
    if (!(factors[i] > 1e20)) {
      throw new Error(`[GGUF] rope_freqs[${i}] = ${factors[i]} — expected 1.0-prefix then ~1e30 sentinels`);
    }
  }
  if (rotated === 0) throw new Error('[GGUF] rope_freqs has no rotating pairs');
  for (const layer of desc.layers) {
    if (layer.kind === 'full_attention' && layer.rope) {
      layer.rope = { ...layer.rope, rotatedPairs: rotated === factors.length ? undefined : rotated };
    }
  }
}
