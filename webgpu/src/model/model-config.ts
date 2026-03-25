/**
 * Model Configuration — Parsed from HuggingFace config.json
 *
 * Model-agnostic interface that parameterizes the entire forward pass.
 * Supports Qwen, Llama, Mistral, Gemma, Phi, DeepSeek, and any model
 * using the standard transformer layout.
 *
 * All architecture-specific values come from config.json — nothing is
 * hardcoded for a particular model family.
 */

export interface ModelConfig {
  /** Model family identifier (e.g., 'qwen2', 'llama', 'mistral') */
  modelType: string;

  // ── Dimensions ─────────────────────────────────────────────────────
  /** Hidden dimension (embedding size). E.g., 3584 for Qwen3.5-9B */
  hiddenSize: number;
  /** Number of transformer layers. E.g., 28 for Qwen3.5-9B */
  numLayers: number;
  /** Number of attention heads for queries. E.g., 28 */
  numAttentionHeads: number;
  /** Number of KV heads (for GQA). Same as numAttentionHeads if not GQA. */
  numKVHeads: number;
  /** Dimension per attention head. Usually hiddenSize / numAttentionHeads */
  headDim: number;
  /** FFN intermediate size (SwiGLU expansion). E.g., 18944 */
  intermediateSize: number;
  /** Vocabulary size. E.g., 152064 for Qwen3.5 */
  vocabSize: number;

  // ── Normalization ──────────────────────────────────────────────────
  /** RMSNorm epsilon. Typically 1e-5 or 1e-6 */
  rmsNormEps: number;

  // ── Position Encoding ──────────────────────────────────────────────
  /** RoPE base frequency. 1000000 for Qwen, 500000 for Llama3 */
  ropeTheta: number;
  /** Maximum sequence length the model supports */
  maxPositionEmbeddings: number;

  // ── Activation ─────────────────────────────────────────────────────
  /** Activation function in FFN. 'silu' for most models. */
  hiddenAct: string;

  // ── Bias ────────────────────────────────────────────────────────────
  /** Whether attention Q/K/V/O projections have bias terms */
  attentionBias: boolean;

  // ── Weight Tying ───────────────────────────────────────────────────
  /** Whether the embedding and lm_head share weights */
  tieWordEmbeddings: boolean;

  // ── Derived ────────────────────────────────────────────────────────
  /** Number of Q heads per KV head group (for GQA). = numAttentionHeads / numKVHeads */
  numQPerKV: number;
  /** Whether this model uses grouped-query attention */
  isGQA: boolean;
}

/**
 * Weight name mapping — translates between HuggingFace weight names and
 * our generic layer-indexed names.
 *
 * MODEL-SPECIFIC: Different model families use different naming conventions.
 * This mapping is needed by the weight loader to find the right tensors.
 */
export interface WeightNameMap {
  /** Embedding table: 'model.embed_tokens.weight' for most models */
  embedTokens: string;
  /** Final RMSNorm: 'model.norm.weight' */
  finalNorm: string;
  /** LM head: 'lm_head.weight' (or same as embedTokens if tied) */
  lmHead: string;

  /** Per-layer weight name pattern. Layer index is substituted for {L}. */
  layer: {
    /** Input layernorm (pre-attention) */
    inputNorm: string;
    /** Q projection */
    qProj: string;
    /** K projection */
    kProj: string;
    /** V projection */
    vProj: string;
    /** Output projection */
    oProj: string;
    /** Bias terms (same names + .bias instead of .weight) */
    qBias: string;
    kBias: string;
    vBias: string;
    oBias: string;
    /** Post-attention layernorm (pre-FFN) */
    postAttnNorm: string;
    /** FFN gate projection (SwiGLU) */
    gateProj: string;
    /** FFN up projection */
    upProj: string;
    /** FFN down projection */
    downProj: string;
  };
}

// ── Known model family configurations ───────────────────────────────────

/** HF config.json field names vary by model family. This maps them. */
interface HFFieldMap {
  numKVHeads: string[];        // field names to try for KV head count
  intermediateSize: string[];  // field names for FFN width
  ropeTheta: string[];         // field names for RoPE base
  headDim: string[];           // explicit head dim (some models have this)
}

const HF_FIELD_MAPS: Record<string, HFFieldMap> = {
  default: {
    numKVHeads: ['num_key_value_heads', 'num_kv_heads'],
    intermediateSize: ['intermediate_size', 'ffn_dim'],
    ropeTheta: ['rope_theta'],
    headDim: ['head_dim'],
  },
  // Phi models use different field names
  phi: {
    numKVHeads: ['num_key_value_heads', 'num_kv_heads'],
    intermediateSize: ['intermediate_size'],
    ropeTheta: ['rope_theta', 'partial_rotary_factor'],
    headDim: ['head_dim'],
  },
};

/**
 * Weight name patterns for known model families.
 * {L} is replaced with the layer index (0-based).
 */
const WEIGHT_NAME_PATTERNS: Record<string, WeightNameMap> = {
  // Qwen, Llama, Mistral, DeepSeek, Gemma (standard HF naming)
  default: {
    embedTokens: 'model.embed_tokens.weight',
    finalNorm: 'model.norm.weight',
    lmHead: 'lm_head.weight',
    layer: {
      inputNorm: 'model.layers.{L}.input_layernorm.weight',
      qProj: 'model.layers.{L}.self_attn.q_proj.weight',
      kProj: 'model.layers.{L}.self_attn.k_proj.weight',
      vProj: 'model.layers.{L}.self_attn.v_proj.weight',
      oProj: 'model.layers.{L}.self_attn.o_proj.weight',
      qBias: 'model.layers.{L}.self_attn.q_proj.bias',
      kBias: 'model.layers.{L}.self_attn.k_proj.bias',
      vBias: 'model.layers.{L}.self_attn.v_proj.bias',
      oBias: 'model.layers.{L}.self_attn.o_proj.bias',
      postAttnNorm: 'model.layers.{L}.post_attention_layernorm.weight',
      gateProj: 'model.layers.{L}.mlp.gate_proj.weight',
      upProj: 'model.layers.{L}.mlp.up_proj.weight',
      downProj: 'model.layers.{L}.mlp.down_proj.weight',
    },
  },
  // Phi models use a different structure
  phi: {
    embedTokens: 'model.embed_tokens.weight',
    finalNorm: 'model.final_layernorm.weight',
    lmHead: 'lm_head.weight',
    layer: {
      inputNorm: 'model.layers.{L}.input_layernorm.weight',
      qProj: 'model.layers.{L}.self_attn.q_proj.weight',
      kProj: 'model.layers.{L}.self_attn.k_proj.weight',
      vProj: 'model.layers.{L}.self_attn.v_proj.weight',
      oProj: 'model.layers.{L}.self_attn.dense.weight',
      qBias: 'model.layers.{L}.self_attn.q_proj.bias',
      kBias: 'model.layers.{L}.self_attn.k_proj.bias',
      vBias: 'model.layers.{L}.self_attn.v_proj.bias',
      oBias: 'model.layers.{L}.self_attn.dense.bias',
      postAttnNorm: 'model.layers.{L}.post_attention_layernorm.weight',
      gateProj: 'model.layers.{L}.mlp.gate_up_proj.weight',
      upProj: 'model.layers.{L}.mlp.gate_up_proj.weight',
      downProj: 'model.layers.{L}.mlp.down_proj.weight',
    },
  },
};

/**
 * Parse a HuggingFace config.json into our ModelConfig.
 * Handles field name differences across model families automatically.
 */
export function parseModelConfig(hfConfig: Record<string, any>): ModelConfig {
  const modelType = (hfConfig.model_type ?? 'unknown').toLowerCase();
  const fieldMap = HF_FIELD_MAPS[modelType] ?? HF_FIELD_MAPS.default;

  // Required fields (present in all models)
  const hiddenSize = hfConfig.hidden_size;
  const numLayers = hfConfig.num_hidden_layers;
  const numAttentionHeads = hfConfig.num_attention_heads;
  const vocabSize = hfConfig.vocab_size;

  if (!hiddenSize || !numLayers || !numAttentionHeads || !vocabSize) {
    throw new Error(
      `Missing required config fields. Got: hidden_size=${hiddenSize}, ` +
      `num_hidden_layers=${numLayers}, num_attention_heads=${numAttentionHeads}, ` +
      `vocab_size=${vocabSize}`
    );
  }

  // Optional fields with fallbacks
  const numKVHeads = firstDefined(hfConfig, fieldMap.numKVHeads) ?? numAttentionHeads;
  const intermediateSize = firstDefined(hfConfig, fieldMap.intermediateSize) ?? hiddenSize * 4;
  const ropeTheta = firstDefined(hfConfig, fieldMap.ropeTheta) ?? 10000;
  const headDim = firstDefined(hfConfig, fieldMap.headDim) ?? Math.floor(hiddenSize / numAttentionHeads);

  const rmsNormEps = hfConfig.rms_norm_eps ?? hfConfig.layer_norm_eps ?? 1e-5;
  const maxPositionEmbeddings = hfConfig.max_position_embeddings ?? 4096;
  const hiddenAct = hfConfig.hidden_act ?? hfConfig.activation_function ?? 'silu';
  const attentionBias = hfConfig.attention_bias ?? false;
  const tieWordEmbeddings = hfConfig.tie_word_embeddings ?? false;

  const numQPerKV = Math.floor(numAttentionHeads / numKVHeads);

  return {
    modelType,
    hiddenSize,
    numLayers,
    numAttentionHeads,
    numKVHeads,
    headDim,
    intermediateSize,
    vocabSize,
    rmsNormEps,
    ropeTheta,
    maxPositionEmbeddings,
    hiddenAct,
    attentionBias,
    tieWordEmbeddings,
    numQPerKV,
    isGQA: numKVHeads !== numAttentionHeads,
  };
}

/**
 * Get the weight name mapping for a given model type.
 */
export function getWeightNameMap(modelType: string): WeightNameMap {
  return WEIGHT_NAME_PATTERNS[modelType] ?? WEIGHT_NAME_PATTERNS.default;
}

/**
 * Resolve a per-layer weight name pattern for a specific layer index.
 */
export function resolveLayerWeightName(pattern: string, layerIndex: number): string {
  return pattern.replace('{L}', String(layerIndex));
}

/**
 * Get all weight names needed for a model (for the weight loader).
 */
export function getAllWeightNames(config: ModelConfig): string[] {
  const map = getWeightNameMap(config.modelType);
  const names: string[] = [
    map.embedTokens,
    map.finalNorm,
  ];

  if (!config.tieWordEmbeddings) {
    names.push(map.lmHead);
  }

  for (let l = 0; l < config.numLayers; l++) {
    for (const key of Object.keys(map.layer) as Array<keyof typeof map.layer>) {
      names.push(resolveLayerWeightName(map.layer[key], l));
    }
  }

  return names;
}

/**
 * Estimate VRAM usage for a model (weights + KV cache + activations).
 */
export function estimateVRAM(config: ModelConfig, opts: {
  bitsPerWeight?: number;   // 4 for INT4, 16 for FP16, 32 for FP32
  seqLength?: number;       // context length
  kvBits?: number;          // KV cache bits per value (3-4 for TurboQuant, 16 for FP16)
} = {}): { weightsBytes: number; kvCacheBytes: number; activationBytes: number; totalBytes: number } {
  const bpw = opts.bitsPerWeight ?? 4;
  const seq = opts.seqLength ?? 2048;
  const kvBits = opts.kvBits ?? 16;

  // Weight parameters (approximate)
  const embParams = config.vocabSize * config.hiddenSize;
  const layerParams = config.numLayers * (
    // attention: Q, K, V, O projections
    config.hiddenSize * config.numAttentionHeads * config.headDim +   // Q
    config.hiddenSize * config.numKVHeads * config.headDim +          // K
    config.hiddenSize * config.numKVHeads * config.headDim +          // V
    config.numAttentionHeads * config.headDim * config.hiddenSize +   // O
    // FFN: gate, up, down
    config.hiddenSize * config.intermediateSize * 2 +                  // gate + up
    config.intermediateSize * config.hiddenSize +                      // down
    // norms
    config.hiddenSize * 2
  );
  const lmHeadParams = config.tieWordEmbeddings ? 0 : config.vocabSize * config.hiddenSize;
  const totalParams = embParams + layerParams + lmHeadParams;

  const weightsBytes = Math.ceil(totalParams * bpw / 8);

  // KV cache: 2 (K+V) × layers × seq × kv_heads × head_dim
  const kvElements = 2 * config.numLayers * seq * config.numKVHeads * config.headDim;
  const kvCacheBytes = Math.ceil(kvElements * kvBits / 8);

  // Activation memory (peak during forward pass, single layer)
  const activationBytes = (
    config.hiddenSize * seq +                              // input
    config.numAttentionHeads * seq * seq +                 // attention scores
    config.intermediateSize * seq * 2                      // FFN intermediates
  ) * 4; // f32

  return {
    weightsBytes,
    kvCacheBytes,
    activationBytes,
    totalBytes: weightsBytes + kvCacheBytes + activationBytes,
  };
}

// ── Helpers ─────────────────────────────────────────────────────────────

function firstDefined(obj: Record<string, any>, keys: string[]): any {
  for (const key of keys) {
    if (obj[key] !== undefined) return obj[key];
  }
  return undefined;
}
