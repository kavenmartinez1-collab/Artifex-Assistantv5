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

  // ── Quantization ────────────────────────────────────────────────────
  /** Quantization method: 'none', 'gptq', 'awq' */
  quantMethod: string;
  /** Quantization bits (4 for GPTQ INT4) */
  quantBits: number;
  /** Quantization group size (typically 128) */
  quantGroupSize: number;
  /** Whether Hadamard rotation was applied during quantization (QuIP#-style) */
  quantHadamard?: boolean;
  /** Whether KLT rotation was fused offline (MambaQuant-style, zero runtime cost) */
  quantKLT?: boolean;
  /** Whether model contains E8 lattice 2-bit quantized layers */
  hasE8?: boolean;
  /** Whether model contains INT8 quantized layers */
  hasQ8?: boolean;

  // ── Hybrid (Mamba-2 / Gated DeltaNet) ──────────────────────────────
  /** Per-layer type: 'full_attention' or 'linear_attention'. Undefined for pure transformers. */
  layerTypes?: string[];
  /** Whether this model has mixed attention types (e.g., Qwen3.5) */
  isHybrid: boolean;
  /** Linear attention key head dimension (e.g., 128) */
  linearKeyHeadDim?: number;
  /** Linear attention value head dimension (e.g., 128) */
  linearValueHeadDim?: number;
  /** Number of key heads in linear attention layers (e.g., 16) */
  linearNumKeyHeads?: number;
  /** Number of value heads in linear attention layers (e.g., 32) */
  linearNumValueHeads?: number;
  /** Conv1d kernel size for linear attention (e.g., 4) */
  linearConvKernelDim?: number;
  /** Fraction of head dims that get RoPE (e.g., 0.25 for Qwen3.5) */
  partialRotaryFactor?: number;
  /** Whether attention output is gated (linear + full attention) */
  attnOutputGate?: boolean;

  // ── Derived ────────────────────────────────────────────────────────
  /** Number of Q heads per KV head group (for GQA). = numAttentionHeads / numKVHeads */
  numQPerKV: number;
  /** Whether this model uses grouped-query attention */
  isGQA: boolean;
  /** Whether this model uses quantized weights */
  isQuantized: boolean;
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
    /** Q/K per-head RMSNorm weights (Qwen3.5 full attention only) */
    qNorm?: string;
    kNorm?: string;
    /** Post-attention layernorm (pre-FFN) */
    postAttnNorm: string;
    /** FFN gate projection (SwiGLU) */
    gateProj: string;
    /** FFN up projection */
    upProj: string;
    /** FFN down projection */
    downProj: string;
  };

  /** Linear attention (Gated DeltaNet / Mamba-2) weight patterns. Only for hybrid models. */
  linearLayer?: {
    inProjQKV: string;     // fused Q/K/V projection
    inProjA: string;       // SSM decay gate input projection
    inProjB: string;       // SSM update gate input projection
    inProjZ: string;       // output gate projection
    outProj: string;       // output projection
    aLog: string;          // diagonal state decay matrix (log-space)
    conv1dWeight: string;  // causal conv1d kernel
    dtBias: string;        // time step bias
    normWeight: string;    // group norm weight
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
      qNorm: 'model.layers.{L}.self_attn.q_norm.weight',
      kNorm: 'model.layers.{L}.self_attn.k_norm.weight',
      postAttnNorm: 'model.layers.{L}.post_attention_layernorm.weight',
      gateProj: 'model.layers.{L}.mlp.gate_proj.weight',
      upProj: 'model.layers.{L}.mlp.up_proj.weight',
      downProj: 'model.layers.{L}.mlp.down_proj.weight',
    },
    linearLayer: {
      inProjQKV: 'model.layers.{L}.linear_attn.in_proj_qkv.weight',
      inProjA: 'model.layers.{L}.linear_attn.in_proj_a.weight',
      inProjB: 'model.layers.{L}.linear_attn.in_proj_b.weight',
      inProjZ: 'model.layers.{L}.linear_attn.in_proj_z.weight',
      outProj: 'model.layers.{L}.linear_attn.out_proj.weight',
      aLog: 'model.layers.{L}.linear_attn.A_log',
      conv1dWeight: 'model.layers.{L}.linear_attn.conv1d.weight',
      dtBias: 'model.layers.{L}.linear_attn.dt_bias',
      normWeight: 'model.layers.{L}.linear_attn.norm.weight',
    },
  },
  // Qwen3.5 adds 'language_model' prefix to all weight paths
  qwen3_5_text: {
    embedTokens: 'model.language_model.embed_tokens.weight',
    finalNorm: 'model.language_model.norm.weight',
    lmHead: 'lm_head.weight',
    layer: {
      inputNorm: 'model.language_model.layers.{L}.input_layernorm.weight',
      qProj: 'model.language_model.layers.{L}.self_attn.q_proj.weight',
      kProj: 'model.language_model.layers.{L}.self_attn.k_proj.weight',
      vProj: 'model.language_model.layers.{L}.self_attn.v_proj.weight',
      oProj: 'model.language_model.layers.{L}.self_attn.o_proj.weight',
      qBias: 'model.language_model.layers.{L}.self_attn.q_proj.bias',
      kBias: 'model.language_model.layers.{L}.self_attn.k_proj.bias',
      vBias: 'model.language_model.layers.{L}.self_attn.v_proj.bias',
      oBias: 'model.language_model.layers.{L}.self_attn.o_proj.bias',
      qNorm: 'model.language_model.layers.{L}.self_attn.q_norm.weight',
      kNorm: 'model.language_model.layers.{L}.self_attn.k_norm.weight',
      postAttnNorm: 'model.language_model.layers.{L}.post_attention_layernorm.weight',
      gateProj: 'model.language_model.layers.{L}.mlp.gate_proj.weight',
      upProj: 'model.language_model.layers.{L}.mlp.up_proj.weight',
      downProj: 'model.language_model.layers.{L}.mlp.down_proj.weight',
    },
    linearLayer: {
      inProjQKV: 'model.language_model.layers.{L}.linear_attn.in_proj_qkv.weight',
      inProjA: 'model.language_model.layers.{L}.linear_attn.in_proj_a.weight',
      inProjB: 'model.language_model.layers.{L}.linear_attn.in_proj_b.weight',
      inProjZ: 'model.language_model.layers.{L}.linear_attn.in_proj_z.weight',
      outProj: 'model.language_model.layers.{L}.linear_attn.out_proj.weight',
      aLog: 'model.language_model.layers.{L}.linear_attn.A_log',
      conv1dWeight: 'model.language_model.layers.{L}.linear_attn.conv1d.weight',
      dtBias: 'model.language_model.layers.{L}.linear_attn.dt_bias',
      normWeight: 'model.language_model.layers.{L}.linear_attn.norm.weight',
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
  // Qwen3.5 multimodal wraps text config under text_config — flatten it.
  // Trigger when text_config is present AND top-level model_type is the multimodal
  // wrapper ('qwen3_5'); some configs (e.g., HailMary) duplicate hidden_size at
  // top level, so don't rely on !hidden_size as the gate.
  if (hfConfig.text_config && hfConfig.model_type === 'qwen3_5') {
    const textCfg = hfConfig.text_config;
    // Merge text_config fields into top level (text_config takes priority)
    hfConfig = { ...hfConfig, ...textCfg };
    // Fix model_type: qwen3_5 → qwen3_5_text (our internal convention)
    if (hfConfig.model_type === 'qwen3_5' || (hfConfig as any).model_type === 'qwen3_5') {
      hfConfig.model_type = 'qwen3_5_text';
    }
    console.log(`[Config] Flattened text_config for ${hfConfig.model_type}`);
  }

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
  const ropeTheta = firstDefined(hfConfig, fieldMap.ropeTheta)
    ?? hfConfig.rope_parameters?.rope_theta  // Qwen3.5 nests rope_theta inside rope_parameters
    ?? 10000;
  const headDim = firstDefined(hfConfig, fieldMap.headDim) ?? Math.floor(hiddenSize / numAttentionHeads);

  const rmsNormEps = hfConfig.rms_norm_eps ?? hfConfig.layer_norm_eps ?? 1e-5;
  const maxPositionEmbeddings = hfConfig.max_position_embeddings ?? 4096;
  const hiddenAct = hfConfig.hidden_act ?? hfConfig.activation_function ?? 'silu';
  // attention_bias defaults depend on model family:
  // Qwen2: true (implicit, not in config.json)
  // Llama/Mistral/Gemma: false
  const attentionBiasDefaults: Record<string, boolean> = {
    qwen2: true,
    qwen2_moe: true,
    qwen3: true,
    qwen3_5_text: false,  // Qwen3.5 explicitly sets attention_bias: false
    qwen3_moe: true,
  };
  const attentionBias = hfConfig.attention_bias ?? attentionBiasDefaults[modelType] ?? false;
  const tieWordEmbeddings = hfConfig.tie_word_embeddings ?? false;

  const numQPerKV = Math.floor(numAttentionHeads / numKVHeads);

  // Quantization detection (GPTQ, AWQ)
  const quantConfig = hfConfig.quantization_config;
  const quantMethod = quantConfig?.quant_method ?? 'none';
  const quantBits = quantConfig?.bits ?? 0;
  const quantGroupSize = quantConfig?.group_size ?? 128;

  // Hybrid model detection (Mamba-2 / Gated DeltaNet)
  const layerTypes: string[] | undefined = hfConfig.layer_types;
  const isHybrid = Array.isArray(layerTypes) && layerTypes.includes('linear_attention');

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
    quantMethod,
    quantBits,
    quantGroupSize,
    quantHadamard: quantConfig?.hadamard ?? false,
    quantKLT: quantConfig?.klt ?? false,
    hasE8: quantConfig?.has_e8 ?? false,
    hasQ8: quantConfig?.has_q8 ?? false,
    numQPerKV,
    isGQA: numKVHeads !== numAttentionHeads,
    isQuantized: quantMethod !== 'none',
    // Hybrid model fields
    layerTypes,
    isHybrid,
    linearKeyHeadDim: isHybrid ? (hfConfig.linear_key_head_dim ?? 128) : undefined,
    linearValueHeadDim: isHybrid ? (hfConfig.linear_value_head_dim ?? 128) : undefined,
    linearNumKeyHeads: isHybrid ? (hfConfig.linear_num_key_heads ?? 16) : undefined,
    linearNumValueHeads: isHybrid ? (hfConfig.linear_num_value_heads ?? 32) : undefined,
    linearConvKernelDim: isHybrid ? (hfConfig.linear_conv_kernel_dim ?? 4) : undefined,
    partialRotaryFactor: hfConfig.partial_rotary_factor
      ?? hfConfig.rope_parameters?.partial_rotary_factor,
    attnOutputGate: hfConfig.attn_output_gate,
  };
}

/**
 * Get the weight name mapping for a given model type.
 */
export function getWeightNameMap(modelType: string): WeightNameMap {
  return WEIGHT_NAME_PATTERNS[modelType] ?? WEIGHT_NAME_PATTERNS.default;
}

/**
 * Auto-detect the weight name mapping by checking which tensor names exist.
 * Falls back to model-type lookup if auto-detect fails.
 * This handles models with non-standard prefixes (e.g., multimodal models
 * that nest text weights under `model.language_model.`).
 */
export function autoDetectWeightNameMap(
  modelType: string,
  tensorNames: Set<string> | Map<string, any>,
): WeightNameMap {
  const has = (name: string) =>
    tensorNames instanceof Set ? tensorNames.has(name) : tensorNames.has(name);

  // Try model-type specific map first
  // Also check for quantized variant (.qweight) since embed may be INT4
  const typeMap = WEIGHT_NAME_PATTERNS[modelType];
  if (typeMap && (has(typeMap.embedTokens) || has(typeMap.embedTokens.replace('.weight', '.qweight')))) return typeMap;

  // Try default pattern
  const defaultMap = WEIGHT_NAME_PATTERNS.default;
  if (has(defaultMap.embedTokens) || has(defaultMap.embedTokens.replace('.weight', '.qweight'))) return defaultMap;

  // Auto-detect: find embed_tokens with any prefix (supports .weight or .qweight)
  for (const name of (tensorNames instanceof Set ? tensorNames : tensorNames.keys())) {
    if (name.endsWith('embed_tokens.weight') || name.endsWith('embed_tokens.qweight')) {
      const prefix = name.replace(/embed_tokens\.(weight|qweight)$/, '');
      console.log(`[ModelConfig] Auto-detected weight prefix: "${prefix}"`);
      // Build a name map by prepending the prefix to the default pattern
      const base = WEIGHT_NAME_PATTERNS.default;
      const reprefix = (s: string) => s.replace('model.', prefix);
      const result: any = {
        embedTokens: reprefix(base.embedTokens),
        finalNorm: reprefix(base.finalNorm),
        lmHead: base.lmHead, // lm_head is usually at the top level
        layer: {
          inputNorm: reprefix(base.layer.inputNorm),
          qProj: reprefix(base.layer.qProj),
          kProj: reprefix(base.layer.kProj),
          vProj: reprefix(base.layer.vProj),
          oProj: reprefix(base.layer.oProj),
          qBias: reprefix(base.layer.qBias),
          kBias: reprefix(base.layer.kBias),
          vBias: reprefix(base.layer.vBias),
          oBias: reprefix(base.layer.oBias),
          qNorm: base.layer.qNorm ? reprefix(base.layer.qNorm) : undefined,
          kNorm: base.layer.kNorm ? reprefix(base.layer.kNorm) : undefined,
          postAttnNorm: reprefix(base.layer.postAttnNorm),
          gateProj: reprefix(base.layer.gateProj),
          upProj: reprefix(base.layer.upProj),
          downProj: reprefix(base.layer.downProj),
        },
      };
      if (base.linearLayer) {
        const lin = base.linearLayer;
        result.linearLayer = {
          inProjQKV: reprefix(lin.inProjQKV),
          inProjA: reprefix(lin.inProjA),
          inProjB: reprefix(lin.inProjB),
          inProjZ: reprefix(lin.inProjZ),
          outProj: reprefix(lin.outProj),
          aLog: reprefix(lin.aLog),
          conv1dWeight: reprefix(lin.conv1dWeight),
          dtBias: reprefix(lin.dtBias),
          normWeight: reprefix(lin.normWeight),
        };
      }
      return result;
    }
  }

  // Final fallback
  return getWeightNameMap(modelType);
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
      const pattern = map.layer[key];
      if (pattern) names.push(resolveLayerWeightName(pattern, l));
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
