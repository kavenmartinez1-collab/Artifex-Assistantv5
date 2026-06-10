/**
 * TensorLocator — canonical tensor roles → storage-specific tensor names.
 *
 * The engine bridges weights by ROLE (qProj, moeRouter, linInProjQKV, ...)
 * instead of doing string surgery on family-specific names. Each weight
 * source provides a locator:
 *   - HF/safetensors: wraps the WEIGHT_NAME_PATTERNS maps + prefix auto-detect
 *   - GGUF (Phase B): blk.{L}.attn_q.weight, ffn_gate_exps, token_embd, ...
 *
 * locate() returns undefined when the family/model has no tensor for a role
 * (e.g. qNorm on Phi, MoE roles on dense models).
 */

import { autoDetectWeightNameMap, resolveLayerWeightName, type WeightNameMap } from './model-config';

export type TensorRole =
  // global
  | 'embedTokens' | 'finalNorm' | 'lmHead'
  // per-layer: norms
  | 'inputNorm' | 'postAttnNorm'
  // per-layer: softmax attention
  | 'qProj' | 'kProj' | 'vProj' | 'oProj'
  | 'qBias' | 'kBias' | 'vBias' | 'oBias'
  | 'qNorm' | 'kNorm'
  // per-layer: dense FFN
  | 'gateProj' | 'upProj' | 'downProj'
  // per-layer: Gated DeltaNet (linear attention)
  | 'linInProjQKV' | 'linInProjA' | 'linInProjB' | 'linInProjZ' | 'linOutProj'
  | 'linALog' | 'linConv1dWeight' | 'linDtBias' | 'linNormWeight'
  // per-layer: MoE (Phase C — GGUF locator only)
  | 'moeRouter' | 'moeExpertGate' | 'moeExpertUp' | 'moeExpertDown'
  | 'moeSharedGateProj' | 'moeSharedUpProj' | 'moeSharedDownProj' | 'moeSharedExpertGate'
  // per-layer: Gemma 4 sandwich norms + layer output scale (Phase D)
  | 'attnPostNorm' | 'ffnPostNorm' | 'layerOutScale'
  // per-layer: Gemma 4 per-layer embeddings (Phase D)
  | 'pleInpGate' | 'pleProj' | 'plePostNorm'
  // global: Gemma 4 PLE projections + RoPE frequency factors (Phase D)
  | 'pleTokenEmbed' | 'pleModelProj' | 'pleProjNorm' | 'ropeFreqs';

export interface TensorLocator {
  /**
   * Canonical tensor name for a role. Per-layer roles require `layer`.
   * Returns undefined if this model family has no tensor for the role.
   */
  locate(role: TensorRole, layer?: number): string | undefined;
}

// Per-layer roles backed by WeightNameMap.layer
const LAYER_ROLE_KEYS: Partial<Record<TensorRole, keyof WeightNameMap['layer']>> = {
  inputNorm: 'inputNorm',
  postAttnNorm: 'postAttnNorm',
  qProj: 'qProj',
  kProj: 'kProj',
  vProj: 'vProj',
  oProj: 'oProj',
  qBias: 'qBias',
  kBias: 'kBias',
  vBias: 'vBias',
  oBias: 'oBias',
  qNorm: 'qNorm',
  kNorm: 'kNorm',
  gateProj: 'gateProj',
  upProj: 'upProj',
  downProj: 'downProj',
};

// Per-layer roles backed by WeightNameMap.linearLayer
const LINEAR_ROLE_KEYS: Partial<Record<TensorRole, keyof NonNullable<WeightNameMap['linearLayer']>>> = {
  linInProjQKV: 'inProjQKV',
  linInProjA: 'inProjA',
  linInProjB: 'inProjB',
  linInProjZ: 'inProjZ',
  linOutProj: 'outProj',
  linALog: 'aLog',
  linConv1dWeight: 'conv1dWeight',
  linDtBias: 'dtBias',
  linNormWeight: 'normWeight',
};

/**
 * HF/safetensors locator. Auto-detects weight-name prefixes from the actual
 * tensor list (same behavior as the legacy autoDetectWeightNameMap path).
 */
export function createHFLocator(
  modelType: string,
  tensorNames: Set<string> | Map<string, unknown>,
): TensorLocator {
  const nameMap = autoDetectWeightNameMap(modelType, tensorNames);

  return {
    locate(role: TensorRole, layer?: number): string | undefined {
      switch (role) {
        case 'embedTokens': return nameMap.embedTokens;
        case 'finalNorm': return nameMap.finalNorm;
        case 'lmHead': return nameMap.lmHead;
      }
      if (layer === undefined) {
        throw new Error(`TensorLocator: role "${role}" requires a layer index`);
      }
      const layerKey = LAYER_ROLE_KEYS[role];
      if (layerKey) {
        const pattern = nameMap.layer[layerKey];
        return pattern ? resolveLayerWeightName(pattern, layer) : undefined;
      }
      const linKey = LINEAR_ROLE_KEYS[role];
      if (linKey) {
        const pattern = nameMap.linearLayer?.[linKey];
        return pattern ? resolveLayerWeightName(pattern, layer) : undefined;
      }
      // MoE / PLE roles: not present in safetensors families we support yet.
      return undefined;
    },
  };
}

// ── GGUF locator ───────────────────────────────────────────────────────

/** Global GGUF tensor names. */
const GGUF_GLOBAL: Partial<Record<TensorRole, string>> = {
  embedTokens: 'token_embd.weight',
  finalNorm: 'output_norm.weight',
  lmHead: 'output.weight',
  // Gemma 4 PLE + RoPE frequency factors (Phase D)
  pleTokenEmbed: 'per_layer_token_embd.weight',
  pleModelProj: 'per_layer_model_proj.weight',
  pleProjNorm: 'per_layer_proj_norm.weight',
  ropeFreqs: 'rope_freqs.weight',
};

/**
 * Per-layer GGUF name suffixes ('blk.{L}.' prefix added at locate time).
 * Mapping verified against llama.cpp gguf-py/gguf/tensor_mapping.py:
 *   ssm_alpha ← linear_attn.in_proj_a, ssm_beta ← in_proj_b,
 *   attn_qkv ← in_proj_qkv, attn_gate ← in_proj_z, ssm_a ← A_log (no .weight),
 *   ssm_dt.bias ← dt_bias.
 * Arrays = fallback chain (first name present in the file wins).
 */
const GGUF_LAYER: Partial<Record<TensorRole, string[]>> = {
  inputNorm: ['attn_norm.weight'],
  postAttnNorm: ['post_attention_norm.weight', 'ffn_norm.weight'],
  qProj: ['attn_q.weight'],
  kProj: ['attn_k.weight'],
  vProj: ['attn_v.weight'],
  oProj: ['attn_output.weight'],
  qBias: ['attn_q.bias'],
  kBias: ['attn_k.bias'],
  vBias: ['attn_v.bias'],
  oBias: ['attn_output.bias'],
  qNorm: ['attn_q_norm.weight'],
  kNorm: ['attn_k_norm.weight'],
  gateProj: ['ffn_gate.weight'],
  upProj: ['ffn_up.weight'],
  downProj: ['ffn_down.weight'],
  // Gated DeltaNet
  linInProjQKV: ['attn_qkv.weight'],
  linInProjA: ['ssm_alpha.weight'],
  linInProjB: ['ssm_beta.weight'],
  linInProjZ: ['attn_gate.weight'],
  linOutProj: ['ssm_out.weight'],
  linALog: ['ssm_a'],
  linConv1dWeight: ['ssm_conv1d.weight'],
  linDtBias: ['ssm_dt.bias'],
  linNormWeight: ['ssm_norm.weight'],
  // MoE
  moeRouter: ['ffn_gate_inp.weight'],
  moeExpertGate: ['ffn_gate_exps.weight'],
  moeExpertUp: ['ffn_up_exps.weight'],
  moeExpertDown: ['ffn_down_exps.weight'],
  moeSharedGateProj: ['ffn_gate_shexp.weight'],
  moeSharedUpProj: ['ffn_up_shexp.weight'],
  moeSharedDownProj: ['ffn_down_shexp.weight'],
  moeSharedExpertGate: ['ffn_gate_inp_shexp.weight'],
  // Gemma 4 sandwich norms + layer output scale + PLE (Phase D)
  ffnPostNorm: ['post_ffw_norm.weight'],
  layerOutScale: ['layer_output_scale.weight'],
  pleInpGate: ['inp_gate.weight'],
  pleProj: ['proj.weight'],
  plePostNorm: ['post_norm.weight'],
};

/**
 * Arch-specific role overrides (checked before GGUF_LAYER).
 *
 * gemma4: 'post_attention_norm.weight' is the SANDWICH post-attention norm
 * (applied to attn output before the residual add), and 'ffn_norm.weight'
 * is the pre-FFN norm. Other archs (e.g. qwen35) use 'post_attention_norm'
 * AS the pre-FFN norm — hence the default fallback chain in GGUF_LAYER.
 */
const GGUF_LAYER_ARCH: Record<string, Partial<Record<TensorRole, string[]>>> = {
  gemma4: {
    postAttnNorm: ['ffn_norm.weight'],
    attnPostNorm: ['post_attention_norm.weight'],
  },
};

/**
 * GGUF locator. Resolves roles to blk.{L}.* / global GGUF tensor names,
 * returning undefined when the tensor isn't present in this file.
 */
export function createGGUFLocator(
  tensorNames: Set<string> | Map<string, unknown>,
  arch?: string,
): TensorLocator {
  const has = (name: string) => tensorNames.has(name);
  const archOverrides = arch ? GGUF_LAYER_ARCH[arch] : undefined;

  return {
    locate(role: TensorRole, layer?: number): string | undefined {
      const globalName = GGUF_GLOBAL[role];
      if (globalName !== undefined) {
        if (role === 'lmHead' && !has(globalName)) {
          // Tied embeddings: llama.cpp dups output from token_embd
          return GGUF_GLOBAL.embedTokens;
        }
        return has(globalName) ? globalName : undefined;
      }
      if (layer === undefined) {
        throw new Error(`TensorLocator: role "${role}" requires a layer index`);
      }
      const suffixes = archOverrides?.[role] ?? GGUF_LAYER[role];
      if (!suffixes) return undefined;
      for (const suffix of suffixes) {
        const name = `blk.${layer}.${suffix}`;
        if (has(name)) return name;
      }
      return undefined;
    },
  };
}
