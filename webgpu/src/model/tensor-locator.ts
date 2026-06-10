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
  // per-layer: Gemma 4 per-layer embeddings (Phase D)
  | 'perLayerEmbed';

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
