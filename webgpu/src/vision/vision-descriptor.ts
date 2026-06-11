/**
 * VisionDescriptor — the model-agnostic contract for vision towers.
 *
 * The vision twin of ModelDescriptor: one parameterized description that the
 * (single) ViT runner consumes, built from either an HF config.json
 * (visionDescriptorFromHFConfig) or a GGUF mmproj / inline v.* header
 * (visionDescriptorFromGGUF, Phase M2).
 *
 * Families share far more than they differ:
 *   - Qwen3-VL:   24×1024 ViT, fused QKV, LayerNorm, GELU-tanh, learned pos
 *                 embeds, 2×2 spatial-merge MLP projector, DeepStack taps.
 *   - Qwen3.5/3.6 multimodal: SigLIP-shaped tower (27×1152) + projector,
 *                 dynamic token counts like Qwen3-VL.
 *   - Gemma 4:    SigLIP-shaped tower, average-pool + linear projector,
 *                 FIXED tokens per image, bidirectional attention over the
 *                 image span on the text side.
 * Where a family's details are not yet parity-verified, the builder marks
 * them via `verified: false` — the loader warns instead of silently
 * producing garbage embeddings.
 */

import type { GGUFFile } from '../model/gguf';

export interface VisionDescriptor {
  family: 'qwen3_vl' | 'qwen_vl_siglip' | 'gemma4' | 'unknown';
  /** Builder confidence: false = best-effort defaults, needs parity run. */
  verified: boolean;

  // ── Tower ──────────────────────────────────────────────────────────
  depth: number;
  hiddenSize: number;
  numHeads: number;
  intermediateSize: number;
  activation: 'gelu_tanh' | 'gelu' | 'silu';
  norm: 'layernorm' | 'rmsnorm';
  /** Projections/norms carry bias terms (true for Qwen3-VL + SigLIP). */
  hasBias: boolean;
  /** Attention blocks store one fused QKV weight (HF Qwen) vs separate
   *  q/k/v tensors (GGUF clip.cpp convention). The locator handles naming;
   *  this flags which compute path the runner takes. */
  fusedQKV: boolean;

  // ── Patchification ─────────────────────────────────────────────────
  patchSize: number;
  /** Frames per temporal patch — images duplicate their single frame. */
  temporalPatchSize: number;
  inChannels: number;

  posEmbed:
    | { kind: 'learned'; count: number; gridSize: number }  // interpolated for other grids
    | { kind: 'none' };

  // ── Projector: tower output → text-embedding-space tokens ─────────
  projector:
    | { kind: 'qwen_merger'; spatialMergeSize: number; outHiddenSize: number }
    | { kind: 'pool_linear'; outTokens: number; outHiddenSize: number }   // Gemma
    | { kind: 'mlp'; outHiddenSize: number };

  /** Qwen3-VL DeepStack: intermediate tower layers whose (merged) features
   *  are residual-added to the first text layers at image positions. */
  deepstackIndexes: number[];

  // ── Text-side placement ────────────────────────────────────────────
  placeholder: {
    imageTokenId: number;
    startTokenId?: number;
    endTokenId?: number;
    /** Fixed token count per image (Gemma). Undefined = grid-derived. */
    fixedTokens?: number;
  };

  // ── Preprocessing ──────────────────────────────────────────────────
  preprocess: {
    imageMean: [number, number, number];
    imageStd: [number, number, number];
    resize:
      | { kind: 'smart'; minPixels: number; maxPixels: number; factor: number }
      | { kind: 'fixed'; width: number; height: number };
  };
}

/**
 * Build a VisionDescriptor from an HF config.json (+ optional
 * preprocessor_config.json). Returns null for text-only models.
 */
export function visionDescriptorFromHFConfig(
  hfConfig: Record<string, any>,
  preprocessorConfig?: Record<string, any>,
): VisionDescriptor | null {
  const vc = hfConfig.vision_config;
  if (!vc) return null;
  const modelType = String(hfConfig.model_type ?? '').toLowerCase();
  const pp = preprocessorConfig ?? {};

  // ── Qwen3-VL (ground truth from Qwen3-VL-4B-Instruct) ─────────────
  if (modelType === 'qwen3_vl' || vc.model_type === 'qwen3_vl') {
    const patch = vc.patch_size ?? 16;
    const merge = vc.spatial_merge_size ?? 2;
    const factor = patch * merge;
    return {
      family: 'qwen3_vl',
      verified: true,
      depth: vc.depth ?? 24,
      hiddenSize: vc.hidden_size ?? 1024,
      numHeads: vc.num_heads ?? 16,
      intermediateSize: vc.intermediate_size ?? 4096,
      activation: vc.hidden_act === 'gelu_pytorch_tanh' ? 'gelu_tanh' : 'gelu',
      norm: 'layernorm',
      hasBias: true,
      fusedQKV: true,
      patchSize: patch,
      temporalPatchSize: vc.temporal_patch_size ?? 2,
      inChannels: vc.in_channels ?? 3,
      posEmbed: {
        kind: 'learned',
        count: vc.num_position_embeddings ?? 2304,
        gridSize: Math.round(Math.sqrt(vc.num_position_embeddings ?? 2304)),
      },
      projector: {
        kind: 'qwen_merger',
        spatialMergeSize: merge,
        outHiddenSize: vc.out_hidden_size ?? hfConfig.hidden_size,
      },
      deepstackIndexes: vc.deepstack_visual_indexes ?? [],
      placeholder: {
        imageTokenId: hfConfig.image_token_id ?? 151655,
        startTokenId: hfConfig.vision_start_token_id ?? 151652,
        endTokenId: hfConfig.vision_end_token_id ?? 151653,
      },
      preprocess: {
        imageMean: (pp.image_mean as [number, number, number]) ?? [0.5, 0.5, 0.5],
        imageStd: (pp.image_std as [number, number, number]) ?? [0.5, 0.5, 0.5],
        resize: {
          kind: 'smart',
          minPixels: pp.size?.shortest_edge ?? pp.min_pixels ?? 65536,
          maxPixels: pp.size?.longest_edge ?? pp.max_pixels ?? 16777216,
          factor,
        },
      },
    };
  }

  // ── Qwen3.5 / Qwen3.6 multimodal (SigLIP-shaped tower) ─────────────
  // Tower dims come from the config; projector/token details are NOT yet
  // parity-verified — flagged so the load path warns.
  if (modelType.startsWith('qwen3_5') || modelType.startsWith('qwen3_6')) {
    const patch = vc.patch_size ?? 16;
    const merge = vc.spatial_merge_size ?? 2;
    return {
      family: 'qwen_vl_siglip',
      verified: false,
      depth: vc.depth ?? 27,
      hiddenSize: vc.hidden_size ?? 1152,
      numHeads: vc.num_heads ?? vc.num_attention_heads ?? 16,
      intermediateSize: vc.intermediate_size ?? 4304,
      activation: 'gelu_tanh',
      norm: 'layernorm',
      hasBias: true,
      fusedQKV: true,
      patchSize: patch,
      temporalPatchSize: vc.temporal_patch_size ?? 2,
      inChannels: vc.in_channels ?? 3,
      posEmbed: vc.num_position_embeddings
        ? { kind: 'learned', count: vc.num_position_embeddings, gridSize: Math.round(Math.sqrt(vc.num_position_embeddings)) }
        : { kind: 'none' },
      projector: {
        kind: 'qwen_merger',
        spatialMergeSize: merge,
        outHiddenSize: vc.out_hidden_size ?? hfConfig.hidden_size ?? 0,
      },
      deepstackIndexes: vc.deepstack_visual_indexes ?? [],
      placeholder: {
        imageTokenId: hfConfig.image_token_id ?? 0,
        startTokenId: hfConfig.vision_start_token_id,
        endTokenId: hfConfig.vision_end_token_id,
      },
      preprocess: {
        imageMean: (pp.image_mean as [number, number, number]) ?? [0.5, 0.5, 0.5],
        imageStd: (pp.image_std as [number, number, number]) ?? [0.5, 0.5, 0.5],
        resize: {
          kind: 'smart',
          minPixels: pp.size?.shortest_edge ?? pp.min_pixels ?? 65536,
          maxPixels: pp.size?.longest_edge ?? pp.max_pixels ?? 16777216,
          factor: patch * merge,
        },
      },
    };
  }

  return visionDescriptorFromGemmaHF(hfConfig, vc, pp);
}

/**
 * Build a VisionDescriptor from a GGUF mmproj header (clip.* metadata).
 * Currently supports projector_type 'qwen3vl_merger' — the same tower
 * architecture parity-verified on the HF qwen3-vl path, at whatever dims
 * the checkpoint declares. Returns null for unsupported projector types
 * (gemma's gated-MLP tower is a different compute graph — its own arc).
 *
 * placeholder.imageTokenId is left 0 — the mmproj knows nothing about the
 * text vocab; the caller resolves '<|image_pad|>' through the tokenizer.
 */
export function visionDescriptorFromGGUF(file: GGUFFile): VisionDescriptor | null {
  if (file.kv.get('clip.has_vision_encoder') !== true) return null;
  const kv = <T>(k: string, fb: T): T => (file.kv.get(k) as T) ?? fb;
  const projType = kv<string>('clip.projector_type', '');
  if (projType !== 'qwen3vl_merger') {
    console.warn(`[Vision] unsupported GGUF projector_type "${projType}" — vision disabled for this model`);
    return null;
  }
  const patch = kv('clip.vision.patch_size', 16);
  const merge = kv('clip.vision.spatial_merge_size', 2);
  const hidden = kv('clip.vision.embedding_length', 1152);
  const dsFlags = kv<boolean[]>('clip.vision.is_deepstack_layers', []);
  const posT = file.tensors.get('v.position_embd.weight');
  const posCount = posT ? posT.shape[0] : 0;
  return {
    family: 'qwen3_vl',
    // Same compute graph as the parity-verified HF qwen3-vl tower; dims are
    // checkpoint-declared. End-to-end quality still owed a reference check.
    verified: true,
    depth: kv('clip.vision.block_count', 27),
    hiddenSize: hidden,
    numHeads: kv('clip.vision.attention.head_count', 16),
    intermediateSize: kv('clip.vision.feed_forward_length', 4304),
    activation: 'gelu_tanh',
    norm: 'layernorm',
    hasBias: true,
    fusedQKV: true,
    patchSize: patch,
    // The GGUF stores the conv's two temporal slices as separate tensors;
    // the loader sums them (frames are duplicated for images), so the
    // browser pipeline runs with T=1 and patchDim = C*P*P.
    temporalPatchSize: 1,
    inChannels: 3,
    posEmbed: posCount > 0
      ? { kind: 'learned', count: posCount, gridSize: Math.round(Math.sqrt(posCount)) }
      : { kind: 'none' },
    projector: {
      kind: 'qwen_merger',
      spatialMergeSize: merge,
      outHiddenSize: kv('clip.vision.projection_dim', 0),
    },
    deepstackIndexes: dsFlags.map((v, i) => (v ? i : -1)).filter(i => i >= 0),
    placeholder: { imageTokenId: 0 },  // resolved via tokenizer by the caller
    preprocess: {
      imageMean: kv<[number, number, number]>('clip.vision.image_mean', [0.5, 0.5, 0.5]),
      imageStd: kv<[number, number, number]>('clip.vision.image_std', [0.5, 0.5, 0.5]),
      resize: { kind: 'smart', minPixels: 65536, maxPixels: 16777216, factor: patch * merge },
    },
  };
}

function visionDescriptorFromGemmaHF(
  hfConfig: Record<string, any>,
  vc: Record<string, any>,
  pp: Record<string, any>,
): VisionDescriptor | null {
  const modelType = String(hfConfig.model_type ?? '').toLowerCase();
  // ── Gemma 4 (SigLIP tower, pool+linear projector, fixed tokens) ────
  if (modelType.startsWith('gemma')) {
    return {
      family: 'gemma4',
      verified: false,
      depth: vc.num_hidden_layers ?? vc.depth ?? 27,
      hiddenSize: vc.hidden_size ?? 1152,
      numHeads: vc.num_attention_heads ?? vc.num_heads ?? 16,
      intermediateSize: vc.intermediate_size ?? 4304,
      activation: 'gelu_tanh',
      norm: 'layernorm',
      hasBias: true,
      fusedQKV: false,
      patchSize: vc.patch_size ?? 14,
      temporalPatchSize: 1,
      inChannels: vc.num_channels ?? 3,
      posEmbed: { kind: 'learned', count: vc.num_positions ?? 4096, gridSize: Math.round(Math.sqrt(vc.num_positions ?? 4096)) },
      projector: {
        kind: 'pool_linear',
        outTokens: hfConfig.mm_tokens_per_image ?? 256,
        outHiddenSize: hfConfig.text_config?.hidden_size ?? hfConfig.hidden_size ?? 0,
      },
      deepstackIndexes: [],
      placeholder: {
        imageTokenId: hfConfig.image_token_index ?? hfConfig.image_token_id ?? 0,
        startTokenId: hfConfig.boi_token_index,
        endTokenId: hfConfig.eoi_token_index,
        fixedTokens: hfConfig.mm_tokens_per_image ?? 256,
      },
      preprocess: {
        imageMean: (pp.image_mean as [number, number, number]) ?? [0.5, 0.5, 0.5],
        imageStd: (pp.image_std as [number, number, number]) ?? [0.5, 0.5, 0.5],
        resize: {
          kind: 'fixed',
          width: pp.size?.width ?? 896,
          height: pp.size?.height ?? 896,
        },
      },
    };
  }

  return null;
}
