/**
 * Forward Pass — Transformer Inference Orchestrator
 *
 * Complete end-to-end transformer forward pass using WGSL compute kernels.
 * Model-agnostic: parameterized entirely by ModelConfig from config.json.
 *
 * Architecture (standard transformer decoder):
 *   embed(token_ids)
 *   for each layer:
 *     x = rmsnorm(x, input_norm_weight)
 *     q, k, v = linear projections
 *     rope(q, k)
 *     kv_cache.write(k, v)
 *     attn_out = attention(q, kv_cache.k, kv_cache.v)
 *     x = x + o_proj(attn_out)
 *     x = rmsnorm(x, post_attn_norm)
 *     x = x + ffn(x)
 *   logits = lm_head(rmsnorm(x))
 *
 * MODEL-SPECIFIC NOTES (search for "MODEL-SPECIFIC"):
 *   1. Activation function: most use SiLU, some use GELU
 *   2. Norm placement: Gemma adds 1.0 to norm weights
 *   3. Fused projections: Phi fuses gate+up into one weight
 *   4. RoPE scaling: some models use NTK-aware or YaRN scaling
 *   5. Sliding window attention: Mistral limits attention span
 */

import type { ModelDescriptor } from '../model/model-descriptor';
import {
  createComputePipeline,
  createBindGroup,
  dispatch,
  workgroupCount,
  BatchedDispatcher,
} from './compute';
import {
  createStorageBuffer,
  createUniformBuffer,
  readBuffer,
} from './buffers';
import { initTurboQuant, buildCodebook } from '../model/turboquant';

import matmulWGSL from '../shaders/matmul.wgsl?raw';
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import ropeWGSL from '../shaders/rope.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import embedWGSL from '../shaders/embed.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import matmulQ4WGSL from '../shaders/matmul_q4.wgsl?raw';
import matmulQ4GemvWGSL from '../shaders/matmul_q4_gemv.wgsl?raw';
import matmulE8WGSL from '../shaders/matmul_e8.wgsl?raw';
import matmulQ8WGSL from '../shaders/matmul_q8.wgsl?raw';
import tqEncodeWGSL from '../shaders/turboquant_encode.wgsl?raw';
import tqDecodeWGSL from '../shaders/turboquant_decode.wgsl?raw';
import attentionTqWGSL from '../shaders/attention_tq.wgsl?raw';
import conv1dWGSL from '../shaders/conv1d.wgsl?raw';
import groupNormWGSL from '../shaders/group_norm.wgsl?raw';
import ssmStepWGSL from '../shaders/ssm_step.wgsl?raw';
import l2normWGSL from '../shaders/l2norm.wgsl?raw';
import deinterleaveWGSL from '../shaders/deinterleave.wgsl?raw';
import hadamardWGSL from '../shaders/hadamard.wgsl?raw';
import dequantQ4BF16WGSL from '../shaders/dequant_q4_bf16.wgsl?raw';
import matmulGgufWGSL from '../shaders/matmul_gguf.wgsl?raw';
import { GGML_TYPES, ggmlTypeTraits } from '../model/gguf';
import { dequantGGML, GGUF_GPU_LAYOUT } from '../model/gguf-dequant';
import { topKSoftmax, MOE_MAX_TOKENS, type MoEBackend } from './moe-cpu';

// ── Limits ───────────────────────────────────────────────────────────────

/** Longest KV sequence the attention kernels can read: attention.wgsl (and
 *  attention_tq.wgsl) hold per-position scores in `var<workgroup> scores:
 *  array<f32, 3840>` — 3840*4 + 256*4 = 16384 bytes, exactly the default
 *  WebGPU workgroup-storage limit. KV caches must not exceed this; callers
 *  sizing persistent chat sessions must clamp to it. */
export const MAX_ATTN_SEQ_LEN = 3840;

// ── CPU embed helpers (BF16 → F32 decode) ──────────────────────────────
const _cpuEmbedBuf = new ArrayBuffer(4);
const _cpuEmbedU32 = new Uint32Array(_cpuEmbedBuf);
const _cpuEmbedF32 = new Float32Array(_cpuEmbedBuf);

// ── Types ────────────────────────────────────────────────────────────────

/** GGUF k-quant weight: repacked block data + its ggml type (Q8_0/Q4_K/Q5_K/Q6_K). */
export interface GGUFWeight {
  data: GPUBuffer;
  ggmlType: number;
  /** View start into `data` (gate+up concat — must be 256-aligned). */
  byteOffset?: number;
}

/** GPU buffers for one transformer layer's weights. */
export interface LayerWeights {
  inputNorm: GPUBuffer;    // [hidden_size]
  qProj: GPUBuffer;        // [num_heads * head_dim, hidden_size]
  kProj: GPUBuffer;        // [num_kv_heads * head_dim, hidden_size]
  vProj: GPUBuffer;        // [num_kv_heads * head_dim, hidden_size]
  oProj: GPUBuffer;        // [hidden_size, num_heads * head_dim]
  qBias?: GPUBuffer;       // [num_heads * head_dim] (only if attention_bias)
  kBias?: GPUBuffer;       // [num_kv_heads * head_dim]
  vBias?: GPUBuffer;       // [num_kv_heads * head_dim]
  oBias?: GPUBuffer;       // [hidden_size]
  postAttnNorm: GPUBuffer; // [hidden_size] (pre-FFN norm)
  // Gemma 4 sandwich norms — applied to the sub-block OUTPUT before its
  // residual add (gemma4.cpp attn_post_norm / ffn_post_norm).
  attnPostNorm?: GPUBuffer; // [hidden_size]
  ffnPostNorm?: GPUBuffer;  // [hidden_size]
  // Gemma 4 PLE sub-block (after the FFN residual) + per-layer output scale
  pleInpGate?: GPUBuffer;     // [ple_dim, hidden_size]
  pleInpGate_gg?: GGUFWeight;
  pleProj?: GPUBuffer;        // [hidden_size, ple_dim]
  pleProj_gg?: GGUFWeight;
  plePostNorm?: GPUBuffer;    // [hidden_size]
  layerOutScale?: GPUBuffer;  // [1] — scales the whole stream after PLE
  gateProj: GPUBuffer;     // [intermediate_size, hidden_size]
  upProj: GPUBuffer;       // [intermediate_size, hidden_size]
  downProj: GPUBuffer;     // [hidden_size, intermediate_size]

  // GPTQ quantized weight buffers (optional, only for INT4 models)
  qProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  kProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  vProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  oProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  gateProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  upProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  downProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };

  // E8 lattice 2-bit quantized weight buffers (optional, per-layer via recipe)
  qProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  kProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  vProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  oProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  gateProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  upProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  downProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };

  // INT8 quantized weight buffers (optional, per-layer via recipe)
  qProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  kProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  vProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  oProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  gateProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  upProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  downProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };

  // GGUF k-quant weight buffers (native llama.cpp blocks, repacked to
  // 4-byte-aligned strides at load — see gguf-dequant.ts repack*)
  qProj_gg?: GGUFWeight;
  kProj_gg?: GGUFWeight;
  vProj_gg?: GGUFWeight;
  oProj_gg?: GGUFWeight;
  gateProj_gg?: GGUFWeight;
  upProj_gg?: GGUFWeight;
  /** Same-A fusion: gate+up concatenated row-wise into one buffer — one
   *  decode GEMV with N=2*ffnDim writes [gate | up] contiguously. When set,
   *  gateProj_gg/upProj_gg are byteOffset views into the same buffer. */
  gateUpProj_gg?: GGUFWeight;
  downProj_gg?: GGUFWeight;
  linearInProjQKV_gg?: GGUFWeight;
  linearInProjA_gg?: GGUFWeight;
  linearInProjB_gg?: GGUFWeight;
  linearInProjZ_gg?: GGUFWeight;
  linearOutProj_gg?: GGUFWeight;

  // Linear attention (Gated DeltaNet) weights — only for hybrid models
  linearInProjQKV?: GPUBuffer;
  linearInProjQKV_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjA?: GPUBuffer;
  linearInProjA_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjB?: GPUBuffer;
  linearInProjB_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjZ?: GPUBuffer;
  linearInProjZ_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearOutProj?: GPUBuffer;
  linearOutProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjQKV_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  linearInProjA_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  linearInProjB_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  linearInProjZ_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  linearOutProj_e8?: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer };
  linearInProjQKV_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjA_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjB_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearInProjZ_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  linearOutProj_q8?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };
  // Full attention Q/K norm weights (Qwen3.5 only)
  qNorm?: GPUBuffer;              // [head_dim] per-head RMSNorm weight for Q
  kNorm?: GPUBuffer;              // [head_dim] per-head RMSNorm weight for K

  linearALog?: GPUBuffer;         // [num_key_heads, key_head_dim] diagonal state decay
  linearConv1dWeight?: GPUBuffer; // [dim, 1, kernel_size] causal conv kernel
  linearDtBias?: GPUBuffer;       // [num_key_heads * key_head_dim] time step bias
  linearNormWeight?: GPUBuffer;   // group norm weight

  /**
   * MoE router (ffn_gate_inp.weight, F32 [num_experts, hidden]). Presence
   * marks the layer as MoE: gateProj/upProj/downProj then hold the SHARED
   * expert and the routed experts run on weights.moe.backend (CPU fleet).
   */
  moeRouter?: GPUBuffer;
}

/** GPU buffers for global (non-layer) weights. */
export interface GlobalWeights {
  embedTokens: GPUBuffer;  // [vocab_size, hidden_size] (f32, f16 packed, or dummy if Q4/split)
  embedIsF16?: boolean;    // true if embedding stored as F16/BF16 (large vocab models)
  embedSplit?: { buffers: GPUBuffer[]; splitPoints: number[] };  // Split BF16 for oversized embeddings (SSM models need lossless embed)
  embedCPU?: { parts: Uint8Array[]; splitPoint: number; hiddenSize: number; isBF16: boolean };
  embedQ4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };  // GPTQ INT4 embedding
  /** E8 codebook buffer, shared across all layers [256*8] f32 */
  e8Codebook?: GPUBuffer;
  finalNorm: GPUBuffer;    // [hidden_size]
  lmHead: GPUBuffer;       // [vocab_size, hidden_size] or same as embedTokens
  lmHeadIsBF16?: boolean;  // true if lm_head stored as BF16 (large vocab models)
  lmHeadQ4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };  // GPTQ INT4 lm_head
  lmHeadSplit?: { buffers: GPUBuffer[]; splitPoints: number[] };  // Split BF16 for oversized lm_head
  lmHeadCPU?: { parts: Uint8Array[]; splitPoint: number; hiddenSize: number; vocabSize: number; isBF16: boolean };
  /** GGUF embed: raw k-quant blocks kept in RAM, CPU row-gather + dequant per token. */
  embedGG?: { data: Uint8Array; ggmlType: number; rowBytes: number };
  /** GGUF lm_head: repacked k-quant blocks on GPU (matmul_gguf kernel). */
  lmHeadGG?: GGUFWeight;
  /**
   * Gemma 4 PLE: per-layer token table, CPU row-gather like embedGG.
   * Sharded into `parts` when the raw blocks exceed the ArrayBuffer cap
   * (E4B Q6_K table is 2.31 GB): row r is in parts[floor(r / rowsPerPart)].
   */
  pleTokenEmbedGG?: {
    data: Uint8Array; parts?: Uint8Array[]; rowsPerPart?: number;
    ggmlType: number; rowBytes: number;
  };
  /** Gemma 4 PLE: hidden → [n_layer * ple_dim] projection. */
  pleModelProj?: GPUBuffer;
  pleModelProj_gg?: GGUFWeight;
  /** Gemma 4 PLE: RMSNorm weight [ple_dim] for the projected inputs. */
  pleProjNorm?: GPUBuffer;
}

/** All model weights on the GPU. */
export interface ModelWeights {
  global: GlobalWeights;
  layers: LayerWeights[];
  /** Set of GPU buffers that store BF16 data (not converted to f32). */
  bf16Buffers?: Set<GPUBuffer>;
  /** Phase C MoE: CPU expert fleet + per-layer shared-expert gate vectors. */
  moe?: {
    backend: MoEBackend;
    /** sharedGateVecs[l] = ffn_gate_inp_shexp.weight (F32 [hidden]) — scalar sigmoid gate. */
    sharedGateVecs: Float32Array[];
  };
}

/** Compressed KV cache data (TurboQuant). */
export interface CompressedKVData {
  scratchK: GPUBuffer;           // shared f32 decode buffer [maxSeqLen, kvDim]
  scratchV: GPUBuffer;           // shared f32 decode buffer [maxSeqLen, kvDim]
  quantizedK: GPUBuffer[];       // packed u32 indices per layer
  quantizedV: GPUBuffer[];
  signBitsK: GPUBuffer[];        // QJL sign bits per layer
  signBitsV: GPUBuffer[];
  normsK: GPUBuffer[];           // f32 norms per layer (||k||)
  normsV: GPUBuffer[];
  residualNormsK: GPUBuffer[];   // f32 residual norms per layer (||r||, for QJL correction)
  residualNormsV: GPUBuffer[];
}

/** SSM state for Gated DeltaNet / Mamba-2 linear attention layers. */
export interface SSMState {
  /** Hidden state per linear layer: [num_key_heads, key_head_dim, value_head_dim] f32 */
  hiddenStates: GPUBuffer[];
  /** Conv sliding window per linear layer: [kernel_size - 1, proj_dim] f32 */
  convStates: GPUBuffer[];
  /** Mapping from global layer index to SSM state index */
  layerToSSMIndex: number[];
}

/** KV cache for all layers. */
export interface KVCache {
  /** K cache per layer: [max_seq, num_kv_heads * head_dim] (unused if compressed or SSM) */
  keys: GPUBuffer[];
  /** V cache per layer: [max_seq, num_kv_heads * head_dim] (unused if compressed or SSM) */
  values: GPUBuffer[];
  /** Current sequence position (number of cached tokens) */
  position: number;
  /** Maximum sequence length */
  maxSeqLen: number;
  /** TurboQuant compressed storage (when enabled) */
  compressed?: CompressedKVData;
  /** SSM state for hybrid models with linear attention layers */
  ssmState?: SSMState;
}

/** Output of a forward pass step. */
export interface ForwardOutput {
  /** Logits for the last token: [vocab_size] */
  logitsBuffer: GPUBuffer;
}

// ── Forward Pass Engine ──────────────────────────────────────────────────

/** Per-call options for forward(). */
export interface ForwardOptions {
  /** Precomputed hidden-state rows for this chunk (seqLen × hiddenSize f32),
   *  e.g. image-patch embeddings from a vision encoder. Bypasses the token
   *  embed stage entirely — the caller is responsible for any model-specific
   *  embedding scaling. tokenIds are still required for cache bookkeeping
   *  (and Gemma PLE row-gather, which is why Gemma vision needs more than
   *  this hook). */
  embeddings?: Float32Array;
  /** Qwen3-VL DeepStack: features[k] (seqLen × hiddenSize f32) is added to
   *  text layer k's output for this chunk. Callers pass this only on
   *  image-segment chunks, so every row is an image row by construction. */
  deepstackFeatures?: Float32Array[];
  /** Image-span chunks attend bidirectionally within the chunk (Gemma). */
  bidirectional?: boolean;
}

export interface ForwardPassEngine {
  /** Run one forward pass step (prefill or single-token decode). */
  forward(tokenIds: Uint32Array, kvCache: KVCache, opts?: ForwardOptions): Promise<ForwardOutput>;

  /** Create an empty KV cache for the given max sequence length.
   *  @param compressed — Use TurboQuant compression (saves ~80% KV memory) */
  createKVCache(maxSeqLen: number, compressed?: boolean): KVCache;

  /** Destroy all buffers in a KV cache. */
  destroyKVCache(kvCache: KVCache): void;

  /** Get the model descriptor. */
  readonly config: ModelDescriptor;
}

/**
 * Create a forward pass engine.
 */
export function createForwardPassEngine(
  device: GPUDevice,
  config: ModelDescriptor,
  weights: ModelWeights,
): ForwardPassEngine {
  const {
    hiddenSize: H,
    numLayers: L,
    numAttentionHeads: nHeads,
    numKVHeads: nKVHeads,
    headDim: dHead,
    intermediateSize: ffnDim,
    vocabSize: V,
    rmsNormEps: eps,
    ropeTheta,
  } = config;

  const kvDim = nKVHeads * dHead;

  // ── Compile all pipelines ──────────────────────────────────────────

  const embedPipeline = createComputePipeline(device, embedWGSL, 'embed', 'embed');
  const embedF16Pipeline = createComputePipeline(device, embedWGSL, 'embed_f16', 'embed-f16');
  const embedF16SplitPipeline = createComputePipeline(device, embedWGSL, 'embed_f16_split', 'embed-f16-split');
  const embedQ4Pipeline = config.isQuantized
    ? createComputePipeline(device, embedWGSL, 'embed_q4', 'embed-q4') : null;
  const rmsnormPipeline = createComputePipeline(device, rmsnormWGSL, 'rmsnorm', 'rmsnorm');
  // MODEL-SPECIFIC: activation. Most use SiLU; Phi/some Gemma use GELU.
  const siluPipeline = createComputePipeline(device, elementwiseWGSL, 'silu', 'silu');
  const mulPipeline = createComputePipeline(device, elementwiseWGSL, 'mul', 'mul');
  const addPipeline = createComputePipeline(device, elementwiseWGSL, 'add', 'add');
  const matmulPipeline = createComputePipeline(device, matmulWGSL, 'matmul', 'matmul');
  // B-transposed matmul for HF weight projections (stored as [out, in])
  const matmulBTPipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt', 'matmul-bt');
  const matmulBTBF16Pipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt_bf16', 'matmul-bt-bf16');
  // INT4 GPTQ dequantizing matmul (weights packed as 4-bit)
  const matmulQ4Pipeline = config.isQuantized
    ? createComputePipeline(device, matmulQ4WGSL, 'matmul_bt_q4', 'matmul-q4')
    : null;
  // GEMV fast-path for M=1 (decode). Same bindings as matmul_bt_q4 so bind
  // groups can be swapped without rebuilding. One thread per output column,
  // so every thread is useful (vs 16/256 with the tiled kernel at M=1).
  //
  // Pipeline variants differ by four WGSL override constants:
  //   USE_ACTORDER     — 0: group_id = k/group_size (fast, no g_idx reads)
  //                      1: group_id = g_idx[k]     (needed for desc_act=true)
  //   USE_SCALE_CACHE  — 0: decode f16 scales from VRAM on every group change
  //                      1: preload workgroup's scale column-slice to LDS once
  //   WG_SIZE          — threads per workgroup = columns processed per workgroup
  //   MAX_GROUPS       — workgroup array capacity (must ≥ num_groups)
  //
  // Variants are compiled lazily via getGemvPipeline() so the set of (MAX_GROUPS,
  // WG_SIZE) pairs is driven by the actual matmul shapes in the loaded model,
  // not hard-coded. See dispatchMatmulQ4 below for the per-tensor selection.
  // E8 lattice 2-bit dequantizing matmul (weights packed as codebook indices)
  const matmulE8Pipeline = config.isQuantized
    ? createComputePipeline(device, matmulE8WGSL, 'matmul_e8', 'matmul-e8')
    : null;
  // INT8 dequantizing matmul (weights packed as 8-bit)
  const matmulQ8Pipeline = config.isQuantized
    ? createComputePipeline(device, matmulQ8WGSL, 'matmul_bt_q8', 'matmul-q8')
    : null;
  // GGUF k-quant dequantizing GEMV/GEMM — one workgroup per output element
  const matmulGgufPipelines: Record<number, GPUComputePipeline> | null =
    config.sourceFormat === 'gguf' ? {
      [GGML_TYPES.Q4_0]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q4_0', 'matmul-gguf-q4_0'),
      [GGML_TYPES.Q5_0]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q5_0', 'matmul-gguf-q5_0'),
      [GGML_TYPES.Q2_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q2_k', 'matmul-gguf-q2_k'),
      [GGML_TYPES.Q3_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q3_k', 'matmul-gguf-q3_k'),
      [GGML_TYPES.Q8_0]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q8_0', 'matmul-gguf-q8_0'),
      [GGML_TYPES.Q4_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q4_k', 'matmul-gguf-q4_k'),
      [GGML_TYPES.Q5_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q5_k', 'matmul-gguf-q5_k'),
      [GGML_TYPES.Q6_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q6_k', 'matmul-gguf-q6_k'),
      [GGML_TYPES.IQ4_NL]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_iq4_nl', 'matmul-gguf-iq4_nl'),
      [GGML_TYPES.IQ4_XS]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_iq4_xs', 'matmul-gguf-iq4_xs'),
    } : null;
  // Lever 2: multi-output tiled GEMV — TN output rows per workgroup share a
  // staged activation tile (legacy kernels re-read the full activation vector
  // per output row). Covers all k-quants (Q2_K..Q6_K); other formats fall
  // through to the legacy one-row kernels in dispatchMatmulGGUF.
  // A/B: ?gemvTile=0 disables. Tile shape tunable via ?gemvTN= / ?gemvTWG=
  // (pipeline-override constants; TWG/TN must be a power of two).
  const gemvSearch = typeof window === 'undefined'
    ? null : new URLSearchParams(window.location.search);
  const gemvTileEnabled = gemvSearch?.get('gemvTile') !== '0';
  // Defaults from the 27B sweep on RDNA2 (2026-06-12): 8/128 beat 8/256 by
  // ~1.6% at 27B shapes (8.81 vs 8.67 tok/s @256 tokens); on the 9B the two
  // were within noise (14.6 vs 14.7), so 128 is the better global default.
  const GEMV_TILE_N = Number(gemvSearch?.get('gemvTN') ?? 8);
  const GEMV_TILE_WG = Number(gemvSearch?.get('gemvTWG') ?? 128);
  const gemvTileConsts = { TN: GEMV_TILE_N, TWG: GEMV_TILE_WG };
  const matmulGgufTiledPipelines: Record<number, GPUComputePipeline> | null =
    config.sourceFormat === 'gguf' && gemvTileEnabled ? {
      [GGML_TYPES.Q2_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q2_k_tiled', 'matmul-gguf-q2_k-tiled', gemvTileConsts),
      [GGML_TYPES.Q3_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q3_k_tiled', 'matmul-gguf-q3_k-tiled', gemvTileConsts),
      [GGML_TYPES.Q4_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q4_k_tiled', 'matmul-gguf-q4_k-tiled', gemvTileConsts),
      [GGML_TYPES.Q5_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q5_k_tiled', 'matmul-gguf-q5_k-tiled', gemvTileConsts),
      [GGML_TYPES.Q6_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q6_k_tiled', 'matmul-gguf-q6_k-tiled', gemvTileConsts),
    } : null;
  // Lever 4: no-stage decode GEMV — same tiled lane→unit mapping but direct
  // vec4 A reads (no a_tile staging, no per-chunk barriers). Bit-identical
  // accumulation to *_tiled; M=1 only. DEFAULT OFF (?gemvNS=1 to enable):
  // on RDNA2 at 27B shapes it REGRESSED 7.51→7.31 tok/s (Q2_K +3%, Q3_K
  // +7.5%, Q6_K +10% per-dispatch) — the LDS a_tile broadcast beats TN×
  // per-lane global A re-reads; the two barriers/chunk were never the cost.
  const gemvNsEnabled = gemvTileEnabled && gemvSearch?.get('gemvNS') === '1';
  const matmulGgufNsPipelines: Record<number, GPUComputePipeline> | null =
    config.sourceFormat === 'gguf' && gemvNsEnabled ? {
      [GGML_TYPES.Q2_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q2_k_tiled_ns', 'matmul-gguf-q2_k-tiled-ns', gemvTileConsts),
      [GGML_TYPES.Q3_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q3_k_tiled_ns', 'matmul-gguf-q3_k-tiled-ns', gemvTileConsts),
      [GGML_TYPES.Q4_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q4_k_tiled_ns', 'matmul-gguf-q4_k-tiled-ns', gemvTileConsts),
      [GGML_TYPES.Q5_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q5_k_tiled_ns', 'matmul-gguf-q5_k-tiled-ns', gemvTileConsts),
      [GGML_TYPES.Q6_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q6_k_tiled_ns', 'matmul-gguf-q6_k-tiled-ns', gemvTileConsts),
    } : null;
  // Lever 4 phase 2: vec4 weight loads on the staged tiled bodies — the W
  // stream dominates decode bandwidth; 18-20 scalar u32 loads per unit
  // become 5 vec4 loads. Bit-identical (gate 4, test-gemv-tiled.mts). Only
  // strides ≡ 0 mod 4 qualify (Q3_K/Q4_K/Q5_K). M=1 only. A/B: ?gemvV4=0.
  const gemvV4Enabled = gemvTileEnabled && gemvSearch?.get('gemvV4') !== '0';
  const matmulGgufV4Pipelines: Record<number, GPUComputePipeline> | null =
    config.sourceFormat === 'gguf' && gemvV4Enabled ? {
      [GGML_TYPES.Q3_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q3_k_tiled_v4', 'matmul-gguf-q3_k-tiled-v4', gemvTileConsts),
      [GGML_TYPES.Q4_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q4_k_tiled_v4', 'matmul-gguf-q4_k-tiled-v4', gemvTileConsts),
      [GGML_TYPES.Q5_K]: createComputePipeline(device, matmulGgufWGSL, 'matmul_gguf_q5_k_tiled_v4', 'matmul-gguf-q5_k-tiled-v4', gemvTileConsts),
    } : null;
  // Same-A GEMV fusion: gate and up projections read the SAME activation
  // vector, have the same shape [ffnDim, H] and (on every GGUF seen so far)
  // the same quant type. Concatenating their block data row-wise lets decode
  // run ONE GEMV with N=2*ffnDim — half the FFN gate/up dispatches and twice
  // the workgroups per dispatch. Row n >= ffnDim of the combined tensor
  // lands exactly at the up tensor's row n-ffnDim words (the kernel computes
  // wordBase = (n*nSB + sb)*stride), so per-row math is untouched →
  // byte-identical. Originals are destroyed and repointed as byteOffset
  // views into the combined buffer (zero net VRAM; prefill + MoE shared
  // expert paths keep working through the views). A/B: ?fuseGU=0.
  const fuseGUEnabled = config.sourceFormat === 'gguf' && gemvSearch?.get('fuseGU') !== '0';
  if (fuseGUEnabled) {
    const enc = device.createCommandEncoder({ label: 'gateup-concat' });
    const deferredWeightDestroys: GPUBuffer[] = [];
    let fusedLayers = 0;
    for (const lw of weights.layers) {
      const g = lw.gateProj_gg, u = lw.upProj_gg;
      const layout = g ? GGUF_GPU_LAYOUT[g.ggmlType] : undefined;
      if (!g || !u || g.ggmlType !== u.ggmlType || !layout) continue;
      const rowBytes = (H / layout.blockElems) * layout.strideU32 * 4;
      const gateBytes = ffnDim * rowBytes;
      // Storage-binding offsets must be 256-aligned (gate_silu's up view at
      // ffnDim*4 too); skip the layer if the shapes don't line up.
      if (!Number.isInteger(rowBytes) || gateBytes % 256 !== 0 || (ffnDim * 4) % 256 !== 0) continue;
      if (gateBytes > g.data.size || gateBytes > u.data.size) continue;
      const combined = device.createBuffer({
        size: gateBytes * 2,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'gateup-gg',
      });
      enc.copyBufferToBuffer(g.data, 0, combined, 0, gateBytes);
      enc.copyBufferToBuffer(u.data, 0, combined, gateBytes, gateBytes);
      // Submitted copies keep the source allocations alive until the GPU is
      // done; destroy() just drops them afterwards.
      const oldG = g.data, oldU = u.data;
      deferredWeightDestroys.push(oldG, oldU);
      lw.gateUpProj_gg = { data: combined, ggmlType: g.ggmlType };
      lw.gateProj_gg = { data: combined, ggmlType: g.ggmlType };
      lw.upProj_gg = { data: combined, ggmlType: u.ggmlType, byteOffset: gateBytes };
      fusedLayers++;
    }
    device.queue.submit([enc.finish()]);
    if (fusedLayers > 0) {
      for (const b of deferredWeightDestroys) b.destroy();
      deferredWeightDestroys.length = 0;
      console.log(`[forward-pass] gate+up same-A fusion: concatenated ${fusedLayers} layers`);
    }
  }

  // Hadamard transform for QuIP#/QuaRot incoherence processing
  const hadamardPipeline = config.isQuantized
    ? createComputePipeline(device, hadamardWGSL, 'hadamard', 'hadamard')
    : null;

  const ropePipeline = createComputePipeline(device, ropeWGSL, 'rope', 'rope');
  // Attention pipelines — two variants each (softmax default, softpick variant).
  // Softpick replaces exp normalization with rectified softmax to eliminate
  // attention-sink saturation on long decodes. Selected at dispatch time via
  // globalThis.__USE_SOFTPICK__.
  const attentionPipeline = createComputePipeline(device, attentionWGSL, 'attention', 'attention',
    { USE_SOFTPICK: 0 });
  const attentionPipelineSoftpick = createComputePipeline(device, attentionWGSL, 'attention', 'attention-softpick',
    { USE_SOFTPICK: 1 });
  const attentionTqPipeline = createComputePipeline(device, attentionTqWGSL, 'attention_tq', 'attention-tq',
    { USE_SOFTPICK: 0 });
  const attentionTqPipelineSoftpick = createComputePipeline(device, attentionTqWGSL, 'attention_tq', 'attention-tq-softpick',
    { USE_SOFTPICK: 1 });

  // ── TurboQuant pipelines & setup ───────────────────────────────────
  // 4 bits for d≤64 (small models), 3 bits for d≥128 (noise averages better)
  const TQ_BITS = dHead >= 128 ? 3 : 4;
  const tqEncodePipeline = createComputePipeline(device, tqEncodeWGSL, 'encode', 'tq-encode');
  const tqDecodePipeline = createComputePipeline(device, tqDecodeWGSL, 'decode', 'tq-decode');
  const tqSetup = initTurboQuant(device, { headDim: dHead, bits: TQ_BITS });
  const tqCodebook = buildCodebook(TQ_BITS);
  const tqPackedWords = Math.ceil(dHead / Math.floor(32 / TQ_BITS));
  const tqSignWords = Math.ceil(dHead / 32);

  // Reusable bind groups for TQ matrices (group 1 — constant across all dispatches)
  const tqEncodeMatBG = createBindGroup(device, tqEncodePipeline, 1, [
    { binding: 0, resource: { buffer: tqSetup.rotationMatrix } },
    { binding: 1, resource: { buffer: tqSetup.jlMatrix } },
    { binding: 2, resource: { buffer: tqSetup.centroids } },
    { binding: 3, resource: { buffer: tqSetup.thresholds } },
  ], 'tq-encode-mat');

  const tqDecodeMatBG = createBindGroup(device, tqDecodePipeline, 1, [
    { binding: 0, resource: { buffer: tqSetup.rotationMatrix } },
    { binding: 1, resource: { buffer: tqSetup.jlMatrix } },
    { binding: 2, resource: { buffer: tqSetup.centroids } },
  ], 'tq-decode-mat');

  // ── Gated DeltaNet / Mamba-2 pipelines (hybrid models only) ──────
  const isHybrid = config.isHybrid;
  // gate_silu is also used by SwiGLU FFN fusion (silu(gate) * up → one dispatch),
  // so create it unconditionally. Keeps platform-generality for non-hybrid models.
  const gateSiluPipeline = createComputePipeline(device, elementwiseWGSL, 'gate_silu', 'gate-silu');
  // Fused gated activation for the dense FFN — Gemma uses GeGLU (gelu_tanh).
  const gateActPipeline = config.activation === 'gelu_tanh'
    ? createComputePipeline(device, elementwiseWGSL, 'gate_gelu', 'gate-gelu')
    : gateSiluPipeline;
  // Gemma: final logit softcapping c·tanh(x/c)
  const softcapPipeline = config.finalLogitSoftcap
    ? createComputePipeline(device, elementwiseWGSL, 'softcap', 'softcap') : null;
  // Gemma 4 PLE: (per_layer_proj + inp_per_layer) × 1/√2 combine
  const addscalePipeline = config.perLayerEmbed
    ? createComputePipeline(device, elementwiseWGSL, 'addscale', 'addscale') : null;
  const softplusPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'softplus', 'softplus') : null;
  const conv1dPipeline = isHybrid
    ? createComputePipeline(device, conv1dWGSL, 'conv1d', 'conv1d') : null;
  const conv1dUpdatePipeline = isHybrid
    ? createComputePipeline(device, conv1dWGSL, 'conv1d_update_state', 'conv1d-update') : null;
  // Fusion lever F4: conv + state update + SiLU in one dispatch. A/B: ?fuseConv=0.
  const conv1dSiluPipeline = isHybrid && gemvSearch?.get('fuseConv') !== '0'
    ? createComputePipeline(device, conv1dWGSL, 'conv1d_silu_update', 'conv1d-silu-update') : null;
  const groupNormPipeline = isHybrid
    ? createComputePipeline(device, groupNormWGSL, 'group_norm', 'group-norm') : null;
  const ssmStepPipeline = isHybrid
    ? createComputePipeline(device, ssmStepWGSL, 'ssm_step', 'ssm-step') : null;
  // Fusion lever F5: beta/decay prologue inside ssm_step. A/B: ?fuseSsm=0.
  const ssmStepFusedPipeline = isHybrid && gemvSearch?.get('fuseSsm') !== '0'
    ? createComputePipeline(device, ssmStepWGSL, 'ssm_step_fused', 'ssm-step-fused') : null;
  // Occupancy lever: thread-per-(kh,v)-column split — grid (nkh,
  // ceil(gvd/SSM_WG)) instead of nkh×256, fusing steps 1+2 and 4+5 in
  // registers (bit-exact; parity in test-fusion-kernels.mts). Same bindings
  // as ssm_step_fused (raw B/A + dt_bias/a_log). A/B: ?ssmSplit=0; WG size
  // tunable via ?ssmWG=.
  const SSM_SPLIT_WG = Number(gemvSearch?.get('ssmWG') ?? 64);
  const ssmStepSplitPipeline = ssmStepFusedPipeline && gemvSearch?.get('ssmSplit') !== '0'
    ? createComputePipeline(device, ssmStepWGSL, 'ssm_step_split', 'ssm-step-split',
        { SSM_WG: SSM_SPLIT_WG }) : null;

  // Linear attention dimensions (only for hybrid models)
  const linKD = config.linearKeyHeadDim ?? 0;
  const linVD = config.linearValueHeadDim ?? 0;
  const linNKH = config.linearNumKeyHeads ?? 0;
  const linNVH = config.linearNumValueHeads ?? 0;
  const linConvK = config.linearConvKernelDim ?? 0;
  const linQKVDim = linNKH * linKD + linNKH * linKD + linNVH * linVD; // fused Q+K+V output
  const linConvDim = linNKH * linKD; // dimension that goes through conv1d (K projection)
  const linVHPerKH = linNKH > 0 ? linNVH / linNKH : 0; // value heads per key head group (e.g., 2)
  const linGroupedVD = linVHPerKH * linVD; // grouped value dim per key head (e.g., 256)
  // DeltaNet z-gate width = nVHeads * dHead. NOT H: on Qwen3.6-35B H=2048 but z=4096
  // (they coincide on Qwen3.5-9B, which masked sizing bugs here).
  const linZDim = linNVH * linVD;

  // Compile additional Mamba-2 kernels
  const sigmoidPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'sigmoid_op', 'sigmoid') : null;
  const decayPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'decay_compute', 'decay') : null;
  const l2NormPipeline = isHybrid
    ? createComputePipeline(device, l2normWGSL, 'l2_normalize', 'l2-norm') : null;
  // Fusion lever F7: attn output gate a*sigmoid(b) in one dispatch. A/B: ?fuseGate=0.
  const gateSigmoidPipeline = config.attnOutputGate && gemvSearch?.get('fuseGate') !== '0'
    ? createComputePipeline(device, elementwiseWGSL, 'gate_sigmoid', 'gate-sigmoid') : null;

  // Fusion lever F1: GPU Q/gate deinterleave for attnOutputGate models
  // (Qwen3.5 interleaves Q and gate per head in the Q projection output).
  // Replaces the per-head CPU copy loop — 57 copies/attn-layer at decode,
  // each breaking the fused compute pass. A/B: ?deint=0 falls back.
  const deinterleavePipeline =
    config.attnOutputGate && gemvSearch?.get('deint') !== '0'
      ? createComputePipeline(device, deinterleaveWGSL, 'deinterleave_qgate', 'deinterleave-qgate')
      : null;

  // Fusion lever F2/F3: in the DeltaNet path, skip the post-conv QKV split
  // and the sigmoid/l2norm copy-backs by binding 256-B-aligned sections of
  // linQKVBuf / normedBuf / attnProjBuf directly (storage-buffer binding
  // offsets must be 256-aligned — verified here, else legacy copies).
  // GGUF/f32/bf16 matmuls accept binding windows; exotic quant paths do not,
  // so this is additionally gated on sourceFormat. A/B: ?fuseOff=0.
  const ssmOffsetBindings = isHybrid
    && config.sourceFormat === 'gguf'
    && gemvSearch?.get('fuseOff') !== '0'
    && (H * 4) % 256 === 0
    && (linNKH * linKD * 4) % 256 === 0;

  // ── On-the-fly Q4 → BF16 dequant for SSM projections (hybrid + quantized) ──
  // GGUF models never need this — k-quant weights dispatch via matmul_gguf directly.
  const needsGpuDequant = isHybrid && config.isQuantized && config.sourceFormat !== 'gguf';
  const dequantQ4Pipeline = needsGpuDequant
    ? createComputePipeline(device, dequantQ4BF16WGSL, 'dequant_q4_to_bf16', 'dequant-q4-bf16')
    : null;

  // Temp buffer sized for the largest SSM projection's BF16 output.
  // SSM projections: QKV [linQKVDim, H], A [linNVH, H], B [linNVH, H], Z [linZDim, H], OutProj [H, outDim]
  // BF16 output size = N * K/2 * 4 bytes (u32 per k-pair)
  const dequantMaxN = needsGpuDequant
    ? Math.max(linQKVDim, linNVH, H)
    : 0;
  const dequantMaxK = needsGpuDequant ? Math.max(H, linNKH * linGroupedVD) : 0;
  const dequantTempBytes = dequantMaxN * Math.ceil(dequantMaxK / 2) * 4;
  const dequantTempBuf = needsGpuDequant
    ? createStorageBuffer(device, null, dequantTempBytes, 'dequant-temp-bf16', true)
    : null;
  if (needsGpuDequant) {
    console.log(`[Engine] GPU dequant enabled: temp buffer ${(dequantTempBytes / 1024 / 1024).toFixed(1)} MB (maxN=${dequantMaxN}, maxK=${dequantMaxK})`);
  }

  // ── Per-kernel timing infrastructure (optional) ─────────────────────
  // Uses WebGPU's optional `timestamp-query` feature to measure each compute
  // pass on-device, then aggregates results by "category" (one category per
  // shader pipeline — e.g. all matmul_q4 dispatches roll up into one total).
  //
  // Registered here because the pipeline→category mapping is stable for the
  // lifetime of the engine. dispatchMatmul* helpers don't need to know; bd()
  // looks up the pipeline identity and emits timestampWrites when the timing
  // context is active. Categories are strings so new shaders auto-surface.
  const supportsTimestamps = device.features.has('timestamp-query');
  const pipelineCategories = new Map<GPUComputePipeline, string>();
  const registerCat = (p: GPUComputePipeline | null, cat: string) => {
    if (p) pipelineCategories.set(p, cat);
  };
  registerCat(embedPipeline, 'embed');
  registerCat(embedF16Pipeline, 'embed');
  registerCat(embedQ4Pipeline, 'embed_q4');
  registerCat(rmsnormPipeline, 'rmsnorm');
  registerCat(siluPipeline, 'silu');
  registerCat(mulPipeline, 'mul');
  registerCat(addPipeline, 'add');
  registerCat(matmulPipeline, 'matmul_f32');
  registerCat(matmulBTPipeline, 'matmul_bt_f32');
  registerCat(matmulBTBF16Pipeline, 'matmul_bt_bf16');
  registerCat(matmulQ4Pipeline, 'matmul_q4_tiled');
  // GEMV variants are registered lazily by getGemvPipeline() on first compile.
  registerCat(matmulE8Pipeline, 'matmul_e8');
  registerCat(matmulQ8Pipeline, 'matmul_q8');
  registerCat(hadamardPipeline, 'hadamard');
  registerCat(ropePipeline, 'rope');
  registerCat(attentionPipeline, 'attention');
  registerCat(attentionPipelineSoftpick, 'attention');
  registerCat(attentionTqPipeline, 'attention_tq');
  registerCat(attentionTqPipelineSoftpick, 'attention_tq');
  registerCat(tqEncodePipeline, 'tq_encode');
  registerCat(tqDecodePipeline, 'tq_decode');
  registerCat(gateSiluPipeline, 'gate_silu');
  if (gateActPipeline !== gateSiluPipeline) registerCat(gateActPipeline, 'gate_gelu');
  registerCat(softcapPipeline, 'softcap');
  registerCat(addscalePipeline, 'addscale');
  registerCat(softplusPipeline, 'softplus');
  registerCat(conv1dPipeline, 'conv1d');
  registerCat(conv1dUpdatePipeline, 'conv1d_update');
  registerCat(groupNormPipeline, 'group_norm');
  registerCat(ssmStepPipeline, 'ssm_step');
  registerCat(ssmStepFusedPipeline, 'ssm_step_fused');
  registerCat(ssmStepSplitPipeline, 'ssm_step_split');
  registerCat(sigmoidPipeline, 'sigmoid');
  registerCat(decayPipeline, 'decay');
  registerCat(l2NormPipeline, 'l2norm');
  registerCat(deinterleavePipeline, 'deinterleave');
  registerCat(conv1dSiluPipeline, 'conv1d_fused');
  registerCat(gateSigmoidPipeline, 'gate_sigmoid');
  // GEMV lever 4 Phase 0: per-quant-type aggregation for GGUF matmuls.
  // Without a category, bd() falls back to the per-site dispatch label
  // (L12-gate, ...) so per-type time never aggregates in __perfTimingRows.
  for (const [t, p] of Object.entries(matmulGgufPipelines ?? {})) {
    registerCat(p, `gguf_${ggmlTypeTraits(+t).name}`);
  }
  for (const [t, p] of Object.entries(matmulGgufTiledPipelines ?? {})) {
    registerCat(p, `gguf_${ggmlTypeTraits(+t).name}_tiled`);
  }
  for (const [t, p] of Object.entries(matmulGgufNsPipelines ?? {})) {
    registerCat(p, `gguf_${ggmlTypeTraits(+t).name}_ns`);
  }
  for (const [t, p] of Object.entries(matmulGgufV4Pipelines ?? {})) {
    registerCat(p, `gguf_${ggmlTypeTraits(+t).name}_v4`);
  }

  // ── Matmul-Q4 GEMV pipeline selection ──────────────────────────────
  // Cached-scales variants are parameterized by (MAX_GROUPS, WG_SIZE). The
  // product MAX_GROUPS * WG_SIZE (= scale cache entries) must fit the 28 KB
  // scale budget (4 KB goes to a_chunk, total ≤ 32 KB Blackwell limit).
  //
  // The table is sorted so pickGemvCacheShape() can linearly scan and return
  // the first entry whose MAX_GROUPS covers num_groups. Larger WG_SIZE is
  // preferred when it fits — fewer workgroup launches, better occupancy for
  // small-K matmuls.
  const GEMV_CACHE_SHAPES: Array<{ maxGroups: number; wgSize: number }> = [
    { maxGroups: 32,  wgSize: 128 },  // 16 KB scales, covers K ≤ 4096 at gs=128
    { maxGroups: 56,  wgSize: 128 },  // 28 KB scales, covers K ≤ 7168
    { maxGroups: 112, wgSize: 64  },  // 28 KB scales, covers K ≤ 14336
    { maxGroups: 224, wgSize: 32  },  // 28 KB scales, covers K ≤ 28672
    { maxGroups: 448, wgSize: 16  },  // 28 KB scales, covers K ≤ 57344 (70B-class FFN)
  ];
  function pickGemvCacheShape(numGroups: number): { maxGroups: number; wgSize: number } | null {
    for (const s of GEMV_CACHE_SHAPES) {
      if (s.maxGroups >= numGroups) return s;
    }
    return null;  // num_groups too large for any cached variant; fall back to uncached
  }

  // Uncached variants don't use scales_wg, so MAX_GROUPS=1 minimizes workgroup
  // storage reservation (the array is declared unconditionally at module scope).
  const GEMV_UNCACHED_SHAPE = { maxGroups: 1, wgSize: 128 };

  // Lazy pipeline compilation + cache. Keyed by the four override constants so
  // every unique (USE_ACTORDER, USE_SCALE_CACHE, MAX_GROUPS, WG_SIZE) tuple
  // compiles at most once. Categories are registered on first compile so the
  // per-kernel timing aggregator rolls variants up into three buckets.
  const gemvPipelineByKey = new Map<string, GPUComputePipeline>();
  function getGemvPipeline(
    useActorder: boolean,
    useScaleCache: boolean,
    maxGroups: number,
    wgSize: number,
  ): GPUComputePipeline | null {
    if (!config.isQuantized) return null;
    const key = `${useActorder ? 1 : 0}:${useScaleCache ? 1 : 0}:${maxGroups}:${wgSize}`;
    let p = gemvPipelineByKey.get(key);
    if (p) return p;
    const variantTag = useScaleCache
      ? `actorder-cached-g${maxGroups}w${wgSize}`
      : useActorder
        ? `actorder-g${maxGroups}w${wgSize}`
        : `noao-g${maxGroups}w${wgSize}`;
    p = createComputePipeline(
      device, matmulQ4GemvWGSL, 'matmul_bt_q4_gemv',
      `matmul-q4-gemv-${variantTag}`,
      {
        USE_ACTORDER: useActorder ? 1 : 0,
        USE_SCALE_CACHE: useScaleCache ? 1 : 0,
        MAX_GROUPS: maxGroups,
        WG_SIZE: wgSize,
      },
    );
    const category = useScaleCache
      ? 'matmul_q4_gemv_actorder_cached'
      : useActorder
        ? 'matmul_q4_gemv_actorder'
        : 'matmul_q4_gemv';
    pipelineCategories.set(p, category);
    gemvPipelineByKey.set(key, p);
    return p;
  }

  // Active timing context — set by forward() for a bounded number of decode
  // calls, then drained and nulled out. bd() checks this each dispatch.
  // The querySet + buffers are lazily created on first use (so engines that
  // never profile pay zero overhead).
  interface TimingCtx {
    querySet: GPUQuerySet;
    resolveBuf: GPUBuffer;      // QUERY_RESOLVE | COPY_SRC
    readBuf: GPUBuffer;         // MAP_READ | COPY_DST
    capacity: number;           // total u64 slots
    used: number;               // slots consumed this forward
    categories: string[];       // category per pass-pair (length = used/2)
  }
  let __timingCtx: TimingCtx | null = null;
  let __timingBuffersInit: { querySet: GPUQuerySet; resolveBuf: GPUBuffer; readBuf: GPUBuffer; capacity: number } | null = null;
  let __timingCallsRemaining = supportsTimestamps ? 2 : 0;  // profile first N eligible decode calls

  function ensureTimingBuffers(): NonNullable<typeof __timingBuffersInit> {
    if (__timingBuffersInit) return __timingBuffersInit;
    const capacity = 4096;  // ~2048 passes; a 32-layer forward has ~730
    const querySet = device.createQuerySet({ type: 'timestamp', count: capacity, label: 'forward-timestamps' });
    const resolveBuf = device.createBuffer({
      size: capacity * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      label: 'timestamp-resolve',
    });
    const readBuf = device.createBuffer({
      size: capacity * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'timestamp-read',
    });
    __timingBuffersInit = { querySet, resolveBuf, readBuf, capacity };
    return __timingBuffersInit;
  }

  // ── Reusable intermediate buffers ──────────────────────────────────
  // Sized for batch prefill (up to MAX_PREFILL tokens per forward pass).
  // Single-token decode uses the same buffers (seqLen=1).
  const MAX_PREFILL = 512;

  const hiddenBuf = createStorageBuffer(device, null, MAX_PREFILL * H * 4, 'hidden', true);
  const residualBuf = createStorageBuffer(device, null, MAX_PREFILL * H * 4, 'residual', true);
  const normedBuf = createStorageBuffer(device, null, MAX_PREFILL * H * 4, 'normed', true);
  // Qwen3.5: Q projection outputs 2x for gate — need double buffer
  const qBufMul = config.attnOutputGate ? 2 : 1;
  const qBuf = createStorageBuffer(device, null, MAX_PREFILL * nHeads * dHead * qBufMul * 4, 'q-proj', true);
  // Separate gate buffer for full attention output gating
  const attnGateBuf = config.attnOutputGate
    ? createStorageBuffer(device, null, MAX_PREFILL * nHeads * dHead * 4, 'attn-gate', true) : null;
  const kBuf = createStorageBuffer(device, null, MAX_PREFILL * kvDim * 4, 'k-proj', true);
  const vBuf = createStorageBuffer(device, null, MAX_PREFILL * kvDim * 4, 'v-proj', true);
  const attnOutBuf = createStorageBuffer(device, null, MAX_PREFILL * nHeads * dHead * 4, 'attn-out', true);
  const attnProjBuf = createStorageBuffer(device, null, MAX_PREFILL * H * 4, 'attn-proj', true);
  const gateBuf = createStorageBuffer(device, null, MAX_PREFILL * ffnDim * 4, 'ffn-gate', true);
  const upBuf = createStorageBuffer(device, null, MAX_PREFILL * ffnDim * 4, 'ffn-up', true);
  const downBuf = createStorageBuffer(device, null, MAX_PREFILL * H * 4, 'ffn-down', true);
  // Temp buffer for in-place elementwise ops (WebGPU can't read+write same buffer).
  // CAUTION: also used as scratch for H-sized copies (hadamard, biases) and
  // nHeads*dHead attention-gate copies — on MoE models ffnDim (=expert dim,
  // e.g. 512) is SMALLER than both, so size by the max of all its uses.
  const scratchDim = Math.max(ffnDim, nHeads * dHead, H, kvDim);
  const ffnTempBuf = createStorageBuffer(device, null, MAX_PREFILL * scratchDim * 4, 'ffn-temp', true);
  const logitsBuf = createStorageBuffer(device, null, V * 4, 'logits', true);
  // Softcap needs a read-side copy (WebGPU can't bind one buffer read + read_write)
  const logitsTempBuf = config.finalLogitSoftcap
    ? createStorageBuffer(device, null, V * 4, 'logits-precap') : null;
  // ── Gemma 4 PLE buffers ─────────────────────────────────────────────
  // Token-major combined per-layer input [tokens][n_layer][dPle].
  // pleInpBuf: CPU row-gather of per_layer_token_embd × √dPle (writeBuffer).
  // pleProjBuf / pleCombinedBuf: rotate through projection → scale → norm →
  // combine (WebGPU forbids binding one buffer read + read_write).
  const dPle = config.perLayerEmbedDim ?? 0;
  const pleDim = config.perLayerEmbed ? L * dPle : 0; // 42 × 256 = 10752
  const pleInpBuf = pleDim > 0
    ? createStorageBuffer(device, null, MAX_PREFILL * pleDim * 4, 'ple-inp') : null;
  const pleProjBuf = pleDim > 0
    ? createStorageBuffer(device, null, MAX_PREFILL * pleDim * 4, 'ple-proj') : null;
  const pleCombinedBuf = pleDim > 0
    ? createStorageBuffer(device, null, MAX_PREFILL * pleDim * 4, 'ple-combined') : null;
  // Per-layer gate output [tokens, dPle]
  const pleGateBuf = pleDim > 0
    ? createStorageBuffer(device, null, MAX_PREFILL * dPle * 4, 'ple-gate') : null;
  // [1] constant 1/√H — per_layer_projection_scale (broadcast mul)
  const pleProjScaleBuf = pleDim > 0
    ? createStorageBuffer(device, new Float32Array([1.0 / Math.sqrt(H)]), 4, 'ple-proj-scale') : null;
  // MoE: router logits [seqLen, numExperts] (read back per layer for CPU top-k)
  const moeNumExperts = config.layers.find((d) => d.moe)?.moe?.numExperts ?? 0;
  const routerLogitsBuf = moeNumExperts > 0
    ? createStorageBuffer(device, null, MAX_PREFILL * moeNumExperts * 4, 'moe-router-logits', true)
    : null;
  // ── C3 fast readback ring ─────────────────────────────────────────
  // Two pumped mapAsyncs per MoE layer: phase 1 (normed + router logits, then
  // the workers are kicked) and phase 2 (shared-expert out, overlapping the
  // workers) — instead of readBuffer() round-trips.
  // Chrome's mapAsync has a ~3 ms resolution floor when the event loop idles;
  // pumping a 4-byte writeBuffer + sub-ms MessageChannel yield while the map
  // is pending forces real queue ticks → ~0.2 ms (measured in gpu-bench.ts).
  // Ring of 2: each phase's buffer is unmapped before the next copy encoding
  // that touches it, so A/B alternation per layer is alias-free.
  const moeStagingBytes = MAX_PREFILL * (2 * H + moeNumExperts) * 4;
  const moeStagingRing = moeNumExperts > 0
    ? [0, 1].map((i) => device.createBuffer({
        size: moeStagingBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        label: `moe-staging-${i}`,
      }))
    : null;
  let moeStagingIdx = 0;
  const pumpDstBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST, label: 'map-pump' });
  const pumpSrcArr = new Uint32Array(1);
  const pumpChan = new MessageChannel();
  pumpChan.port1.start();
  const fastYield = () => new Promise<void>((resolve) => {
    pumpChan.port1.addEventListener('message', () => resolve(), { once: true });
    pumpChan.port2.postMessage(0);
  });
  /** mapAsync with the writeBuffer-pump trick. Never await a bare mapAsync. */
  async function mapWithPump(staging: GPUBuffer, bytes: number): Promise<void> {
    let done = false;
    const p = staging.mapAsync(GPUMapMode.READ, 0, bytes).then(() => { done = true; });
    while (!done) {
      device.queue.writeBuffer(pumpDstBuf, 0, pumpSrcArr);
      await fastYield();
    }
    await p;
  }
  const tokenIdBuf = createStorageBuffer(device, null, MAX_PREFILL * 4, 'token-ids', true);
  // Small buffer for extracting last token's hidden state (for LM head after batch prefill)
  const lastHiddenBuf = createStorageBuffer(device, null, H * 4, 'last-hidden', true);

  // DeepStack feature staging buffers — one per tap layer, lazily created.
  const dsFeatBufs: GPUBuffer[] = [];

  // Option A (chunked-prefill for hybrid models): per-token scratch buffers used
  // to wrap the existing single-token SSM block in a per-token JS loop while the
  // surrounding non-SSM stages process seqLen > 1 tokens in batched fashion.
  // ssmInputBuf holds one token-row sliced out of the multi-token normedBuf;
  // ssmOutBuf holds the SSM block's output for one token before being copied
  // into the multi-token attnProjBuf at that token's offset.
  const ssmInputBuf = isHybrid ? createStorageBuffer(device, null, H * 4, 'ssm-input', true) : null;
  // ssmOutBuf doubles as scratch for the RMSNormGated + silu(z) gate output,
  // which is linZDim (= nVHeads*dHead) wide — larger than H on the 35B MoE.
  const ssmOutBuf = isHybrid ? createStorageBuffer(device, null, Math.max(H, linZDim) * 4, 'ssm-out', true) : null;

  // Linear attention intermediate buffers (only for hybrid models)
  const linQKVBuf = isHybrid ? createStorageBuffer(device, null, linQKVDim * 4, 'lin-qkv', true) : null;
  const linQBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linKD * 4, 'lin-q', true) : null;
  const linKBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linKD * 4, 'lin-k', true) : null;
  const linVBuf = isHybrid ? createStorageBuffer(device, null, linNVH * linVD * 4, 'lin-v', true) : null;
  const linABuf = isHybrid ? createStorageBuffer(device, null, linConvDim * 4, 'lin-a', true) : null;
  const linBBuf = isHybrid ? createStorageBuffer(device, null, linNVH * 4, 'lin-beta', true) : null;
  const linZBuf = isHybrid ? createStorageBuffer(device, null, linZDim * 4, 'lin-z', true) : null;
  const linOutBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linGroupedVD * 4, 'lin-out', true) : null;
  const linConvOutBuf = isHybrid ? createStorageBuffer(device, null, linQKVDim * 4, 'lin-conv-out', true) : null;
  const linDecayBuf = isHybrid ? createStorageBuffer(device, null, linNVH * 4, 'lin-decay', true) : null;
  const linDtBuf = isHybrid ? createStorageBuffer(device, null, Math.max(linNVH, linNKH) * 4, 'lin-dt', true) : null;
  // F3: sigmoid(beta) lands here directly. The legacy path wrote it to
  // linDtBuf and copied back into linBBuf because linDtBuf is reused by
  // softplus before ssm_step consumes beta.
  const linBetaBuf = isHybrid ? createStorageBuffer(device, null, linNVH * 4, 'lin-beta', true) : null;

  // ── Uniform Buffer Cache ──────────────────────────────────────────────
  // During M=1 generation, dispatch parameters are identical every token.
  // Cache uniform buffers to avoid ~25 GPU allocations per forward pass.
  const uniformCache = new Map<string, GPUBuffer>();
  function getCachedUniform(data: Uint32Array | Uint8Array, label: string): GPUBuffer {
    // Build cache key from raw bytes
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const key = `${bytes.length}:${Array.from(bytes).join(',')}`;
    let buf = uniformCache.get(key);
    if (!buf) {
      buf = createUniformBuffer(device, data, label);
      uniformCache.set(key, buf);
    }
    return buf;
  }

  // ── Bind group cache ───────────────────────────────────────────────
  // Bind groups are immutable, and at M=1 decode the same (pipeline, buffers)
  // tuples recur every token — re-creating ~950 bind groups per token via
  // device.createBindGroup is pure encoding overhead. Cache by pipeline +
  // group index + buffer identities (+ offset/size). Uniform params buffers
  // are content-cached above, so their identity is stable per dispatch site.
  // Buffer contents may change (writeBuffer/copies); that never invalidates a
  // bind group — only buffer *identity* matters. A/B toggle: ?bgCache=0.
  const bgCacheEnabled = typeof window === 'undefined'
    || new URLSearchParams(window.location.search).get('bgCache') !== '0';
  const bindGroupCache = new Map<string, GPUBindGroup>();
  const bgKeyIds = new WeakMap<object, number>();
  let bgNextKeyId = 1;
  function bgKeyId(o: object): number {
    let id = bgKeyIds.get(o);
    if (id === undefined) {
      id = bgNextKeyId++;
      bgKeyIds.set(o, id);
    }
    return id;
  }
  function cachedBindGroup(
    pipeline: GPUComputePipeline,
    groupIndex: number,
    entries: Array<{ binding: number; resource: GPUBindingResource }>,
    label = '',
  ): GPUBindGroup {
    if (bgCacheEnabled) {
      let key = `${bgKeyId(pipeline)}|${groupIndex}`;
      let cacheable = true;
      for (const e of entries) {
        const r = e.resource as Partial<GPUBufferBinding>;
        if (!r.buffer) { cacheable = false; break; } // non-buffer resource
        key += `|${e.binding}:${bgKeyId(r.buffer)}:${r.offset ?? 0}:${r.size ?? 'w'}`;
      }
      if (cacheable) {
        let bg = bindGroupCache.get(key);
        if (!bg) {
          __perfBindGroupCount++;
          bg = createBindGroup(device, pipeline, groupIndex, entries, label);
          bindGroupCache.set(key, bg);
        }
        return bg;
      }
    }
    __perfBindGroupCount++;
    return createBindGroup(device, pipeline, groupIndex, entries, label);
  }

  /** C[M,N] = A[M,K] @ dequant(q4_packed, scales, zeros)^T — GPTQ INT4 */
  function dispatchMatmulQ4(
    aBuf: GPUBuffer,
    q4: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean },
    cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    if (!matmulQ4Pipeline) throw new Error('INT4 matmul not compiled (model is not quantized)');
    if (!q4.qweight || !q4.scales || !q4.qzeros || !q4.g_idx) {
      const missing = [
        !q4.qweight && 'qweight', !q4.scales && 'scales',
        !q4.qzeros && 'qzeros', !q4.g_idx && 'g_idx',
      ].filter(Boolean).join(', ');
      throw new Error(`matmul_q4 "${label}": missing buffers: ${missing}`);
    }
    const params = getCachedUniform(
      new Uint32Array([M, N, K, config.quantGroupSize]), `${label}-p`);
    // GEMV fast path: at M=1 the tiled kernel wastes 15/16 threads per
    // workgroup. Swap to the dedicated GEMV shader which is one-thread-per-
    // output-column. Bindings are identical → bind group unchanged.
    // Variant selection:
    //   - trivial g_idx → USE_ACTORDER=0, USE_SCALE_CACHE=0 (fast path)
    //   - actorder, cache shape fits num_groups → cached path
    //   - actorder, num_groups exceeds every candidate shape → uncached fallback
    const useGemv = M === 1 && config.isQuantized;
    const gs = config.quantGroupSize || 128;
    const numGroups = Math.ceil(K / gs);
    let gemvPipeline: GPUComputePipeline | null = null;
    let gemvWgSize = 128;
    if (useGemv) {
      if (q4.hasActOrder) {
        const shape = pickGemvCacheShape(numGroups);
        if (shape) {
          gemvPipeline = getGemvPipeline(true, true, shape.maxGroups, shape.wgSize);
          gemvWgSize = shape.wgSize;
        } else {
          gemvPipeline = getGemvPipeline(true, false, GEMV_UNCACHED_SHAPE.maxGroups, GEMV_UNCACHED_SHAPE.wgSize);
          gemvWgSize = GEMV_UNCACHED_SHAPE.wgSize;
        }
      } else {
        gemvPipeline = getGemvPipeline(false, false, GEMV_UNCACHED_SHAPE.maxGroups, GEMV_UNCACHED_SHAPE.wgSize);
        gemvWgSize = GEMV_UNCACHED_SHAPE.wgSize;
      }
    }
    const pipeline = useGemv && gemvPipeline ? gemvPipeline : matmulQ4Pipeline;
    const bg = cachedBindGroup(pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: q4.qweight } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: q4.scales } },
      { binding: 5, resource: { buffer: q4.qzeros } },
      { binding: 6, resource: { buffer: q4.g_idx } },
    ], label);
    const dispatchDims: [number, number, number] = useGemv && gemvPipeline
      ? [Math.ceil(N / gemvWgSize), 1, 1]
      : [Math.ceil(M / 16), Math.ceil(N / 16), 1];
    bd(pipeline, [bg], dispatchDims, label);
  }

  /** C[M,N] = A[M,K] @ dequant_e8(indices, scales, offsets, codebook)^T — E8 2-bit */
  function dispatchMatmulE8(
    aBuf: GPUBuffer,
    e8: { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer },
    cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    if (!matmulE8Pipeline) throw new Error('E8 matmul not compiled (model is not quantized)');
    const codebookBuf = weights.global.e8Codebook;
    if (!codebookBuf) throw new Error(`matmul_e8 "${label}": no E8 codebook loaded`);
    if (!e8.indices || !e8.scales || !e8.offsets) {
      const missing = [
        !e8.indices && 'indices', !e8.scales && 'scales', !e8.offsets && 'offsets',
      ].filter(Boolean).join(', ');
      throw new Error(`matmul_e8 "${label}": missing buffers: ${missing}`);
    }
    const params = getCachedUniform(
      new Uint32Array([M, N, K, config.quantGroupSize]), `${label}-p`);
    const bg = cachedBindGroup(matmulE8Pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: e8.indices } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: e8.scales } },
      { binding: 5, resource: { buffer: e8.offsets } },
      { binding: 6, resource: { buffer: codebookBuf } },
    ], label);
    bd(matmulE8Pipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
  }

  /** C[M,N] = A[M,K] @ dequant_q8(packed, scales, zeros)^T — INT8 */
  let q8DispatchCount = 0;
  function dispatchMatmulQ8(
    aBuf: GPUBuffer,
    q8: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean },
    cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    if (!matmulQ8Pipeline) throw new Error('INT8 matmul not compiled (model is not quantized)');
    if (q8DispatchCount < 1) {
      console.log(`[Q8] First dispatch ${label}: M=${M} N=${N} K=${K}`);
    }
    q8DispatchCount++;
    if (!q8.qweight || !q8.scales || !q8.qzeros || !q8.g_idx) {
      const missing = [
        !q8.qweight && 'qweight', !q8.scales && 'scales',
        !q8.qzeros && 'qzeros', !q8.g_idx && 'g_idx',
      ].filter(Boolean).join(', ');
      throw new Error(`matmul_q8 "${label}": missing buffers: ${missing}`);
    }
    const params = getCachedUniform(
      new Uint32Array([M, N, K, config.quantGroupSize]), `${label}-p`);
    const bg = cachedBindGroup(matmulQ8Pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: q8.qweight } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: q8.scales } },
      { binding: 5, resource: { buffer: q8.qzeros } },
      { binding: 6, resource: { buffer: q8.g_idx } },
    ], label);
    bd(matmulQ8Pipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);

  }

  /** Optional byte-offset windows into the A (input) and C (output) buffers.
   *  Offsets must be 256-B aligned (WebGPU minStorageBufferOffsetAlignment).
   *  Lets fusion-lever call sites bind a token/section slice of a larger
   *  buffer instead of staging it through a copy. */
  interface MatmulIO { aOffset?: number; cOffset?: number }
  function ioRes(buf: GPUBuffer, offset?: number): GPUBindingResource {
    return offset ? { buffer: buf, offset } : { buffer: buf };
  }

  /** C[M,N] = A[M,K] @ dequant_gguf(W)^T — native GGUF k-quant blocks. */
  function dispatchMatmulGGUF(
    aBuf: GPUBuffer, gg: GGUFWeight, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
    io?: MatmulIO,
  ) {
    // Decode (M=1): prefer vec4-W, then no-stage (opt-in), then staged
    // tiled; prefill keeps staged tiled. Same grid math for all three.
    const fast = M === 1
      ? matmulGgufV4Pipelines?.[gg.ggmlType] ?? matmulGgufNsPipelines?.[gg.ggmlType] ?? null
      : null;
    const tiled = fast ?? matmulGgufTiledPipelines?.[gg.ggmlType] ?? null;
    const pipeline = tiled ?? matmulGgufPipelines?.[gg.ggmlType];
    if (!pipeline) throw new Error(`matmul_gguf "${label}": no pipeline for ggml type ${gg.ggmlType}`);
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = cachedBindGroup(pipeline, 0, [
      { binding: 0, resource: ioRes(aBuf, io?.aOffset) },
      { binding: 1, resource: ioRes(gg.data, gg.byteOffset) },
      { binding: 2, resource: ioRes(cBuf, io?.cOffset) },
      { binding: 3, resource: { buffer: params } },
    ], label);
    if (tiled) {
      // One workgroup per TN output elements; z chunks past the 65535 cap.
      const nWG = Math.ceil(N / GEMV_TILE_N);
      bd(pipeline, [bg], [Math.min(nWG, 65535), M, Math.ceil(nWG / 65535)], label);
    } else {
      // One workgroup per output element (n, m); z chunks n past the 65535 cap.
      bd(pipeline, [bg], [Math.min(N, 65535), M, Math.ceil(N / 65535)], label);
    }
  }

  // Set of GPU buffers stored as BF16 (need matmul_bt_bf16 kernel)
  const bf16Set = weights.bf16Buffers ?? new Set<GPUBuffer>();

  /** Dequant Q4 weights → temp BF16 buffer, then matmul with BF16 kernel.
   *  Two dispatches per call: dequant compute + BF16 matmul. */
  function dispatchDequantAndMatmul(
    inputBuf: GPUBuffer,
    q4: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer },
    outputBuf: GPUBuffer, M: number, N: number, K: number, label: string,
  ) {
    const Khalf = Math.ceil(K / 2);
    const dqParams = getCachedUniform(new Uint32Array([N, K, config.quantGroupSize, 0]), `${label}-dq-p`);
    const dqBG = cachedBindGroup(dequantQ4Pipeline!, 0, [
      { binding: 0, resource: { buffer: q4.qweight } },
      { binding: 1, resource: { buffer: q4.scales } },
      { binding: 2, resource: { buffer: q4.qzeros } },
      { binding: 3, resource: { buffer: q4.g_idx } },
      { binding: 4, resource: { buffer: dequantTempBuf! } },
      { binding: 5, resource: { buffer: dqParams } },
    ], `${label}-dequant`);
    const totalWGs = Math.ceil((N * Khalf) / 256);
    const wgX = Math.min(totalWGs, 65535);
    const wgY = Math.ceil(totalWGs / wgX);
    bd(dequantQ4Pipeline!, [dqBG], [wgX, wgY], `${label}-dequant`);

    dispatchMatmulBTBF16(inputBuf, dequantTempBuf!, outputBuf, M, N, K, label);
  }

  /** Dispatch f32/BF16/INT4/E8/INT8 matmul depending on weight type.
   *  When gpuDequant=true and weight is Q4, dequants to temp BF16 first. */
  function dispatchProjection(
    inputBuf: GPUBuffer, lw: LayerWeights, proj: string,
    outputBuf: GPUBuffer, M: number, N: number, K: number, label: string,
    gpuDequant = false,
    io?: MatmulIO,
  ) {
    // Priority: GGUF k-quant > E8 2-bit > INT8 > INT4 GPTQ > BF16 > f32
    const ggKey = `${proj}_gg` as keyof LayerWeights;
    const gg = lw[ggKey] as GGUFWeight | undefined;
    if (gg) {
      dispatchMatmulGGUF(inputBuf, gg, outputBuf, M, N, K, label, io);
      return;
    }
    // io windows are only supported on paths whose kernels take plain A/C
    // storage bindings (GGUF above, f32/BF16 below). The exotic quant paths
    // never co-occur with the ssmOffsetBindings gate (GGUF-only).
    if (io && (io.aOffset || io.cOffset)) {
      const wkeyIO = proj as keyof LayerWeights;
      const wBufIO = lw[wkeyIO] as GPUBuffer | undefined;
      if (!wBufIO) throw new Error(`dispatchProjection "${label}": io windows unsupported for quantized non-GGUF path`);
    }

    const e8key = `${proj}_e8` as keyof LayerWeights;
    const e8 = lw[e8key] as { indices: GPUBuffer; scales: GPUBuffer; offsets: GPUBuffer } | undefined;
    if (e8) {
      dispatchMatmulE8(inputBuf, e8, outputBuf, M, N, K, label);
      return;
    }

    const q8key = `${proj}_q8` as keyof LayerWeights;
    const q8 = lw[q8key] as { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean } | undefined;
    if (q8) {
      dispatchMatmulQ8(inputBuf, q8, outputBuf, M, N, K, label);
      return;
    }

    const q4key = `${proj}_q4` as keyof LayerWeights;
    const q4 = lw[q4key] as { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean } | undefined;
    if (q4) {
      if (gpuDequant && dequantQ4Pipeline && dequantTempBuf) {
        dispatchDequantAndMatmul(inputBuf, q4, outputBuf, M, N, K, label);
      } else {
        dispatchMatmulQ4(inputBuf, q4, outputBuf, M, N, K, label);
      }
    } else {
      const wkey = proj as keyof LayerWeights;
      const wBuf = lw[wkey] as GPUBuffer;
      if (bf16Set.has(wBuf)) {
        dispatchMatmulBTBF16(inputBuf, wBuf, outputBuf, M, N, K, label, io);
      } else {
        dispatchMatmulBT(inputBuf, wBuf, outputBuf, M, N, K, label, io);
      }
    }
  }

  // ── Debug: read first N values from a GPU buffer ────────────────────
  let debugCallCount = 0;
  async function debugRead(buf: GPUBuffer, label: string, n = 8) {
    await device.queue.onSubmittedWorkDone();
    const raw = await readBuffer(device, buf, n * 4);
    const vals = new Float32Array(raw);
    const str = Array.from(vals).map(v => v.toFixed(4)).join(', ');
    console.log(`[DEBUG ${label}] first ${n}: [${str}]`);
  }

  // ── Debug: full-buffer stats + sampled last-row vector for divergence probe ──
  // Reads the whole buffer, computes min/max/absMean/std/nan/inf over all
  // elements, and extracts a sample vector of up to `sampleLen` f32 values
  // starting at `sampleOffset`. Pushes {label, size, stats, sample} to the
  // supplied array. Caller is responsible for flushing the batch + awaiting
  // GPU completion before calling.
  async function dumpBufStats(
    buf: GPUBuffer,
    label: string,
    totalFloats: number,
    sampleOffset: number,
    sampleLen: number,
    results: any[],
  ) {
    const raw = await readBuffer(device, buf, totalFloats * 4);
    const vals = new Float32Array(raw);
    let mn = Infinity, mx = -Infinity, sum = 0, absSum = 0, sqSum = 0;
    let nanCount = 0, infCount = 0;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (Number.isNaN(v)) { nanCount++; continue; }
      if (!Number.isFinite(v)) { infCount++; continue; }
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
      absSum += Math.abs(v);
      sqSum += v * v;
    }
    const valid = vals.length - nanCount - infCount;
    const mean = valid > 0 ? sum / valid : 0;
    const absMean = valid > 0 ? absSum / valid : 0;
    const variance = valid > 0 ? (sqSum / valid) - mean * mean : 0;
    const std = Math.sqrt(Math.max(0, variance));
    const end = Math.min(sampleOffset + sampleLen, vals.length);
    const sample = Array.from(vals.slice(sampleOffset, end));
    const stats = {
      size: vals.length,
      min: mn === Infinity ? 0 : mn,
      max: mx === -Infinity ? 0 : mx,
      mean, absMean, std, nanCount, infCount,
    };
    results.push({ label, stats, sample });
    console.log(
      `[DUMP ${label}] n=${vals.length} min=${stats.min.toFixed(4)} max=${stats.max.toFixed(4)} `
      + `absMean=${absMean.toFixed(4)} std=${std.toFixed(4)} NaN=${nanCount} Inf=${infCount}`
    );
  }

  // ── Batched dispatch support ─────────────────────────────────────
  // When currentBatch is set, all dispatches and buffer copies are accumulated
  // in a single GPUCommandEncoder and submitted together via flushBatch().
  // This reduces ~900 queue.submit() calls per forward pass to 1.
  let currentBatch: BatchedDispatcher | null = null;
  let deferredDestroys: GPUBuffer[] = [];

  /** Dispatch a compute pass — batched if currentBatch is set, else immediate. */
  // ── Performance instrumentation (Step 1: measurement layer) ────────
  // Module-scope counters so we can attribute per-forward dispatch/copy load
  // without threading a context object through every helper. The counters are
  // sampled at forward() entry/exit; the deltas are what gets logged.
  let __perfDispatchCount = 0;
  let __perfCopyCount = 0;
  let __perfBindGroupCount = 0;
  // C3: MoE step breakdown — staging map wait vs CPU expert compute.
  let __perfMoESyncMs = 0;
  let __perfMoEExpertMs = 0;

  function bd(
    pipeline: GPUComputePipeline, bindGroups: GPUBindGroup[],
    workgroupCounts: [number, number?, number?], label?: string,
  ) {
    __perfDispatchCount++;
    if (currentBatch) {
      let tsWrites: GPUComputePassTimestampWrites | undefined;
      if (__timingCtx && __timingCtx.used + 2 <= __timingCtx.capacity) {
        const begin = __timingCtx.used;
        tsWrites = {
          querySet: __timingCtx.querySet,
          beginningOfPassWriteIndex: begin,
          endOfPassWriteIndex: begin + 1,
        };
        const cat = pipelineCategories.get(pipeline) ?? label ?? 'unknown';
        __timingCtx.categories.push(cat);
        __timingCtx.used += 2;
      }
      currentBatch.dispatch(pipeline, bindGroups, workgroupCounts, label, tsWrites);
    } else {
      dispatch(device, pipeline, bindGroups, workgroupCounts, label ?? '');
    }
  }

  /** Destroy a buffer — deferred if batching, immediate otherwise. */
  function deferDestroy(buf: GPUBuffer) {
    if (currentBatch) {
      deferredDestroys.push(buf);
    } else {
      buf.destroy();
    }
  }

  /** Copy buffer — batched if currentBatch is set, else immediate submit. */
  function batchCopy(
    src: GPUBuffer, srcOff: number,
    dst: GPUBuffer, dstOff: number,
    size: number,
  ) {
    __perfCopyCount++;
    if (currentBatch) {
      currentBatch.copyBuffer(src, srcOff, dst, dstOff, size);
    } else {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, srcOff, dst, dstOff, size);
      device.queue.submit([enc.finish()]);
    }
  }

  /** Flush current batch, destroy deferred buffers, reset for more work. */
  function flushBatch() {
    if (currentBatch) {
      currentBatch.flush();
      for (const buf of deferredDestroys) buf.destroy();
      deferredDestroys = [];
      currentBatch.reset('forward-cont');
    }
  }

  // ── Dispatch helpers ───────────────────────────────────────────────

  function dispatchMatmul(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = cachedBindGroup(matmulPipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
    ], label);
    bd(matmulPipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
  }

  /** C[M,N] = A[M,K] @ B^T[K,N] where B is stored as [N,K] (HF weight format) */
  function dispatchMatmulBT(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
    io?: MatmulIO,
  ) {
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = cachedBindGroup(matmulBTPipeline, 0, [
      { binding: 0, resource: ioRes(aBuf, io?.aOffset) },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: ioRes(cBuf, io?.cOffset) },
      { binding: 3, resource: { buffer: params } },
    ], label);
    bd(matmulBTPipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
  }

  /** C[M,N] = A[M,K] @ B_bf16^T[K,N] where B is stored as BF16 packed [N,K/2] u32 */
  function dispatchMatmulBTBF16(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
    io?: MatmulIO,
  ) {
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = cachedBindGroup(matmulBTBF16Pipeline, 0, [
      { binding: 0, resource: ioRes(aBuf, io?.aOffset) },
      { binding: 2, resource: ioRes(cBuf, io?.cOffset) },
      { binding: 3, resource: { buffer: params } },
      { binding: 5, resource: { buffer: bBuf } },
    ], label);
    bd(matmulBTBF16Pipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
  }

  // Qwen3_5 uses (1+weight) in RMSNorm — detect from model type
  const useResidualWeight = config.modelType === 'qwen3_5_text' ? 1 : 0;

  // Hadamard rotation flag — set when model was quantized with --hadamard (online rotation)
  // KLT models use offline fusion — no runtime rotation needed, so disable Hadamard
  const useHadamard = !!(config.quantHadamard && !config.quantKLT && hadamardPipeline);

  /** Apply Fast Walsh-Hadamard Transform to buffer (in-place via copy). */
  function dispatchHadamard(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    rows: number, cols: number, signSeed: number, label: string,
  ) {
    if (!hadamardPipeline) return;
    const params = getCachedUniform(
      new Uint32Array([cols, rows, signSeed]), `${label}-p`);
    const bg = cachedBindGroup(hadamardPipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ], label);
    bd(hadamardPipeline, [bg], [rows], label);
  }

  function dispatchRMSNorm(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    weightBuf: GPUBuffer, rows: number, label: string,
    dim: number = H, skipWeight = false,
  ) {
    const paramData = new ArrayBuffer(16);
    new Uint32Array(paramData, 0, 1)[0] = dim;
    new Float32Array(paramData, 4, 1)[0] = eps;
    new Uint32Array(paramData, 8, 1)[0] = useResidualWeight;
    new Uint32Array(paramData, 12, 1)[0] = skipWeight ? 1 : 0;
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);
    const bg = cachedBindGroup(rmsnormPipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: paramBuf } },
    ], label);
    bd(rmsnormPipeline, [bg], [rows], label);
  }

  function dispatchElementwise(
    pipeline: GPUComputePipeline,
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    size: number, label: string, secondBuf?: GPUBuffer,
    broadcastB?: number, scale?: number,
    aSlice?: { len: number; stride: number; off: number },
  ) {
    const ewParamData = new ArrayBuffer(32);
    new Uint32Array(ewParamData, 0, 2).set([size, broadcastB ?? 0]);
    new Float32Array(ewParamData, 8, 1)[0] = scale ?? 0;
    new Uint32Array(ewParamData, 12, 3).set([aSlice?.len ?? 0, aSlice?.stride ?? 0, aSlice?.off ?? 0]);
    const params = getCachedUniform(new Uint8Array(ewParamData), `${label}-p`);
    const entries: Array<{ binding: number; resource: GPUBindingResource }> = [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ];
    if (secondBuf) entries.push({ binding: 3, resource: { buffer: secondBuf } });
    const bg = cachedBindGroup(pipeline, 0, entries, label);
    bd(pipeline, [bg], [workgroupCount(size, 256)], label);
  }

  function dispatchRoPE(
    qkBuf: GPUBuffer, seqLen: number, numHeads: number,
    posOffset: number, label: string,
    headDimOverride?: number, rotaryDimOverride?: number,
    thetaOverride?: number, rotatedPairs?: number,
  ) {
    const hd = headDimOverride ?? dHead;
    const rd = rotaryDimOverride ?? 0;
    const rp = rotatedPairs ?? 0;
    // RoPE params struct: [seq_len, head_dim, num_heads, pos_offset, rope_base, rotary_dim, rotated_pairs]
    const paramData = new ArrayBuffer(32);
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = seqLen;
    u32View[1] = hd;
    u32View[2] = numHeads;
    u32View[3] = posOffset;
    f32View[4] = thetaOverride ?? ropeTheta;
    u32View[5] = rd;
    u32View[6] = rp;
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);

    const bg = cachedBindGroup(ropePipeline, 0, [
      { binding: 0, resource: { buffer: qkBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
    ], label);

    const rotDim = rd > 0 ? rd : hd;
    const activePairs = rp > 0 ? rp : rotDim / 2;
    const totalPairs = seqLen * numHeads * activePairs;
    bd(ropePipeline, [bg], [workgroupCount(totalPairs, 256)], label);
  }

  const MAX_ATTN_CACHE = MAX_ATTN_SEQ_LEN;

  function dispatchAttention(
    qBuf: GPUBuffer, kCacheBuf: GPUBuffer, vCacheBuf: GPUBuffer,
    outputBuf: GPUBuffer, newSeqLen: number, cacheLen: number,
    isCausal: boolean, posOffset: number, label: string,
    headDimOverride?: number, window?: number,
  ) {
    if (cacheLen > MAX_ATTN_CACHE) {
      throw new Error(
        `[attention] cacheLen ${cacheLen} exceeds workgroup limit ${MAX_ATTN_CACHE}. ` +
        `Silent clamping would discard tokens and produce incorrect output. ` +
        `Cap input + generation at ${MAX_ATTN_CACHE} tokens or rebuild attention.wgsl ` +
        `with a larger var<workgroup> scores array.`,
      );
    }
    const hd = headDimOverride ?? dHead;
    const paramData = new ArrayBuffer(48);
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = nHeads;
    u32View[1] = nKVHeads;
    u32View[2] = hd;
    u32View[3] = newSeqLen;
    u32View[4] = cacheLen;
    f32View[5] = config.attnScale ?? 1.0 / Math.sqrt(hd);
    u32View[6] = isCausal ? 1 : 0;
    u32View[7] = posOffset;
    u32View[8] = window ?? 0;
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);

    const useSoftpick = (globalThis as any).__USE_SOFTPICK__ === true;
    const pipe = useSoftpick ? attentionPipelineSoftpick : attentionPipeline;
    const bg = cachedBindGroup(pipe, 0, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: paramBuf } },
    ], label);

    bd(pipe, [bg], [newSeqLen, nHeads], label);
  }

  function dispatchAttentionTQ(
    qBuf: GPUBuffer, kCacheBuf: GPUBuffer, vCacheBuf: GPUBuffer,
    outputBuf: GPUBuffer, newSeqLen: number, cacheLen: number,
    isCausal: boolean, posOffset: number,
    signBitsK: GPUBuffer, normsK: GPUBuffer, residualNormsK: GPUBuffer,
    label: string,
  ) {
    if (cacheLen > MAX_ATTN_CACHE) {
      throw new Error(
        `[attentionTQ] cacheLen ${cacheLen} exceeds workgroup limit ${MAX_ATTN_CACHE}. ` +
        `Silent clamping would discard tokens and produce incorrect output. ` +
        `Cap input + generation at ${MAX_ATTN_CACHE} tokens or rebuild attention_tq.wgsl ` +
        `with a larger var<workgroup> scores array.`,
      );
    }
    // Params: standard attention params + qjl_constant + sign_words_per_vec
    const paramData = new ArrayBuffer(48); // 10 fields * 4 bytes, padded to 48
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = nHeads;
    u32View[1] = nKVHeads;
    u32View[2] = dHead;
    u32View[3] = newSeqLen;
    u32View[4] = cacheLen;
    f32View[5] = 1.0 / Math.sqrt(dHead);
    u32View[6] = isCausal ? 1 : 0;
    u32View[7] = posOffset;
    f32View[8] = Math.sqrt(Math.PI / 2) / Math.sqrt(dHead); // qjl_constant
    u32View[9] = tqSignWords; // sign_words_per_vec
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);

    const useSoftpick = (globalThis as any).__USE_SOFTPICK__ === true;
    const pipe = useSoftpick ? attentionTqPipelineSoftpick : attentionTqPipeline;
    const bg0 = cachedBindGroup(pipe, 0, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: paramBuf } },
    ], `${label}-g0`);

    const bg1 = cachedBindGroup(pipe, 1, [
      { binding: 0, resource: { buffer: signBitsK } },
      { binding: 1, resource: { buffer: normsK } },
      { binding: 2, resource: { buffer: residualNormsK } },
      { binding: 3, resource: { buffer: tqSetup.spiMatrix } },
    ], `${label}-g1`);

    bd(pipe, [bg0, bg1], [newSeqLen, nHeads], label);
  }

  function copyToKVCache(
    srcBuf: GPUBuffer, cacheBuf: GPUBuffer,
    seqLen: number, dim: number, position: number,
  ) {
    const srcBytes = seqLen * dim * 4;
    const dstOffset = position * dim * 4;
    batchCopy(srcBuf, 0, cacheBuf, dstOffset, srcBytes);
  }

  // ── TurboQuant dispatch helpers ─────────────────────────────────────

  function dispatchTQEncode(
    inputBuf: GPUBuffer,
    outQuantBuf: GPUBuffer, outSignBuf: GPUBuffer, outNormsBuf: GPUBuffer,
    outResidualNormsBuf: GPUBuffer,
    numVecs: number, outVecOffset: number, label: string,
  ) {
    const params = getCachedUniform(
      new Uint32Array([dHead, TQ_BITS, tqCodebook.centroids.length,
        tqCodebook.thresholds.length, outVecOffset]),
      `${label}-p`);
    const bg0 = cachedBindGroup(tqEncodePipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outQuantBuf } },
      { binding: 2, resource: { buffer: outSignBuf } },
      { binding: 3, resource: { buffer: outNormsBuf } },
      { binding: 4, resource: { buffer: outResidualNormsBuf } },
    ], `${label}-g0`);
    const bg2 = cachedBindGroup(tqEncodePipeline, 2, [
      { binding: 0, resource: { buffer: params } },
    ], `${label}-g2`);
    bd(tqEncodePipeline, [bg0, tqEncodeMatBG, bg2], [numVecs], label);
  }

  function dispatchTQDecode(
    inQuantBuf: GPUBuffer, inSignBuf: GPUBuffer, inNormsBuf: GPUBuffer,
    outputBuf: GPUBuffer, numVecs: number, label: string,
  ) {
    const params = getCachedUniform(
      new Uint32Array([dHead, TQ_BITS, tqCodebook.centroids.length,
        tqCodebook.thresholds.length]),
      `${label}-p`);
    const bg0 = cachedBindGroup(tqDecodePipeline, 0, [
      { binding: 0, resource: { buffer: inQuantBuf } },
      { binding: 1, resource: { buffer: inSignBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: inNormsBuf } },
    ], `${label}-g0`);
    const bg2 = cachedBindGroup(tqDecodePipeline, 2, [
      { binding: 0, resource: { buffer: params } },
    ], `${label}-g2`);
    bd(tqDecodePipeline, [bg0, tqDecodeMatBG, bg2], [numVecs], label);
  }

  // ── Forward Pass ───────────────────────────────────────────────────

  async function forward(tokenIds: Uint32Array, kvCache: KVCache, opts?: ForwardOptions): Promise<ForwardOutput> {
    const seqLen = tokenIds.length;
    const pos = kvCache.position;
    const cacheLen = pos + seqLen;
    // Check global debug flag — auto-test sets this before each run
    const g = globalThis as any;
    const debugLastPos = g.__DEBUG_LAST_PREFILL_POS__;
    const fireAtLastPrefill = typeof debugLastPos === 'number' && pos === debugLastPos;
    const isDebug = (g.__DEBUG_FORWARD_PASS__ === true && pos === 0) || fireAtLastPrefill;
    const isDebug2 = false;
    if (isDebug && pos === 0) g.__DEBUG_FORWARD_PASS__ = false; // first-call one-shot
    if (fireAtLastPrefill) g.__DEBUG_LAST_PREFILL_POS__ = undefined; // last-prefill one-shot
    debugCallCount++;

    // Divergence-probe dump: when __DEBUG_DUMP_STATS__ is set, collect
    // per-layer stats + last-token hidden states for comparison vs a PyTorch
    // reference. The flag is consumed (set to false) so caller can re-arm
    // at subsequent steps with distinct tags. Results accumulate in
    // g.__DEBUG_DUMP_RESULT__, keyed by tag, so multiple dump points (prefill-
    // end, decode-1, decode-10, ...) co-exist in a single run.
    //
    // Flag shape: `true` → default tag "dump"; string → use as tag.
    const dumpFlag = g.__DEBUG_DUMP_STATS__;
    const dumpFire = dumpFlag === true || typeof dumpFlag === 'string';
    const dumpTag: string = typeof dumpFlag === 'string' ? dumpFlag : 'dump';
    const dumpResults: any[] | null = dumpFire ? [] : null;
    if (dumpFire) {
      g.__DEBUG_DUMP_STATS__ = false; // consume; caller re-arms for next step
      console.log(`[DUMP ${dumpTag}] armed for forward seqLen=${seqLen} pos=${pos}`);
    }
    const lastRowOffset = (seqLen - 1) * H;

    // ── Per-tensor audit ───────────────────────────────────────────
    // When __DEBUG_AUDIT_LAYERS__ is set to a list of layer indices, dump the
    // last-row output of each linear projection inside those layers. Only
    // fires on dump-active forward calls (so the cost is paid alongside the
    // existing per-layer dump). Labels are generic (`L${l}-${proj}-out`) so
    // this works for any future model family — the engine emits a flat label
    // list, and the comparison script maps labels to HF submodule paths via a
    // per-family JSON file (see scripts/compare_label_map.*.json).
    const auditLayersRaw = g.__DEBUG_AUDIT_LAYERS__;
    const auditLayers = new Set<number>(
      Array.isArray(auditLayersRaw) ? (auditLayersRaw as number[]) : []
    );
    async function maybeAudit(layer: number, buf: GPUBuffer, label: string, M: number, N: number) {
      if (!dumpResults || !auditLayers.has(layer)) return;
      flushBatch();
      await device.queue.onSubmittedWorkDone();
      const lastRowStart = Math.max(0, (M - 1) * N);
      await dumpBufStats(buf, label, M * N, lastRowStart, N, dumpResults);
    }

    // ── Performance instrumentation (Step 1) ─────────────────────────
    // Count CPU time and dispatch/copy load for this forward pass. The numbers
    // are written to globalThis.__perfLastForward so generate.ts can compose a
    // per-decode-step breakdown, and an auto-log fires for the first few calls.
    const __perfT0 = performance.now();
    const __perfDispStart = __perfDispatchCount;
    const __perfCopyStart = __perfCopyCount;
    const __perfBGStart = __perfBindGroupCount;
    __perfMoESyncMs = 0;
    __perfMoEExpertMs = 0;

    // Upload token IDs
    device.queue.writeBuffer(tokenIdBuf, 0, tokenIds.buffer, tokenIds.byteOffset, tokenIds.byteLength);

    // Batch all GPU work into a single submit for speed
    currentBatch = new BatchedDispatcher(device, 'forward');
    deferredDestroys = [];

    // Activate per-kernel timing on the first N eligible decode calls. Decode
    // (seqLen==1) is where the optimization effort is focused and dispatch
    // counts are stable, so profiling there gives the most actionable numbers.
    if (supportsTimestamps && __timingCallsRemaining > 0 && seqLen === 1) {
      const bufs = ensureTimingBuffers();
      __timingCtx = {
        querySet: bufs.querySet,
        resolveBuf: bufs.resolveBuf,
        readBuf: bufs.readBuf,
        capacity: bufs.capacity,
        used: 0,
        categories: [],
      };
    }

    // ── Embedding (injected, CPU BF16, split BF16, single BF16/f32, GPTQ INT4) ──
    if (opts?.embeddings) {
      // Multimodal injection: the caller computed this chunk's hidden rows
      // (e.g. vision-encoder image patches) — skip the token embed stage.
      const injected = opts.embeddings;
      if (injected.length !== seqLen * H) {
        throw new Error(`[forward] injected embeddings: ${injected.length} floats != seqLen*H = ${seqLen * H}`);
      }
      device.queue.writeBuffer(
        hiddenBuf, 0,
        injected.buffer as ArrayBuffer, injected.byteOffset, injected.byteLength);
    } else if (weights.global.embedGG) {
      // GGUF embed: CPU row-gather + k-quant dequant of the token's row
      const { data, ggmlType, rowBytes } = weights.global.embedGG;
      const f32 = new Float32Array(seqLen * H);
      for (let t = 0; t < seqLen; t++) {
        const row = data.subarray(tokenIds[t] * rowBytes, (tokenIds[t] + 1) * rowBytes);
        if (row.byteLength !== rowBytes) {
          throw new Error(`[EMBED GGUF] token ${tokenIds[t]}: row slice ${row.byteLength} != ${rowBytes} bytes`);
        }
        f32.set(dequantGGML(ggmlType, row, H), t * H);
      }
      // Gemma: scale embeddings by √hiddenSize (gemma4.cpp:155)
      if (config.embedScale) {
        for (let i = 0; i < f32.length; i++) f32[i] *= config.embedScale;
      }
      device.queue.writeBuffer(hiddenBuf, 0, f32.buffer, 0, seqLen * H * 4);
    } else if (weights.global.embedCPU) {
      const { parts, splitPoint, hiddenSize: embedH, isBF16 } = weights.global.embedCPU;
      const f32 = new Float32Array(seqLen * embedH);
      for (let t = 0; t < seqLen; t++) {
        const tokenId = tokenIds[t];
        const partIdx = Math.min(Math.floor(tokenId / splitPoint), parts.length - 1);
        const adjustedId = tokenId - partIdx * splitPoint;
        const rowByteOff = adjustedId * embedH * 2;
        const row = new Uint16Array(parts[partIdx].buffer, parts[partIdx].byteOffset + rowByteOff, embedH);
        const outOff = t * embedH;
        if (isBF16) {
          for (let i = 0; i < embedH; i++) {
            _cpuEmbedU32[0] = row[i] << 16;
            f32[outOff + i] = _cpuEmbedF32[0];
          }
        } else {
          for (let i = 0; i < embedH; i++) {
            const bits = row[i];
            const sign = (bits >> 15) & 1;
            const exp = (bits >> 10) & 0x1F;
            const frac = bits & 0x3FF;
            if (exp === 0) { f32[outOff + i] = frac === 0 ? 0 : (sign ? -1 : 1) * frac / 1024 * 2 ** -14; }
            else if (exp === 31) { f32[outOff + i] = sign ? -1e30 : 1e30; }
            else { _cpuEmbedU32[0] = (sign << 31) | ((exp + 112) << 23) | (frac << 13); f32[outOff + i] = _cpuEmbedF32[0]; }
          }
        }
      }
      device.queue.writeBuffer(hiddenBuf, 0, f32.buffer, 0, seqLen * embedH * 4);
    } else if (weights.global.embedSplit) {
      // Split BF16 embedding — lossless, needed for SSM models where INT4 causes drift
      const split = weights.global.embedSplit;
      const embedParams = getCachedUniform(
        new Uint32Array([H, seqLen, split.splitPoints[0], 0]), 'embed-split-p');
      const embedBG = cachedBindGroup(embedF16SplitPipeline, 0, [
        { binding: 0, resource: { buffer: tokenIdBuf } },
        { binding: 1, resource: { buffer: hiddenBuf } },
        { binding: 2, resource: { buffer: split.buffers[0] } },
        { binding: 3, resource: { buffer: embedParams } },
        { binding: 4, resource: { buffer: split.buffers[1] } },
      ], 'embed-split');
      bd(embedF16SplitPipeline, [embedBG], [seqLen], 'embed-split');
      if (isDebug) console.log(`[EMBED-PATH] split-bf16, splitPoint=${split.splitPoints[0]}, bufs=${split.buffers.length}`);
    } else if (weights.global.embedQ4 && embedQ4Pipeline) {
      // GPTQ INT4 embedding — dequant on the fly per token
      const eq4 = weights.global.embedQ4;
      const embedParams = getCachedUniform(
        new Uint32Array([H, seqLen, config.quantGroupSize || 128, V]), 'embed-q4-p');
      const embedBG = cachedBindGroup(embedQ4Pipeline, 0, [
        { binding: 0, resource: { buffer: tokenIdBuf } },
        { binding: 1, resource: { buffer: hiddenBuf } },
        { binding: 2, resource: { buffer: eq4.qweight } },
        { binding: 3, resource: { buffer: embedParams } },
        { binding: 4, resource: { buffer: eq4.scales } },
        { binding: 5, resource: { buffer: eq4.qzeros } },
      ], 'embed-q4');
      bd(embedQ4Pipeline, [embedBG], [seqLen], 'embed-q4');
      if (isDebug) console.log(`[EMBED-PATH] q4, qweight.size=${eq4.qweight.size}`);
    } else {
      const useF16Embed = weights.global.embedIsF16 === true;
      const embedPipe = useF16Embed ? embedF16Pipeline : embedPipeline;
      const embedParams = getCachedUniform(new Uint32Array([H, seqLen]), 'embed-p');
      const embedBG = cachedBindGroup(embedPipe, 0, [
        { binding: 0, resource: { buffer: tokenIdBuf } },
        { binding: 1, resource: { buffer: hiddenBuf } },
        { binding: 2, resource: { buffer: weights.global.embedTokens } },
        { binding: 3, resource: { buffer: embedParams } },
      ], 'embed');
      bd(embedPipe, [embedBG], [seqLen], 'embed');
      if (isDebug) {
        const tbl = weights.global.embedTokens;
        const expectedBF16 = V * H * 2;
        const expectedF32 = V * H * 4;
        console.log(
          `[EMBED-PATH] ${useF16Embed ? 'f16/bf16' : 'f32'}, `
          + `embedTokens.size=${tbl.size} (expected BF16=${expectedBF16}, F32=${expectedF32}), `
          + `H=${H}, V=${V}, seqLen=${seqLen}`
        );
      }
    }

    // Debug: dump embed output on first call (uncomment to diagnose INT4 embed issues)
    // if (debugCallCount <= 1) {
    //   flushBatch();
    //   await device.queue.onSubmittedWorkDone();
    //   const raw = new Float32Array(await readBuffer(device, hiddenBuf, 16 * 4));
    //   const isQ4 = !!(weights.global.embedQ4 && embedQ4Pipeline);
    //   console.log(`[EMBED ${isQ4 ? 'Q4' : 'BF16'}] first 16: [${Array.from(raw).map(v => v.toFixed(6)).join(', ')}]`);
    //   const allZero = raw.every(v => v === 0);
    //   const hasNaN = raw.some(v => isNaN(v));
    //   const hasInf = raw.some(v => !isFinite(v));
    //   if (allZero) console.error('[EMBED] WARNING: all zeros — embed dispatch likely failed');
    //   if (hasNaN) console.error('[EMBED] WARNING: NaN detected — dequant bug');
    //   if (hasInf) console.error('[EMBED] WARNING: Inf detected — overflow');
    // }
    if (dumpResults) {
      flushBatch();
      await device.queue.onSubmittedWorkDone();
      await dumpBufStats(hiddenBuf, 'embed-out', seqLen * H, lastRowOffset, H, dumpResults);
    }
    if (isDebug) {
      flushBatch();
      await debugRead(hiddenBuf, 'embed-out', 8);
      // Also sample the embed table directly to verify binding/data
      try {
        const tbl = weights.global.embedTokens;
        const firstTok = tokenIds[0];
        const useF16 = weights.global.embedIsF16 === true;
        const isQ4 = !!(weights.global.embedQ4 && embedQ4Pipeline);
        if (!isQ4) {
          // Read 16 bytes from start of embed table (row 0) and from row for first token
          const head = new Uint8Array(await readBuffer(device, tbl, 16));
          const rowOffset = useF16 ? firstTok * H * 2 : firstTok * H * 4;
          const rowBytes = useF16 ? 16 : 32;
          // Only read if within buffer
          if (rowOffset + rowBytes <= tbl.size) {
            const row = new Uint8Array(await readBuffer(device, tbl, rowBytes, rowOffset));
            const hex = (b: Uint8Array) => Array.from(b).map(x => x.toString(16).padStart(2, '0')).join(' ');
            console.log(`[EMBED-TABLE] head16=[${hex(head)}]`);
            console.log(`[EMBED-TABLE] token ${firstTok} row (first ${rowBytes} bytes) =[${hex(row)}]`);
          } else {
            console.log(`[EMBED-TABLE] row offset ${rowOffset} + ${rowBytes} exceeds buffer size ${tbl.size}`);
          }
        }
      } catch (e) {
        console.error('[EMBED-TABLE] readback failed:', e);
      }
    }

    // ── Gemma 4 PLE input (once per forward) ─────────────────────────
    // gemma4.cpp:145-173:
    //   inp_per_layer = get_rows(per_layer_tok_embd, tokens) × √dPle
    //   per_layer_proj = RMSNorm(per_layer_model_proj @ inpL × 1/√H, proj_norm)
    //   combined = (per_layer_proj + inp_per_layer) × 1/√2
    // Token-major layout [tokens][L][dPle] (ggml permutes to layer-major;
    // we instead use a strided slice view in the per-layer gate_gelu).
    if (pleDim > 0 && weights.global.pleTokenEmbedGG
        && pleInpBuf && pleProjBuf && pleCombinedBuf && pleProjScaleBuf) {
      const { data, parts, rowsPerPart, ggmlType, rowBytes } = weights.global.pleTokenEmbedGG;
      const tokScale = Math.sqrt(dPle);
      const pleF32 = new Float32Array(seqLen * pleDim);
      for (let t = 0; t < seqLen; t++) {
        let row: Uint8Array;
        if (parts && rowsPerPart) {
          const r = tokenIds[t] % rowsPerPart;
          row = parts[Math.floor(tokenIds[t] / rowsPerPart)]
            .subarray(r * rowBytes, (r + 1) * rowBytes);
        } else {
          row = data.subarray(tokenIds[t] * rowBytes, (tokenIds[t] + 1) * rowBytes);
        }
        if (row.byteLength !== rowBytes) {
          throw new Error(`[PLE GGUF] token ${tokenIds[t]}: row slice ${row.byteLength} != ${rowBytes} bytes`);
        }
        const vals = dequantGGML(ggmlType, row, pleDim);
        for (let i = 0; i < pleDim; i++) pleF32[t * pleDim + i] = vals[i] * tokScale;
      }
      device.queue.writeBuffer(pleInpBuf, 0, pleF32.buffer, 0, seqLen * pleDim * 4);
      // per_layer_proj = model_proj @ inpL (inpL = embed stream, already ×√H)
      dispatchProjection(hiddenBuf, weights.global as unknown as LayerWeights, 'pleModelProj',
        pleProjBuf, seqLen, pleDim, H, 'ple-model-proj');
      // × 1/√H BEFORE the norm — RMSNorm is scale-invariant in the limit, but
      // skipping this would scale the effective eps by H (1e-6 → 2.56e-3).
      dispatchElementwise(mulPipeline, pleProjBuf, pleCombinedBuf, seqLen * pleDim,
        'ple-proj-scale', pleProjScaleBuf, 1);
      // RMSNorm over each dPle-sized row (seqLen × L rows)
      dispatchRMSNorm(pleCombinedBuf, pleProjBuf, weights.global.pleProjNorm!,
        seqLen * L, 'ple-proj-norm', dPle);
      // combined = (per_layer_proj + inp_per_layer) × 1/√2
      dispatchElementwise(addscalePipeline!, pleProjBuf, pleCombinedBuf, seqLen * pleDim,
        'ple-combine', pleInpBuf, 0, Math.SQRT1_2);
    }

    // ── Transformer layers ───────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const lw = weights.layers[l];
      const isLinearLayer = config.layers[l].kind === 'linear_attention';

      // Save hidden state for residual connection
      batchCopy(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);

      // Pre-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.inputNorm, seqLen, `L${l}-norm1`);

      // QuIP#/QuaRot: Hadamard rotation after norm, before attention projections
      // Only for full attention layers — linear attention weights are BF16 (not rotated)
      if (useHadamard && !isLinearLayer) {
        batchCopy(normedBuf, 0, ffnTempBuf, 0, seqLen * H * 4);
        dispatchHadamard(ffnTempBuf, normedBuf, seqLen, H, 0, `L${l}-had1`);
      }

      if (isLinearLayer && kvCache.ssmState) {
        // ── GATED DELTANET LINEAR ATTENTION ──────────────────────────
        const ssmIdx = kvCache.ssmState.layerToSSMIndex[l];
        const hBuf = kvCache.ssmState.hiddenStates[ssmIdx];
        const csBuf = kvCache.ssmState.convStates[ssmIdx];

        // Option A (chunked-prefill): the gated DeltaNet recurrence is sequential
        // by construction, but the surrounding stages (embed, full attention, FFN,
        // final norm, lm_head) all support seqLen > 1. Wrap the existing single-
        // token SSM block in a per-token JS loop so multi-token chunks work without
        // changing any SSM/conv kernel. ssmInputBuf is the per-token slice of
        // normedBuf; ssmOutBuf is the per-token output written back into
        // attnProjBuf at offset ssmT*H*4 so downstream stages see all tokens.
        for (let ssmT = 0; ssmT < seqLen; ssmT++) {
          // F2: bind this token's slice of normedBuf directly via a 256-B-
          // aligned offset window instead of staging it through ssmInputBuf.
          const ssmInSrc = ssmOffsetBindings ? normedBuf : ssmInputBuf!;
          const ssmInIO = ssmOffsetBindings ? { aOffset: ssmT * H * 4 } : undefined;
          if (!ssmOffsetBindings) {
            // Slice this token's normed input out of the multi-token buffer.
            batchCopy(normedBuf, ssmT * H * 4, ssmInputBuf!, 0, H * 4);
          }

          // Debug: normed input before projections (first token only to keep logs sane)
          if (isDebug && l === 0 && ssmT === 0) {
            flushBatch();
            await debugRead(ssmInSrc, 'L0-normed-input', 8);
          }

          // 1. Fused QKV projection
          dispatchProjection(ssmInSrc, lw, 'linearInProjQKV', linQKVBuf!, 1, linQKVDim, H, `L${l}-lin-qkv`, needsGpuDequant, ssmInIO);
          if (ssmT === seqLen - 1) await maybeAudit(l, linQKVBuf!, `L${l}-lin-qkv-out`, 1, linQKVDim);

        // Debug: raw QKV projection output (before conv1d) — first token only
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          // Read first 8 values of Q, K, V sections from the fused QKV buffer
          // Q at [0..2047], K at [2048..4095], V at [4096..8191]
          const fullQKV = new Float32Array(await readBuffer(device, linQKVBuf!, (4104) * 4));
          console.log(`[REF] QKV raw Q[0:8]: [${Array.from(fullQKV.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
          console.log(`[REF] QKV raw K[0:8]: [${Array.from(fullQKV.slice(2048, 2056)).map(v => v.toFixed(4)).join(', ')}]`);
          console.log(`[REF] QKV raw V[0:8]: [${Array.from(fullQKV.slice(4096, 4104)).map(v => v.toFixed(4)).join(', ')}]`);
          // PyTorch ref: Q=[0.138, 0.203, 0.464, 0.496, 0.108, -0.119, -3.181, 0.365]
          // PyTorch ref: K=[1.925, 2.020, 0.374, 0.841, 0.669, 2.595, -0.756, 0.715]
          // PyTorch ref: V=[-0.386, 4.992, -3.642, -3.278, -5.948, 1.927, 2.007, -0.526]
        }

        // 2. Conv1d on ENTIRE QKV (8192 channels) BEFORE split
        // Conv1d weight is [8192, 1, 4] — all channels go through causal conv
        // F4: fused conv+state+SiLU writes to linConvOutBuf (it must not alias
        // its input linQKVBuf), so downstream consumers read postConvQKV.
        // Legacy path SiLUs back into linQKVBuf.
        const postConvQKV = conv1dSiluPipeline ? linConvOutBuf! : linQKVBuf!;
        if (conv1dSiluPipeline) {
          const convParams = getCachedUniform(
            new Uint32Array([linQKVDim, linConvK]), `L${l}-conv-p`);
          const convBG = cachedBindGroup(conv1dSiluPipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 2, resource: { buffer: lw.linearConv1dWeight! } },
            { binding: 3, resource: { buffer: linConvOutBuf! } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv-fused`);
          bd(conv1dSiluPipeline, [convBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d-silu`);
        } else if (conv1dPipeline && conv1dUpdatePipeline) {
          const convParams = getCachedUniform(
            new Uint32Array([linQKVDim, linConvK]), `L${l}-conv-p`);
          const convBG = cachedBindGroup(conv1dPipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 2, resource: { buffer: lw.linearConv1dWeight! } },
            { binding: 3, resource: { buffer: linConvOutBuf! } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv`);
          bd(conv1dPipeline, [convBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d`);

          // Update conv state (shift + append raw QKV)
          const updateBG = cachedBindGroup(conv1dUpdatePipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv-upd`);
          bd(conv1dUpdatePipeline, [updateBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d-update`);

          // SiLU on entire conv output (all 8192 channels)
          dispatchElementwise(siluPipeline, linConvOutBuf!, linQKVBuf!, linQKVDim, `L${l}-conv-silu`);
        }

        // 3. Split QKV AFTER conv+silu
        // F2: under offset bindings the split copies are skipped — the l2norm
        // dispatches below read the Q/K sections of linQKVBuf via offset
        // windows (writing normed Q/K into linQBuf/linKBuf), and ssm_step
        // binds the V section of linQKVBuf directly.
        const qSize = linNKH * linKD * 4;
        const kSize = linNKH * linKD * 4;
        const vSize = linNVH * linVD * 4;
        if (!ssmOffsetBindings) {
          batchCopy(postConvQKV, 0, linQBuf!, 0, qSize);
          batchCopy(postConvQKV, qSize, linKBuf!, 0, kSize);
          batchCopy(postConvQKV, qSize + kSize, linVBuf!, 0, vSize);
        }

        // Debug: conv1d output and silu output — first token only
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const qkvAfterSilu = new Float32Array(await readBuffer(device, postConvQKV, (4104) * 4));
          console.log(`[REF] After conv+silu Q[0:8]: [${Array.from(qkvAfterSilu.slice(0, 8)).map(v => v.toFixed(6)).join(', ')}]`);
          console.log(`[REF] After conv+silu K[0:8]: [${Array.from(qkvAfterSilu.slice(2048, 2056)).map(v => v.toFixed(6)).join(', ')}]`);
          console.log(`[REF] After conv+silu V[0:8]: [${Array.from(qkvAfterSilu.slice(4096, 4104)).map(v => v.toFixed(6)).join(', ')}]`);
          // PyTorch ref: Q=[-0.007, 0.011, 0.021, 0.020, -0.004, -7e-5, 0.007, -0.014]
          // PyTorch ref: K=[0.008, -0.007, 0.001, -0.001, 0.001, 0.153, -0.030, -0.001]
          // PyTorch ref: V=[0.002, 0.049, 0.034, 0.043, 0.066, 0.016, 0.018, 0.003]
        }

        // 4. Project A, B, Z (these use the ORIGINAL normed input, not conv output)
        // NOTE: in_proj_b output is num_v_heads (32), NOT num_k_heads (16)!
        // Option A: read from per-token slice ssmInputBuf instead of multi-token normedBuf.
        dispatchProjection(ssmInSrc, lw, 'linearInProjA', linABuf!, 1, linNVH, H, `L${l}-lin-a`, needsGpuDequant, ssmInIO);
        dispatchProjection(ssmInSrc, lw, 'linearInProjB', linBBuf!, 1, linNVH, H, `L${l}-lin-b`, needsGpuDequant, ssmInIO);
        dispatchProjection(ssmInSrc, lw, 'linearInProjZ', linZBuf!, 1, linZDim, H, `L${l}-lin-z`, needsGpuDequant, ssmInIO);
        if (ssmT === seqLen - 1) {
          await maybeAudit(l, linABuf!, `L${l}-lin-a-out`, 1, linNVH);
          await maybeAudit(l, linBBuf!, `L${l}-lin-b-out`, 1, linNVH);
          await maybeAudit(l, linZBuf!, `L${l}-lin-z-out`, 1, linZDim);
        }

        // Debug: A, B, Z projection outputs — first token only
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const aVals = new Float32Array(await readBuffer(device, linABuf!, 4 * 4));
          const bVals = new Float32Array(await readBuffer(device, linBBuf!, 4 * 4));
          const zVals = new Float32Array(await readBuffer(device, linZBuf!, 8 * 4));
          console.log(`[REF] A (raw) first 4: [${Array.from(aVals).map(v => v.toFixed(4)).join(', ')}]`);
          console.log(`[REF] B (raw) first 4: [${Array.from(bVals).map(v => v.toFixed(4)).join(', ')}]`);
          console.log(`[REF] Z first 8: [${Array.from(zVals).map(v => v.toFixed(4)).join(', ')}]`);
          // PyTorch ref: A=[6.650, 0.602, 8.459, 12.811]
          // PyTorch ref: B=[-3.521, 1.663, -1.692, 0.444]
          // PyTorch ref: Z=[1.414, -0.271, -1.269, -0.706, -0.792, -0.701, 0.285, -0.571]
        }

        // NOTE: NO RoPE in linear attention layers! Only full attention uses RoPE.

        // 5. Sigmoid on beta (num_v_heads=32 scalars)
        // F3: write sigmoid(beta) to its own buffer; the legacy copy-back to
        // linBBuf existed only because linDtBuf is reused by softplus below.
        // F5: sigmoid/softplus/decay fold into the ssm_step_fused prologue —
        // skip all three elementwise dispatches.
        const fuseSsm = !!(ssmStepFusedPipeline && lw.linearDtBias && lw.linearALog);
        const betaBuf = ssmOffsetBindings ? linBetaBuf! : linBBuf!;
        if (sigmoidPipeline && !fuseSsm) {
          if (ssmOffsetBindings) {
            dispatchElementwise(sigmoidPipeline, linBBuf!, linBetaBuf!, linNVH, `L${l}-sigmoid-beta`);
          } else {
            dispatchElementwise(sigmoidPipeline, linBBuf!, linDtBuf!, linNVH, `L${l}-sigmoid-beta`);
            batchCopy(linDtBuf!, 0, linBBuf!, 0, linNVH * 4);
          }
        }

        // 5c. L2-normalize Q and K (use_qk_l2norm_in_kernel=True in reference)
        // Uses separate shader (l2norm.wgsl) to avoid binding conflicts with elementwise.wgsl
        if (l2NormPipeline) {
          const qDim = linNKH * linKD;
          const kDim = linNKH * linKD;

          if (ssmOffsetBindings) {
            // F2+F3: read Q/K sections of postConvQKV via offset windows and
            // write normed values straight into linQBuf/linKBuf — no split
            // copies, no copy-backs through linConvOutBuf.
            const l2pQ = getCachedUniform(new Uint32Array([qDim, linKD]), `L${l}-l2q-p`);
            const l2bgQ = cachedBindGroup(l2NormPipeline, 0, [
              { binding: 0, resource: { buffer: postConvQKV } },
              { binding: 1, resource: { buffer: linQBuf! } },
              { binding: 2, resource: { buffer: l2pQ } },
            ], `L${l}-l2norm-q`);
            bd(l2NormPipeline, [l2bgQ], [workgroupCount(qDim, 256)], `L${l}-l2norm-q`);

            const l2pK = getCachedUniform(new Uint32Array([kDim, linKD]), `L${l}-l2k-p`);
            const l2bgK = cachedBindGroup(l2NormPipeline, 0, [
              { binding: 0, resource: { buffer: postConvQKV, offset: qSize } },
              { binding: 1, resource: { buffer: linKBuf! } },
              { binding: 2, resource: { buffer: l2pK } },
            ], `L${l}-l2norm-k`);
            bd(l2NormPipeline, [l2bgK], [workgroupCount(kDim, 256)], `L${l}-l2norm-k`);

            // Debug: verify L2 norm actually wrote to output — first token only
            if (isDebug && l === 0 && ssmT === 0) {
              flushBatch();
              await device.queue.onSubmittedWorkDone();
              const rawQKV = await readBuffer(device, postConvQKV, qSize + 8 * 4);
              const rawK = new Float32Array(rawQKV).slice(qSize / 4, qSize / 4 + 8);
              const normK = await readBuffer(device, linKBuf!, 8 * 4);
              console.log(`[L2 DEBUG] K input first 8: [${Array.from(rawK).map(v => v.toFixed(4)).join(', ')}]`);
              console.log(`[L2 DEBUG] K output (linKBuf) first 8: [${Array.from(new Float32Array(normK)).map(v => v.toFixed(4)).join(', ')}]`);
            }
          } else {
            const l2pQ = getCachedUniform(new Uint32Array([qDim, linKD]), `L${l}-l2q-p`);
            const l2bgQ = cachedBindGroup(l2NormPipeline, 0, [
              { binding: 0, resource: { buffer: linQBuf! } },
              { binding: 1, resource: { buffer: linConvOutBuf! } },
              { binding: 2, resource: { buffer: l2pQ } },
            ], `L${l}-l2norm-q`);
            bd(l2NormPipeline, [l2bgQ], [workgroupCount(qDim, 256)], `L${l}-l2norm-q`);
            batchCopy(linConvOutBuf!, 0, linQBuf!, 0, qDim * 4);

            const l2pK = getCachedUniform(new Uint32Array([kDim, linKD]), `L${l}-l2k-p`);
            const l2bgK = cachedBindGroup(l2NormPipeline, 0, [
              { binding: 0, resource: { buffer: linKBuf! } },
              { binding: 1, resource: { buffer: linConvOutBuf! } },
              { binding: 2, resource: { buffer: l2pK } },
            ], `L${l}-l2norm-k`);
            bd(l2NormPipeline, [l2bgK], [workgroupCount(kDim, 256)], `L${l}-l2norm-k`);

            // Debug: verify L2 norm actually wrote to output — first token only
            if (isDebug && l === 0 && ssmT === 0) {
              flushBatch();
              await device.queue.onSubmittedWorkDone();
              const rawK = await readBuffer(device, linKBuf!, 8 * 4);
              const normK = await readBuffer(device, linConvOutBuf!, 8 * 4);
              console.log(`[L2 DEBUG] K input first 8: [${Array.from(new Float32Array(rawK)).map(v => v.toFixed(4)).join(', ')}]`);
              console.log(`[L2 DEBUG] K output (linConvOutBuf) first 8: [${Array.from(new Float32Array(normK)).map(v => v.toFixed(4)).join(', ')}]`);
            }

            batchCopy(linConvOutBuf!, 0, linKBuf!, 0, kDim * 4);
          }
        }

        // 6. Decay per VALUE HEAD [32], not per key dim
        // A_log is [32], dt_bias is [32], in_proj_a output is [32]
        if (softplusPipeline && lw.linearDtBias && !fuseSsm) {
          dispatchElementwise(softplusPipeline, linABuf!, linDtBuf!, linNVH, `L${l}-softplus`, lw.linearDtBias);
        }
        if (decayPipeline && lw.linearALog && !fuseSsm) {
          dispatchElementwise(decayPipeline, lw.linearALog, linDecayBuf!, linNVH, `L${l}-decay`, linDtBuf!);
        }

        // Debug SSM intermediates (layer 0, first token only)
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          if (!fuseSsm) {
            const betaRaw = await readBuffer(device, betaBuf, linNKH * 4);
            const beta = new Float32Array(betaRaw);
            console.log(`[SSM DEBUG] beta (sigmoid) first 4: [${Array.from(beta.slice(0, 4)).map(v => v.toFixed(4)).join(', ')}]`);

            const decayRaw = await readBuffer(device, linDecayBuf!, Math.min(linNVH, 8) * 4);
            const decay = new Float32Array(decayRaw);
            console.log(`[SSM DEBUG] decay (per vh, ${linNVH} total) first 8: [${Array.from(decay.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
          }

          const kNormRaw = await readBuffer(device, linKBuf!, 8 * 4);
          const kNorm = new Float32Array(kNormRaw);
          console.log(`[SSM DEBUG] K (L2-normed) first 8: [${Array.from(kNorm.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

          const vVals = ssmOffsetBindings
            ? new Float32Array(await readBuffer(device, postConvQKV, qSize + kSize + 8 * 4))
              .slice((qSize + kSize) / 4, (qSize + kSize) / 4 + 8)
            : new Float32Array(await readBuffer(device, linVBuf!, 8 * 4));
          console.log(`[SSM DEBUG] V first 8: [${Array.from(vVals.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

          const qRaw = await readBuffer(device, linQBuf!, 8 * 4);
          const qVals = new Float32Array(qRaw);
          console.log(`[SSM DEBUG] Q first 8: [${Array.from(qVals.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
        }

        // 7. SSM step: update hidden state, readout via Q
        // Decay is per value head [32], not per key dim
        if (ssmStepPipeline) {
          // layout_tiled: GGUF V/beta/decay buffers are in llama.cpp's tiled
          // v-head order (convert_hf_to_gguf reorders grouped → tiled); HF
          // safetensors are grouped. ssm_step must index accordingly.
          const ssmLayoutTiled = config.sourceFormat === 'gguf' ? 1 : 0;
          const ssmParams = getCachedUniform(
            new Uint32Array([linNKH, linNVH, linKD, linGroupedVD, ssmLayoutTiled, 0, 0, 0]), `L${l}-ssm-p`);
          // F5: fused entry takes RAW B (binding 3) / RAW A (binding 4) plus
          // dt_bias/A_log and computes beta/decay in its prologue.
          // Occupancy lever: split entry uses the same bindings but a
          // (nkh, ceil(gvd/SSM_WG)) grid with one thread per (kh, v) column.
          const ssmSplit = fuseSsm && !!ssmStepSplitPipeline;
          const ssmPipe = ssmSplit ? ssmStepSplitPipeline!
            : fuseSsm ? ssmStepFusedPipeline! : ssmStepPipeline;
          const ssmBG0 = cachedBindGroup(ssmPipe, 0, [
            { binding: 0, resource: { buffer: linQBuf! } },
            { binding: 1, resource: { buffer: linKBuf! } }, // K after conv1d + silu + L2 norm
            // F2: V is never split out — bind its section of postConvQKV.
            { binding: 2, resource: ssmOffsetBindings
              ? { buffer: postConvQKV, offset: qSize + kSize }
              : { buffer: linVBuf! } },
            ...(fuseSsm ? [
              { binding: 3, resource: { buffer: linBBuf! } },         // raw B proj
              { binding: 4, resource: { buffer: linABuf! } },         // raw A proj
              { binding: 5, resource: { buffer: lw.linearDtBias! } },
              { binding: 6, resource: { buffer: lw.linearALog! } },
            ] : [
              { binding: 3, resource: { buffer: betaBuf } },        // beta (after sigmoid)
              { binding: 4, resource: { buffer: linDecayBuf! } },   // decay = exp(-exp(A_log)*dt)
            ]),
          ], `L${l}-ssm-g0${fuseSsm ? '-f' : ''}`);
          const ssmBG1 = cachedBindGroup(ssmPipe, 1, [
            { binding: 0, resource: { buffer: hBuf } },
            { binding: 1, resource: { buffer: linOutBuf! } },
          ], `L${l}-ssm-g1`);
          const ssmBG2 = cachedBindGroup(ssmPipe, 2, [
            { binding: 0, resource: { buffer: ssmParams } },
          ], `L${l}-ssm-g2`);
          bd(ssmPipe, [ssmBG0, ssmBG1, ssmBG2],
            ssmSplit ? [linNKH, Math.ceil(linGroupedVD / SSM_SPLIT_WG)] : [linNKH],
            `L${l}-ssm-step`);
        }

        // ── SSM state-drift probe ────────────────────────────────────
        // Gated on __DEBUG_SSM_STATE__ (number): how many forward() calls between
        // samples. Small value (1) = every step, expensive. Larger (e.g. 50) =
        // sparse sampling, cheap. Set to undefined/0 to disable.
        //
        // Reads back the per-layer hidden state `h`, computes ‖h‖₂ and max|h| on
        // CPU, logs one line per SSM layer. Only fires on the LAST token of a
        // chunk (so prefill chunks log once at chunk end; decode logs every
        // sampled step). This is the primary diagnostic for the Move-2
        // hypothesis that INT4 quantization causes the DeltaNet recurrent state
        // to drift into a degenerate attractor over long decodes.
        {
          const ssmProbeInterval = (globalThis as any).__DEBUG_SSM_STATE__ as number | undefined;
          const ssmProbeEnabled = typeof ssmProbeInterval === 'number'
            && ssmProbeInterval > 0
            && (debugCallCount % ssmProbeInterval === 0)
            && ssmT === seqLen - 1;
          if (ssmProbeEnabled) {
            flushBatch();
            await device.queue.onSubmittedWorkDone();
            const hBytes = await readBuffer(device, hBuf, hBuf.size);
            const hArr = new Float32Array(hBytes);
            let sumSq = 0;
            let maxAbs = 0;
            let nanCount = 0;
            let nonzeroCount = 0;
            for (let i = 0; i < hArr.length; i++) {
              const v = hArr[i];
              if (Number.isNaN(v)) { nanCount++; continue; }
              const a = Math.abs(v);
              if (a > 0) nonzeroCount++;
              sumSq += v * v;
              if (a > maxAbs) maxAbs = a;
            }
            const norm = Math.sqrt(sumSq);
            const pct = ((nonzeroCount / hArr.length) * 100).toFixed(1);

            // Channel-saturation probe: collapse h[NKH, KD, VD] over (NKH, KD)
            // to a per-channel max over VD, then log the top-K channels.
            // Stable aggregate norm can hide a single channel saturating.
            // A frozen attractor would show the SAME top channels at every probe.
            const probeChannels = (globalThis as any).__DEBUG_SSM_CHANNELS__ as number | undefined;
            const topK = typeof probeChannels === 'number' && probeChannels > 0 ? probeChannels : 0;
            if (topK > 0) {
              const VD = linGroupedVD;
              const nGroups = Math.floor(hArr.length / VD);
              // Per-channel (VD index) max over all (head, kd) groups.
              const chanMax = new Float32Array(VD);
              for (let g = 0; g < nGroups; g++) {
                const base = g * VD;
                for (let c = 0; c < VD; c++) {
                  const a = Math.abs(hArr[base + c]);
                  if (a > chanMax[c]) chanMax[c] = a;
                }
              }
              // Top-K selection via simple partial sort (VD is small, e.g. 256).
              const idx = new Array<number>(VD);
              for (let i = 0; i < VD; i++) idx[i] = i;
              idx.sort((a, b) => chanMax[b] - chanMax[a]);
              const top = idx.slice(0, Math.min(topK, VD))
                .map(i => `${i}:${chanMax[i].toExponential(2)}`)
                .join(',');
              console.log(
                `[SSM-PROBE fwd#${debugCallCount} pos=${pos + ssmT} L${l}] `
                + `‖h‖=${norm.toExponential(3)} `
                + `max|h|=${maxAbs.toExponential(3)} `
                + `nonzero=${pct}% `
                + (nanCount > 0 ? `NaN=${nanCount} ` : '')
                + `n=${hArr.length} `
                + `topVD=[${top}]`
              );
            } else {
              console.log(
                `[SSM-PROBE fwd#${debugCallCount} pos=${pos + ssmT} L${l}] `
                + `‖h‖=${norm.toExponential(3)} `
                + `max|h|=${maxAbs.toExponential(3)} `
                + `nonzero=${pct}% `
                + (nanCount > 0 ? `NaN=${nanCount} ` : '')
                + `n=${hArr.length}`
              );
            }
          }
        }

        // Debug SSM output (layer 0, first token only)
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const ssmOutRaw = await readBuffer(device, linOutBuf!, 8 * 4);
          const ssmOut = new Float32Array(ssmOutRaw);
          console.log(`[SSM DEBUG] SSM output first 8: [${Array.from(ssmOut.slice(0, 8)).map(v => v.toFixed(6)).join(', ')}]`);
        }

        // 8. RMSNormGated on output — per-head RMSNorm then multiply by silu(Z)
        // Reference: self.norm(core_attn_out, z) using Qwen3_5RMSNormGated
        // Treats output as [32 heads, 128 dims], normalizes each head independently
        // Uses weight directly (NOT 1+weight — RMSNormGated initializes to ones)
        if (lw.linearNormWeight) {
          const gnParamData = new ArrayBuffer(16);
          new Uint32Array(gnParamData, 0, 1)[0] = linVD; // hidden_size = 128 (per head)
          new Float32Array(gnParamData, 4, 1)[0] = eps;
          new Uint32Array(gnParamData, 8, 1)[0] = 0; // use_residual_weight = 0 (weight directly)
          const gnParams = getCachedUniform(new Uint8Array(gnParamData), `L${l}-gn-p`);
          // Option A: RMSNormGated and the subsequent silu-gate write to a per-token
          // scratch (ssmOutBuf) instead of attnProjBuf. Otherwise this iteration
          // would clobber tokens already written at offsets 0..(ssmT-1)*H by prior
          // iterations of the per-token loop. The final value of ssmOutBuf is
          // copied into attnProjBuf at the correct per-token offset below.
          const gnBG = cachedBindGroup(rmsnormPipeline, 0, [
            { binding: 0, resource: { buffer: linOutBuf! } },
            { binding: 1, resource: { buffer: ssmOutBuf! } },
            { binding: 2, resource: { buffer: lw.linearNormWeight } },
            { binding: 3, resource: { buffer: gnParams } },
          ], `L${l}-rms-gated`);
          // Dispatch 32 workgroups — one per value head (each normalizes 128 elements)
          bd(rmsnormPipeline, [gnBG], [linNVH], `L${l}-rms-gated`);

          // Multiply by silu(Z): output = normed * silu(z)
          if (gateSiluPipeline) {
            const outDim = linNKH * linGroupedVD;
            dispatchElementwise(gateSiluPipeline, ssmOutBuf!, linOutBuf!, outDim, `L${l}-gate`, linZBuf!);
          }
        }

        // 9. Output projection — F3: write straight into this token's slice of
        // the multi-token attnProjBuf via an output offset window (legacy path
        // staged through ssmOutBuf and copied). Downstream stages (residual
        // add, post-attn norm, FFN) already process all seqLen tokens batched.
        const outDim = linNKH * linGroupedVD;
        if (ssmOffsetBindings) {
          dispatchProjection(linOutBuf!, lw, 'linearOutProj', attnProjBuf, 1, H, outDim, `L${l}-lin-out`, needsGpuDequant, { cOffset: ssmT * H * 4 });
          // Audit reads tokens 0..ssmT from attnProjBuf; last row = this token.
          if (ssmT === seqLen - 1) await maybeAudit(l, attnProjBuf, `L${l}-lin-out-out`, ssmT + 1, H);
        } else {
          dispatchProjection(linOutBuf!, lw, 'linearOutProj', ssmOutBuf!, 1, H, outDim, `L${l}-lin-out`, needsGpuDequant);
          if (ssmT === seqLen - 1) await maybeAudit(l, ssmOutBuf!, `L${l}-lin-out-out`, 1, H);
        }

        // Q8 out_proj debug readback (first token only — token 0 sits at
        // offset 0 of attnProjBuf under offset bindings, so either read works)
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const outData = new Float32Array(await readBuffer(device, ssmOffsetBindings ? attnProjBuf : ssmOutBuf!, 8 * 4));
          console.log(`[Q8 DEBUG] L0 out_proj output[0:8]: [${Array.from(outData).map(v => v.toFixed(6)).join(', ')}]`);
        }

        if (!ssmOffsetBindings) {
          // Option A: write this token's SSM output into the multi-token attnProjBuf
          // at offset ssmT*H*4.
          batchCopy(ssmOutBuf!, 0, attnProjBuf, ssmT * H * 4, H * 4);
        }
      } // end per-token SSM loop (Option A)

      } else {
        // ── STANDARD SOFTMAX ATTENTION ────────────────────────────────

        // Per-layer attention geometry (Gemma 4: sliding layers use head_dim
        // 256, full-attention layers 512). Uniform models fall back to globals.
        const dHeadL = config.layers[l].headDim ?? dHead;
        const kvDimL = nKVHeads * dHeadL;
        // KV sharing (Gemma 4 layers 24-41): consume another layer's cache,
        // skip K/V projection + norm + RoPE + cache write entirely.
        const kvSrc = config.layers[l].kvSourceLayer;
        const kvLayer = kvSrc ?? l;
        const slidingWindow = config.layers[l].slidingWindow ?? 0;

        // Q, K, V projections (auto-selects f32 or INT4 matmul)
        // F1: when the deinterleave kernel runs, Q lands in attnOutBuf (not
        // qBuf); the qNorm below reads/writes in the opposite direction to
        // land normed Q in qBuf without any copy-backs.
        let qInAttnOut = false;
        if (config.attnOutputGate) {
          // Qwen3.5: Q proj outputs [nHeads, dHead*2] — interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
          // Project to qBuf (sized 2x), then deinterleave → attnOutBuf (Q) + attnGateBuf (gate)
          dispatchProjection(normedBuf, lw, 'qProj', qBuf, seqLen, nHeads * dHead * 2, H, `L${l}-q`);
          await maybeAudit(l, qBuf, `L${l}-q-out`, seqLen, nHeads * dHead * 2);
          if (deinterleavePipeline && attnGateBuf) {
            const nElems = seqLen * nHeads * dHead;
            const params = getCachedUniform(
              new Uint32Array([nElems, dHead]), `L${l}-deint-p`);
            const bg = cachedBindGroup(deinterleavePipeline, 0, [
              { binding: 0, resource: { buffer: qBuf } },
              { binding: 1, resource: { buffer: attnOutBuf } },
              { binding: 2, resource: { buffer: attnGateBuf } },
              { binding: 3, resource: { buffer: params } },
            ], `L${l}-deint`);
            bd(deinterleavePipeline, [bg], [workgroupCount(nElems, 256)], `L${l}-deint`);
            qInAttnOut = true;
          } else {
            // Legacy CPU copy loop (?deint=0 fallback)
            for (let s = 0; s < seqLen; s++) {
              for (let h = 0; h < nHeads; h++) {
                const srcOffset = (s * nHeads * dHead * 2 + h * dHead * 2) * 4;
                const qDst = (s * nHeads * dHead + h * dHead) * 4;
                batchCopy(qBuf, srcOffset, attnOutBuf, qDst, dHead * 4);
                batchCopy(qBuf, srcOffset + dHead * 4, attnGateBuf!, qDst, dHead * 4);
              }
            }
            // Copy deinterleaved Q from attnOutBuf back to qBuf
            batchCopy(attnOutBuf, 0, qBuf, 0, seqLen * nHeads * dHead * 4);
          }
        } else {
          dispatchProjection(normedBuf, lw, 'qProj', qBuf, seqLen, nHeads * dHeadL, H, `L${l}-q`);
          await maybeAudit(l, qBuf, `L${l}-q-out`, seqLen, nHeads * dHeadL);
        }
        if (kvSrc === undefined) {
          dispatchProjection(normedBuf, lw, 'kProj', kBuf, seqLen, kvDimL, H, `L${l}-k`);
          await maybeAudit(l, kBuf, `L${l}-k-out`, seqLen, kvDimL);
          dispatchProjection(normedBuf, lw, 'vProj', vBuf, seqLen, kvDimL, H, `L${l}-v`);
          await maybeAudit(l, vBuf, `L${l}-v-out`, seqLen, kvDimL);
        }

        // Per-head RMSNorm on Q and K (q_norm, k_norm) — Qwen3.5, Gemma 4.
        // Weight convention follows useResidualWeight (same as input layernorm).
        // Gated independently: Gemma 4 KV-sharing layers (24-41) have qNorm
        // but no kNorm (attn_k_norm is dead weight, skipped by the loader).
        if (lw.qNorm) {
          // Q norm: treat [seqLen * nHeads, dHead] as rows of dHead
          if (qInAttnOut) {
            // F1: Q is in attnOutBuf — norm it straight back into qBuf,
            // eliminating the copy-back.
            dispatchRMSNorm(attnOutBuf, qBuf, lw.qNorm, seqLen * nHeads, `L${l}-qnorm`, dHeadL);
          } else {
            dispatchRMSNorm(qBuf, attnOutBuf, lw.qNorm, seqLen * nHeads, `L${l}-qnorm`, dHeadL);
            batchCopy(attnOutBuf, 0, qBuf, 0, seqLen * nHeads * dHeadL * 4);
          }
        } else if (qInAttnOut) {
          // No qNorm: move deinterleaved Q home in one bulk copy.
          batchCopy(attnOutBuf, 0, qBuf, 0, seqLen * nHeads * dHead * 4);
        }
        if (lw.kNorm && kvSrc === undefined) {
          dispatchRMSNorm(kBuf, ffnTempBuf, lw.kNorm, seqLen * nKVHeads, `L${l}-knorm`, dHeadL);
          batchCopy(ffnTempBuf, 0, kBuf, 0, seqLen * kvDimL * 4);
        }

        // Gemma 4: weightless RMSNorm on V before caching (gemma4.cpp Vcur_normed).
        // The weight buffer is a dummy — skip_weight=1 ignores it.
        if (config.modelType === 'gemma4_text' && kvSrc === undefined) {
          batchCopy(vBuf, 0, ffnTempBuf, 0, seqLen * kvDimL * 4);
          dispatchRMSNorm(ffnTempBuf, vBuf, lw.inputNorm, seqLen * nKVHeads, `L${l}-vnorm`, dHeadL, true);
        }

        // Add bias if model has attention_bias
        if (config.attentionBias && lw.qBias && lw.kBias && lw.vBias) {
          const qDim = nHeads * dHead;
          batchCopy(qBuf, 0, attnOutBuf, 0, seqLen * qDim * 4);
          dispatchElementwise(addPipeline, attnOutBuf, qBuf, seqLen * qDim, `L${l}-qb`, lw.qBias, qDim);

          batchCopy(kBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
          dispatchElementwise(addPipeline, ffnTempBuf, kBuf, seqLen * kvDim, `L${l}-kb`, lw.kBias, kvDim);

          batchCopy(vBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
          dispatchElementwise(addPipeline, ffnTempBuf, vBuf, seqLen * kvDim, `L${l}-vb`, lw.vBias, kvDim);
        }

        // Apply RoPE to Q and K
        // Qwen3.5 full attention uses partial RoPE (25% of head dims).
        // Gemma 4: per-layer theta (sliding 10k / full 1e6) + proportional
        // RoPE on full layers (only the first rotatedPairs pairs rotate).
        const fullAttnRotaryDim = config.partialRotaryFactor
          ? Math.floor(config.partialRotaryFactor * dHeadL) : 0;
        const layerRope = config.layers[l].rope;
        dispatchRoPE(qBuf, seqLen, nHeads, pos, `L${l}-rope-q`, dHeadL, fullAttnRotaryDim,
          layerRope?.theta, layerRope?.rotatedPairs);
        if (kvSrc === undefined) {
          dispatchRoPE(kBuf, seqLen, nKVHeads, pos, `L${l}-rope-k`, dHeadL, fullAttnRotaryDim,
            layerRope?.theta, layerRope?.rotatedPairs);
        }

        if (isDebug && l === 0) {
          flushBatch();
          await debugRead(qBuf, 'L0-Q-after-rope', 8);
          await debugRead(kBuf, 'L0-K-after-rope', 8);
        }

        // Write new K, V to cache and run attention
        // Gemma image spans attend bidirectionally: when the caller marks a
        // multimodal chunk bidirectional, its queries see the whole chunk
        // (plus the full cache prefix) instead of the causal triangle.
        const isCausal = seqLen > 1 && !opts?.bidirectional;

        if (kvCache.compressed) {
          const c = kvCache.compressed;
          const numVecs = seqLen * nKVHeads;
          const outOffset = pos * nKVHeads;

          // Encode K/V to compressed cache (now also stores residual norms)
          dispatchTQEncode(kBuf, c.quantizedK[l], c.signBitsK[l], c.normsK[l],
            c.residualNormsK[l], numVecs, outOffset, `L${l}-tq-enc-k`);
          dispatchTQEncode(vBuf, c.quantizedV[l], c.signBitsV[l], c.normsV[l],
            c.residualNormsV[l], numVecs, outOffset, `L${l}-tq-enc-v`);

          // Decode previous K/V (PolarQuant only — QJL correction applied in attention)
          if (pos > 0) {
            const prevVecs = pos * nKVHeads;
            dispatchTQDecode(c.quantizedK[l], c.signBitsK[l], c.normsK[l],
              c.scratchK, prevVecs, `L${l}-tq-dec-k`);
            dispatchTQDecode(c.quantizedV[l], c.signBitsV[l], c.normsV[l],
              c.scratchV, prevVecs, `L${l}-tq-dec-v`);
          }

          // Copy current (exact) K/V to scratch at the current position
          const curOffset = pos * kvDim * 4;
          const curSize = seqLen * kvDim * 4;
          batchCopy(kBuf, 0, c.scratchK, curOffset, curSize);
          batchCopy(vBuf, 0, c.scratchV, curOffset, curSize);

          // Attention on decoded K + exact current K in scratch buffers.
          // NOTE: Asymmetric attention (dispatchAttentionTQ) is available but currently
          // disabled — QJL correction variance compounds through autoregressive generation
          // with RTN-quantized weights. Re-enable once GPTQ weights reduce FFN noise.
          dispatchAttention(
            qBuf, c.scratchK, c.scratchV, attnOutBuf,
            seqLen, cacheLen, isCausal, pos, `L${l}-attn`,
          );
        } else {
          if (kvSrc === undefined) {
            copyToKVCache(kBuf, kvCache.keys[l], seqLen, kvDimL, pos);
            copyToKVCache(vBuf, kvCache.values[l], seqLen, kvDimL, pos);
          }

          // ── Attention KV-cache probe ───────────────────────────────
          // Gated on __DEBUG_ATTN_KV__ (number): interval between samples
          // in forward() calls. Only fires on attention layers (this branch)
          // and only at the last sampled position of the chunk. Reads back
          // the valid portion of kvCache.keys[l] / kvCache.values[l] up to
          // pos+seqLen and computes ‖K‖, ‖V‖, max|K|, max|V|. Smoking gun
          // for long-context collapse originating in attention (sink
          // saturation, cache dequant drift, RoPE accumulation).
          {
            const attnProbeInterval = (globalThis as any).__DEBUG_ATTN_KV__ as number | undefined;
            const attnProbeEnabled = typeof attnProbeInterval === 'number'
              && attnProbeInterval > 0
              && (debugCallCount % attnProbeInterval === 0);
            if (attnProbeEnabled) {
              flushBatch();
              await device.queue.onSubmittedWorkDone();
              const validTokens = pos + seqLen;
              const validBytes = validTokens * kvDimL * 4;
              const kBytes = await readBuffer(device, kvCache.keys[kvLayer], validBytes);
              const vBytes = await readBuffer(device, kvCache.values[kvLayer], validBytes);
              const kArr = new Float32Array(kBytes);
              const vArr = new Float32Array(vBytes);
              let kSumSq = 0, vSumSq = 0;
              let kMax = 0, vMax = 0;
              let kNaN = 0, vNaN = 0;
              for (let i = 0; i < kArr.length; i++) {
                const v = kArr[i];
                if (Number.isNaN(v)) { kNaN++; continue; }
                const a = Math.abs(v);
                kSumSq += v * v;
                if (a > kMax) kMax = a;
              }
              for (let i = 0; i < vArr.length; i++) {
                const v = vArr[i];
                if (Number.isNaN(v)) { vNaN++; continue; }
                const a = Math.abs(v);
                vSumSq += v * v;
                if (a > vMax) vMax = a;
              }
              // Last-token slice stats — isolates what was just written.
              const lastOff = pos * kvDimL;
              let kLastSq = 0, vLastSq = 0;
              let kLastMax = 0, vLastMax = 0;
              for (let i = 0; i < seqLen * kvDimL; i++) {
                const kv = kArr[lastOff + i];
                const vv = vArr[lastOff + i];
                if (!Number.isNaN(kv)) {
                  kLastSq += kv * kv;
                  const a = Math.abs(kv);
                  if (a > kLastMax) kLastMax = a;
                }
                if (!Number.isNaN(vv)) {
                  vLastSq += vv * vv;
                  const a = Math.abs(vv);
                  if (a > vLastMax) vLastMax = a;
                }
              }
              console.log(
                `[ATTN-PROBE fwd#${debugCallCount} pos=${pos + seqLen - 1} L${l}] `
                + `‖K_all‖=${Math.sqrt(kSumSq).toExponential(3)} `
                + `max|K|=${kMax.toExponential(3)} `
                + `‖V_all‖=${Math.sqrt(vSumSq).toExponential(3)} `
                + `max|V|=${vMax.toExponential(3)} `
                + `‖K_last‖=${Math.sqrt(kLastSq).toExponential(3)} `
                + `max|K_last|=${kLastMax.toExponential(3)} `
                + `‖V_last‖=${Math.sqrt(vLastSq).toExponential(3)} `
                + `max|V_last|=${vLastMax.toExponential(3)} `
                + (kNaN > 0 ? `K_NaN=${kNaN} ` : '')
                + (vNaN > 0 ? `V_NaN=${vNaN} ` : '')
                + `tokens=${validTokens} kvDim=${kvDimL}`
              );
            }
          }

          dispatchAttention(
            qBuf, kvCache.keys[kvLayer], kvCache.values[kvLayer], attnOutBuf,
            seqLen, cacheLen, isCausal, pos, `L${l}-attn`,
            dHeadL, slidingWindow,
          );
        }

        // Debug at key full_attn layers: L23 (last normal) and L27 (first blowup)
        if (isDebug && (l === 23 || l === 27)) {
          flushBatch();
          await debugRead(vBuf, `L${l}-V`, 8);
          await debugRead(attnOutBuf, `L${l}-attn-out-preGate`, 8);
          if (attnGateBuf) await debugRead(attnGateBuf, `L${l}-gate-raw`, 8);
        }

        // Qwen3.5: output gating — attn_output = attn_output * sigmoid(gate)
        // gate is in attnGateBuf (from Q projection split)
        // gate shape is [seqLen, nHeads * dHead] — reshaped from [seqLen, nHeads, dHead]
        // F7: one gate_sigmoid dispatch (a * sigmoid(b)) → ffnTempBuf; o-proj
        // reads it from there. Skipped under useHadamard (it scratches
        // ffnTempBuf) — that path keeps the legacy 3-command sequence.
        let attnGated = attnOutBuf;
        if (config.attnOutputGate && attnGateBuf && gateSigmoidPipeline && !useHadamard) {
          const gateDim = seqLen * nHeads * dHead;
          dispatchElementwise(gateSigmoidPipeline, attnOutBuf, ffnTempBuf, gateDim, `L${l}-attn-gate`, attnGateBuf);
          attnGated = ffnTempBuf;
        } else if (config.attnOutputGate && attnGateBuf && sigmoidPipeline) {
          const gateDim = seqLen * nHeads * dHead;
          // sigmoid(gate) → normedBuf
          dispatchElementwise(sigmoidPipeline, attnGateBuf, normedBuf, gateDim, `L${l}-gate-sig`);
          // attn * sigmoid(gate) → attnOutBuf (in-place via copy pattern)
          batchCopy(attnOutBuf, 0, ffnTempBuf, 0, gateDim * 4);
          dispatchElementwise(mulPipeline, ffnTempBuf, attnOutBuf, gateDim, `L${l}-attn-gate`, normedBuf);
        }

        if (isDebug && (l === 23 || l === 27)) {
          flushBatch();
          await debugRead(attnGated, `L${l}-attn-out-postGate`, 8);
        }

        // QuIP#/QuaRot: Hadamard rotation before o_proj (its weights were also rotated)
        if (useHadamard) {
          const oDim = nHeads * dHeadL;
          if ((oDim & (oDim - 1)) === 0) {
            batchCopy(attnOutBuf, 0, ffnTempBuf, 0, seqLen * oDim * 4);
            dispatchHadamard(ffnTempBuf, attnOutBuf, seqLen, oDim, 0, `L${l}-had-o`);
          }
        }

        // Output projection: [seq, nHeads*dHead] → [seq, H]
        dispatchProjection(attnGated, lw, 'oProj', attnProjBuf, seqLen, H, nHeads * dHeadL, `L${l}-o`);
        await maybeAudit(l, attnProjBuf, `L${l}-o-out`, seqLen, H);

        // O projection bias
        if (config.attentionBias && lw.oBias) {
          batchCopy(attnProjBuf, 0, normedBuf, 0, seqLen * H * 4);
          dispatchElementwise(addPipeline, normedBuf, attnProjBuf, seqLen * H, `L${l}-ob`, lw.oBias, H);
        }
      }

      // Gemma 4 sandwich norm: attn_post_norm on the attn output BEFORE the
      // residual add (gemma4.cpp attn_post_norm).
      if (lw.attnPostNorm) {
        batchCopy(attnProjBuf, 0, normedBuf, 0, seqLen * H * 4);
        dispatchRMSNorm(normedBuf, attnProjBuf, lw.attnPostNorm, seqLen, `L${l}-attn-post-norm`);
      }

      // Residual: hidden = residual + attn_output (shared for both layer types)
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res1`, attnProjBuf);

      if (isDebug && l === 0) {
        flushBatch();
        await debugRead(attnProjBuf, `L0-attn-proj(pos=${pos})`, 8);
        await debugRead(hiddenBuf, `L0-after-attn-residual(pos=${pos})`, 8);
      }
      if (isDebug && (l === 23 || l === 27)) {
        flushBatch();
        await debugRead(attnProjBuf, `L${l}-attn-proj-out`, 8);
      }

      // Save for second residual
      batchCopy(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);

      // Post-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.postAttnNorm, seqLen, `L${l}-norm2`);

      // QuIP#/QuaRot: Hadamard rotation after norm, before FFN projections
      if (useHadamard) {
        batchCopy(normedBuf, 0, ffnTempBuf, 0, seqLen * H * 4);
        dispatchHadamard(ffnTempBuf, normedBuf, seqLen, H, 0, `L${l}-had2`);
      }

      // ── FFN (SwiGLU) ───────────────────────────────────────────────
      // MODEL-SPECIFIC: Phi fuses gate+up into one projection.
      // MoE layers skip this block: the SHARED expert (same dense slots) is
      // dispatched inside the MoE branch below, AFTER the workers are kicked,
      // so its GPU time + readback overlap the CPU expert GEMVs.
      const isMoELayer = !!(lw.moeRouter && weights.moe && routerLogitsBuf);
      if (!isMoELayer) {
        if (lw.gateUpProj_gg && seqLen === 1 && !dumpResults) {
          // Same-A fusion (decode only): one GEMV computes [gate | up] into
          // gateBuf, then gate_silu reads the halves via 256-B-aligned
          // binding offsets. Byte-identical to the two-dispatch path — the
          // concat preserves every row's words (see gateup-concat above).
          dispatchMatmulGGUF(normedBuf, lw.gateUpProj_gg, gateBuf, 1, 2 * ffnDim, H, `L${l}-gateup`);
          const guParamData = new ArrayBuffer(32);
          new Uint32Array(guParamData, 0, 2).set([ffnDim, 0]);
          const guP = getCachedUniform(new Uint8Array(guParamData), `L${l}-silumul-p`);
          const guBG = cachedBindGroup(gateActPipeline, 0, [
            { binding: 0, resource: { buffer: gateBuf, offset: ffnDim * 4 } }, // input_a = up half
            { binding: 1, resource: { buffer: ffnTempBuf } },
            { binding: 2, resource: { buffer: guP } },
            { binding: 3, resource: { buffer: gateBuf, size: ffnDim * 4 } },   // input_b = gate half
          ], `L${l}-silumul-gu`);
          bd(gateActPipeline, [guBG], [workgroupCount(ffnDim, 256)], `L${l}-silumul`);
        } else {
          dispatchProjection(normedBuf, lw, 'gateProj', gateBuf, seqLen, ffnDim, H, `L${l}-gate`);
          await maybeAudit(l, gateBuf, `L${l}-gate-out`, seqLen, ffnDim);
          dispatchProjection(normedBuf, lw, 'upProj', upBuf, seqLen, ffnDim, H, `L${l}-up`);
          await maybeAudit(l, upBuf, `L${l}-up-out`, seqLen, ffnDim);

          // Fused gated activation: ffnTemp = up * act(gate) in a single dispatch
          // (act = silu for most models, gelu_tanh for Gemma — config.activation).
          // Saves one dispatch + one full ffnDim read/write of VRAM traffic per layer.
          // Old two-dispatch path (kept for reference):
          // dispatchElementwise(siluPipeline, gateBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silu`);
          // dispatchElementwise(mulPipeline, ffnTempBuf, gateBuf, seqLen * ffnDim, `L${l}-mul`, upBuf);
          dispatchElementwise(gateActPipeline, upBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silumul`, gateBuf);
        }

        dispatchProjection(ffnTempBuf, lw, 'downProj', downBuf, seqLen, H, ffnDim, `L${l}-down`);
        await maybeAudit(l, downBuf, `L${l}-down-out`, seqLen, H);
      }

      // ── MoE routed experts (Phase C) ──────────────────────────────
      // Two-phase overlap (C3): phase 1 reads back normed hidden + router
      // logits and kicks the worker fleet for token 0; phase 2 then dispatches
      // the SHARED expert (dense-FFN slots) and reads it back WHILE the
      // workers compute. Combine routed + sigmoid(g)·shared → write back into
      // downBuf so the res2 add below is unchanged.
      // Ground truth: llama-graph.cpp build_moe_ffn + models/qwen35moe.cpp.
      if (lw.moeRouter && weights.moe && routerLogitsBuf) {
        const spec = config.layers[l].moe!;
        const E = spec.numExperts;
        const topK = spec.numExpertsPerToken;
        const hBytes = seqLen * H * 4;
        const eBytes = seqLen * E * 4;

        // Phase 1: router logits + normed hidden.
        // Ordering invariant: copies are recorded AFTER the router dispatch in
        // the same batch, so a single flush covers compute + copy-to-staging.
        dispatchProjection(normedBuf, lw, 'moeRouter', routerLogitsBuf, seqLen, E, H, `L${l}-router`);
        const stagingA = moeStagingRing![moeStagingIdx];
        moeStagingIdx = (moeStagingIdx + 1) % moeStagingRing!.length;
        batchCopy(normedBuf, 0, stagingA, 0, hBytes);
        batchCopy(routerLogitsBuf, 0, stagingA, hBytes, eBytes);
        flushBatch();
        let __syncT0 = performance.now();
        await mapWithPump(stagingA, hBytes + eBytes);
        __perfMoESyncMs += performance.now() - __syncT0;
        const mappedA = stagingA.getMappedRange(0, hBytes + eBytes);
        if (mappedA.byteLength !== hBytes + eBytes) {
          throw new Error(`[MoE L${l}] staging map ${mappedA.byteLength} B, expected ${hBytes + eBytes} B`);
        }
        // slice() copies out — the views must outlive unmap().
        const normed = new Float32Array(mappedA.slice(0, hBytes));
        const logits = new Float32Array(mappedA.slice(hBytes, hBytes + eBytes));
        stagingA.unmap();

        // Route ALL tokens and kick the WHOLE chunk as one worker generation
        // (C4) — decode (seqLen === 1) and prefill chunks alike fully overlap
        // phase 2. Workers iterate the chunk's (token, expert) pairs
        // expert-major so each strip is read once per chunk.
        if (seqLen > MOE_MAX_TOKENS) {
          throw new Error(
            `[MoE L${l}] seqLen ${seqLen} > MOE_MAX_TOKENS ${MOE_MAX_TOKENS} — `
            + `keep PREFILL_CHUNK ≤ ${MOE_MAX_TOKENS} for MoE models (generate.ts)`,
          );
        }
        const routes: Array<{ ids: Int32Array; weights: Float32Array }> = [];
        for (let t = 0; t < seqLen; t++) {
          routes.push(topKSoftmax(logits.subarray(t * E, (t + 1) * E), topK));
        }
        const pendingBatch = weights.moe.backend.computeExpertsBatch(l, normed, routes);

        // Phase 2: shared expert on the dense-FFN slots + readback, racing
        // the workers. (gateSilu fusion identical to the dense path above.)
        dispatchProjection(normedBuf, lw, 'gateProj', gateBuf, seqLen, ffnDim, H, `L${l}-gate`);
        dispatchProjection(normedBuf, lw, 'upProj', upBuf, seqLen, ffnDim, H, `L${l}-up`);
        dispatchElementwise(gateActPipeline, upBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silumul`, gateBuf);
        dispatchProjection(ffnTempBuf, lw, 'downProj', downBuf, seqLen, H, ffnDim, `L${l}-down`);
        const stagingB = moeStagingRing![moeStagingIdx];
        moeStagingIdx = (moeStagingIdx + 1) % moeStagingRing!.length;
        batchCopy(downBuf, 0, stagingB, 0, hBytes);
        flushBatch();
        __syncT0 = performance.now();
        await mapWithPump(stagingB, hBytes);
        __perfMoESyncMs += performance.now() - __syncT0;
        const mappedB = stagingB.getMappedRange(0, hBytes);
        if (mappedB.byteLength !== hBytes) {
          throw new Error(`[MoE L${l}] shared staging map ${mappedB.byteLength} B, expected ${hBytes} B`);
        }
        const shared = new Float32Array(mappedB.slice(0, hBytes));
        stagingB.unmap();
        const gateVec = weights.moe.sharedGateVecs[l];
        const combined = new Float32Array(seqLen * H);

        // The batch was kicked before phase 2 — only the EXPOSED wait (after
        // the shared-expert readback) lands in moe_experts.
        const __expT0 = performance.now();
        const routedAll = await pendingBatch;
        __perfMoEExpertMs += performance.now() - __expT0;

        for (let t = 0; t < seqLen; t++) {
          const hidden = normed.subarray(t * H, (t + 1) * H);
          const { ids, weights: rw } = routes[t];
          const routed = routedAll.subarray(t * H, (t + 1) * H);
          // Scalar shared-expert gate: sigmoid(x · ffn_gate_inp_shexp)
          let g = 0;
          for (let i = 0; i < H; i++) g += hidden[i] * gateVec[i];
          g = 1 / (1 + Math.exp(-g));
          for (let i = 0; i < H; i++) combined[t * H + i] = routed[i] + g * shared[t * H + i];

          // One-shot per-layer MoE probe: set __DEBUG_MOE__ = true in the
          // console, send one message — logs every MoE layer for the first
          // forward pass, then self-clears.
          if (t === 0 && ((globalThis as any).__DEBUG_MOE__ || (globalThis as any).__DEBUG_MOE)) {
            const st = (v: Float32Array) => {
              let s = 0, mx = 0, nan = 0;
              for (let i = 0; i < v.length; i++) {
                const a = v[i];
                if (Number.isNaN(a)) nan++;
                s += a * a;
                if (Math.abs(a) > mx) mx = Math.abs(a);
              }
              return `${Math.sqrt(s).toFixed(3)}/${mx.toFixed(3)}${nan ? `/NaN${nan}` : ''}`;
            };
            console.log(
              `[MoE L${l}] normed ${st(hidden as Float32Array)} | logits ${st(logits.subarray(t * E, (t + 1) * E))} `
              + `| top8 ${Array.from(ids).join(',')} w0 ${rw[0].toFixed(3)} | routed ${st(routed)} `
              + `| shexp ${st(shared.subarray(t * H, (t + 1) * H))} g ${g.toFixed(3)} `
              + `| combined ${st(combined.subarray(t * H, (t + 1) * H))}`,
            );
            if (l === config.numLayers - 1) {
              (globalThis as any).__DEBUG_MOE__ = false;
              (globalThis as any).__DEBUG_MOE = false;
            }
          }
        }
        // writeBuffer executes in queue order BEFORE the res2 dispatch below.
        device.queue.writeBuffer(downBuf, 0, combined.buffer, 0, combined.byteLength);
      }

      // Gemma 4 sandwich norm: ffn_post_norm on the FFN output BEFORE the
      // residual add (gemma4.cpp ffn_post_norm).
      if (lw.ffnPostNorm) {
        batchCopy(downBuf, 0, normedBuf, 0, seqLen * H * 4);
        dispatchRMSNorm(normedBuf, downBuf, lw.ffnPostNorm, seqLen, `L${l}-ffn-post-norm`);
      }

      // Residual: hidden = residual + ffn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res2`, downBuf);

      // ── Gemma 4 PLE sub-block (gemma4.cpp:397-457) ───────────────────
      // x = x + RMSNorm(proj @ (gelu(inp_gate @ x) ⊙ combined[l]), post_norm)
      // then ×out_scale (broadcast [1], applied to the whole stream).
      if (pleDim > 0 && pleCombinedBuf && pleGateBuf
          && (lw.pleInpGate || lw.pleInpGate_gg)) {
        // gate = inp_gate @ x  [seqLen, dPle]
        dispatchProjection(hiddenBuf, lw, 'pleInpGate', pleGateBuf, seqLen, dPle, H, `L${l}-ple-gate`);
        // gelu(gate) ⊙ combined[l] — strided slice of token-major [tok][L][dPle]
        dispatchElementwise(gateActPipeline, pleCombinedBuf, normedBuf, seqLen * dPle,
          `L${l}-ple-gate-gelu`, pleGateBuf, 0, 0, { len: dPle, stride: pleDim, off: l * dPle });
        // proj: dPle → H
        dispatchProjection(normedBuf, lw, 'pleProj', attnProjBuf, seqLen, H, dPle, `L${l}-ple-proj`);
        dispatchRMSNorm(attnProjBuf, normedBuf, lw.plePostNorm!, seqLen, `L${l}-ple-post-norm`);
        // residual: pe_in (hiddenBuf) + ple_out (normedBuf) → attnProjBuf
        dispatchElementwise(addPipeline, hiddenBuf, attnProjBuf, seqLen * H, `L${l}-ple-res`, normedBuf);
        if (lw.layerOutScale) {
          dispatchElementwise(mulPipeline, attnProjBuf, hiddenBuf, seqLen * H,
            `L${l}-out-scale`, lw.layerOutScale, 1);
        } else {
          batchCopy(attnProjBuf, 0, hiddenBuf, 0, seqLen * H * 4);
        }
      }

      // ── DeepStack injection (Qwen3-VL) ─────────────────────────────
      // Vision features from the tower's intermediate mergers are added to
      // the first text layers' outputs at image positions. One buffer per
      // tap layer: all writeBuffers execute before this forward()'s single
      // batched submit, so a shared buffer would be overwritten.
      if (opts?.deepstackFeatures && l < opts.deepstackFeatures.length) {
        const feat = opts.deepstackFeatures[l];
        if (feat.length !== seqLen * H) {
          throw new Error(`[forward] deepstack[${l}]: ${feat.length} floats != seqLen*H = ${seqLen * H}`);
        }
        if (!dsFeatBufs[l]) {
          dsFeatBufs[l] = createStorageBuffer(device, null, MAX_PREFILL * H * 4, `deepstack-feat-${l}`, true);
        }
        device.queue.writeBuffer(dsFeatBufs[l], 0, feat.buffer as ArrayBuffer, feat.byteOffset, feat.byteLength);
        // hidden += feat (ping-pong through normedBuf — no read+write aliasing)
        batchCopy(hiddenBuf, 0, normedBuf, 0, seqLen * H * 4);
        dispatchElementwise(addPipeline, normedBuf, hiddenBuf, seqLen * H, `L${l}-deepstack`, dsFeatBufs[l]);
      }

      // Dump hidden state after every layer for comparison with PyTorch reference
      if (isDebug) {
        flushBatch();
        await device.queue.onSubmittedWorkDone();
        const layerOut = new Float32Array(await readBuffer(device, hiddenBuf, 8 * 4));
        console.log(`[LAYER ${l}] output: [${Array.from(layerOut).map(v => v.toFixed(4)).join(', ')}]`);
      }
      if (dumpResults) {
        flushBatch();
        await device.queue.onSubmittedWorkDone();
        await dumpBufStats(hiddenBuf, `layer-${l}-out`, seqLen * H, lastRowOffset, H, dumpResults);
      }
    }

    // ── Final norm + LM head ─────────────────────────────────────────
    dispatchRMSNorm(hiddenBuf, normedBuf, weights.global.finalNorm, seqLen, 'final-norm');

    if (dumpResults) {
      flushBatch();
      await device.queue.onSubmittedWorkDone();
      await dumpBufStats(normedBuf, 'final-norm-out', seqLen * H, lastRowOffset, H, dumpResults);
    }

    // LM head projection (last token only for generation)
    const lmHeadBuf = config.tieWordEmbeddings
      ? weights.global.embedTokens
      : weights.global.lmHead;

    // For batch prefill (seqLen > 1), extract last row of normedBuf for LM head
    let lmInputBuf = normedBuf;
    if (seqLen > 1) {
      const lastRowOffset = (seqLen - 1) * H * 4;
      batchCopy(normedBuf, lastRowOffset, lastHiddenBuf, 0, H * 4);
      lmInputBuf = lastHiddenBuf;
    }
    // LM head — select kernel based on weight format (CPU BF16, split BF16, BF16, INT4 GPTQ, or f32)
    const lmIsQ4 = !!(weights.global.lmHeadQ4 && matmulQ4Pipeline);
    if (weights.global.lmHeadCPU) {
      // CPU BF16 lm_head: flush GPU, read hidden state, matmul on CPU, write logits back.
      // This saves ~2.4 GB VRAM at the cost of slower lm_head computation.
      if (__timingCtx && currentBatch && __timingCtx.used > 0) {
        currentBatch.resolveQuerySet(
          __timingCtx.querySet, 0, __timingCtx.used,
          __timingCtx.resolveBuf, 0,
        );
        currentBatch.copyBuffer(
          __timingCtx.resolveBuf, 0,
          __timingCtx.readBuf, 0,
          __timingCtx.used * 8,
        );
      }
      if (currentBatch) {
        currentBatch.flush();
        for (const buf of deferredDestroys) buf.destroy();
        deferredDestroys = [];
        currentBatch = null;
      }
      await device.queue.onSubmittedWorkDone();

      const { parts, splitPoint, hiddenSize: lmH, vocabSize: lmV, isBF16 } = weights.global.lmHeadCPU;
      const hiddenBytes = await readBuffer(device, lmInputBuf, lmH * 4);
      const hidden = new Float32Array(hiddenBytes);
      const logits = new Float32Array(lmV);

      const t0 = performance.now();
      for (let v = 0; v < lmV; v++) {
        const partIdx = Math.min(Math.floor(v / splitPoint), parts.length - 1);
        const adjustedV = v - partIdx * splitPoint;
        const rowByteOff = adjustedV * lmH * 2;
        const row = new Uint16Array(parts[partIdx].buffer, parts[partIdx].byteOffset + rowByteOff, lmH);
        let sum = 0;
        if (isBF16) {
          for (let h = 0; h < lmH; h++) {
            _cpuEmbedU32[0] = row[h] << 16;
            sum += hidden[h] * _cpuEmbedF32[0];
          }
        } else {
          for (let h = 0; h < lmH; h++) {
            const bits = row[h];
            const sign = (bits >> 15) & 1;
            const exp = (bits >> 10) & 0x1F;
            const frac = bits & 0x3FF;
            if (exp === 0) { _cpuEmbedU32[0] = 0; }
            else if (exp === 31) { _cpuEmbedU32[0] = (sign << 31) | 0x7F800000; }
            else { _cpuEmbedU32[0] = (sign << 31) | ((exp + 112) << 23) | (frac << 13); }
            sum += hidden[h] * _cpuEmbedF32[0];
          }
        }
        logits[v] = sum;
      }
      console.log(`[LM-HEAD CPU] ${lmV} logits in ${(performance.now() - t0).toFixed(0)}ms`);
      device.queue.writeBuffer(logitsBuf, 0, logits.buffer, 0, lmV * 4);
    } else if (weights.global.lmHeadSplit) {
      // Split BF16 lm_head: two matmuls, one per buffer half, writing to offset logitsBuf
      const split = weights.global.lmHeadSplit;
      const V_lo = split.splitPoints[0];
      const V_hi = V - V_lo;
      const align = device.limits.minStorageBufferOffsetAlignment || 256;

      const params0 = getCachedUniform(new Uint32Array([1, V_lo, H, 0]), 'lm-head-s0-p');
      const bg0 = cachedBindGroup(matmulBTBF16Pipeline, 0, [
        { binding: 0, resource: { buffer: lmInputBuf } },
        { binding: 2, resource: { buffer: logitsBuf, offset: 0, size: V_lo * 4 } },
        { binding: 3, resource: { buffer: params0 } },
        { binding: 5, resource: { buffer: split.buffers[0] } },
      ], 'lm-head-s0');
      bd(matmulBTBF16Pipeline, [bg0], [Math.ceil(1 / 16), Math.ceil(V_lo / 16)], 'lm-head-s0');

      const offsetBytes = V_lo * 4;
      if (offsetBytes % align !== 0) {
        console.error(`[LM-HEAD] Split offset ${offsetBytes} not aligned to ${align} — output will be corrupted!`);
      }
      const params1 = getCachedUniform(new Uint32Array([1, V_hi, H, 0]), 'lm-head-s1-p');
      const bg1 = cachedBindGroup(matmulBTBF16Pipeline, 0, [
        { binding: 0, resource: { buffer: lmInputBuf } },
        { binding: 2, resource: { buffer: logitsBuf, offset: offsetBytes, size: V_hi * 4 } },
        { binding: 3, resource: { buffer: params1 } },
        { binding: 5, resource: { buffer: split.buffers[1] } },
      ], 'lm-head-s1');
      bd(matmulBTBF16Pipeline, [bg1], [Math.ceil(1 / 16), Math.ceil(V_hi / 16)], 'lm-head-s1');
    } else if (weights.global.lmHeadGG) {
      // GGUF k-quant lm_head (kernel z-chunks N past the 65535 workgroup cap)
      dispatchMatmulGGUF(lmInputBuf, weights.global.lmHeadGG, logitsBuf, 1, V, H, 'lm-head');
    } else if (lmIsQ4) {
      // GPTQ INT4 lm_head (saves ~1.4 GB vs BF16)
      dispatchMatmulQ4(lmInputBuf, weights.global.lmHeadQ4!, logitsBuf, 1, V, H, 'lm-head');
    } else if (weights.global.lmHeadIsBF16) {
      const params = getCachedUniform(new Uint32Array([1, V, H, 0]), 'lm-head-p');
      const bg = cachedBindGroup(matmulBTBF16Pipeline, 0, [
        { binding: 0, resource: { buffer: lmInputBuf } },
        { binding: 2, resource: { buffer: logitsBuf } },
        { binding: 3, resource: { buffer: params } },
        { binding: 5, resource: { buffer: lmHeadBuf } },
      ], 'lm-head');
      bd(matmulBTBF16Pipeline, [bg], [Math.ceil(1 / 16), Math.ceil(V / 16)], 'lm-head');
    } else {
      dispatchMatmulBT(lmInputBuf, lmHeadBuf, logitsBuf, 1, V, H, 'lm-head');
    }

    // Gemma: final logit softcapping — logits = c · tanh(logits / c)
    // (gemma4.cpp result_output; lmHeadCPU path can't reach here for GGUF models)
    if (config.finalLogitSoftcap && softcapPipeline && logitsTempBuf) {
      batchCopy(logitsBuf, 0, logitsTempBuf, 0, V * 4);
      dispatchElementwise(softcapPipeline, logitsTempBuf, logitsBuf, V, 'logit-softcap',
        undefined, undefined, config.finalLogitSoftcap);
    }

    // If timing is active, encode the resolve + staging copy into the same
    // batch so they ride along with the forward pass (single submit).
    const __timingDrained = __timingCtx;
    if (__timingDrained && currentBatch && __timingDrained.used > 0) {
      currentBatch.resolveQuerySet(
        __timingDrained.querySet, 0, __timingDrained.used,
        __timingDrained.resolveBuf, 0,
      );
      currentBatch.copyBuffer(
        __timingDrained.resolveBuf, 0,
        __timingDrained.readBuf, 0,
        __timingDrained.used * 8,
      );
    }
    __timingCtx = null;

    // Flush batched GPU work (if batching is enabled)
    if (currentBatch) {
      currentBatch.flush();
      for (const buf of deferredDestroys) buf.destroy();
      deferredDestroys = [];
      currentBatch = null;
    }

    // Drain timing data (async — waits for GPU). Only runs on profiled calls.
    if (__timingDrained && __timingDrained.used > 0) {
      __timingCallsRemaining--;
      const ctx = __timingDrained;
      const bytes = ctx.used * 8;
      await ctx.readBuf.mapAsync(GPUMapMode.READ, 0, bytes);
      const copy = new BigUint64Array(ctx.readBuf.getMappedRange(0, bytes).slice(0));
      ctx.readBuf.unmap();

      // Aggregate (beginNs, endNs) pairs by category.
      const totals = new Map<string, { ns: bigint; count: number }>();
      let grandNs = 0n;
      for (let i = 0; i < ctx.categories.length; i++) {
        const b = copy[i * 2];
        const e = copy[i * 2 + 1];
        // Skip invalid writes (driver may emit 0 for dropped passes).
        if (e <= b) continue;
        const dt = e - b;
        grandNs += dt;
        const cat = ctx.categories[i];
        const rec = totals.get(cat) ?? { ns: 0n, count: 0 };
        rec.ns += dt;
        rec.count += 1;
        totals.set(cat, rec);
      }

      const grandMs = Number(grandNs) / 1e6;
      const rows = [...totals.entries()]
        .map(([cat, r]) => ({
          category: cat,
          count: r.count,
          total_ms: Number(r.ns) / 1e6,
          pct: grandMs > 0 ? (Number(r.ns) / 1e6 / grandMs) * 100 : 0,
          avg_us: r.count > 0 ? Number(r.ns) / 1e3 / r.count : 0,
        }))
        .sort((a, b) => b.total_ms - a.total_ms);

      console.log(`[timing forward #${debugCallCount} seqLen=${seqLen} pos=${pos}] GPU-sum=${grandMs.toFixed(2)}ms across ${ctx.categories.length} dispatches`);
      console.table(rows.map(r => ({
        category: r.category,
        count: r.count,
        total_ms: r.total_ms.toFixed(3),
        pct: r.pct.toFixed(1) + '%',
        avg_us: r.avg_us.toFixed(1),
      })));
      // Headless drivers can't see console.table objects — stash for evaluate().
      (globalThis as any).__perfTimingRows = rows;
    }

    // Divergence probe: dump full logits and stash collected results for
    // download. Main.ts serializes __DEBUG_DUMP_RESULT__ to JSON.
    // Results are keyed by tag so multiple fires (prefill-end, decode-1, ...)
    // accumulate into one object.
    if (dumpResults) {
      await device.queue.onSubmittedWorkDone();
      // Logits: sample the full vector (V is bounded; 150K * 4 = 600 KB).
      await dumpBufStats(logitsBuf, 'logits', V, 0, V, dumpResults);
      const acc = (globalThis as any).__DEBUG_DUMP_RESULT__ ?? {};
      acc[dumpTag] = {
        tag: dumpTag,
        seqLen, pos,
        H, V,
        layers: dumpResults,
        timestamp: Date.now(),
      };
      (globalThis as any).__DEBUG_DUMP_RESULT__ = acc;
      console.log(`[DUMP ${dumpTag}] collected ${dumpResults.length} entries → __DEBUG_DUMP_RESULT__[${dumpTag}]`);
    }

    // Debug: dump logits with global argmax on first forward pass
    if (isDebug) {
      await device.queue.onSubmittedWorkDone();
      const allLogits = new Float32Array(await readBuffer(device, logitsBuf, V * 4));
      let globalMax = -Infinity, globalArgmax = 0;
      let globalMin = Infinity;
      let nanCount = 0, infCount = 0;
      for (let i = 0; i < V; i++) {
        if (isNaN(allLogits[i])) nanCount++;
        else if (!isFinite(allLogits[i])) infCount++;
        else {
          if (allLogits[i] > globalMax) { globalMax = allLogits[i]; globalArgmax = i; }
          if (allLogits[i] < globalMin) globalMin = allLogits[i];
        }
      }
      const indices = Array.from({length: V}, (_, i) => i);
      indices.sort((a, b) => allLogits[b] - allLogits[a]);
      const top10 = indices.slice(0, 10).map(i => `${i}:${allLogits[i].toFixed(2)}`);
      console.log(`[FWD #${debugCallCount} pos=${kvCache.position}] argmax=${globalArgmax} max=${globalMax.toFixed(2)} min=${globalMin.toFixed(2)} NaN=${nanCount} Inf=${infCount}`);
      console.log(`  top10: ${top10.join(', ')}`);
    }

    // Update cache position
    kvCache.position += seqLen;

    // ── Performance instrumentation (Step 1, exit) ────────────────────
    // CPU-side ms = wall time spent inside forward() (encoder build + submit).
    // Dispatch/copy deltas attribute GPU-work load to this single call.
    const __perfT1 = performance.now();
    const __perfDisp = __perfDispatchCount - __perfDispStart;
    const __perfCopy = __perfCopyCount - __perfCopyStart;
    const __perfBG = __perfBindGroupCount - __perfBGStart;
    const __perfMs = __perfT1 - __perfT0;
    (globalThis as any).__perfLastForward = {
      seqLen, cpuMs: __perfMs, dispatches: __perfDisp, copies: __perfCopy,
      bgCreates: __perfBG,
      moeSyncMs: __perfMoESyncMs, moeExpertMs: __perfMoEExpertMs,
    };
    // Auto-log for the first 5 forward calls so the user sees the breakdown
    // without needing to set any global flag. After that, silence (avoids log
    // spam during long generations).
    if (debugCallCount <= 5) {
      console.log(
        `[perf forward #${debugCallCount} seqLen=${seqLen} pos=${pos}] `
        + `cpu=${__perfMs.toFixed(1)}ms `
        + `dispatches=${__perfDisp} copies=${__perfCopy} bg_creates=${__perfBG} `
        + `(per-token: ${(__perfMs / seqLen).toFixed(1)}ms, `
        + `${(__perfDisp / seqLen).toFixed(0)} dispatches)`
        + (__perfMoESyncMs + __perfMoEExpertMs > 0
          ? ` moe_sync=${__perfMoESyncMs.toFixed(1)}ms moe_experts=${__perfMoEExpertMs.toFixed(1)}ms`
          : '')
      );
    }

    return { logitsBuffer: logitsBuf };
  }

  // ── KV Cache ───────────────────────────────────────────────────────

  function createKVCache(maxSeqLen: number, compressed = false): KVCache {
    if (maxSeqLen > MAX_ATTN_CACHE) {
      throw new Error(
        `[KVCache] maxSeqLen ${maxSeqLen} exceeds attention workgroup limit ${MAX_ATTN_CACHE}. ` +
        `Allocating a cache that the attention kernel cannot fully read would silently ` +
        `discard tokens once generation passed ${MAX_ATTN_CACHE}. ` +
        `Cap maxSeqLen at ${MAX_ATTN_CACHE} or rebuild attention.wgsl with a larger workgroup array.`,
      );
    }
    // TurboQuant audit guard: the compressed branch reconstructs full K/V into
    // global-kvDim scratch and attends without sliding-window / KV-sharing /
    // per-layer-head-dim awareness. Those features (Gemma 4) would silently
    // produce wrong output under compression. Refuse rather than mislead —
    // TQ is supported on uniform full-attention models (dense Qwen2.5/3) and
    // the full-attention layers of DeltaNet hybrids (Qwen3.5/3.6).
    if (compressed) {
      const offenders = config.layers
        .map((d, i) => ({ d, i }))
        .filter(({ d }) =>
          d.kvSourceLayer !== undefined
          || (d.slidingWindow ?? 0) > 0
          || (d.headDim !== undefined && d.headDim !== dHead));
      if (offenders.length > 0) {
        const kinds = [...new Set(offenders.map(({ d }) =>
          d.kvSourceLayer !== undefined ? 'kv-sharing'
            : (d.slidingWindow ?? 0) > 0 ? 'sliding-window' : 'per-layer-head-dim'))];
        throw new Error(
          `[KVCache] TurboQuant compression is not supported for this model: `
          + `${offenders.length} layer(s) use ${kinds.join(' / ')}, which the compressed `
          + `attention path does not yet honor. Disable TurboQuant for this model `
          + `(${config.modelType}) or extend the compressed branch.`);
      }
    }
    // For hybrid models, allocate SSM state for linear attention layers
    let ssmState: SSMState | undefined;
    if (config.isHybrid) {
      const hiddenStates: GPUBuffer[] = [];
      const convStates: GPUBuffer[] = [];
      const layerToSSMIndex: number[] = new Array(L).fill(-1);

      const linKD = config.linearKeyHeadDim!;
      const linVD = config.linearValueHeadDim!;
      const linNKH = config.linearNumKeyHeads!;
      const linConvK = config.linearConvKernelDim!;
      // Conv state covers the full QKV projection dim (conv1d applied before split)
      const convDim = linQKVDim;

      let ssmIdx = 0;
      for (let l = 0; l < L; l++) {
        if (config.layers[l].kind === 'linear_attention') {
          // h: [num_key_heads, key_head_dim, grouped_value_dim]
          // grouped_value_dim = (num_value_heads / num_key_heads) * value_head_dim
          const hSize = linNKH * linKD * linGroupedVD * 4;
          hiddenStates.push(createStorageBuffer(device, null, hSize, `ssm-h-${l}`, true));
          // conv_state: [kernel_size - 1, proj_dim]
          const csSize = (linConvK - 1) * convDim * 4;
          convStates.push(createStorageBuffer(device, null, csSize, `ssm-conv-${l}`, false));
          layerToSSMIndex[l] = ssmIdx++;
        }
      }

      const totalSSM = ssmIdx * (linNKH * linKD * linGroupedVD * 4 + (linConvK - 1) * convDim * 4);
      console.log(`[KVCache] SSM state: ${(totalSSM / 1024 / 1024).toFixed(1)} MB for ${ssmIdx} linear layers (fixed, sequence-independent)`);
      ssmState = { hiddenStates, convStates, layerToSSMIndex };
    }

    if (!compressed) {
      const keys: GPUBuffer[] = [];
      const values: GPUBuffer[] = [];
      for (let l = 0; l < L; l++) {
        // For hybrid models, only full attention layers need KV cache
        if (config.isHybrid && config.layers[l].kind === 'linear_attention') {
          keys.push(null as any);   // placeholder — not used for linear layers
          values.push(null as any);
        } else if (config.layers[l].kvSourceLayer !== undefined) {
          // KV sharing (Gemma 4 layers 24-41): alias the source layer's
          // buffers — no allocation, attention reads the shared cache.
          const src = config.layers[l].kvSourceLayer!;
          keys.push(keys[src]);
          values.push(values[src]);
        } else {
          // Per-layer head_dim (Gemma 4: sliding 256, full 512)
          const kvDimL = nKVHeads * (config.layers[l].headDim ?? dHead);
          keys.push(createStorageBuffer(device, null, maxSeqLen * kvDimL * 4, `kv-k-${l}`, true));
          values.push(createStorageBuffer(device, null, maxSeqLen * kvDimL * 4, `kv-v-${l}`, true));
        }
      }
      return { keys, values, position: 0, maxSeqLen, ssmState };
    }

    // TurboQuant compressed KV cache
    const totalVecs = maxSeqLen * nKVHeads;
    const quantBufSize = totalVecs * tqPackedWords * 4;
    const signBufSize = totalVecs * tqSignWords * 4;
    const normBufSize = totalVecs * 4;

    const quantizedK: GPUBuffer[] = [];
    const quantizedV: GPUBuffer[] = [];
    const signBitsK: GPUBuffer[] = [];
    const signBitsV: GPUBuffer[] = [];
    const normsK: GPUBuffer[] = [];
    const normsV: GPUBuffer[] = [];
    const residualNormsK: GPUBuffer[] = [];
    const residualNormsV: GPUBuffer[] = [];

    // Only full-attention layers touch the compressed cache. In a DeltaNet
    // hybrid, the linear-attention layers go through the SSM path and never
    // index these buffers — allocate a 4-byte placeholder for them so the
    // arrays stay layer-indexed without wasting hundreds of MB (e.g. 54 of
    // the 27B's 64 layers are linear).
    let tqLayers = 0;
    for (let l = 0; l < L; l++) {
      const usesKV = config.layers[l].kind !== 'linear_attention';
      const sz = (bytes: number) => usesKV ? bytes : 4;
      quantizedK.push(createStorageBuffer(device, null, sz(quantBufSize), `tq-qk-${l}`, false));
      quantizedV.push(createStorageBuffer(device, null, sz(quantBufSize), `tq-qv-${l}`, false));
      signBitsK.push(createStorageBuffer(device, null, sz(signBufSize), `tq-sk-${l}`, false));
      signBitsV.push(createStorageBuffer(device, null, sz(signBufSize), `tq-sv-${l}`, false));
      normsK.push(createStorageBuffer(device, null, sz(normBufSize), `tq-nk-${l}`, false));
      normsV.push(createStorageBuffer(device, null, sz(normBufSize), `tq-nv-${l}`, false));
      residualNormsK.push(createStorageBuffer(device, null, sz(normBufSize), `tq-rnk-${l}`, false));
      residualNormsV.push(createStorageBuffer(device, null, sz(normBufSize), `tq-rnv-${l}`, false));
      if (usesKV) tqLayers++;
    }

    // Shared scratch f32 buffers (reused across layers during decode)
    const scratchK = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-k', true);
    const scratchV = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-v', true);

    const compressedBytes = tqLayers * (quantBufSize + signBufSize + normBufSize * 2) * 2;
    const f32Bytes = tqLayers * maxSeqLen * kvDim * 4 * 2;
    console.log(`[KVCache] TurboQuant ${TQ_BITS}-bit: ${(compressedBytes / 1024 / 1024).toFixed(1)} MB ` +
      `over ${tqLayers} full-attention layer(s) ` +
      `(vs ${(f32Bytes / 1024 / 1024).toFixed(1)} MB f32, saving ${((1 - compressedBytes / f32Bytes) * 100).toFixed(0)}%)`);

    return {
      keys: [], values: [], // unused in compressed mode
      position: 0,
      maxSeqLen,
      compressed: {
        scratchK, scratchV,
        quantizedK, quantizedV,
        signBitsK, signBitsV,
        normsK, normsV,
        residualNormsK, residualNormsV,
      },
      ssmState,
    };
  }

  function destroyKVCache(kvCache: KVCache): void {
    // KV-sharing layers alias their source layer's buffers — dedupe so each
    // GPUBuffer is destroyed exactly once.
    const seen = new Set<GPUBuffer>();
    for (const buf of [...kvCache.keys, ...kvCache.values]) {
      if (buf && !seen.has(buf)) { seen.add(buf); buf.destroy(); }
    }
    if (kvCache.compressed) {
      const c = kvCache.compressed;
      c.scratchK.destroy();
      c.scratchV.destroy();
      for (const b of c.quantizedK) b.destroy();
      for (const b of c.quantizedV) b.destroy();
      for (const b of c.signBitsK) b.destroy();
      for (const b of c.signBitsV) b.destroy();
      for (const b of c.normsK) b.destroy();
      for (const b of c.normsV) b.destroy();
      for (const b of c.residualNormsK) b.destroy();
      for (const b of c.residualNormsV) b.destroy();
    }
    if (kvCache.ssmState) {
      for (const b of kvCache.ssmState.hiddenStates) b.destroy();
      for (const b of kvCache.ssmState.convStates) b.destroy();
    }
  }

  return { forward, createKVCache, destroyKVCache, config };
}
