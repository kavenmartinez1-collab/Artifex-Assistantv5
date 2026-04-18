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

import type { ModelConfig } from '../model/model-config';
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
import hadamardWGSL from '../shaders/hadamard.wgsl?raw';

// ── Types ────────────────────────────────────────────────────────────────

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
  postAttnNorm: GPUBuffer; // [hidden_size]
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
}

/** GPU buffers for global (non-layer) weights. */
export interface GlobalWeights {
  embedTokens: GPUBuffer;  // [vocab_size, hidden_size] (f32, f16 packed, or dummy if Q4)
  embedIsF16?: boolean;    // true if embedding stored as F16/BF16 (large vocab models)
  embedQ4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };  // GPTQ INT4 embedding
  /** E8 codebook buffer, shared across all layers [256*8] f32 */
  e8Codebook?: GPUBuffer;
  finalNorm: GPUBuffer;    // [hidden_size]
  lmHead: GPUBuffer;       // [vocab_size, hidden_size] or same as embedTokens
  lmHeadIsBF16?: boolean;  // true if lm_head stored as BF16 (large vocab models)
  lmHeadQ4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer; g_idx: GPUBuffer; hasActOrder?: boolean };  // GPTQ INT4 lm_head
}

/** All model weights on the GPU. */
export interface ModelWeights {
  global: GlobalWeights;
  layers: LayerWeights[];
  /** Set of GPU buffers that store BF16 data (not converted to f32). */
  bf16Buffers?: Set<GPUBuffer>;
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

export interface ForwardPassEngine {
  /** Run one forward pass step (prefill or single-token decode). */
  forward(tokenIds: Uint32Array, kvCache: KVCache): Promise<ForwardOutput>;

  /** Create an empty KV cache for the given max sequence length.
   *  @param compressed — Use TurboQuant compression (saves ~80% KV memory) */
  createKVCache(maxSeqLen: number, compressed?: boolean): KVCache;

  /** Destroy all buffers in a KV cache. */
  destroyKVCache(kvCache: KVCache): void;

  /** Get the model config. */
  readonly config: ModelConfig;
}

/**
 * Create a forward pass engine.
 */
export function createForwardPassEngine(
  device: GPUDevice,
  config: ModelConfig,
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
  // so all 256 threads are useful (vs 16/256 with the tiled kernel at M=1).
  //
  // Two pipeline variants are compiled upfront, differing only by the WGSL
  // override constant USE_ACTORDER. The fast variant (USE_ACTORDER=0) skips
  // the per-K g_idx VRAM read and computes group_id = k / group_size directly.
  // The correct-but-slower variant (USE_ACTORDER=1) consults g_idx, needed for
  // GPTQ models quantized with desc_act=true. dispatchMatmulQ4 picks per-tensor
  // based on the q4.hasActOrder flag.
  const matmulQ4GemvPipeline = config.isQuantized
    ? createComputePipeline(
        device, matmulQ4GemvWGSL, 'matmul_bt_q4_gemv', 'matmul-q4-gemv',
        { USE_ACTORDER: 0 })
    : null;
  const matmulQ4GemvActOrderPipeline = config.isQuantized
    ? createComputePipeline(
        device, matmulQ4GemvWGSL, 'matmul_bt_q4_gemv', 'matmul-q4-gemv-actorder',
        { USE_ACTORDER: 1 })
    : null;
  // E8 lattice 2-bit dequantizing matmul (weights packed as codebook indices)
  const matmulE8Pipeline = config.isQuantized
    ? createComputePipeline(device, matmulE8WGSL, 'matmul_e8', 'matmul-e8')
    : null;
  // INT8 dequantizing matmul (weights packed as 8-bit)
  const matmulQ8Pipeline = config.isQuantized
    ? createComputePipeline(device, matmulQ8WGSL, 'matmul_bt_q8', 'matmul-q8')
    : null;
  // Hadamard transform for QuIP#/QuaRot incoherence processing
  const hadamardPipeline = config.isQuantized
    ? createComputePipeline(device, hadamardWGSL, 'hadamard', 'hadamard')
    : null;

  const ropePipeline = createComputePipeline(device, ropeWGSL, 'rope', 'rope');
  const attentionPipeline = createComputePipeline(device, attentionWGSL, 'attention', 'attention');
  const attentionTqPipeline = createComputePipeline(device, attentionTqWGSL, 'attention_tq', 'attention-tq');

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
  const softplusPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'softplus', 'softplus') : null;
  const conv1dPipeline = isHybrid
    ? createComputePipeline(device, conv1dWGSL, 'conv1d', 'conv1d') : null;
  const conv1dUpdatePipeline = isHybrid
    ? createComputePipeline(device, conv1dWGSL, 'conv1d_update_state', 'conv1d-update') : null;
  const groupNormPipeline = isHybrid
    ? createComputePipeline(device, groupNormWGSL, 'group_norm', 'group-norm') : null;
  const ssmStepPipeline = isHybrid
    ? createComputePipeline(device, ssmStepWGSL, 'ssm_step', 'ssm-step') : null;

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

  // Compile additional Mamba-2 kernels
  const sigmoidPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'sigmoid_op', 'sigmoid') : null;
  const decayPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'decay_compute', 'decay') : null;
  const l2NormPipeline = isHybrid
    ? createComputePipeline(device, l2normWGSL, 'l2_normalize', 'l2-norm') : null;

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
  // Temp buffer for in-place elementwise ops (WebGPU can't read+write same buffer)
  const ffnTempBuf = createStorageBuffer(device, null, MAX_PREFILL * ffnDim * 4, 'ffn-temp', true);
  const logitsBuf = createStorageBuffer(device, null, V * 4, 'logits', true);
  const tokenIdBuf = createStorageBuffer(device, null, MAX_PREFILL * 4, 'token-ids', true);
  // Small buffer for extracting last token's hidden state (for LM head after batch prefill)
  const lastHiddenBuf = createStorageBuffer(device, null, H * 4, 'last-hidden', true);

  // Option A (chunked-prefill for hybrid models): per-token scratch buffers used
  // to wrap the existing single-token SSM block in a per-token JS loop while the
  // surrounding non-SSM stages process seqLen > 1 tokens in batched fashion.
  // ssmInputBuf holds one token-row sliced out of the multi-token normedBuf;
  // ssmOutBuf holds the SSM block's output for one token before being copied
  // into the multi-token attnProjBuf at that token's offset.
  const ssmInputBuf = isHybrid ? createStorageBuffer(device, null, H * 4, 'ssm-input', true) : null;
  const ssmOutBuf = isHybrid ? createStorageBuffer(device, null, H * 4, 'ssm-out', true) : null;

  // Linear attention intermediate buffers (only for hybrid models)
  const linQKVBuf = isHybrid ? createStorageBuffer(device, null, linQKVDim * 4, 'lin-qkv', true) : null;
  const linQBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linKD * 4, 'lin-q', true) : null;
  const linKBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linKD * 4, 'lin-k', true) : null;
  const linVBuf = isHybrid ? createStorageBuffer(device, null, linNVH * linVD * 4, 'lin-v', true) : null;
  const linABuf = isHybrid ? createStorageBuffer(device, null, linConvDim * 4, 'lin-a', true) : null;
  const linBBuf = isHybrid ? createStorageBuffer(device, null, linNVH * 4, 'lin-beta', true) : null;
  const linZBuf = isHybrid ? createStorageBuffer(device, null, H * 4, 'lin-z', true) : null;
  const linOutBuf = isHybrid ? createStorageBuffer(device, null, linNKH * linGroupedVD * 4, 'lin-out', true) : null;
  const linConvOutBuf = isHybrid ? createStorageBuffer(device, null, linQKVDim * 4, 'lin-conv-out', true) : null;
  const linDecayBuf = isHybrid ? createStorageBuffer(device, null, linNVH * 4, 'lin-decay', true) : null;
  const linDtBuf = isHybrid ? createStorageBuffer(device, null, Math.max(linNVH, linNKH) * 4, 'lin-dt', true) : null;

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
    // Pick the actorder variant if this tensor's g_idx is non-trivial;
    // otherwise use the faster variant that skips the g_idx VRAM reads.
    const useGemv = M === 1 && matmulQ4GemvPipeline !== null;
    const gemvPipeline = q4.hasActOrder ? matmulQ4GemvActOrderPipeline : matmulQ4GemvPipeline;
    const pipeline = useGemv ? gemvPipeline! : matmulQ4Pipeline;
    const bg = createBindGroup(device, pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: q4.qweight } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: q4.scales } },
      { binding: 5, resource: { buffer: q4.qzeros } },
      { binding: 6, resource: { buffer: q4.g_idx } },
    ], label);
    const dispatchDims: [number, number, number] = useGemv
      ? [Math.ceil(N / 256), 1, 1]
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
    const bg = createBindGroup(device, matmulE8Pipeline, 0, [
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
    const bg = createBindGroup(device, matmulQ8Pipeline, 0, [
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

  // Set of GPU buffers stored as BF16 (need matmul_bt_bf16 kernel)
  const bf16Set = weights.bf16Buffers ?? new Set<GPUBuffer>();

  /** Dispatch f32/BF16/INT4/E8/INT8 matmul depending on weight type */
  function dispatchProjection(
    inputBuf: GPUBuffer, lw: LayerWeights, proj: string,
    outputBuf: GPUBuffer, M: number, N: number, K: number, label: string,
  ) {
    // Priority: E8 2-bit > INT8 > INT4 GPTQ > BF16 > f32
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
      dispatchMatmulQ4(inputBuf, q4, outputBuf, M, N, K, label);
    } else {
      const wkey = proj as keyof LayerWeights;
      const wBuf = lw[wkey] as GPUBuffer;
      if (bf16Set.has(wBuf)) {
        dispatchMatmulBTBF16(inputBuf, wBuf, outputBuf, M, N, K, label);
      } else {
        dispatchMatmulBT(inputBuf, wBuf, outputBuf, M, N, K, label);
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

  function bd(
    pipeline: GPUComputePipeline, bindGroups: GPUBindGroup[],
    workgroupCounts: [number, number?, number?], label?: string,
  ) {
    __perfDispatchCount++;
    if (currentBatch) {
      currentBatch.dispatch(pipeline, bindGroups, workgroupCounts, label);
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
    const bg = createBindGroup(device, matmulPipeline, 0, [
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
  ) {
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = createBindGroup(device, matmulBTPipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
    ], label);
    bd(matmulBTPipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
  }

  /** C[M,N] = A[M,K] @ B_bf16^T[K,N] where B is stored as BF16 packed [N,K/2] u32 */
  function dispatchMatmulBTBF16(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    const params = getCachedUniform(new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = createBindGroup(device, matmulBTBF16Pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 2, resource: { buffer: cBuf } },
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
    const bg = createBindGroup(device, hadamardPipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ], label);
    bd(hadamardPipeline, [bg], [rows], label);
  }

  function dispatchRMSNorm(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    weightBuf: GPUBuffer, rows: number, label: string,
  ) {
    const paramData = new ArrayBuffer(16);
    new Uint32Array(paramData, 0, 1)[0] = H;
    new Float32Array(paramData, 4, 1)[0] = eps;
    new Uint32Array(paramData, 8, 1)[0] = useResidualWeight;
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);
    const bg = createBindGroup(device, rmsnormPipeline, 0, [
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
    broadcastB?: number,
  ) {
    const params = getCachedUniform(new Uint32Array([size, broadcastB ?? 0]), `${label}-p`);
    const entries: Array<{ binding: number; resource: GPUBindingResource }> = [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ];
    if (secondBuf) entries.push({ binding: 3, resource: { buffer: secondBuf } });
    const bg = createBindGroup(device, pipeline, 0, entries, label);
    bd(pipeline, [bg], [workgroupCount(size, 256)], label);
  }

  function dispatchRoPE(
    qkBuf: GPUBuffer, seqLen: number, numHeads: number,
    posOffset: number, label: string,
    headDimOverride?: number, rotaryDimOverride?: number,
  ) {
    const hd = headDimOverride ?? dHead;
    const rd = rotaryDimOverride ?? 0;
    // RoPE params struct: [seq_len, head_dim, num_heads, pos_offset, rope_base, rotary_dim]
    const paramData = new ArrayBuffer(24);
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = seqLen;
    u32View[1] = hd;
    u32View[2] = numHeads;
    u32View[3] = posOffset;
    f32View[4] = ropeTheta;
    u32View[5] = rd;
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);

    const bg = createBindGroup(device, ropePipeline, 0, [
      { binding: 0, resource: { buffer: qkBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
    ], label);

    const rotDim = rd > 0 ? rd : hd;
    const halfDim = rotDim / 2;
    const totalPairs = seqLen * numHeads * halfDim;
    bd(ropePipeline, [bg], [workgroupCount(totalPairs, 256)], label);
  }

  function dispatchAttention(
    qBuf: GPUBuffer, kCacheBuf: GPUBuffer, vCacheBuf: GPUBuffer,
    outputBuf: GPUBuffer, newSeqLen: number, cacheLen: number,
    isCausal: boolean, posOffset: number, label: string,
  ) {
    const paramData = new ArrayBuffer(32);
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
    const paramBuf = getCachedUniform(new Uint8Array(paramData), `${label}-p`);

    const bg = createBindGroup(device, attentionPipeline, 0, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: paramBuf } },
    ], label);

    bd(attentionPipeline, [bg], [newSeqLen, nHeads], label);
  }

  function dispatchAttentionTQ(
    qBuf: GPUBuffer, kCacheBuf: GPUBuffer, vCacheBuf: GPUBuffer,
    outputBuf: GPUBuffer, newSeqLen: number, cacheLen: number,
    isCausal: boolean, posOffset: number,
    signBitsK: GPUBuffer, normsK: GPUBuffer, residualNormsK: GPUBuffer,
    label: string,
  ) {
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

    const bg0 = createBindGroup(device, attentionTqPipeline, 0, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: paramBuf } },
    ], `${label}-g0`);

    const bg1 = createBindGroup(device, attentionTqPipeline, 1, [
      { binding: 0, resource: { buffer: signBitsK } },
      { binding: 1, resource: { buffer: normsK } },
      { binding: 2, resource: { buffer: residualNormsK } },
      { binding: 3, resource: { buffer: tqSetup.spiMatrix } },
    ], `${label}-g1`);

    bd(attentionTqPipeline, [bg0, bg1], [newSeqLen, nHeads], label);
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
    const bg0 = createBindGroup(device, tqEncodePipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outQuantBuf } },
      { binding: 2, resource: { buffer: outSignBuf } },
      { binding: 3, resource: { buffer: outNormsBuf } },
      { binding: 4, resource: { buffer: outResidualNormsBuf } },
    ], `${label}-g0`);
    const bg2 = createBindGroup(device, tqEncodePipeline, 2, [
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
    const bg0 = createBindGroup(device, tqDecodePipeline, 0, [
      { binding: 0, resource: { buffer: inQuantBuf } },
      { binding: 1, resource: { buffer: inSignBuf } },
      { binding: 2, resource: { buffer: outputBuf } },
      { binding: 3, resource: { buffer: inNormsBuf } },
    ], `${label}-g0`);
    const bg2 = createBindGroup(device, tqDecodePipeline, 2, [
      { binding: 0, resource: { buffer: params } },
    ], `${label}-g2`);
    bd(tqDecodePipeline, [bg0, tqDecodeMatBG, bg2], [numVecs], label);
  }

  // ── Forward Pass ───────────────────────────────────────────────────

  async function forward(tokenIds: Uint32Array, kvCache: KVCache): Promise<ForwardOutput> {
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

    // ── Performance instrumentation (Step 1) ─────────────────────────
    // Count CPU time and dispatch/copy load for this forward pass. The numbers
    // are written to globalThis.__perfLastForward so generate.ts can compose a
    // per-decode-step breakdown, and an auto-log fires for the first few calls.
    const __perfT0 = performance.now();
    const __perfDispStart = __perfDispatchCount;
    const __perfCopyStart = __perfCopyCount;

    // Upload token IDs
    device.queue.writeBuffer(tokenIdBuf, 0, tokenIds.buffer, tokenIds.byteOffset, tokenIds.byteLength);

    // Batch all GPU work into a single submit for speed
    currentBatch = new BatchedDispatcher(device, 'forward');
    deferredDestroys = [];

    // ── Embedding (f32, BF16/F16, or GPTQ INT4) ────────────────────
    if (weights.global.embedQ4 && embedQ4Pipeline) {
      // GPTQ INT4 embedding — dequant on the fly per token
      const eq4 = weights.global.embedQ4;
      const embedParams = getCachedUniform(
        new Uint32Array([H, seqLen, config.quantGroupSize || 128, V]), 'embed-q4-p');
      const embedBG = createBindGroup(device, embedQ4Pipeline, 0, [
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
      const embedBG = createBindGroup(device, embedPipe, 0, [
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

    // ── Transformer layers ───────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const lw = weights.layers[l];
      const isLinearLayer = config.layerTypes?.[l] === 'linear_attention';

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
          // Slice this token's normed input out of the multi-token buffer.
          batchCopy(normedBuf, ssmT * H * 4, ssmInputBuf!, 0, H * 4);

          // Debug: normed input before projections (first token only to keep logs sane)
          if (isDebug && l === 0 && ssmT === 0) {
            flushBatch();
            await debugRead(ssmInputBuf!, 'L0-normed-input', 8);
          }

          // 1. Fused QKV projection
          dispatchProjection(ssmInputBuf!, lw, 'linearInProjQKV', linQKVBuf!, 1, linQKVDim, H, `L${l}-lin-qkv`);

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
        if (conv1dPipeline && conv1dUpdatePipeline) {
          const convParams = getCachedUniform(
            new Uint32Array([linQKVDim, linConvK]), `L${l}-conv-p`);
          const convBG = createBindGroup(device, conv1dPipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 2, resource: { buffer: lw.linearConv1dWeight! } },
            { binding: 3, resource: { buffer: linConvOutBuf! } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv`);
          bd(conv1dPipeline, [convBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d`);

          // Update conv state (shift + append raw QKV)
          const updateBG = createBindGroup(device, conv1dUpdatePipeline, 0, [
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
        const qSize = linNKH * linKD * 4;
        const kSize = linNKH * linKD * 4;
        const vSize = linNVH * linVD * 4;
        batchCopy(linQKVBuf!, 0, linQBuf!, 0, qSize);
        batchCopy(linQKVBuf!, qSize, linKBuf!, 0, kSize);
        batchCopy(linQKVBuf!, qSize + kSize, linVBuf!, 0, vSize);

        // Debug: conv1d output and silu output — first token only
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const qkvAfterSilu = new Float32Array(await readBuffer(device, linQKVBuf!, (4104) * 4));
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
        dispatchProjection(ssmInputBuf!, lw, 'linearInProjA', linABuf!, 1, linNVH, H, `L${l}-lin-a`);
        dispatchProjection(ssmInputBuf!, lw, 'linearInProjB', linBBuf!, 1, linNVH, H, `L${l}-lin-b`);
        dispatchProjection(ssmInputBuf!, lw, 'linearInProjZ', linZBuf!, 1, H, H, `L${l}-lin-z`);

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
        if (sigmoidPipeline) {
          dispatchElementwise(sigmoidPipeline, linBBuf!, linDtBuf!, linNVH, `L${l}-sigmoid-beta`);
          batchCopy(linDtBuf!, 0, linBBuf!, 0, linNVH * 4);
        }

        // 5c. L2-normalize Q and K (use_qk_l2norm_in_kernel=True in reference)
        // Uses separate shader (l2norm.wgsl) to avoid binding conflicts with elementwise.wgsl
        if (l2NormPipeline) {
          const qDim = linNKH * linKD;
          const kDim = linNKH * linKD;

          const l2pQ = getCachedUniform(new Uint32Array([qDim, linKD]), `L${l}-l2q-p`);
          const l2bgQ = createBindGroup(device, l2NormPipeline, 0, [
            { binding: 0, resource: { buffer: linQBuf! } },
            { binding: 1, resource: { buffer: linConvOutBuf! } },
            { binding: 2, resource: { buffer: l2pQ } },
          ], `L${l}-l2norm-q`);
          bd(l2NormPipeline, [l2bgQ], [workgroupCount(qDim, 256)], `L${l}-l2norm-q`);
          batchCopy(linConvOutBuf!, 0, linQBuf!, 0, qDim * 4);

          const l2pK = getCachedUniform(new Uint32Array([kDim, linKD]), `L${l}-l2k-p`);
          const l2bgK = createBindGroup(device, l2NormPipeline, 0, [
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

        // 6. Decay per VALUE HEAD [32], not per key dim
        // A_log is [32], dt_bias is [32], in_proj_a output is [32]
        if (softplusPipeline && lw.linearDtBias) {
          dispatchElementwise(softplusPipeline, linABuf!, linDtBuf!, linNVH, `L${l}-softplus`, lw.linearDtBias);
        }
        if (decayPipeline && lw.linearALog) {
          dispatchElementwise(decayPipeline, lw.linearALog, linDecayBuf!, linNVH, `L${l}-decay`, linDtBuf!);
        }

        // Debug SSM intermediates (layer 0, first token only)
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const betaRaw = await readBuffer(device, linBBuf!, linNKH * 4);
          const beta = new Float32Array(betaRaw);
          console.log(`[SSM DEBUG] beta (sigmoid) first 4: [${Array.from(beta.slice(0, 4)).map(v => v.toFixed(4)).join(', ')}]`);

          const decayRaw = await readBuffer(device, linDecayBuf!, Math.min(linNVH, 8) * 4);
          const decay = new Float32Array(decayRaw);
          console.log(`[SSM DEBUG] decay (per vh, ${linNVH} total) first 8: [${Array.from(decay.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

          const kNormRaw = await readBuffer(device, linKBuf!, 8 * 4);
          const kNorm = new Float32Array(kNormRaw);
          console.log(`[SSM DEBUG] K (L2-normed) first 8: [${Array.from(kNorm.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

          const vRaw = await readBuffer(device, linVBuf!, 8 * 4);
          const vVals = new Float32Array(vRaw);
          console.log(`[SSM DEBUG] V first 8: [${Array.from(vVals.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);

          const qRaw = await readBuffer(device, linQBuf!, 8 * 4);
          const qVals = new Float32Array(qRaw);
          console.log(`[SSM DEBUG] Q first 8: [${Array.from(qVals.slice(0, 8)).map(v => v.toFixed(4)).join(', ')}]`);
        }

        // 7. SSM step: update hidden state, readout via Q
        // Decay is per value head [32], not per key dim
        if (ssmStepPipeline) {
          const ssmParams = getCachedUniform(
            new Uint32Array([linNKH, linNVH, linKD, linGroupedVD]), `L${l}-ssm-p`);
          const ssmBG0 = createBindGroup(device, ssmStepPipeline, 0, [
            { binding: 0, resource: { buffer: linQBuf! } },
            { binding: 1, resource: { buffer: linKBuf! } }, // K after conv1d + silu + L2 norm
            { binding: 2, resource: { buffer: linVBuf! } },
            { binding: 3, resource: { buffer: linBBuf! } },       // beta (after sigmoid)
            { binding: 4, resource: { buffer: linDecayBuf! } },   // decay = exp(-exp(A_log)*dt)
          ], `L${l}-ssm-g0`);
          const ssmBG1 = createBindGroup(device, ssmStepPipeline, 1, [
            { binding: 0, resource: { buffer: hBuf } },
            { binding: 1, resource: { buffer: linOutBuf! } },
          ], `L${l}-ssm-g1`);
          const ssmBG2 = createBindGroup(device, ssmStepPipeline, 2, [
            { binding: 0, resource: { buffer: ssmParams } },
          ], `L${l}-ssm-g2`);
          bd(ssmStepPipeline, [ssmBG0, ssmBG1, ssmBG2],
            [linNKH], `L${l}-ssm-step`);
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
          const gnBG = createBindGroup(device, rmsnormPipeline, 0, [
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

        // 9. Output projection → ssmOutBuf [1, H] (per-token scratch)
        const outDim = linNKH * linGroupedVD;
        dispatchProjection(linOutBuf!, lw, 'linearOutProj', ssmOutBuf!, 1, H, outDim, `L${l}-lin-out`);

        // Q8 out_proj debug readback (first token only)
        if (isDebug && l === 0 && ssmT === 0) {
          flushBatch();
          await device.queue.onSubmittedWorkDone();
          const outData = new Float32Array(await readBuffer(device, ssmOutBuf!, 8 * 4));
          console.log(`[Q8 DEBUG] L0 out_proj output[0:8]: [${Array.from(outData).map(v => v.toFixed(6)).join(', ')}]`);
        }

        // Option A: write this token's SSM output into the multi-token attnProjBuf
        // at offset ssmT*H*4. Downstream stages (residual add, post-attn norm, FFN)
        // already process all seqLen tokens in batched fashion.
        batchCopy(ssmOutBuf!, 0, attnProjBuf, ssmT * H * 4, H * 4);
      } // end per-token SSM loop (Option A)

      } else {
        // ── STANDARD SOFTMAX ATTENTION ────────────────────────────────

        // Q, K, V projections (auto-selects f32 or INT4 matmul)
        if (config.attnOutputGate) {
          // Qwen3.5: Q proj outputs [nHeads, dHead*2] — interleaved [Q_h0, gate_h0, Q_h1, gate_h1, ...]
          // Project to qBuf (sized 2x), then deinterleave in-place
          dispatchProjection(normedBuf, lw, 'qProj', qBuf, seqLen, nHeads * dHead * 2, H, `L${l}-q`);
          // Deinterleave from qBuf → attnOutBuf (Q) + attnGateBuf (gate)
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
        } else {
          dispatchProjection(normedBuf, lw, 'qProj', qBuf, seqLen, nHeads * dHead, H, `L${l}-q`);
        }
        dispatchProjection(normedBuf, lw, 'kProj', kBuf, seqLen, kvDim, H, `L${l}-k`);
        dispatchProjection(normedBuf, lw, 'vProj', vBuf, seqLen, kvDim, H, `L${l}-v`);

        // Qwen3.5: per-head RMSNorm on Q and K (q_norm, k_norm)
        // Uses (1+weight) convention — same as input layernorm
        if (config.attnOutputGate && lw.qNorm && lw.kNorm) {
          // Q norm: treat [seqLen * nHeads, dHead] as rows of dHead
          const qNormParams = new ArrayBuffer(16);
          new Uint32Array(qNormParams, 0, 1)[0] = dHead;
          new Float32Array(qNormParams, 4, 1)[0] = eps;
          new Uint32Array(qNormParams, 8, 1)[0] = useResidualWeight;
          const qnp = getCachedUniform(new Uint8Array(qNormParams), `L${l}-qn-p`);
          const qnBG = createBindGroup(device, rmsnormPipeline, 0, [
            { binding: 0, resource: { buffer: qBuf } },
            { binding: 1, resource: { buffer: attnOutBuf } },
            { binding: 2, resource: { buffer: lw.qNorm } },
            { binding: 3, resource: { buffer: qnp } },
          ], `L${l}-qnorm`);
          bd(rmsnormPipeline, [qnBG], [seqLen * nHeads], `L${l}-qnorm`);
          batchCopy(attnOutBuf, 0, qBuf, 0, seqLen * nHeads * dHead * 4);

          // K norm
          const knp = getCachedUniform(new Uint8Array(qNormParams), `L${l}-kn-p`);
          new Uint32Array(qNormParams, 0, 1)[0] = dHead; // same params
          const knBG = createBindGroup(device, rmsnormPipeline, 0, [
            { binding: 0, resource: { buffer: kBuf } },
            { binding: 1, resource: { buffer: ffnTempBuf } },
            { binding: 2, resource: { buffer: lw.kNorm } },
            { binding: 3, resource: { buffer: knp } },
          ], `L${l}-knorm`);
          bd(rmsnormPipeline, [knBG], [seqLen * nKVHeads], `L${l}-knorm`);
          batchCopy(ffnTempBuf, 0, kBuf, 0, seqLen * kvDim * 4);
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
        // Qwen3.5 full attention uses partial RoPE (25% of head dims)
        const fullAttnRotaryDim = config.partialRotaryFactor
          ? Math.floor(config.partialRotaryFactor * dHead) : 0;
        dispatchRoPE(qBuf, seqLen, nHeads, pos, `L${l}-rope-q`, undefined, fullAttnRotaryDim);
        dispatchRoPE(kBuf, seqLen, nKVHeads, pos, `L${l}-rope-k`, undefined, fullAttnRotaryDim);

        if (isDebug && l === 0) {
          flushBatch();
          await debugRead(qBuf, 'L0-Q-after-rope', 8);
          await debugRead(kBuf, 'L0-K-after-rope', 8);
        }

        // Write new K, V to cache and run attention
        const isCausal = seqLen > 1;

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
          copyToKVCache(kBuf, kvCache.keys[l], seqLen, kvDim, pos);
          copyToKVCache(vBuf, kvCache.values[l], seqLen, kvDim, pos);

          dispatchAttention(
            qBuf, kvCache.keys[l], kvCache.values[l], attnOutBuf,
            seqLen, cacheLen, isCausal, pos, `L${l}-attn`,
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
        // gate is in linZBuf (from Q projection split)
        // gate shape is [seqLen, nHeads * dHead] — reshaped from [seqLen, nHeads, dHead]
        if (config.attnOutputGate && attnGateBuf && sigmoidPipeline) {
          const gateDim = seqLen * nHeads * dHead;
          // sigmoid(gate) → normedBuf
          dispatchElementwise(sigmoidPipeline, attnGateBuf, normedBuf, gateDim, `L${l}-gate-sig`);
          // attn * sigmoid(gate) → attnOutBuf (in-place via copy pattern)
          batchCopy(attnOutBuf, 0, ffnTempBuf, 0, gateDim * 4);
          dispatchElementwise(mulPipeline, ffnTempBuf, attnOutBuf, gateDim, `L${l}-attn-gate`, normedBuf);
        }

        if (isDebug && (l === 23 || l === 27)) {
          flushBatch();
          await debugRead(attnOutBuf, `L${l}-attn-out-postGate`, 8);
        }

        // QuIP#/QuaRot: Hadamard rotation before o_proj (its weights were also rotated)
        if (useHadamard) {
          const oDim = nHeads * dHead;
          if ((oDim & (oDim - 1)) === 0) {
            batchCopy(attnOutBuf, 0, ffnTempBuf, 0, seqLen * oDim * 4);
            dispatchHadamard(ffnTempBuf, attnOutBuf, seqLen, oDim, 0, `L${l}-had-o`);
          }
        }

        // Output projection: [seq, nHeads*dHead] → [seq, H]
        dispatchProjection(attnOutBuf, lw, 'oProj', attnProjBuf, seqLen, H, nHeads * dHead, `L${l}-o`);

        // O projection bias
        if (config.attentionBias && lw.oBias) {
          batchCopy(attnProjBuf, 0, normedBuf, 0, seqLen * H * 4);
          dispatchElementwise(addPipeline, normedBuf, attnProjBuf, seqLen * H, `L${l}-ob`, lw.oBias, H);
        }
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
      dispatchProjection(normedBuf, lw, 'gateProj', gateBuf, seqLen, ffnDim, H, `L${l}-gate`);
      dispatchProjection(normedBuf, lw, 'upProj', upBuf, seqLen, ffnDim, H, `L${l}-up`);

      // MODEL-SPECIFIC: SiLU for most models, GELU for some
      // Fused SwiGLU activation: ffnTemp = up * silu(gate) in a single dispatch.
      // Saves one dispatch + one full ffnDim read/write of VRAM traffic per layer.
      // Old two-dispatch path (kept for reference / GELU variants):
      // dispatchElementwise(siluPipeline, gateBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silu`);
      // dispatchElementwise(mulPipeline, ffnTempBuf, gateBuf, seqLen * ffnDim, `L${l}-mul`, upBuf);
      dispatchElementwise(gateSiluPipeline, upBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silumul`, gateBuf);

      dispatchProjection(ffnTempBuf, lw, 'downProj', downBuf, seqLen, H, ffnDim, `L${l}-down`);

      // Residual: hidden = residual + ffn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res2`, downBuf);

      // Dump hidden state after every layer for comparison with PyTorch reference
      if (isDebug) {
        flushBatch();
        await device.queue.onSubmittedWorkDone();
        const layerOut = new Float32Array(await readBuffer(device, hiddenBuf, 8 * 4));
        console.log(`[LAYER ${l}] output: [${Array.from(layerOut).map(v => v.toFixed(4)).join(', ')}]`);
      }
    }

    // ── Final norm + LM head ─────────────────────────────────────────
    dispatchRMSNorm(hiddenBuf, normedBuf, weights.global.finalNorm, seqLen, 'final-norm');

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
    // LM head — select kernel based on weight format (BF16, INT4 GPTQ, or f32)
    const lmIsQ4 = !!(weights.global.lmHeadQ4 && matmulQ4Pipeline);
    if (lmIsQ4) {
      // GPTQ INT4 lm_head (saves ~1.4 GB vs BF16)
      dispatchMatmulQ4(lmInputBuf, weights.global.lmHeadQ4!, logitsBuf, 1, V, H, 'lm-head');
    } else if (weights.global.lmHeadIsBF16) {
      const params = getCachedUniform(new Uint32Array([1, V, H, 0]), 'lm-head-p');
      const bg = createBindGroup(device, matmulBTBF16Pipeline, 0, [
        { binding: 0, resource: { buffer: lmInputBuf } },
        { binding: 2, resource: { buffer: logitsBuf } },
        { binding: 3, resource: { buffer: params } },
        { binding: 5, resource: { buffer: lmHeadBuf } },
      ], 'lm-head');
      bd(matmulBTBF16Pipeline, [bg], [Math.ceil(1 / 16), Math.ceil(V / 16)], 'lm-head');
    } else {
      dispatchMatmulBT(lmInputBuf, lmHeadBuf, logitsBuf, 1, V, H, 'lm-head');
    }

    // Flush batched GPU work (if batching is enabled)
    if (currentBatch) {
      currentBatch.flush();
      for (const buf of deferredDestroys) buf.destroy();
      deferredDestroys = [];
      currentBatch = null;
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
    const __perfMs = __perfT1 - __perfT0;
    (globalThis as any).__perfLastForward = {
      seqLen, cpuMs: __perfMs, dispatches: __perfDisp, copies: __perfCopy,
    };
    // Auto-log for the first 5 forward calls so the user sees the breakdown
    // without needing to set any global flag. After that, silence (avoids log
    // spam during long generations).
    if (debugCallCount <= 5) {
      console.log(
        `[perf forward #${debugCallCount} seqLen=${seqLen} pos=${pos}] `
        + `cpu=${__perfMs.toFixed(1)}ms `
        + `dispatches=${__perfDisp} copies=${__perfCopy} `
        + `(per-token: ${(__perfMs / seqLen).toFixed(1)}ms, `
        + `${(__perfDisp / seqLen).toFixed(0)} dispatches)`
      );
    }

    return { logitsBuffer: logitsBuf };
  }

  // ── KV Cache ───────────────────────────────────────────────────────

  function createKVCache(maxSeqLen: number, compressed = false): KVCache {
    // For hybrid models, allocate SSM state for linear attention layers
    let ssmState: SSMState | undefined;
    if (config.isHybrid && config.layerTypes) {
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
        if (config.layerTypes[l] === 'linear_attention') {
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
        if (config.isHybrid && config.layerTypes?.[l] === 'linear_attention') {
          keys.push(null as any);   // placeholder — not used for linear layers
          values.push(null as any);
        } else {
          keys.push(createStorageBuffer(device, null, maxSeqLen * kvDim * 4, `kv-k-${l}`, true));
          values.push(createStorageBuffer(device, null, maxSeqLen * kvDim * 4, `kv-v-${l}`, true));
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

    for (let l = 0; l < L; l++) {
      quantizedK.push(createStorageBuffer(device, null, quantBufSize, `tq-qk-${l}`, false));
      quantizedV.push(createStorageBuffer(device, null, quantBufSize, `tq-qv-${l}`, false));
      signBitsK.push(createStorageBuffer(device, null, signBufSize, `tq-sk-${l}`, false));
      signBitsV.push(createStorageBuffer(device, null, signBufSize, `tq-sv-${l}`, false));
      normsK.push(createStorageBuffer(device, null, normBufSize, `tq-nk-${l}`, false));
      normsV.push(createStorageBuffer(device, null, normBufSize, `tq-nv-${l}`, false));
      residualNormsK.push(createStorageBuffer(device, null, normBufSize, `tq-rnk-${l}`, false));
      residualNormsV.push(createStorageBuffer(device, null, normBufSize, `tq-rnv-${l}`, false));
    }

    // Shared scratch f32 buffers (reused across layers during decode)
    const scratchK = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-k', true);
    const scratchV = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-v', true);

    const compressedBytes = L * (quantBufSize + signBufSize + normBufSize * 2) * 2;
    const f32Bytes = L * maxSeqLen * kvDim * 4 * 2;
    console.log(`[KVCache] TurboQuant ${TQ_BITS}-bit: ${(compressedBytes / 1024 / 1024).toFixed(1)} MB ` +
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
    for (const buf of kvCache.keys) if (buf) buf.destroy();
    for (const buf of kvCache.values) if (buf) buf.destroy();
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
