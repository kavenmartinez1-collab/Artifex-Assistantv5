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
import tqEncodeWGSL from '../shaders/turboquant_encode.wgsl?raw';
import tqDecodeWGSL from '../shaders/turboquant_decode.wgsl?raw';
import conv1dWGSL from '../shaders/conv1d.wgsl?raw';
import groupNormWGSL from '../shaders/group_norm.wgsl?raw';
import ssmStepWGSL from '../shaders/ssm_step.wgsl?raw';
import l2normWGSL from '../shaders/l2norm.wgsl?raw';

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
  qProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  kProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  vProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  oProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  gateProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  upProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  downProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };

  // Linear attention (Gated DeltaNet) weights — only for hybrid models
  linearInProjQKV?: GPUBuffer;
  linearInProjQKV_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  linearInProjA?: GPUBuffer;
  linearInProjA_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  linearInProjB?: GPUBuffer;
  linearInProjB_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  linearInProjZ?: GPUBuffer;
  linearInProjZ_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  linearOutProj?: GPUBuffer;
  linearOutProj_q4?: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer };
  linearALog?: GPUBuffer;         // [num_key_heads, key_head_dim] diagonal state decay
  linearConv1dWeight?: GPUBuffer; // [dim, 1, kernel_size] causal conv kernel
  linearDtBias?: GPUBuffer;       // [num_key_heads * key_head_dim] time step bias
  linearNormWeight?: GPUBuffer;   // group norm weight
}

/** GPU buffers for global (non-layer) weights. */
export interface GlobalWeights {
  embedTokens: GPUBuffer;  // [vocab_size, hidden_size] (f32 or f16 packed)
  embedIsF16?: boolean;    // true if embedding stored as F16 (large vocab models)
  finalNorm: GPUBuffer;    // [hidden_size]
  lmHead: GPUBuffer;       // [vocab_size, hidden_size] or same as embedTokens
  lmHeadIsBF16?: boolean;  // true if lm_head stored as BF16 (large vocab models)
}

/** All model weights on the GPU. */
export interface ModelWeights {
  global: GlobalWeights;
  layers: LayerWeights[];
}

/** Compressed KV cache data (TurboQuant). */
export interface CompressedKVData {
  scratchK: GPUBuffer;       // shared f32 decode buffer [maxSeqLen, kvDim]
  scratchV: GPUBuffer;       // shared f32 decode buffer [maxSeqLen, kvDim]
  quantizedK: GPUBuffer[];   // packed u32 indices per layer
  quantizedV: GPUBuffer[];
  signBitsK: GPUBuffer[];    // QJL sign bits per layer
  signBitsV: GPUBuffer[];
  normsK: GPUBuffer[];       // f32 norms per layer
  normsV: GPUBuffer[];
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
  const ropePipeline = createComputePipeline(device, ropeWGSL, 'rope', 'rope');
  const attentionPipeline = createComputePipeline(device, attentionWGSL, 'attention', 'attention');

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
  const gateSiluPipeline = isHybrid
    ? createComputePipeline(device, elementwiseWGSL, 'gate_silu', 'gate-silu') : null;
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
  const qBuf = createStorageBuffer(device, null, MAX_PREFILL * nHeads * dHead * 4, 'q-proj', true);
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

  /** C[M,N] = A[M,K] @ dequant(q4_packed, scales, zeros)^T — GPTQ INT4 */
  function dispatchMatmulQ4(
    aBuf: GPUBuffer,
    q4: { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer },
    cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    if (!matmulQ4Pipeline) throw new Error('INT4 matmul not compiled (model is not quantized)');
    const params = createUniformBuffer(device,
      new Uint32Array([M, N, K, config.quantGroupSize]), `${label}-p`);
    const bg = createBindGroup(device, matmulQ4Pipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: q4.qweight } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
      { binding: 4, resource: { buffer: q4.scales } },
      { binding: 5, resource: { buffer: q4.qzeros } },
    ], label);
    dispatch(device, matmulQ4Pipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
    params.destroy();
  }

  /** Dispatch either f32 matmul_bt or INT4 matmul_q4 depending on weight type */
  function dispatchProjection(
    inputBuf: GPUBuffer, lw: LayerWeights, proj: string,
    outputBuf: GPUBuffer, M: number, N: number, K: number, label: string,
  ) {
    const q4key = `${proj}_q4` as keyof LayerWeights;
    const q4 = lw[q4key] as { qweight: GPUBuffer; scales: GPUBuffer; qzeros: GPUBuffer } | undefined;
    if (q4) {
      dispatchMatmulQ4(inputBuf, q4, outputBuf, M, N, K, label);
    } else {
      const wkey = proj as keyof LayerWeights;
      dispatchMatmulBT(inputBuf, lw[wkey] as GPUBuffer, outputBuf, M, N, K, label);
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

  // ── Dispatch helpers ───────────────────────────────────────────────

  function dispatchMatmul(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    const params = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = createBindGroup(device, matmulPipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
    ], label);
    // Shader: row = wid.x (M dim), col = wid.y (N dim)
    dispatch(device, matmulPipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
    params.destroy();
  }

  /** C[M,N] = A[M,K] @ B^T[K,N] where B is stored as [N,K] (HF weight format) */
  function dispatchMatmulBT(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    const params = createUniformBuffer(device, new Uint32Array([M, N, K, 0]), `${label}-p`);
    const bg = createBindGroup(device, matmulBTPipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
    ], label);
    // Shader: row = wid.x (M dim), col = wid.y (N dim)
    dispatch(device, matmulBTPipeline, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], label);
    params.destroy();
  }

  // Qwen3_5 uses (1+weight) in RMSNorm — detect from model type
  const useResidualWeight = config.modelType === 'qwen3_5_text' ? 1 : 0;

  function dispatchRMSNorm(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    weightBuf: GPUBuffer, rows: number, label: string,
  ) {
    const paramData = new ArrayBuffer(16);
    new Uint32Array(paramData, 0, 1)[0] = H;
    new Float32Array(paramData, 4, 1)[0] = eps;
    new Uint32Array(paramData, 8, 1)[0] = useResidualWeight;
    const paramBuf = createUniformBuffer(device, new Uint8Array(paramData), `${label}-p`);
    const bg = createBindGroup(device, rmsnormPipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: paramBuf } },
    ], label);
    dispatch(device, rmsnormPipeline, [bg], [rows], label);
    paramBuf.destroy();
  }

  function dispatchElementwise(
    pipeline: GPUComputePipeline,
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    size: number, label: string, secondBuf?: GPUBuffer,
    broadcastB?: number,
  ) {
    const params = createUniformBuffer(device, new Uint32Array([size, broadcastB ?? 0]), `${label}-p`);
    const entries: Array<{ binding: number; resource: GPUBindingResource }> = [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ];
    if (secondBuf) entries.push({ binding: 3, resource: { buffer: secondBuf } });
    const bg = createBindGroup(device, pipeline, 0, entries, label);
    dispatch(device, pipeline, [bg], [workgroupCount(size, 256)], label);
    params.destroy();
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
    // MODEL-SPECIFIC: RoPE theta comes from config. NTK/YaRN scaling
    // would modify the base here based on sequence length.
    const paramBuf = createUniformBuffer(device, new Uint8Array(paramData), `${label}-p`);

    const bg = createBindGroup(device, ropePipeline, 0, [
      { binding: 0, resource: { buffer: qkBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
    ], label);

    const rotDim = rd > 0 ? rd : hd;
    const halfDim = rotDim / 2;
    const totalPairs = seqLen * numHeads * halfDim;
    dispatch(device, ropePipeline, [bg], [workgroupCount(totalPairs, 256)], label);
    paramBuf.destroy();
  }

  function dispatchAttention(
    qBuf: GPUBuffer, kCacheBuf: GPUBuffer, vCacheBuf: GPUBuffer,
    outputBuf: GPUBuffer, newSeqLen: number, cacheLen: number,
    isCausal: boolean, posOffset: number, label: string,
  ) {
    // Attention params struct
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
    const paramBuf = createUniformBuffer(device, new Uint8Array(paramData), `${label}-p`);

    const bg = createBindGroup(device, attentionPipeline, 0, [
      { binding: 0, resource: { buffer: qBuf } },
      { binding: 1, resource: { buffer: kCacheBuf } },
      { binding: 2, resource: { buffer: vCacheBuf } },
      { binding: 3, resource: { buffer: outputBuf } },
      { binding: 4, resource: { buffer: paramBuf } },
    ], label);

    // One workgroup per (query_position, head) pair
    dispatch(device, attentionPipeline, [bg], [newSeqLen, nHeads], label);
    paramBuf.destroy();
  }

  function copyToKVCache(
    srcBuf: GPUBuffer, cacheBuf: GPUBuffer,
    seqLen: number, dim: number, position: number,
  ) {
    // Copy new K or V vectors into the cache at the current position
    const srcBytes = seqLen * dim * 4;
    const dstOffset = position * dim * 4;
    const encoder = device.createCommandEncoder({ label: 'kv-cache-copy' });
    encoder.copyBufferToBuffer(srcBuf, 0, cacheBuf, dstOffset, srcBytes);
    device.queue.submit([encoder.finish()]);
  }

  // ── TurboQuant dispatch helpers ─────────────────────────────────────

  function dispatchTQEncode(
    inputBuf: GPUBuffer,
    outQuantBuf: GPUBuffer, outSignBuf: GPUBuffer, outNormsBuf: GPUBuffer,
    numVecs: number, outVecOffset: number, label: string,
  ) {
    const params = createUniformBuffer(device,
      new Uint32Array([dHead, TQ_BITS, tqCodebook.centroids.length,
        tqCodebook.thresholds.length, outVecOffset]),
      `${label}-p`);
    const bg0 = createBindGroup(device, tqEncodePipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outQuantBuf } },
      { binding: 2, resource: { buffer: outSignBuf } },
      { binding: 3, resource: { buffer: outNormsBuf } },
    ], `${label}-g0`);
    const bg2 = createBindGroup(device, tqEncodePipeline, 2, [
      { binding: 0, resource: { buffer: params } },
    ], `${label}-g2`);
    dispatch(device, tqEncodePipeline, [bg0, tqEncodeMatBG, bg2], [numVecs], label);
    params.destroy();
  }

  function dispatchTQDecode(
    inQuantBuf: GPUBuffer, inSignBuf: GPUBuffer, inNormsBuf: GPUBuffer,
    outputBuf: GPUBuffer, numVecs: number, label: string,
  ) {
    const params = createUniformBuffer(device,
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
    dispatch(device, tqDecodePipeline, [bg0, tqDecodeMatBG, bg2], [numVecs], label);
    params.destroy();
  }

  // ── Forward Pass ───────────────────────────────────────────────────

  async function forward(tokenIds: Uint32Array, kvCache: KVCache): Promise<ForwardOutput> {
    const seqLen = tokenIds.length;
    const pos = kvCache.position;
    const cacheLen = pos + seqLen;
    const isDebug = debugCallCount < 2; // debug first two forward passes
    const isDebug2 = debugCallCount === 1; // extra detail on second pass
    debugCallCount++;

    // Upload token IDs
    device.queue.writeBuffer(tokenIdBuf, 0, tokenIds.buffer, tokenIds.byteOffset, tokenIds.byteLength);

    // ── Embedding (f32 or f16 depending on buffer size) ────────────
    const useF16Embed = weights.global.embedIsF16 === true;
    const embedPipe = useF16Embed ? embedF16Pipeline : embedPipeline;
    const embedParams = createUniformBuffer(device, new Uint32Array([H, seqLen]), 'embed-p');
    const embedBG = createBindGroup(device, embedPipe, 0, [
      { binding: 0, resource: { buffer: tokenIdBuf } },
      { binding: 1, resource: { buffer: hiddenBuf } },
      { binding: 2, resource: { buffer: weights.global.embedTokens } },
      { binding: 3, resource: { buffer: embedParams } },
    ], 'embed');
    dispatch(device, embedPipe, [embedBG], [seqLen], 'embed');
    embedParams.destroy();

    if (isDebug) {
      await debugRead(hiddenBuf, 'embed-out', 8);
    }

    // ── Transformer layers ───────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const lw = weights.layers[l];
      const isLinearLayer = config.layerTypes?.[l] === 'linear_attention';

      // Save hidden state for residual connection
      const enc1 = device.createCommandEncoder({ label: `res1-${l}` });
      enc1.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([enc1.finish()]);

      // Pre-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.inputNorm, seqLen, `L${l}-norm1`);

      if (isLinearLayer && kvCache.ssmState) {
        // ── GATED DELTANET LINEAR ATTENTION ──────────────────────────
        const ssmIdx = kvCache.ssmState.layerToSSMIndex[l];
        const hBuf = kvCache.ssmState.hiddenStates[ssmIdx];
        const csBuf = kvCache.ssmState.convStates[ssmIdx];

        // Debug: normed input before projections
        if (isDebug && l === 0) {
          await debugRead(normedBuf, 'L0-normed-input', 8);
        }

        // 1. Fused QKV projection
        dispatchProjection(normedBuf, lw, 'linearInProjQKV', linQKVBuf!, 1, linQKVDim, H, `L${l}-lin-qkv`);

        // Debug: raw QKV projection output (before conv1d)
        if (isDebug && l === 0) {
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
          const convParams = createUniformBuffer(device,
            new Uint32Array([linQKVDim, linConvK]), `L${l}-conv-p`);
          const convBG = createBindGroup(device, conv1dPipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 2, resource: { buffer: lw.linearConv1dWeight! } },
            { binding: 3, resource: { buffer: linConvOutBuf! } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv`);
          dispatch(device, conv1dPipeline, [convBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d`);

          // Update conv state (shift + append raw QKV)
          const updateBG = createBindGroup(device, conv1dUpdatePipeline, 0, [
            { binding: 0, resource: { buffer: linQKVBuf! } },
            { binding: 1, resource: { buffer: csBuf } },
            { binding: 4, resource: { buffer: convParams } },
          ], `L${l}-conv-upd`);
          dispatch(device, conv1dUpdatePipeline, [updateBG],
            [workgroupCount(linQKVDim, 256)], `L${l}-conv1d-update`);
          convParams.destroy();

          // SiLU on entire conv output (all 8192 channels)
          dispatchElementwise(siluPipeline, linConvOutBuf!, linQKVBuf!, linQKVDim, `L${l}-conv-silu`);
        }

        // 3. Split QKV AFTER conv+silu
        const qSize = linNKH * linKD * 4;
        const kSize = linNKH * linKD * 4;
        const vSize = linNVH * linVD * 4;
        const encSplit = device.createCommandEncoder({ label: `L${l}-qkv-split` });
        encSplit.copyBufferToBuffer(linQKVBuf!, 0, linQBuf!, 0, qSize);
        encSplit.copyBufferToBuffer(linQKVBuf!, qSize, linKBuf!, 0, kSize);
        encSplit.copyBufferToBuffer(linQKVBuf!, qSize + kSize, linVBuf!, 0, vSize);
        device.queue.submit([encSplit.finish()]);

        // Debug: conv1d output and silu output
        if (isDebug && l === 0) {
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
        dispatchProjection(normedBuf, lw, 'linearInProjA', linABuf!, 1, linNVH, H, `L${l}-lin-a`);
        dispatchProjection(normedBuf, lw, 'linearInProjB', linBBuf!, 1, linNVH, H, `L${l}-lin-b`);
        dispatchProjection(normedBuf, lw, 'linearInProjZ', linZBuf!, 1, H, H, `L${l}-lin-z`);

        // Debug: A, B, Z projection outputs
        if (isDebug && l === 0) {
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
          const encSig = device.createCommandEncoder({ label: `L${l}-sig-cp` });
          encSig.copyBufferToBuffer(linDtBuf!, 0, linBBuf!, 0, linNVH * 4);
          device.queue.submit([encSig.finish()]);
        }

        // 5c. L2-normalize Q and K (use_qk_l2norm_in_kernel=True in reference)
        // Uses separate shader (l2norm.wgsl) to avoid binding conflicts with elementwise.wgsl
        if (l2NormPipeline) {
          const qDim = linNKH * linKD;
          const kDim = linNKH * linKD;

          const l2pQ = createUniformBuffer(device, new Uint32Array([qDim, linKD]), `L${l}-l2q-p`);
          const l2bgQ = createBindGroup(device, l2NormPipeline, 0, [
            { binding: 0, resource: { buffer: linQBuf! } },
            { binding: 1, resource: { buffer: linConvOutBuf! } },
            { binding: 2, resource: { buffer: l2pQ } },
          ], `L${l}-l2norm-q`);
          dispatch(device, l2NormPipeline, [l2bgQ], [workgroupCount(qDim, 256)], `L${l}-l2norm-q`);
          l2pQ.destroy();
          const encQ = device.createCommandEncoder({ label: `L${l}-l2q-cp` });
          encQ.copyBufferToBuffer(linConvOutBuf!, 0, linQBuf!, 0, qDim * 4);
          device.queue.submit([encQ.finish()]);

          const l2pK = createUniformBuffer(device, new Uint32Array([kDim, linKD]), `L${l}-l2k-p`);
          const l2bgK = createBindGroup(device, l2NormPipeline, 0, [
            { binding: 0, resource: { buffer: linKBuf! } },
            { binding: 1, resource: { buffer: linConvOutBuf! } },
            { binding: 2, resource: { buffer: l2pK } },
          ], `L${l}-l2norm-k`);
          dispatch(device, l2NormPipeline, [l2bgK], [workgroupCount(kDim, 256)], `L${l}-l2norm-k`);
          l2pK.destroy();

          // Debug: verify L2 norm actually wrote to output
          if (isDebug && l === 0) {
            await device.queue.onSubmittedWorkDone();
            const rawK = await readBuffer(device, linKBuf!, 8 * 4);
            const normK = await readBuffer(device, linConvOutBuf!, 8 * 4);
            console.log(`[L2 DEBUG] K input first 8: [${Array.from(new Float32Array(rawK)).map(v => v.toFixed(4)).join(', ')}]`);
            console.log(`[L2 DEBUG] K output (linConvOutBuf) first 8: [${Array.from(new Float32Array(normK)).map(v => v.toFixed(4)).join(', ')}]`);
          }

          const encK = device.createCommandEncoder({ label: `L${l}-l2k-cp` });
          encK.copyBufferToBuffer(linConvOutBuf!, 0, linKBuf!, 0, kDim * 4);
          device.queue.submit([encK.finish()]);
        }

        // 6. Decay per VALUE HEAD [32], not per key dim
        // A_log is [32], dt_bias is [32], in_proj_a output is [32]
        if (softplusPipeline && lw.linearDtBias) {
          dispatchElementwise(softplusPipeline, linABuf!, linDtBuf!, linNVH, `L${l}-softplus`, lw.linearDtBias);
        }
        if (decayPipeline && lw.linearALog) {
          dispatchElementwise(decayPipeline, lw.linearALog, linDecayBuf!, linNVH, `L${l}-decay`, linDtBuf!);
        }

        // Debug SSM intermediates (layer 0, first pass)
        if (isDebug && l === 0) {
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
          const ssmParams = createUniformBuffer(device,
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
          dispatch(device, ssmStepPipeline, [ssmBG0, ssmBG1, ssmBG2],
            [linNKH], `L${l}-ssm-step`);
          ssmParams.destroy();
        }

        // Debug SSM output (layer 0, first pass)
        if (isDebug && l === 0) {
          await device.queue.onSubmittedWorkDone();
          const ssmOutRaw = await readBuffer(device, linOutBuf!, 8 * 4);
          const ssmOut = new Float32Array(ssmOutRaw);
          console.log(`[SSM DEBUG] SSM output first 8: [${Array.from(ssmOut.slice(0, 8)).map(v => v.toFixed(6)).join(', ')}]`);
        }

        // 7. GroupNorm on output
        if (groupNormPipeline && lw.linearNormWeight) {
          const outDim = linNKH * linGroupedVD; // = num_value_heads * value_head_dim = 4096
          const numGroups = linNVH; // one group per value head = 32
          const cpg = linVD; // channels per group = 128
          // GroupNorm params: [num_channels, num_groups, channels_per_group, eps]
          const gnParamData = new ArrayBuffer(16);
          const gnU32 = new Uint32Array(gnParamData);
          const gnF32 = new Float32Array(gnParamData);
          gnU32[0] = outDim;
          gnU32[1] = numGroups;
          gnU32[2] = cpg;
          gnF32[3] = eps;
          const gnParams = createUniformBuffer(device, new Uint8Array(gnParamData), `L${l}-gn-p`);
          const gnBG = createBindGroup(device, groupNormPipeline, 0, [
            { binding: 0, resource: { buffer: linOutBuf! } },
            { binding: 1, resource: { buffer: attnProjBuf } }, // reuse as temp output
            { binding: 2, resource: { buffer: lw.linearNormWeight } },
            { binding: 3, resource: { buffer: gnParams } },
          ], `L${l}-gn`);
          dispatch(device, groupNormPipeline, [gnBG], [numGroups], `L${l}-group-norm`);
          gnParams.destroy();
        }

        // 8. Output gating: out = normed_output * silu(Z)
        if (gateSiluPipeline) {
          const outDim = linNKH * linGroupedVD;
          // attnProjBuf has the normed output, linZBuf has the gate
          dispatchElementwise(gateSiluPipeline, attnProjBuf, linOutBuf!, outDim, `L${l}-gate`, linZBuf!);
        }

        // 9. Output projection → attnProjBuf [1, H]
        const outDim = linNKH * linGroupedVD;
        dispatchProjection(linOutBuf!, lw, 'linearOutProj', attnProjBuf, 1, H, outDim, `L${l}-lin-out`);

      } else {
        // ── STANDARD SOFTMAX ATTENTION ────────────────────────────────

        // Q, K, V projections (auto-selects f32 or INT4 matmul)
        dispatchProjection(normedBuf, lw, 'qProj', qBuf, seqLen, nHeads * dHead, H, `L${l}-q`);
        dispatchProjection(normedBuf, lw, 'kProj', kBuf, seqLen, kvDim, H, `L${l}-k`);
        dispatchProjection(normedBuf, lw, 'vProj', vBuf, seqLen, kvDim, H, `L${l}-v`);

        // Add bias if model has attention_bias
        if (config.attentionBias && lw.qBias && lw.kBias && lw.vBias) {
          const qDim = nHeads * dHead;
          const encQb = device.createCommandEncoder({ label: `L${l}-qb-copy` });
          encQb.copyBufferToBuffer(qBuf, 0, attnOutBuf, 0, seqLen * qDim * 4);
          device.queue.submit([encQb.finish()]);
          dispatchElementwise(addPipeline, attnOutBuf, qBuf, seqLen * qDim, `L${l}-qb`, lw.qBias, qDim);

          const encKb = device.createCommandEncoder({ label: `L${l}-kb-copy` });
          encKb.copyBufferToBuffer(kBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
          device.queue.submit([encKb.finish()]);
          dispatchElementwise(addPipeline, ffnTempBuf, kBuf, seqLen * kvDim, `L${l}-kb`, lw.kBias, kvDim);

          const encVb = device.createCommandEncoder({ label: `L${l}-vb-copy` });
          encVb.copyBufferToBuffer(vBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
          device.queue.submit([encVb.finish()]);
          dispatchElementwise(addPipeline, ffnTempBuf, vBuf, seqLen * kvDim, `L${l}-vb`, lw.vBias, kvDim);
        }

        if (isDebug && l === 0) {
          await debugRead(normedBuf, 'L0-normed', 8);
          await debugRead(qBuf, 'L0-Q-before-rope', 8);
          await debugRead(kBuf, 'L0-K-before-rope', 8);
          await debugRead(vBuf, 'L0-V', 8);
        }

        // Apply RoPE to Q and K
        dispatchRoPE(qBuf, seqLen, nHeads, pos, `L${l}-rope-q`);
        dispatchRoPE(kBuf, seqLen, nKVHeads, pos, `L${l}-rope-k`);

        if (isDebug && l === 0) {
          await debugRead(qBuf, 'L0-Q-after-rope', 8);
          await debugRead(kBuf, 'L0-K-after-rope', 8);
        }

        // Write new K, V to cache and run attention
        const isCausal = seqLen > 1;

        if (kvCache.compressed) {
          const c = kvCache.compressed;
          const numVecs = seqLen * nKVHeads;
          const outOffset = pos * nKVHeads;

          dispatchTQEncode(kBuf, c.quantizedK[l], c.signBitsK[l], c.normsK[l],
            numVecs, outOffset, `L${l}-tq-enc-k`);
          dispatchTQEncode(vBuf, c.quantizedV[l], c.signBitsV[l], c.normsV[l],
            numVecs, outOffset, `L${l}-tq-enc-v`);

          if (pos > 0) {
            const prevVecs = pos * nKVHeads;
            dispatchTQDecode(c.quantizedK[l], c.signBitsK[l], c.normsK[l],
              c.scratchK, prevVecs, `L${l}-tq-dec-k`);
            dispatchTQDecode(c.quantizedV[l], c.signBitsV[l], c.normsV[l],
              c.scratchV, prevVecs, `L${l}-tq-dec-v`);
          }

          const curOffset = pos * kvDim * 4;
          const curSize = seqLen * kvDim * 4;
          const encCK = device.createCommandEncoder({ label: `L${l}-tq-cpK` });
          encCK.copyBufferToBuffer(kBuf, 0, c.scratchK, curOffset, curSize);
          device.queue.submit([encCK.finish()]);
          const encCV = device.createCommandEncoder({ label: `L${l}-tq-cpV` });
          encCV.copyBufferToBuffer(vBuf, 0, c.scratchV, curOffset, curSize);
          device.queue.submit([encCV.finish()]);

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

        // Output projection: [seq, nHeads*dHead] → [seq, H]
        dispatchProjection(attnOutBuf, lw, 'oProj', attnProjBuf, seqLen, H, nHeads * dHead, `L${l}-o`);

        // O projection bias
        if (config.attentionBias && lw.oBias) {
          const encOb = device.createCommandEncoder({ label: `L${l}-ob-copy` });
          encOb.copyBufferToBuffer(attnProjBuf, 0, normedBuf, 0, seqLen * H * 4);
          device.queue.submit([encOb.finish()]);
          dispatchElementwise(addPipeline, normedBuf, attnProjBuf, seqLen * H, `L${l}-ob`, lw.oBias, H);
        }
      }

      // Residual: hidden = residual + attn_output (shared for both layer types)
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res1`, attnProjBuf);

      if (isDebug && l === 0) {
        await debugRead(attnProjBuf, `L0-attn-proj(pos=${pos})`, 8);
        await debugRead(hiddenBuf, `L0-after-attn-residual(pos=${pos})`, 8);
      }

      // Save for second residual
      const enc2 = device.createCommandEncoder({ label: `res2-${l}` });
      enc2.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([enc2.finish()]);

      // Post-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.postAttnNorm, seqLen, `L${l}-norm2`);

      // ── FFN (SwiGLU) ───────────────────────────────────────────────
      // MODEL-SPECIFIC: Phi fuses gate+up into one projection.
      dispatchProjection(normedBuf, lw, 'gateProj', gateBuf, seqLen, ffnDim, H, `L${l}-gate`);
      dispatchProjection(normedBuf, lw, 'upProj', upBuf, seqLen, ffnDim, H, `L${l}-up`);

      // MODEL-SPECIFIC: SiLU for most models, GELU for some
      // Cannot use same buffer for input and output in WebGPU — use dedicated temp
      dispatchElementwise(siluPipeline, gateBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silu`);
      dispatchElementwise(mulPipeline, ffnTempBuf, gateBuf, seqLen * ffnDim, `L${l}-mul`, upBuf);

      dispatchProjection(gateBuf, lw, 'downProj', downBuf, seqLen, H, ffnDim, `L${l}-down`);

      // Residual: hidden = residual + ffn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res2`, downBuf);

      if (isDebug && l === 0) {
        await debugRead(hiddenBuf, 'L0-after-ffn-residual', 8);
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
      const encLH = device.createCommandEncoder({ label: 'last-hidden-copy' });
      encLH.copyBufferToBuffer(normedBuf, lastRowOffset, lastHiddenBuf, 0, H * 4);
      device.queue.submit([encLH.finish()]);
      lmInputBuf = lastHiddenBuf;
    }
    // LM head — use BF16 kernel if weight is stored as BF16 (too large for f32)
    if (weights.global.lmHeadIsBF16) {
      const params = createUniformBuffer(device, new Uint32Array([1, V, H, 0]), 'lm-head-p');
      const bg = createBindGroup(device, matmulBTBF16Pipeline, 0, [
        { binding: 0, resource: { buffer: lmInputBuf } },
        { binding: 2, resource: { buffer: logitsBuf } },
        { binding: 3, resource: { buffer: params } },
        { binding: 5, resource: { buffer: lmHeadBuf } },
      ], 'lm-head');
      dispatch(device, matmulBTBF16Pipeline, [bg], [Math.ceil(1 / 16), Math.ceil(V / 16)], 'lm-head');
      params.destroy();
    } else {
      dispatchMatmulBT(lmInputBuf, lmHeadBuf, logitsBuf, 1, V, H, 'lm-head');
    }

    // Update cache position
    kvCache.position += seqLen;

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

    for (let l = 0; l < L; l++) {
      quantizedK.push(createStorageBuffer(device, null, quantBufSize, `tq-qk-${l}`, false));
      quantizedV.push(createStorageBuffer(device, null, quantBufSize, `tq-qv-${l}`, false));
      signBitsK.push(createStorageBuffer(device, null, signBufSize, `tq-sk-${l}`, false));
      signBitsV.push(createStorageBuffer(device, null, signBufSize, `tq-sv-${l}`, false));
      normsK.push(createStorageBuffer(device, null, normBufSize, `tq-nk-${l}`, false));
      normsV.push(createStorageBuffer(device, null, normBufSize, `tq-nv-${l}`, false));
    }

    // Shared scratch f32 buffers (reused across layers during decode)
    const scratchK = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-k', true);
    const scratchV = createStorageBuffer(device, null, maxSeqLen * kvDim * 4, 'tq-scratch-v', true);

    const compressedBytes = L * (quantBufSize + signBufSize + normBufSize) * 2;
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
    }
    if (kvCache.ssmState) {
      for (const b of kvCache.ssmState.hiddenStates) b.destroy();
      for (const b of kvCache.ssmState.convStates) b.destroy();
    }
  }

  return { forward, createKVCache, destroyKVCache, config };
}
