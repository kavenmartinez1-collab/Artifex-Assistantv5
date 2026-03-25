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

import matmulWGSL from '../shaders/matmul.wgsl?raw';
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import ropeWGSL from '../shaders/rope.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import embedWGSL from '../shaders/embed.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';

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
}

/** GPU buffers for global (non-layer) weights. */
export interface GlobalWeights {
  embedTokens: GPUBuffer;  // [vocab_size, hidden_size]
  finalNorm: GPUBuffer;    // [hidden_size]
  lmHead: GPUBuffer;       // [vocab_size, hidden_size] or same as embedTokens
}

/** All model weights on the GPU. */
export interface ModelWeights {
  global: GlobalWeights;
  layers: LayerWeights[];
}

/** KV cache for all layers. */
export interface KVCache {
  /** K cache per layer: [max_seq, num_kv_heads * head_dim] */
  keys: GPUBuffer[];
  /** V cache per layer: [max_seq, num_kv_heads * head_dim] */
  values: GPUBuffer[];
  /** Current sequence position (number of cached tokens) */
  position: number;
  /** Maximum sequence length */
  maxSeqLen: number;
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

  /** Create an empty KV cache for the given max sequence length. */
  createKVCache(maxSeqLen: number): KVCache;

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
  const rmsnormPipeline = createComputePipeline(device, rmsnormWGSL, 'rmsnorm', 'rmsnorm');
  // MODEL-SPECIFIC: activation. Most use SiLU; Phi/some Gemma use GELU.
  const siluPipeline = createComputePipeline(device, elementwiseWGSL, 'silu', 'silu');
  const mulPipeline = createComputePipeline(device, elementwiseWGSL, 'mul', 'mul');
  const addPipeline = createComputePipeline(device, elementwiseWGSL, 'add', 'add');
  const matmulPipeline = createComputePipeline(device, matmulWGSL, 'matmul', 'matmul');
  // B-transposed matmul for HF weight projections (stored as [out, in])
  const matmulBTPipeline = createComputePipeline(device, matmulWGSL, 'matmul_bt', 'matmul-bt');
  const ropePipeline = createComputePipeline(device, ropeWGSL, 'rope', 'rope');
  const attentionPipeline = createComputePipeline(device, attentionWGSL, 'attention', 'attention');

  // ── Reusable intermediate buffers ──────────────────────────────────
  // Single-token decode (seqLen=1). Prefill would need dynamic sizing.
  const maxSeq = 1;

  const hiddenBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'hidden', true);
  const residualBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'residual', true);
  const normedBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'normed', true);
  const qBuf = createStorageBuffer(device, null, maxSeq * nHeads * dHead * 4, 'q-proj', true);
  const kBuf = createStorageBuffer(device, null, maxSeq * kvDim * 4, 'k-proj', true);
  const vBuf = createStorageBuffer(device, null, maxSeq * kvDim * 4, 'v-proj', true);
  const attnOutBuf = createStorageBuffer(device, null, maxSeq * nHeads * dHead * 4, 'attn-out', true);
  const attnProjBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'attn-proj', true);
  const gateBuf = createStorageBuffer(device, null, maxSeq * ffnDim * 4, 'ffn-gate', true);
  const upBuf = createStorageBuffer(device, null, maxSeq * ffnDim * 4, 'ffn-up', true);
  const downBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'ffn-down', true);
  // Temp buffer for in-place elementwise ops (WebGPU can't read+write same buffer)
  const ffnTempBuf = createStorageBuffer(device, null, maxSeq * ffnDim * 4, 'ffn-temp', true);
  const logitsBuf = createStorageBuffer(device, null, V * 4, 'logits', true);
  const tokenIdBuf = createStorageBuffer(device, null, 256 * 4, 'token-ids', true); // up to 256 tokens

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

  function dispatchRMSNorm(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    weightBuf: GPUBuffer, rows: number, label: string,
  ) {
    const paramData = new ArrayBuffer(16);
    new Uint32Array(paramData, 0, 1)[0] = H;
    new Float32Array(paramData, 4, 1)[0] = eps;
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
  ) {
    const params = createUniformBuffer(device, new Uint32Array([size]), `${label}-p`);
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
  ) {
    // RoPE params struct: [seq_len, head_dim, num_heads, pos_offset, rope_base]
    const paramData = new ArrayBuffer(20);
    const u32View = new Uint32Array(paramData);
    const f32View = new Float32Array(paramData);
    u32View[0] = seqLen;
    u32View[1] = dHead;
    u32View[2] = numHeads;
    u32View[3] = posOffset;
    f32View[4] = ropeTheta;
    // MODEL-SPECIFIC: RoPE theta comes from config. NTK/YaRN scaling
    // would modify the base here based on sequence length.
    const paramBuf = createUniformBuffer(device, new Uint8Array(paramData), `${label}-p`);

    const bg = createBindGroup(device, ropePipeline, 0, [
      { binding: 0, resource: { buffer: qkBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
    ], label);

    const halfDim = dHead / 2;
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

    // ── Embedding ────────────────────────────────────────────────────
    const embedParams = createUniformBuffer(device, new Uint32Array([H, seqLen]), 'embed-p');
    const embedBG = createBindGroup(device, embedPipeline, 0, [
      { binding: 0, resource: { buffer: tokenIdBuf } },
      { binding: 1, resource: { buffer: hiddenBuf } },
      { binding: 2, resource: { buffer: weights.global.embedTokens } },
      { binding: 3, resource: { buffer: embedParams } },
    ], 'embed');
    dispatch(device, embedPipeline, [embedBG], [seqLen], 'embed');
    embedParams.destroy();

    if (isDebug) {
      await debugRead(hiddenBuf, 'embed-out', 8);
    }

    // ── Transformer layers ───────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const lw = weights.layers[l];

      // Save hidden state for residual connection
      const enc1 = device.createCommandEncoder({ label: `res1-${l}` });
      enc1.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([enc1.finish()]);

      // Pre-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.inputNorm, seqLen, `L${l}-norm1`);

      // Q, K, V projections
      dispatchMatmulBT(normedBuf, lw.qProj, qBuf, seqLen, nHeads * dHead, H, `L${l}-q`);
      dispatchMatmulBT(normedBuf, lw.kProj, kBuf, seqLen, kvDim, H, `L${l}-k`);
      dispatchMatmulBT(normedBuf, lw.vProj, vBuf, seqLen, kvDim, H, `L${l}-v`);

      // Add bias if model has attention_bias (Qwen2, etc.)
      // Use copy-add pattern to avoid same-buffer read+write conflict:
      //   copy qBuf → temp, then temp + bias → qBuf
      if (config.attentionBias && lw.qBias && lw.kBias && lw.vBias) {
        // Q bias: copy Q to attnOutBuf (temp), add bias back to qBuf
        const encQb = device.createCommandEncoder({ label: `L${l}-qb-copy` });
        encQb.copyBufferToBuffer(qBuf, 0, attnOutBuf, 0, seqLen * nHeads * dHead * 4);
        device.queue.submit([encQb.finish()]);
        dispatchElementwise(addPipeline, attnOutBuf, qBuf, seqLen * nHeads * dHead, `L${l}-qb`, lw.qBias);

        // K bias: copy K to ffnTempBuf (temp, reused), add bias
        const encKb = device.createCommandEncoder({ label: `L${l}-kb-copy` });
        encKb.copyBufferToBuffer(kBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
        device.queue.submit([encKb.finish()]);
        dispatchElementwise(addPipeline, ffnTempBuf, kBuf, seqLen * kvDim, `L${l}-kb`, lw.kBias);

        // V bias: reuse ffnTempBuf
        const encVb = device.createCommandEncoder({ label: `L${l}-vb-copy` });
        encVb.copyBufferToBuffer(vBuf, 0, ffnTempBuf, 0, seqLen * kvDim * 4);
        device.queue.submit([encVb.finish()]);
        dispatchElementwise(addPipeline, ffnTempBuf, vBuf, seqLen * kvDim, `L${l}-vb`, lw.vBias);
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

      // Write new K, V to cache
      copyToKVCache(kBuf, kvCache.keys[l], seqLen, kvDim, pos);
      copyToKVCache(vBuf, kvCache.values[l], seqLen, kvDim, pos);

      // Multi-head attention (reads full K,V cache including new tokens)
      const isCausal = seqLen > 1; // causal mask only needed for prefill
      dispatchAttention(
        qBuf, kvCache.keys[l], kvCache.values[l], attnOutBuf,
        seqLen, cacheLen, isCausal, pos, `L${l}-attn`,
      );

      // Output projection: [seq, nHeads*dHead] → [seq, H]
      dispatchMatmulBT(attnOutBuf, lw.oProj, attnProjBuf, seqLen, H, nHeads * dHead, `L${l}-o`);

      // O projection bias
      if (config.attentionBias && lw.oBias) {
        const encOb = device.createCommandEncoder({ label: `L${l}-ob-copy` });
        encOb.copyBufferToBuffer(attnProjBuf, 0, normedBuf, 0, seqLen * H * 4);
        device.queue.submit([encOb.finish()]);
        dispatchElementwise(addPipeline, normedBuf, attnProjBuf, seqLen * H, `L${l}-ob`, lw.oBias);
      }

      // Residual: hidden = residual + attn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `L${l}-res1`, attnProjBuf);

      if (isDebug && l === 0) {
        await debugRead(attnOutBuf, `L0-attn-out(pos=${pos})`, 8);
        await debugRead(attnProjBuf, `L0-attn-proj(pos=${pos})`, 8);
        await debugRead(hiddenBuf, `L0-after-attn-residual(pos=${pos})`, 8);
      }
      if (isDebug2 && l === 0) {
        // On second token, check if attention differs from just V (it should, with 2 cache entries)
        await debugRead(vBuf, `L0-V(pos=${pos})`, 8);
        console.log(`[DEBUG] cache_len=${cacheLen}, pos=${pos}`);
      }

      // Save for second residual
      const enc2 = device.createCommandEncoder({ label: `res2-${l}` });
      enc2.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([enc2.finish()]);

      // Post-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.postAttnNorm, seqLen, `L${l}-norm2`);

      // ── FFN (SwiGLU) ───────────────────────────────────────────────
      // MODEL-SPECIFIC: Phi fuses gate+up into one projection.
      dispatchMatmulBT(normedBuf, lw.gateProj, gateBuf, seqLen, ffnDim, H, `L${l}-gate`);
      dispatchMatmulBT(normedBuf, lw.upProj, upBuf, seqLen, ffnDim, H, `L${l}-up`);

      // MODEL-SPECIFIC: SiLU for most models, GELU for some
      // Cannot use same buffer for input and output in WebGPU — use dedicated temp
      dispatchElementwise(siluPipeline, gateBuf, ffnTempBuf, seqLen * ffnDim, `L${l}-silu`);
      dispatchElementwise(mulPipeline, ffnTempBuf, gateBuf, seqLen * ffnDim, `L${l}-mul`, upBuf);

      dispatchMatmulBT(gateBuf, lw.downProj, downBuf, seqLen, H, ffnDim, `L${l}-down`);

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
    // For multi-token input, we'd offset to the last row of normedBuf.
    // For seqLen=1 (decode), normedBuf IS the last token.
    dispatchMatmulBT(normedBuf, lmHeadBuf, logitsBuf, 1, V, H, 'lm-head');

    // Update cache position
    kvCache.position += seqLen;

    return { logitsBuffer: logitsBuf };
  }

  // ── KV Cache ───────────────────────────────────────────────────────

  function createKVCache(maxSeqLen: number): KVCache {
    const keys: GPUBuffer[] = [];
    const values: GPUBuffer[] = [];
    for (let l = 0; l < L; l++) {
      keys.push(createStorageBuffer(device, null, maxSeqLen * kvDim * 4, `kv-k-${l}`, true));
      values.push(createStorageBuffer(device, null, maxSeqLen * kvDim * 4, `kv-v-${l}`, true));
    }
    return { keys, values, position: 0, maxSeqLen };
  }

  return { forward, createKVCache, config };
}
