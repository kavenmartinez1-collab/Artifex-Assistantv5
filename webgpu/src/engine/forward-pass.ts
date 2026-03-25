/**
 * Forward Pass — Transformer Inference Orchestrator
 *
 * Chains WGSL compute kernels into a full transformer forward pass.
 * Model-agnostic: parameterized entirely by ModelConfig from config.json.
 *
 * Architecture (standard transformer decoder):
 *   embed(token_ids)
 *   for each layer:
 *     x = rmsnorm(x, input_norm_weight)
 *     x = attention(x, q/k/v/o weights, rope, kv_cache)  + residual
 *     x = rmsnorm(x, post_attn_norm_weight)
 *     x = ffn(x, gate/up/down weights)  + residual
 *   x = rmsnorm(x, final_norm_weight)
 *   logits = x @ lm_head
 *
 * Supported model families:
 *   Qwen (ChatML, GQA, RoPE θ=1M)
 *   Llama (GQA, RoPE θ=500K)
 *   Mistral (GQA, sliding window)
 *   Gemma (different norm placement — noted below)
 *   Phi (fused gate/up projection — noted below)
 *   DeepSeek (standard layout)
 *
 * MODEL-SPECIFIC NOTES (search for "MODEL-SPECIFIC" in this file):
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
} from './buffers';

import matmulWGSL from '../shaders/matmul.wgsl?raw';
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import ropeWGSL from '../shaders/rope.wgsl?raw';
import softmaxWGSL from '../shaders/softmax.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import embedWGSL from '../shaders/embed.wgsl?raw';

// ── Types ────────────────────────────────────────────────────────────────

/** GPU buffers for one transformer layer's weights. */
export interface LayerWeights {
  inputNorm: GPUBuffer;    // [hidden_size]
  qProj: GPUBuffer;        // [num_heads * head_dim, hidden_size]
  kProj: GPUBuffer;        // [num_kv_heads * head_dim, hidden_size]
  vProj: GPUBuffer;        // [num_kv_heads * head_dim, hidden_size]
  oProj: GPUBuffer;        // [hidden_size, num_heads * head_dim]
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
  forward(tokenIds: Uint32Array, kvCache: KVCache): ForwardOutput;

  /** Create an empty KV cache for the given max sequence length. */
  createKVCache(maxSeqLen: number): KVCache;

  /** Get the model config. */
  readonly config: ModelConfig;
}

/**
 * Create a forward pass engine.
 *
 * @param device  - WebGPU device
 * @param config  - Model configuration (from parseModelConfig)
 * @param weights - All model weights on the GPU
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
  } = config;

  const kvDim = nKVHeads * dHead; // total KV dimension

  // ── Compile pipelines ──────────────────────────────────────────────

  const embedPipeline = createComputePipeline(device, embedWGSL, 'embed', 'embed');
  const rmsnormPipeline = createComputePipeline(device, rmsnormWGSL, 'rmsnorm', 'rmsnorm');
  // MODEL-SPECIFIC: activation function. Most models use SiLU.
  // If config.hiddenAct === 'gelu', use the 'gelu' entry point instead.
  const siluPipeline = createComputePipeline(device, elementwiseWGSL, 'silu', 'silu');
  const mulPipeline = createComputePipeline(device, elementwiseWGSL, 'mul', 'mul');
  const addPipeline = createComputePipeline(device, elementwiseWGSL, 'add', 'add');
  const matmulPipeline = createComputePipeline(device, matmulWGSL, 'matmul_tiled', 'matmul');
  const softmaxPipeline = createComputePipeline(device, softmaxWGSL, 'softmax', 'softmax');
  const ropePipeline = createComputePipeline(device, ropeWGSL, 'rope', 'rope');

  // ── Reusable intermediate buffers ──────────────────────────────────
  // Allocated once, reused across layers to minimize VRAM usage.
  // Sized for the maximum possible dimensions.

  // Attention intermediates (sized for single-token decode, seq_len=1)
  // For prefill with seq_len > 1, we'd need larger buffers.
  const maxSeq = 1; // single-token decode for now

  // hidden state: [seq_len, hidden_size]
  const hiddenBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'hidden', true);
  const residualBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'residual', true);
  const normedBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'normed', true);

  // Q, K, V projections
  const qBuf = createStorageBuffer(device, null, maxSeq * nHeads * dHead * 4, 'q-proj', true);
  const kBuf = createStorageBuffer(device, null, maxSeq * kvDim * 4, 'k-proj', true);
  const vBuf = createStorageBuffer(device, null, maxSeq * kvDim * 4, 'v-proj', true);

  // Attention output
  const attnOutBuf = createStorageBuffer(device, null, maxSeq * nHeads * dHead * 4, 'attn-out', true);
  const attnProjBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'attn-proj', true);

  // FFN intermediates
  const gateBuf = createStorageBuffer(device, null, maxSeq * ffnDim * 4, 'ffn-gate', true);
  const upBuf = createStorageBuffer(device, null, maxSeq * ffnDim * 4, 'ffn-up', true);
  const downBuf = createStorageBuffer(device, null, maxSeq * H * 4, 'ffn-down', true);

  // Final logits
  const logitsBuf = createStorageBuffer(device, null, V * 4, 'logits', true);

  // Token ID input buffer
  const tokenIdBuf = createStorageBuffer(device, null, maxSeq * 4, 'token-ids', true);

  // ── Helper: dispatch matmul (C = A × B) ────────────────────────────
  // A: [M, K], B: [K, N], C: [M, N]
  function dispatchMatmul(
    aBuf: GPUBuffer, bBuf: GPUBuffer, cBuf: GPUBuffer,
    M: number, N: number, K: number, label: string,
  ) {
    const params = createUniformBuffer(
      device,
      new Uint32Array([M, N, K, 0]),
      `${label}-params`,
    );
    const bg = createBindGroup(device, matmulPipeline, 0, [
      { binding: 0, resource: { buffer: aBuf } },
      { binding: 1, resource: { buffer: bBuf } },
      { binding: 2, resource: { buffer: cBuf } },
      { binding: 3, resource: { buffer: params } },
    ], label);

    // Tiled matmul uses 16×16 tiles
    dispatch(device, matmulPipeline, [bg],
      [Math.ceil(N / 16), Math.ceil(M / 16)], label);
    params.destroy();
  }

  // ── Helper: dispatch rmsnorm ────────────────────────────────────────
  function dispatchRMSNorm(
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    weightBuf: GPUBuffer, rows: number, label: string,
  ) {
    const params = createUniformBuffer(
      device,
      new Float32Array([H, eps]),  // packed as [hidden_size(u32), eps(f32)]
      `${label}-params`,
    );
    // Rewrite params as mixed u32/f32
    const paramData = new ArrayBuffer(8);
    new Uint32Array(paramData, 0, 1)[0] = H;
    new Float32Array(paramData, 4, 1)[0] = eps;
    const paramBuf = createUniformBuffer(device, new Uint8Array(paramData), `${label}-params`);

    const bg = createBindGroup(device, rmsnormPipeline, 0, [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: weightBuf } },
      { binding: 3, resource: { buffer: paramBuf } },
    ], label);

    dispatch(device, rmsnormPipeline, [bg], [rows], label);
    params.destroy();
    paramBuf.destroy();
  }

  // ── Helper: dispatch elementwise op ─────────────────────────────────
  function dispatchElementwise(
    pipeline: GPUComputePipeline,
    inputBuf: GPUBuffer, outputBuf: GPUBuffer,
    size: number, label: string, secondBuf?: GPUBuffer,
  ) {
    const params = createUniformBuffer(
      device, new Uint32Array([size]), `${label}-params`,
    );
    const entries: Array<{ binding: number; resource: GPUBindingResource }> = [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
      { binding: 2, resource: { buffer: params } },
    ];
    if (secondBuf) {
      entries.push({ binding: 3, resource: { buffer: secondBuf } });
    }

    const bg = createBindGroup(device, pipeline, 0, entries, label);
    dispatch(device, pipeline, [bg], [workgroupCount(size, 256)], label);
    params.destroy();
  }

  // ── Forward Pass ───────────────────────────────────────────────────

  function forward(tokenIds: Uint32Array, kvCache: KVCache): ForwardOutput {
    const seqLen = tokenIds.length;
    const pos = kvCache.position;

    // Upload token IDs
    device.queue.writeBuffer(tokenIdBuf, 0, tokenIds.buffer, tokenIds.byteOffset, tokenIds.byteLength);

    // ── Embedding lookup ─────────────────────────────────────────────
    const embedParams = createUniformBuffer(
      device, new Uint32Array([H, seqLen]), 'embed-params',
    );
    const embedBG = createBindGroup(device, embedPipeline, 0, [
      { binding: 0, resource: { buffer: tokenIdBuf } },
      { binding: 1, resource: { buffer: hiddenBuf } },
      { binding: 2, resource: { buffer: weights.global.embedTokens } },
      { binding: 3, resource: { buffer: embedParams } },
    ], 'embed');
    dispatch(device, embedPipeline, [embedBG], [seqLen], 'embed');
    embedParams.destroy();

    // ── Transformer layers ───────────────────────────────────────────
    for (let l = 0; l < L; l++) {
      const lw = weights.layers[l];

      // Copy hidden state to residual for skip connection
      const copyEncoder = device.createCommandEncoder({ label: `copy-residual-${l}` });
      copyEncoder.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([copyEncoder.finish()]);

      // Pre-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.inputNorm, seqLen, `layer${l}-pre-attn-norm`);

      // Q, K, V projections: [seq, hidden] × [hidden, proj_dim]^T
      // Weight matrices are stored as [out_dim, in_dim] in HF format
      // So matmul is: normed[seq, H] × W^T[H, out_dim] = output[seq, out_dim]
      dispatchMatmul(normedBuf, lw.qProj, qBuf, seqLen, nHeads * dHead, H, `layer${l}-q`);
      dispatchMatmul(normedBuf, lw.kProj, kBuf, seqLen, kvDim, H, `layer${l}-k`);
      dispatchMatmul(normedBuf, lw.vProj, vBuf, seqLen, kvDim, H, `layer${l}-v`);

      // MODEL-SPECIFIC: RoPE
      // Apply rotary position embeddings to Q and K.
      // The RoPE theta and head_dim are model-specific (from config).
      // MODEL-SPECIFIC: Some models use NTK-aware or YaRN RoPE scaling
      // for extended context. This would modify the frequency computation.
      // TODO: dispatch RoPE on Q and K with position offset

      // MODEL-SPECIFIC: KV Cache
      // Store K and V in cache at position `pos`.
      // For TurboQuant-compressed cache, encode here.
      // TODO: copy K, V to kvCache at position offset
      // TODO: for attention, read full K, V history from cache

      // Attention: scores = Q @ K^T, softmax, output = scores @ V
      // For single-token decode: Q is [1, nHeads*dHead], K_cache is [pos+1, kvDim]
      // Full attention implementation requires head-aware batched matmul.
      // TODO: implement multi-head attention dispatch
      //   - Reshape Q to [nHeads, 1, dHead]
      //   - Reshape K to [nKVHeads, pos+1, dHead] (with GQA head expansion)
      //   - scores = Q @ K^T / sqrt(dHead)    [nHeads, 1, pos+1]
      //   - Apply causal mask (only for prefill, not needed for single-token)
      //   - weights = softmax(scores)
      //   - output = weights @ V               [nHeads, 1, dHead]
      //   - Reshape back to [1, nHeads*dHead]

      // Output projection: [seq, nHeads*dHead] × W_o^T[nHeads*dHead, H]
      dispatchMatmul(qBuf, lw.oProj, attnProjBuf, seqLen, H, nHeads * dHead, `layer${l}-o`);

      // Residual connection: hidden = residual + attn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `layer${l}-attn-residual`, attnProjBuf);

      // Copy for second residual
      const copyEncoder2 = device.createCommandEncoder({ label: `copy-residual2-${l}` });
      copyEncoder2.copyBufferToBuffer(hiddenBuf, 0, residualBuf, 0, seqLen * H * 4);
      device.queue.submit([copyEncoder2.finish()]);

      // Post-attention RMSNorm
      dispatchRMSNorm(hiddenBuf, normedBuf, lw.postAttnNorm, seqLen, `layer${l}-post-attn-norm`);

      // ── FFN (SwiGLU) ───────────────────────────────────────────────
      // MODEL-SPECIFIC: Phi models fuse gate+up into one projection.
      // For fused models, split the output in half after projection.
      dispatchMatmul(normedBuf, lw.gateProj, gateBuf, seqLen, ffnDim, H, `layer${l}-gate`);
      dispatchMatmul(normedBuf, lw.upProj, upBuf, seqLen, ffnDim, H, `layer${l}-up`);

      // MODEL-SPECIFIC: activation function
      // Most: SiLU(gate) * up.  GELU models: GELU(gate) * up.
      dispatchElementwise(siluPipeline, gateBuf, gateBuf, seqLen * ffnDim, `layer${l}-silu`);
      dispatchElementwise(mulPipeline, gateBuf, gateBuf, seqLen * ffnDim, `layer${l}-gate-mul`, upBuf);

      // Down projection
      dispatchMatmul(gateBuf, lw.downProj, downBuf, seqLen, H, ffnDim, `layer${l}-down`);

      // Residual connection: hidden = residual + ffn_output
      dispatchElementwise(addPipeline, residualBuf, hiddenBuf, seqLen * H, `layer${l}-ffn-residual`, downBuf);
    }

    // ── Final norm + LM head ─────────────────────────────────────────
    dispatchRMSNorm(hiddenBuf, normedBuf, weights.global.finalNorm, seqLen, 'final-norm');

    // For generation, we only need logits for the LAST token
    // LM head: normed[last_token] @ lm_head^T = logits[vocab_size]
    // TODO: for seq_len > 1, offset normedBuf to the last row
    const lmHeadBuf = config.tieWordEmbeddings
      ? weights.global.embedTokens
      : weights.global.lmHead;
    dispatchMatmul(normedBuf, lmHeadBuf, logitsBuf, 1, V, H, 'lm-head');

    // Update KV cache position
    kvCache.position += seqLen;

    return { logitsBuffer: logitsBuf };
  }

  // ── KV Cache ───────────────────────────────────────────────────────

  function createKVCache(maxSeqLen: number): KVCache {
    const keys: GPUBuffer[] = [];
    const values: GPUBuffer[] = [];

    for (let l = 0; l < L; l++) {
      keys.push(createStorageBuffer(
        device, null, maxSeqLen * kvDim * 4, `kv-cache-k-${l}`, true,
      ));
      values.push(createStorageBuffer(
        device, null, maxSeqLen * kvDim * 4, `kv-cache-v-${l}`, true,
      ));
    }

    return { keys, values, position: 0, maxSeqLen };
  }

  return {
    forward,
    createKVCache,
    config,
  };
}
