/**
 * Vision encoder — WGSL ViT forward pass, descriptor-driven.
 *
 * Runs the full tower for one preprocessed image:
 *   patches → patch-embed matmul (+bias) → +interpolated pos embeds
 *   → depth × [LN → QKV → MHA (non-causal) → proj (+residual)
 *              → LN → fc1 → GELU-tanh → fc2 (+residual)]
 *   → merger: LN(hidden) → reinterpret merge² merge-grouped rows as one
 *     (zero-copy: preprocess emits merge-window-contiguous patch order)
 *     → fc1 → GELU → fc2 → text-space embeddings
 *   → DeepStack taps: tower hidden after desc.deepstackIndexes blocks, each
 *     through its own merger (norm is POST-concat [merge²·H], unlike the
 *     main merger's pre-concat [H] norm).
 *
 * v1 is f32 end-to-end and dispatch-per-op (the tower runs once per image —
 * parity first, batching later). Reuses the text engine's WGSL sources
 * (matmul_bt, attention with is_causal=0, elementwise add/gelu) plus the
 * new layernorm kernel. WebGPU forbids aliasing a buffer as read +
 * read_write in one dispatch, so every op ping-pongs between buffers.
 *
 * Fused QKV is computed as three matmuls against 256-byte-aligned row
 * slices of the fused weight (rows [0,H)=Q, [H,2H)=K, [2H,3H)=V) so Q/K/V
 * land directly in the attention kernel's expected buffers.
 */

import { createComputePipeline, createBindGroup, dispatch, workgroupCount } from '../engine/compute';
import { createStorageBuffer, createUniformBuffer, readBuffer } from '../engine/buffers';
import type { VisionDescriptor } from './vision-descriptor';
import type { VisionWeights, VisionMergerWeights } from './vision-loader';
import type { PreprocessedImage } from './preprocess';

import matmulWGSL from '../shaders/matmul.wgsl?raw';
import layernormWGSL from '../shaders/layernorm.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';

/** attention.wgsl shared-memory ceiling — patches per image must fit. */
export const MAX_VIT_PATCHES = 3840;

export interface VisionEncodeResult {
  /** numTokens × outHidden f32 — inject via MultimodalPrompt. */
  embeddings: Float32Array;
  /** One per deepstackIndexes entry, same shape as embeddings. */
  deepstack: Float32Array[];
  numTokens: number;
}

export interface VisionEncoder {
  encode(pre: PreprocessedImage): Promise<VisionEncodeResult>;
  destroy(): void;
}

export function createVisionEncoder(
  device: GPUDevice,
  desc: VisionDescriptor,
  weights: VisionWeights,
): VisionEncoder {
  if (!desc.fusedQKV) {
    throw new Error('[Vision] split-QKV (GGUF clip) towers land in M2 — fused QKV only for now');
  }
  const H = desc.hiddenSize;
  const heads = desc.numHeads;
  const headDim = H / heads;
  const I = desc.intermediateSize;
  const merge = desc.projector.kind === 'qwen_merger' ? desc.projector.spatialMergeSize : 1;
  const mergeDim = H * merge * merge;
  const outH = desc.projector.outHiddenSize;
  const patchDim = desc.inChannels * desc.temporalPatchSize * desc.patchSize * desc.patchSize;

  // ── Pipelines ───────────────────────────────────────────────────────
  const matmulBT = createComputePipeline(device, matmulWGSL, 'matmul_bt', 'vis-matmul-bt');
  const layernormPipe = createComputePipeline(device, layernormWGSL, 'layernorm', 'vis-layernorm');
  const addPipe = createComputePipeline(device, elementwiseWGSL, 'add', 'vis-add');
  const geluPipe = createComputePipeline(device, elementwiseWGSL, 'gelu', 'vis-gelu');
  const attnPipe = createComputePipeline(device, attentionWGSL, 'attention', 'vis-attention', { USE_SOFTPICK: 0 });

  // ── Dispatch helpers (immediate submits; per-image one-shot) ────────
  function mm(
    A: GPUBuffer, B: GPUBuffer, C: GPUBuffer, M: number, N: number, K: number,
    bOffsetBytes = 0, bSizeBytes?: number,
  ) {
    const params = createUniformBuffer(device, new Uint32Array([M, N, K]), 'vis-mm-p');
    const bg = createBindGroup(device, matmulBT, 0, [
      { binding: 0, resource: { buffer: A } },
      { binding: 1, resource: bSizeBytes !== undefined
          ? { buffer: B, offset: bOffsetBytes, size: bSizeBytes }
          : { buffer: B } },
      { binding: 2, resource: { buffer: C } },
      { binding: 3, resource: { buffer: params } },
    ], 'vis-mm');
    dispatch(device, matmulBT, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], 'vis-mm');
  }

  function lnorm(x: GPUBuffer, out: GPUBuffer, w: GPUBuffer, b: GPUBuffer, rows: number, hidden: number) {
    const p = new ArrayBuffer(8);
    new Uint32Array(p, 0, 1)[0] = hidden;
    new Float32Array(p, 4, 1)[0] = 1e-6;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'vis-ln-p');
    const bg = createBindGroup(device, layernormPipe, 0, [
      { binding: 0, resource: { buffer: x } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: w } },
      { binding: 3, resource: { buffer: b } },
      { binding: 4, resource: { buffer: pbuf } },
    ], 'vis-ln');
    dispatch(device, layernormPipe, [bg], [rows], 'vis-ln');
  }

  /** out = a + b. b broadcasts every `broadcast` elements (0 = same length).
   *  Optional byte offset/size view into b (fused-bias slices, 256-aligned). */
  function add(
    a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, n: number, broadcast = 0,
    bOffsetBytes = 0, bSizeBytes?: number,
  ) {
    const p = new ArrayBuffer(28);
    const u = new Uint32Array(p);
    u[0] = n; u[1] = broadcast;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'vis-add-p');
    const bg = createBindGroup(device, addPipe, 0, [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: pbuf } },
      { binding: 3, resource: bSizeBytes !== undefined
          ? { buffer: b, offset: bOffsetBytes, size: bSizeBytes }
          : { buffer: b } },
    ], 'vis-add');
    dispatch(device, addPipe, [bg], [workgroupCount(n, 256)], 'vis-add');
  }

  function gelu(x: GPUBuffer, out: GPUBuffer, n: number) {
    const p = new ArrayBuffer(28);
    new Uint32Array(p)[0] = n;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'vis-gelu-p');
    const bg = createBindGroup(device, geluPipe, 0, [
      { binding: 0, resource: { buffer: x } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: pbuf } },
    ], 'vis-gelu');
    dispatch(device, geluPipe, [bg], [workgroupCount(n, 256)], 'vis-gelu');
  }

  function attention(q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, out: GPUBuffer, seqLen: number) {
    const p = new ArrayBuffer(48);
    const u = new Uint32Array(p);
    const f = new Float32Array(p);
    u[0] = heads; u[1] = heads; u[2] = headDim;   // ViTs: no GQA
    u[3] = seqLen; u[4] = seqLen;                 // all queries see all keys
    f[5] = 1 / Math.sqrt(headDim);
    u[6] = 0;                                     // is_causal = 0 (bidirectional)
    u[7] = 0; u[8] = 0;                           // pos_offset, window
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'vis-attn-p');
    const bg = createBindGroup(device, attnPipe, 0, [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: k } },
      { binding: 2, resource: { buffer: v } },
      { binding: 3, resource: { buffer: out } },
      { binding: 4, resource: { buffer: pbuf } },
    ], 'vis-attn');
    dispatch(device, attnPipe, [bg], [seqLen, heads], 'vis-attn');
  }

  function copyBuf(src: GPUBuffer, dst: GPUBuffer, bytes: number) {
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
    device.queue.submit([enc.finish()]);
  }

  /** Bilinearly interpolate the learned pos table (gridSize² × H, raster
   *  order) to gridH×gridW, emitted merge-window-grouped to match patch
   *  rows. align_corners=False convention — parity-checked in the harness. */
  function interpolatePosEmbed(gridH: number, gridW: number): Float32Array | null {
    if (!weights.posEmbedCPU || desc.posEmbed.kind !== 'learned') return null;
    const src = weights.posEmbedCPU;
    const S = desc.posEmbed.gridSize;
    const out = new Float32Array(gridH * gridW * H);
    let outRow = 0;
    const writePos = (gy: number, gx: number) => {
      const sy = ((gy + 0.5) * S) / gridH - 0.5;
      const sx = ((gx + 0.5) * S) / gridW - 0.5;
      const y0 = Math.max(0, Math.min(S - 1, Math.floor(sy)));
      const x0 = Math.max(0, Math.min(S - 1, Math.floor(sx)));
      const y1 = Math.min(S - 1, y0 + 1);
      const x1 = Math.min(S - 1, x0 + 1);
      const fy = Math.min(1, Math.max(0, sy - y0));
      const fx = Math.min(1, Math.max(0, sx - x0));
      const r00 = (y0 * S + x0) * H, r01 = (y0 * S + x1) * H;
      const r10 = (y1 * S + x0) * H, r11 = (y1 * S + x1) * H;
      const dst = outRow * H;
      outRow++;
      for (let c = 0; c < H; c++) {
        const top = src[r00 + c] * (1 - fx) + src[r01 + c] * fx;
        const bot = src[r10 + c] * (1 - fx) + src[r11 + c] * fx;
        out[dst + c] = top * (1 - fy) + bot * fy;
      }
    };
    const mh = Math.floor(gridH / merge), mw = Math.floor(gridW / merge);
    if (merge > 1) {
      for (let by = 0; by < mh; by++) for (let bx = 0; bx < mw; bx++)
        for (let wy = 0; wy < merge; wy++) for (let wx = 0; wx < merge; wx++)
          writePos(by * merge + wy, bx * merge + wx);
    } else {
      for (let gy = 0; gy < gridH; gy++) for (let gx = 0; gx < gridW; gx++) writePos(gy, gx);
    }
    return out;
  }

  // ── Encode ──────────────────────────────────────────────────────────
  async function encode(pre: PreprocessedImage): Promise<VisionEncodeResult> {
    const N = pre.gridH * pre.gridW;
    if (N > MAX_VIT_PATCHES) {
      throw new Error(`[Vision] ${N} patches exceeds attention limit ${MAX_VIT_PATCHES} — reduce maxPixels`);
    }
    if (pre.patches.length !== N * patchDim) {
      throw new Error(`[Vision] patch matrix ${pre.patches.length} != N*patchDim ${N * patchDim}`);
    }
    if (N % (merge * merge) !== 0) {
      throw new Error(`[Vision] ${N} patches not divisible by merge² = ${merge * merge}`);
    }
    const rows = N / (merge * merge);

    const t0 = performance.now();
    const bufs: GPUBuffer[] = [];
    const sb = (bytes: number, label: string) => {
      const b = createStorageBuffer(device, null, bytes, label, true);
      bufs.push(b);
      return b;
    };
    const patchBuf = createStorageBuffer(device, pre.patches, pre.patches.byteLength, 'vis-patches');
    bufs.push(patchBuf);
    // Ping-pong working set (WebGPU forbids read+write aliasing in a dispatch)
    const xBuf = sb(N * H * 4, 'vis-x');           // residual stream
    const x2Buf = sb(N * H * 4, 'vis-x2');         // residual ping-pong
    const normBuf = sb(N * H * 4, 'vis-norm');
    const qBuf = sb(N * H * 4, 'vis-q');
    const kBuf = sb(N * H * 4, 'vis-k');
    const vBuf = sb(N * H * 4, 'vis-v');
    const q2Buf = sb(N * H * 4, 'vis-q2');
    const k2Buf = sb(N * H * 4, 'vis-k2');
    const v2Buf = sb(N * H * 4, 'vis-v2');
    const attnBuf = sb(N * H * 4, 'vis-attn-o');
    const hBuf = sb(N * H * 4, 'vis-h');           // proj / fc2 outputs
    const h2Buf = sb(N * H * 4, 'vis-h2');         // post-bias
    const mlpBuf = sb(N * I * 4, 'vis-mlp');
    const mlp2Buf = sb(N * I * 4, 'vis-mlp2');
    const scratchA = sb(N * H * 4, 'vis-scr-a');   // merger LN out (= rows×mergeDim)
    const scratchB = sb(N * H * 4, 'vis-scr-b');   // merger fc1 out
    const scratchC = sb(N * H * 4, 'vis-scr-c');   // merger gelu out
    const mergeOut = sb(rows * outH * 4, 'vis-merge-o');
    const merge2 = sb(rows * outH * 4, 'vis-merge-o2');
    const dsTaps = new Map<number, GPUBuffer>();
    for (const idx of desc.deepstackIndexes) dsTaps.set(idx, sb(N * H * 4, `vis-ds-${idx}`));

    /** Merger (main: LN pre-concat over [N,H]; deepstack: LN post-concat
     *  over [rows, mergeDim]). Reads back rows×outH. */
    async function runMerger(m: VisionMergerWeights, src: GPUBuffer, preConcatNorm: boolean): Promise<Float32Array> {
      if (preConcatNorm) lnorm(src, scratchA, m.norm, m.normBias, N, H);
      else lnorm(src, scratchA, m.norm, m.normBias, rows, mergeDim);
      // scratchA is [rows, mergeDim] by reinterpretation (merge-grouped rows)
      mm(scratchA, m.fc1, scratchB, rows, mergeDim, mergeDim);
      add(scratchB, m.fc1Bias, scratchC, rows * mergeDim, mergeDim);
      gelu(scratchC, scratchB, rows * mergeDim);
      mm(scratchB, m.fc2, mergeOut, rows, outH, mergeDim);
      add(mergeOut, m.fc2Bias, merge2, rows * outH, outH);
      await device.queue.onSubmittedWorkDone();
      const raw = await readBuffer(device, merge2, rows * outH * 4);
      return new Float32Array(raw);
    }

    try {
      // Patch embed (+bias) and position embeds
      mm(patchBuf, weights.patchEmbed, hBuf, N, H, patchDim);
      add(hBuf, weights.patchEmbedBias, xBuf, N * H, H);
      const pos = interpolatePosEmbed(pre.gridH, pre.gridW);
      if (pos) {
        const posBuf = createStorageBuffer(device, pos, pos.byteLength, 'vis-pos');
        bufs.push(posBuf);
        add(xBuf, posBuf, x2Buf, N * H, 0);
        copyBuf(x2Buf, xBuf, N * H * 4);
      }

      // Tower blocks
      const Hbytes = H * H * 4;
      for (let l = 0; l < desc.depth; l++) {
        const blk = weights.blocks[l];
        lnorm(xBuf, normBuf, blk.ln1, blk.ln1Bias, N, H);
        // Q/K/V via row-slices of the fused [3H, H] weight + [3H] bias
        mm(normBuf, blk.qkv, qBuf, N, H, H, 0, Hbytes);
        mm(normBuf, blk.qkv, kBuf, N, H, H, Hbytes, Hbytes);
        mm(normBuf, blk.qkv, vBuf, N, H, H, 2 * Hbytes, Hbytes);
        add(qBuf, blk.qkvBias, q2Buf, N * H, H, 0, H * 4);
        add(kBuf, blk.qkvBias, k2Buf, N * H, H, H * 4, H * 4);
        add(vBuf, blk.qkvBias, v2Buf, N * H, H, 2 * H * 4, H * 4);
        attention(q2Buf, k2Buf, v2Buf, attnBuf, N);
        mm(attnBuf, blk.attnOut, hBuf, N, H, H);
        add(hBuf, blk.attnOutBias, h2Buf, N * H, H);
        add(xBuf, h2Buf, x2Buf, N * H, 0);                 // residual → x2
        lnorm(x2Buf, normBuf, blk.ln2, blk.ln2Bias, N, H);
        mm(normBuf, blk.fc1, mlpBuf, N, I, H);
        add(mlpBuf, blk.fc1Bias, mlp2Buf, N * I, I);
        gelu(mlp2Buf, mlpBuf, N * I);
        mm(mlpBuf, blk.fc2, hBuf, N, H, I);
        add(hBuf, blk.fc2Bias, h2Buf, N * H, H);
        add(x2Buf, h2Buf, xBuf, N * H, 0);                 // residual → x

        const tap = dsTaps.get(l);
        if (tap) copyBuf(xBuf, tap, N * H * 4);
      }

      const embeddings = await runMerger(weights.merger, xBuf, true);
      const deepstack: Float32Array[] = [];
      for (let d = 0; d < desc.deepstackIndexes.length; d++) {
        const tap = dsTaps.get(desc.deepstackIndexes[d])!;
        deepstack.push(await runMerger(weights.deepstack[d], tap, false));
      }

      console.log(`[Vision] encoded ${N} patches → ${rows} tokens in ${(performance.now() - t0).toFixed(0)}ms`);
      return { embeddings, deepstack, numTokens: rows };
    } finally {
      for (const b of bufs) b.destroy();
    }
  }

  return {
    encode,
    destroy() { weights.destroy(); },
  };
}
