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
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import attentionWGSL from '../shaders/attention.wgsl?raw';
import visionRopeWGSL from '../shaders/vision_rope.wgsl?raw';
import visionOpsWGSL from '../shaders/vision_ops.wgsl?raw';
import type { GemmaVisionWeights, ClampRange } from './vision-loader-gguf';

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
  weights: VisionWeights | GemmaVisionWeights,
): VisionEncoder {
  if (desc.towerVariant === 'gemma4') {
    return createGemmaVisionEncoder(device, desc, weights as GemmaVisionWeights);
  }
  return createQwenVisionEncoder(device, desc, weights as VisionWeights);
}

function createQwenVisionEncoder(
  device: GPUDevice,
  desc: VisionDescriptor,
  weights: VisionWeights,
): VisionEncoder {
  if (!desc.fusedQKV) {
    throw new Error('[Vision] split-QKV qwen-clip towers not yet wired — fused QKV only');
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
  const ropePipe = createComputePipeline(device, visionRopeWGSL, 'vision_rope', 'vis-rope');

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

  /** Apply 2D vision RoPE to a Q or K buffer using precomputed phases. */
  function visionRope(x: GPUBuffer, out: GPUBuffer, phases: GPUBuffer, rows: number) {
    const pbuf = createUniformBuffer(device, new Uint32Array([rows, H, headDim]), 'vis-rope-p');
    const bg = createBindGroup(device, ropePipe, 0, [
      { binding: 0, resource: { buffer: x } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: phases } },
      { binding: 3, resource: { buffer: pbuf } },
    ], 'vis-rope');
    dispatch(device, ropePipe, [bg], [workgroupCount(rows * H, 256)], 'vis-rope');
  }

  /** Per-patch 2D RoPE phases, merge-grouped to match patch row order.
   *  Layout per row: [r·f_0..r·f_{q-1}, c·f_0..c·f_{q-1}], q = headDim/4 —
   *  matches HF rot_pos_emb (freq_table[pos_ids].flatten(1)), theta 10000. */
  function buildRopePhases(gridH: number, gridW: number): Float32Array {
    const half = headDim / 2;           // phases per patch (32 for headDim 64)
    const q = half / 2;                 // freqs per axis (16)
    const invFreq = new Float32Array(q);
    for (let k = 0; k < q; k++) invFreq[k] = 1 / Math.pow(10000, (2 * k) / half);
    const out = new Float32Array(gridH * gridW * half);
    let row = 0;
    const writePhases = (gy: number, gx: number) => {
      const base = row * half;
      row++;
      for (let k = 0; k < q; k++) {
        out[base + k] = gy * invFreq[k];
        out[base + q + k] = gx * invFreq[k];
      }
    };
    const mh = Math.floor(gridH / merge), mw = Math.floor(gridW / merge);
    if (merge > 1) {
      for (let by = 0; by < mh; by++) for (let bx = 0; bx < mw; bx++)
        for (let wy = 0; wy < merge; wy++) for (let wx = 0; wx < merge; wx++)
          writePhases(by * merge + wy, bx * merge + wx);
    } else {
      for (let gy = 0; gy < gridH; gy++) for (let gx = 0; gx < gridW; gx++) writePhases(gy, gx);
    }
    return out;
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
      // HF fast_pos_embed_interpolate: linspace(0, S-1, grid) — align-corners
      const sy = gridH === 1 ? 0 : (gy * (S - 1)) / (gridH - 1);
      const sx = gridW === 1 ? 0 : (gx * (S - 1)) / (gridW - 1);
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

      // 2D RoPE phases — applied to Q/K inside every block
      const phases = buildRopePhases(pre.gridH, pre.gridW);
      const phasesBuf = createStorageBuffer(device, phases, phases.byteLength, 'vis-rope-phases');
      bufs.push(phasesBuf);

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
        // 2D RoPE on Q/K (rotated back into qBuf/kBuf — free post-bias)
        visionRope(q2Buf, qBuf, phasesBuf, N);
        visionRope(k2Buf, kBuf, phasesBuf, N);
        attention(qBuf, kBuf, v2Buf, attnBuf, N);
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

// ─────────────────────────────────────────────────────────────────────────
// Gemma 4 tower (spec: llama.cpp clip_graph_gemma4v — see vision_ops.wgsl
// header and the descriptor builder for the extracted constants).
//
// Per block: RMS ln1 → clamped Q/K/V matmuls → per-head RMS Q/K norms →
// x/y-half RoPE (θ=100) → weightless RMS on V → attention (scale 1.0) →
// clamped out-proj → RMS attn_post_norm → residual → RMS ln2 → clamped
// gated-GELU FFN → RMS ffn_post_norm → residual.
// Projector: 3×3 avg pool (×√H folded in) → clamp-free input projection →
// weightless RMS. Pos embeds: integer x/y table lookups summed on CPU.
// ─────────────────────────────────────────────────────────────────────────

function createGemmaVisionEncoder(
  device: GPUDevice,
  desc: VisionDescriptor,
  weights: GemmaVisionWeights,
): VisionEncoder {
  const H = desc.hiddenSize;
  const heads = desc.numHeads;
  const headDim = H / heads;
  const I = desc.intermediateSize;
  const pk = desc.gemma?.poolKernel ?? 3;
  const ropeTheta = desc.gemma?.ropeTheta ?? 100;
  const attnScale = desc.gemma?.attnScale ?? 1.0;
  const outH = desc.projector.outHiddenSize;
  const patchDim = desc.inChannels * desc.patchSize * desc.patchSize;

  const matmulBT = createComputePipeline(device, matmulWGSL, 'matmul_bt', 'gm-matmul');
  const rmsPipe = createComputePipeline(device, rmsnormWGSL, 'rmsnorm', 'gm-rms');
  const gateGeluPipe = createComputePipeline(device, elementwiseWGSL, 'gate_gelu', 'gm-gate-gelu');
  const attnPipe = createComputePipeline(device, attentionWGSL, 'attention', 'gm-attention', { USE_SOFTPICK: 0 });
  const clampPipe = createComputePipeline(device, visionOpsWGSL, 'clamp_op', 'gm-clamp');
  const poolPipe = createComputePipeline(device, visionOpsWGSL, 'avgpool2d', 'gm-pool');
  const ropeXYPipe = createComputePipeline(device, visionOpsWGSL, 'vision_rope_xy', 'gm-rope-xy');
  const addPipe = createComputePipeline(device, elementwiseWGSL, 'add', 'gm-add');

  /** out = a + b (distinct buffers — no aliasing). */
  function addInto(a: GPUBuffer, b: GPUBuffer, out: GPUBuffer, n: number) {
    const p = new ArrayBuffer(28);
    new Uint32Array(p)[0] = n;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-add-p');
    const bg = createBindGroup(device, addPipe, 0, [
      { binding: 0, resource: { buffer: a } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: pbuf } },
      { binding: 3, resource: { buffer: b } },
    ], 'gm-add');
    dispatch(device, addPipe, [bg], [workgroupCount(n, 256)], 'gm-add');
  }

  // Weightless RMS needs a bound weight buffer even when skipped
  const dummyWeight = createStorageBuffer(device, new Float32Array([1]), 4, 'gm-dummy-w');

  function mm(A: GPUBuffer, B: GPUBuffer, C: GPUBuffer, M: number, N: number, K: number) {
    const params = createUniformBuffer(device, new Uint32Array([M, N, K]), 'gm-mm-p');
    const bg = createBindGroup(device, matmulBT, 0, [
      { binding: 0, resource: { buffer: A } },
      { binding: 1, resource: { buffer: B } },
      { binding: 2, resource: { buffer: C } },
      { binding: 3, resource: { buffer: params } },
    ], 'gm-mm');
    dispatch(device, matmulBT, [bg], [Math.ceil(M / 16), Math.ceil(N / 16)], 'gm-mm');
  }

  /** RMS norm: rows × hidden. weightless=true skips the weight multiply. */
  function rms(x: GPUBuffer, out: GPUBuffer, w: GPUBuffer | null, rows: number, hidden: number) {
    const p = new ArrayBuffer(16);
    const u = new Uint32Array(p);
    new Uint32Array(p, 0, 1)[0] = hidden;
    new Float32Array(p, 4, 1)[0] = 1e-6;
    u[2] = 0;                      // use_residual_weight = 0 (plain ×w)
    u[3] = w === null ? 1 : 0;     // skip_weight (weightless RMS)
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-rms-p');
    const bg = createBindGroup(device, rmsPipe, 0, [
      { binding: 0, resource: { buffer: x } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: w ?? dummyWeight } },
      { binding: 3, resource: { buffer: pbuf } },
    ], 'gm-rms');
    dispatch(device, rmsPipe, [bg], [rows], 'gm-rms');
  }

  function clampOp(x: GPUBuffer, out: GPUBuffer, n: number, lo: number, hi: number) {
    const p = new ArrayBuffer(12);
    new Uint32Array(p, 0, 1)[0] = n;
    new Float32Array(p, 4, 1)[0] = lo;
    new Float32Array(p, 8, 1)[0] = hi;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-clamp-p');
    const bg = createBindGroup(device, clampPipe, 0, [
      { binding: 0, resource: { buffer: x } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: pbuf } },
    ], 'gm-clamp');
    dispatch(device, clampPipe, [bg], [workgroupCount(n, 256)], 'gm-clamp');
  }

  /** Gemma4ClippableLinear: clamp input, matmul, clamp output.
   *  scratch must hold M×K floats; out and post share M×N. */
  function clampedMM(
    inBuf: GPUBuffer, W: GPUBuffer, outBuf: GPUBuffer, postBuf: GPUBuffer,
    M: number, N: number, K: number, clamp: ClampRange | undefined, scratch: GPUBuffer,
  ): GPUBuffer {
    if (!clamp) {
      mm(inBuf, W, outBuf, M, N, K);
      return outBuf;
    }
    clampOp(inBuf, scratch, M * K, clamp.inLo, clamp.inHi);
    mm(scratch, W, outBuf, M, N, K);
    clampOp(outBuf, postBuf, M * N, clamp.outLo, clamp.outHi);
    return postBuf;
  }

  /** out = up * gelu(gate) — elementwise gate_gelu (a=up, b=gate). */
  function gateGelu(up: GPUBuffer, gate: GPUBuffer, out: GPUBuffer, n: number) {
    const p = new ArrayBuffer(28);
    new Uint32Array(p)[0] = n;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-gg-p');
    const bg = createBindGroup(device, gateGeluPipe, 0, [
      { binding: 0, resource: { buffer: up } },
      { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: pbuf } },
      { binding: 3, resource: { buffer: gate } },
    ], 'gm-gg');
    dispatch(device, gateGeluPipe, [bg], [workgroupCount(n, 256)], 'gm-gg');
  }

  function attention(q: GPUBuffer, k: GPUBuffer, v: GPUBuffer, out: GPUBuffer, seqLen: number) {
    const p = new ArrayBuffer(48);
    const u = new Uint32Array(p);
    const f = new Float32Array(p);
    u[0] = heads; u[1] = heads; u[2] = headDim;
    u[3] = seqLen; u[4] = seqLen;
    f[5] = attnScale;            // gemma4v: 1.0, not 1/sqrt(d)
    u[6] = 0; u[7] = 0; u[8] = 0;
    const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-attn-p');
    const bg = createBindGroup(device, attnPipe, 0, [
      { binding: 0, resource: { buffer: q } },
      { binding: 1, resource: { buffer: k } },
      { binding: 2, resource: { buffer: v } },
      { binding: 3, resource: { buffer: out } },
      { binding: 4, resource: { buffer: pbuf } },
    ], 'gm-attn');
    dispatch(device, attnPipe, [bg], [seqLen, heads], 'gm-attn');
  }

  function ropeXY(x: GPUBuffer, out: GPUBuffer, phases: GPUBuffer, rows: number) {
    const pbuf = createUniformBuffer(device, new Uint32Array([rows, H, headDim]), 'gm-rope-p');
    const bg = createBindGroup(device, ropeXYPipe, 0, [
      { binding: 6, resource: { buffer: x } },
      { binding: 7, resource: { buffer: out } },
      { binding: 8, resource: { buffer: phases } },
      { binding: 9, resource: { buffer: pbuf } },
    ], 'gm-rope');
    dispatch(device, ropeXYPipe, [bg], [workgroupCount(rows * H, 256)], 'gm-rope');
  }

  async function encode(pre: PreprocessedImage): Promise<VisionEncodeResult> {
    const N = pre.gridH * pre.gridW;
    if (N > MAX_VIT_PATCHES) {
      throw new Error(`[Vision] ${N} patches exceeds attention limit ${MAX_VIT_PATCHES}`);
    }
    if (pre.patches.length !== N * patchDim) {
      throw new Error(`[Vision] patch matrix ${pre.patches.length} != N*patchDim ${N * patchDim}`);
    }
    if (pre.gridH % pk !== 0 || pre.gridW % pk !== 0) {
      throw new Error(`[Vision] grid ${pre.gridH}x${pre.gridW} not divisible by pool kernel ${pk}`);
    }
    const outW = pre.gridW / pk, outHrows = pre.gridH / pk;
    const R = outW * outHrows;
    const t0 = performance.now();

    const bufs: GPUBuffer[] = [];
    const sb = (bytes: number, label: string) => {
      const b = createStorageBuffer(device, null, bytes, label, true);
      bufs.push(b);
      return b;
    };
    const patchBuf = createStorageBuffer(device, pre.patches, pre.patches.byteLength, 'gm-patches');
    bufs.push(patchBuf);
    const xBuf = sb(N * H * 4, 'gm-x');
    const x2Buf = sb(N * H * 4, 'gm-x2');
    const normBuf = sb(N * H * 4, 'gm-norm');
    const scratchH = sb(N * H * 4, 'gm-scratch');     // clamp input scratch (K=H)
    const qa = sb(N * H * 4, 'gm-qa'); const qb = sb(N * H * 4, 'gm-qb');
    const ka = sb(N * H * 4, 'gm-ka'); const kb = sb(N * H * 4, 'gm-kb');
    const va = sb(N * H * 4, 'gm-va'); const vb = sb(N * H * 4, 'gm-vb');
    const attnBuf = sb(N * H * 4, 'gm-attn-o');
    const hBuf = sb(N * H * 4, 'gm-h');
    const h2Buf = sb(N * H * 4, 'gm-h2');
    const gateA = sb(N * I * 4, 'gm-gate-a'); const gateB = sb(N * I * 4, 'gm-gate-b');
    const upA = sb(N * I * 4, 'gm-up-a'); const upB = sb(N * I * 4, 'gm-up-b');
    const mlpBuf = sb(N * I * 4, 'gm-mlp');
    const scratchI = sb(N * I * 4, 'gm-scratch-i');   // clamp scratch for down (K=I)
    const poolBuf = sb(R * H * 4, 'gm-pool');
    const projBuf = sb(R * outH * 4, 'gm-proj');
    const outBuf = sb(R * outH * 4, 'gm-out');

    try {
      // Patch embed (no bias)
      mm(patchBuf, weights.patchEmbed, xBuf, N, H, patchDim);

      // Factorized pos embeds: CPU gather tbl_x[col] + tbl_y[row], raster order
      {
        const pos = new Float32Array(N * H);
        for (let i = 0; i < N; i++) {
          const px = i % pre.gridW, py = Math.floor(i / pre.gridW);
          const xr = px * H, yr = py * H, dst = i * H;
          for (let c = 0; c < H; c++) {
            pos[dst + c] = weights.posTableX[xr + c] + weights.posTableY[yr + c];
          }
        }
        const posBuf = createStorageBuffer(device, pos, pos.byteLength, 'gm-pos');
        bufs.push(posBuf);
        addInto(xBuf, posBuf, x2Buf, N * H);
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(x2Buf, 0, xBuf, 0, N * H * 4);
        device.queue.submit([enc.finish()]);
      }

      // RoPE phases: raster (col=x, row=y), θ_j = pos · theta^(-2j/half)
      const half = headDim / 2, qFreqs = half / 2;
      const inv = new Float32Array(qFreqs);
      for (let j = 0; j < qFreqs; j++) inv[j] = 1 / Math.pow(ropeTheta, (2 * j) / half);
      const ph = new Float32Array(N * half);
      for (let i = 0; i < N; i++) {
        const px = i % pre.gridW, py = Math.floor(i / pre.gridW);
        for (let j = 0; j < qFreqs; j++) {
          ph[i * half + j] = px * inv[j];
          ph[i * half + qFreqs + j] = py * inv[j];
        }
      }
      const phasesBuf = createStorageBuffer(device, ph, ph.byteLength, 'gm-phases');
      bufs.push(phasesBuf);

      for (let l = 0; l < desc.depth; l++) {
        const blk = weights.blocks[l];
        rms(xBuf, normBuf, blk.ln1, N, H);
        const qOut = clampedMM(normBuf, blk.q, qa, qb, N, H, H, blk.qClamp, scratchH);
        const kOut = clampedMM(normBuf, blk.k, ka, kb, N, H, H, blk.kClamp, scratchH);
        const vOut = clampedMM(normBuf, blk.v, va, vb, N, H, H, blk.vClamp, scratchH);
        // Per-head RMS norms (rows = N·heads over head_dim) + weightless V norm
        const qN = qOut === qa ? qb : qa;
        const kN = kOut === ka ? kb : ka;
        const vN = vOut === va ? vb : va;
        rms(qOut, qN, blk.qNorm, N * heads, headDim);
        rms(kOut, kN, blk.kNorm, N * heads, headDim);
        rms(vOut, vN, null, N * heads, headDim);
        // x/y-half RoPE on Q/K (back into the other buffer of each pair)
        ropeXY(qN, qOut, phasesBuf, N);
        ropeXY(kN, kOut, phasesBuf, N);
        attention(qOut, kOut, vN, attnBuf, N);
        const oOut = clampedMM(attnBuf, blk.attnOut, hBuf, h2Buf, N, H, H, blk.attnOutClamp, scratchH);
        const oNormed = oOut === hBuf ? h2Buf : hBuf;
        rms(oOut, oNormed, blk.attnPostNorm, N, H);
        // residual → x2
        addInto(xBuf, oNormed, x2Buf, N * H);
        rms(x2Buf, normBuf, blk.ln2, N, H);
        const gOut = clampedMM(normBuf, blk.gate, gateA, gateB, N, I, H, blk.gateClamp, scratchH);
        const uOut = clampedMM(normBuf, blk.up, upA, upB, N, I, H, blk.upClamp, scratchH);
        gateGelu(uOut, gOut, mlpBuf, N * I);
        const dOut = clampedMM(mlpBuf, blk.down, hBuf, h2Buf, N, H, I, blk.downClamp, scratchI);
        const dNormed = dOut === hBuf ? h2Buf : hBuf;
        rms(dOut, dNormed, blk.ffnPostNorm, N, H);
        addInto(x2Buf, dNormed, xBuf, N * H);
      }

      // Pooler: kernel×kernel avg, ×√H — then projection and weightless RMS
      {
        const p = new ArrayBuffer(28);
        const u = new Uint32Array(p);
        u[0] = pre.gridW; u[1] = pre.gridH; u[2] = H; u[3] = pk; u[4] = outW; u[5] = outHrows;
        new Float32Array(p, 24, 1)[0] = Math.sqrt(H);
        const pbuf = createUniformBuffer(device, new Uint32Array(p), 'gm-pool-p');
        const bg = createBindGroup(device, poolPipe, 0, [
          { binding: 3, resource: { buffer: xBuf } },
          { binding: 4, resource: { buffer: poolBuf } },
          { binding: 5, resource: { buffer: pbuf } },
        ], 'gm-pool');
        dispatch(device, poolPipe, [bg], [workgroupCount(R * H, 256)], 'gm-pool');
      }
      mm(poolBuf, weights.inputProjection, projBuf, R, outH, H);
      rms(projBuf, outBuf, null, R, outH);

      await device.queue.onSubmittedWorkDone();
      const raw = await readBuffer(device, outBuf, R * outH * 4);
      console.log(`[Vision] gemma encoded ${N} patches → ${R} tokens in ${(performance.now() - t0).toFixed(0)}ms`);
      return { embeddings: new Float32Array(raw), deepstack: [], numTokens: R };
    } finally {
      for (const b of bufs) b.destroy();
    }
  }

  return {
    encode,
    destroy() {
      weights.destroy();
      dummyWeight.destroy();
    },
  };
}
