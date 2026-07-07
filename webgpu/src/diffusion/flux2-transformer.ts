// FLUX.2-klein DiT (Flux2Transformer2DModel) — one denoise step on WebGPU.
//
// Math verified against venv diffusers 0.39 transformer_flux2.py:
//   - inner 3072 (24h x 128d), 5 double + 20 single blocks, all Linear bias=False
//   - temb = TimestepEmbedding(sinusoid_256(t*1000))  [flip_sin_to_cos, shift 0]
//   - THREE shared modulation vectors computed ONCE per step from
//     Linear(SiLU(temb)): double_img [18432], double_txt [18432], single [9216];
//     each set is (shift, scale, gate) x 3072, sets ordered (msa, mlp)
//   - joint sequence is [txt(512) | img]; per-head RMSNorm (w-direct, eps 1e-6)
//     on q,k; adjacent-pair RoPE (theta 2000, axes [32,32,32,32]); non-causal
//     SDPA; SwiGLU FFN (18432 -> gate -> 9216 -> 3072)
//   - single block: fused qkv+mlp proj [27648], one gated residual over
//     to_out([attn | mlp_gated])
//   - norm_out = AdaLayerNormContinuous: (scale, shift) = chunk(Linear(SiLU(temb)));
//     x = LN(x)*(1+scale)+shift; then proj_out -> [Simg, 128]
//
// All GEMMs run matmul_bt_bf16_fast directly over the raw bf16 weight buffers
// (64x64 tile, plain f32 accumulation — 3.3-3.7 TFLOPS effective vs 0.63 for
// the Kahan matmul_bt_bf16; DiT parity re-gated after the swap).
// TDR budgeting: dispatches are cost-weighted and the queue is flushed at
// ~FLOP_BUDGET per submit; GEMMs and attention are M/query-sliced so no single
// dispatch exceeds the budget (worst 1024px GEMM now ~214 ms unsliced).
//
// NOTE: the device must be created with maxBufferSize/maxStorageBufferBindingSize
// raised to the adapter limits (mlp activations reach 339 MB at 1024px).

import matmulWGSL from '../shaders/matmul.wgsl?raw';
import rmsnormWGSL from '../shaders/rmsnorm.wgsl?raw';
import layernormWGSL from '../shaders/layernorm.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import adalnWGSL from '../shaders/adaln.wgsl?raw';
import ropePairsWGSL from '../shaders/rope_pairs.wgsl?raw';
import attnStreamWGSL from '../shaders/attention_stream.wgsl?raw';
import type { Flux2DitWeights, Flux2MatWeight } from './flux2-loader';

export const FLUX2_TXT_LEN = 512;
const DIM = 3072;
const HEADS = 24;
const HEAD_DIM = 128;
const MLP_IN = 18432; // 2 * 9216
const MLP_OUT = 9216;
const CAT_DIM = DIM + MLP_OUT; // 12288
const PE_DIM = 7680;
const EPS = 1e-6;

// Element offsets into the shared per-step modulation buffer.
const MOD_IMG = 0;
const MOD_TXT = 18432;
const MOD_SINGLE = 36864;
const MOD_NORM_OUT = 46080; // 6144: [scale | shift]
const MOD_TOTAL = 52224;

const F32 = Math.fround;

export interface Bind { b: number; buf: GPUBuffer; off?: number; size?: number }

/** Uniform slot arena: one writeBuffer per dispatch into 256-byte slots so
 *  encoders never see a uniform mutated between recording and submit. */
export class UniArena {
  private bufs: GPUBuffer[] = [];
  private used = 0;
  private readonly perBuf = 1024; // slots
  constructor(private device: GPUDevice) {}
  alloc(bytes: ArrayBuffer): { buf: GPUBuffer; off: number } {
    const bi = Math.floor(this.used / this.perBuf);
    if (bi >= this.bufs.length) {
      this.bufs.push(this.device.createBuffer({
        size: this.perBuf * 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }));
    }
    const off = (this.used % this.perBuf) * 256;
    this.device.queue.writeBuffer(this.bufs[bi], off, bytes);
    this.used++;
    return { buf: this.bufs[bi], off };
  }
  reset() { this.used = 0; }
  destroy() { for (const b of this.bufs) b.destroy(); this.bufs = []; }
}

export class Ctx {
  private enc: GPUCommandEncoder | null = null;
  private pass: GPUComputePassEncoder | null = null;
  private cost = 0;
  constructor(private device: GPUDevice, readonly budget: number) {}

  dispatch(pipe: GPUComputePipeline, binds: Bind[], gx: number, gy: number, cost: number) {
    if (this.cost > 0 && this.cost + cost > this.budget) this.flush();
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    if (!this.pass) this.pass = this.enc.beginComputePass();
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: binds.map((e) => ({
        binding: e.b,
        resource: { buffer: e.buf, offset: e.off ?? 0, ...(e.size !== undefined ? { size: e.size } : {}) },
      })),
    });
    this.pass.setPipeline(pipe);
    this.pass.setBindGroup(0, bg);
    this.pass.dispatchWorkgroups(gx, gy, 1);
    this.cost += cost;
  }

  flush() {
    if (this.pass) { this.pass.end(); this.pass = null; }
    if (this.enc) { this.device.queue.submit([this.enc.finish()]); this.enc = null; }
    this.cost = 0;
  }

  async read(buf: GPUBuffer, byteOff: number, elems: number): Promise<Float32Array> {
    if (this.pass) { this.pass.end(); this.pass = null; }
    if (!this.enc) this.enc = this.device.createCommandEncoder();
    const staging = this.device.createBuffer({
      size: elems * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    this.enc.copyBufferToBuffer(buf, byteOff, staging, 0, elems * 4);
    this.device.queue.submit([this.enc.finish()]);
    this.enc = null;
    this.cost = 0;
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange()).slice();
    staging.unmap();
    staging.destroy();
    return out;
  }
}

export class Flux2Transformer {
  private pipes!: Record<string, GPUComputePipeline>;
  private arena: UniArena;
  private ctx: Ctx;

  // Persistent per-resolution state
  private imgTokens = 0;
  private S = 0;
  private bufs: Record<string, GPUBuffer> = {};
  private promptEmbedsSet = false;

  constructor(
    private device: GPUDevice,
    private weights: Flux2DitWeights,
    flopBudget = 0.35e12,
  ) {
    this.arena = new UniArena(device);
    this.ctx = new Ctx(device, flopBudget);
    this.compile();
  }

  private compile() {
    const d = this.device;
    const mk = (code: string, entry: string) => d.createComputePipeline({
      layout: 'auto',
      compute: { module: d.createShaderModule({ code }), entryPoint: entry },
    });
    this.pipes = {
      gemm: mk(matmulWGSL, 'matmul_bt_bf16_fast'),
      rms: mk(rmsnormWGSL, 'rmsnorm'),
      ln: mk(layernormWGSL, 'layernorm'),
      silu: mk(elementwiseWGSL, 'silu'),
      modulate: mk(adalnWGSL, 'adaln_modulate'),
      swiglu: mk(adalnWGSL, 'swiglu_gate'),
      gateAdd: mk(adalnWGSL, 'gate_add'),
      concat: mk(adalnWGSL, 'concat_cols'),
      rope: mk(ropePairsWGSL, 'rope_pairs'),
      attn: mk(attnStreamWGSL, 'attention_stream_qt'),
      transpose: mk(attnStreamWGSL, 'transpose_khds'),
    };
  }

  private mat(name: string): Flux2MatWeight {
    const m = this.weights.mats.get(name);
    if (!m) throw new Error(`[Flux2 DiT] missing matrix weight: ${name}`);
    return m;
  }
  private vec(name: string): GPUBuffer {
    const v = this.weights.vecs.get(name);
    if (!v) throw new Error(`[Flux2 DiT] missing vector weight: ${name}`);
    return v;
  }

  /** Allocate activation buffers + upload RoPE tables for a resolution.
   *  cos/sin: [S, 128] f32 joint tables (txt rows 0..511 first, then img). */
  setup(imgTokens: number, cos: Float32Array, sin: Float32Array) {
    const S = FLUX2_TXT_LEN + imgTokens;
    if (cos.length !== S * HEAD_DIM || sin.length !== S * HEAD_DIM) {
      throw new Error(`[Flux2 DiT] rope table length ${cos.length} != ${S * HEAD_DIM}`);
    }
    this.releaseActivations();
    this.imgTokens = imgTokens;
    this.S = S;
    const d = this.device;
    const alloc = (elems: number) => d.createBuffer({
      size: elems * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const b = this.bufs;
    b.pe = alloc(FLUX2_TXT_LEN * PE_DIM);
    b.lat = alloc(imgTokens * 128);
    b.streamA = alloc(S * DIM);
    b.streamB = alloc(S * DIM);
    b.normedA = alloc(S * DIM);
    b.normedB = alloc(S * DIM);
    b.q = alloc(S * DIM);
    b.k = alloc(S * DIM);
    b.v = alloc(S * DIM);
    b.kt = alloc(S * DIM);
    b.attn = alloc(S * DIM);
    b.proj = alloc(S * DIM);
    b.mlp = alloc(S * MLP_IN);
    b.gate = alloc(S * MLP_OUT);
    b.cat = alloc(S * CAT_DIM);
    b.mod = alloc(MOD_TOTAL);
    b.sinusoid = alloc(256);
    b.t1 = alloc(DIM);
    b.t2 = alloc(DIM);
    b.temb = alloc(DIM);
    b.tembSilu = alloc(DIM);
    b.out = alloc(imgTokens * 128);
    b.ropeCos = alloc(S * HEAD_DIM);
    b.ropeSin = alloc(S * HEAD_DIM);
    b.ones = alloc(DIM);
    b.zeros = alloc(DIM);
    d.queue.writeBuffer(b.ropeCos, 0, cos.buffer as ArrayBuffer, cos.byteOffset, cos.byteLength);
    d.queue.writeBuffer(b.ropeSin, 0, sin.buffer as ArrayBuffer, sin.byteOffset, sin.byteLength);
    d.queue.writeBuffer(b.ones, 0, new Float32Array(DIM).fill(1));
    d.queue.writeBuffer(b.zeros, 0, new Float32Array(DIM));
    this.promptEmbedsSet = false;
  }

  setPromptEmbeds(pe: Float32Array) {
    if (pe.length !== FLUX2_TXT_LEN * PE_DIM) {
      throw new Error(`[Flux2 DiT] prompt_embeds length ${pe.length} != ${FLUX2_TXT_LEN * PE_DIM}`);
    }
    this.device.queue.writeBuffer(this.bufs.pe, 0, pe.buffer as ArrayBuffer, pe.byteOffset, pe.byteLength);
    this.promptEmbedsSet = true;
  }

  // ── dispatch helpers ─────────────────────────────────────────────────

  private uni(words: (number | ['f32', number])[]): { buf: GPUBuffer; off: number } {
    const ab = new ArrayBuffer(48);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    words.forEach((w, i) => {
      if (Array.isArray(w)) f[i] = w[1]; else u[i] = w;
    });
    return this.arena.alloc(ab);
  }

  /** C[M,N](+cOff) = A[M,K](+aOff) @ W^T, M-sliced to the flush budget.
   *  wRowOff/wRows select a row band of the [N_total, K] bf16 weight. */
  private gemm(
    a: GPUBuffer, aOff: number, w: Flux2MatWeight, wRowOff: number, wRows: number,
    c: GPUBuffer, cOff: number, M: number,
  ) {
    const N = wRows, K = w.k;
    const per = 2 * N * K;
    const maxRows = Math.max(16, Math.floor(this.ctx.budget / per));
    for (let m0 = 0; m0 < M; m0 += maxRows) {
      const mS = Math.min(maxRows, M - m0);
      const u = this.uni([mS, N, K]);
      this.ctx.dispatch(this.pipes.gemm, [
        { b: 0, buf: a, off: aOff + m0 * K * 4, size: mS * K * 4 },
        { b: 2, buf: c, off: cOff + m0 * N * 4, size: mS * N * 4 },
        { b: 3, buf: u.buf, off: u.off, size: 48 },
        { b: 5, buf: w.buffer, off: wRowOff * K * 2, size: wRows * K * 2 },
      ], Math.ceil(mS / 64), Math.ceil(N / 64), per * mS);
    }
  }

  /** LayerNorm (no affine: ones/zeros), rows of width DIM. */
  private ln(src: GPUBuffer, srcOff: number, rows: number, dst: GPUBuffer, dstOff: number) {
    const u = this.uni([DIM, ['f32', EPS]]);
    this.ctx.dispatch(this.pipes.ln, [
      { b: 0, buf: src, off: srcOff, size: rows * DIM * 4 },
      { b: 1, buf: dst, off: dstOff, size: rows * DIM * 4 },
      { b: 2, buf: this.bufs.ones },
      { b: 3, buf: this.bufs.zeros },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], rows, 1, rows * DIM * 8);
  }

  /** adaln_modulate rows of width DIM with Mod offsets (shift, scale). */
  private modulate(src: GPUBuffer, off: number, rows: number, dst: GPUBuffer, shiftOff: number, scaleOff: number) {
    const u = this.uni([rows, DIM, shiftOff, scaleOff]);
    this.ctx.dispatch(this.pipes.modulate, [
      { b: 0, buf: src, off, size: rows * DIM * 4 },
      { b: 2, buf: dst, off, size: rows * DIM * 4 },
      { b: 3, buf: this.bufs.mod },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(DIM / 256), rows, rows * DIM * 4);
  }

  /** dst = x + Mod[gateOff..]*y over rows of width DIM. */
  private gateAdd(x: GPUBuffer, y: GPUBuffer, off: number, rows: number, dst: GPUBuffer, gateOff: number) {
    const u = this.uni([rows, DIM, gateOff, 0]);
    this.ctx.dispatch(this.pipes.gateAdd, [
      { b: 0, buf: x, off, size: rows * DIM * 4 },
      { b: 1, buf: y, off, size: rows * DIM * 4 },
      { b: 2, buf: dst, off, size: rows * DIM * 4 },
      { b: 3, buf: this.bufs.mod },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(DIM / 256), rows, rows * DIM * 4);
  }

  /** Per-head RMSNorm on a [rows128, 128] view (torch w-direct, eps 1e-6).
   *  Splits >65535-row dispatches via 512-byte-aligned row offsets. */
  private qkNorm(src: GPUBuffer, byteOff: number, rows128: number, dst: GPUBuffer, weight: GPUBuffer) {
    const MAXR = 65280;
    for (let r0 = 0; r0 < rows128; r0 += MAXR) {
      const r = Math.min(MAXR, rows128 - r0);
      const u = this.uni([HEAD_DIM, ['f32', EPS], 0, 0]);
      this.ctx.dispatch(this.pipes.rms, [
        { b: 0, buf: src, off: byteOff + r0 * HEAD_DIM * 4, size: r * HEAD_DIM * 4 },
        { b: 1, buf: dst, off: byteOff + r0 * HEAD_DIM * 4, size: r * HEAD_DIM * 4 },
        { b: 2, buf: weight },
        { b: 3, buf: u.buf, off: u.off, size: 48 },
      ], r, 1, r * HEAD_DIM * 8);
    }
  }

  private rope(x: GPUBuffer) {
    const u = this.uni([HEADS, HEAD_DIM, this.S, 0]);
    this.ctx.dispatch(this.pipes.rope, [
      { b: 0, buf: x, size: this.S * DIM * 4 },
      { b: 1, buf: this.bufs.ropeCos },
      { b: 2, buf: this.bufs.ropeSin },
      { b: 3, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil((HEADS * HEAD_DIM / 2) / 256), this.S, this.S * DIM * 4);
  }

  /** Q,V: [S, 3072]; K transposed to kt then streamed SDPA -> attn buffer. */
  private attention(q: GPUBuffer, k: GPUBuffer) {
    const S = this.S;
    const ut = this.uni([HEADS, HEAD_DIM, S, S, 0]);
    this.ctx.dispatch(this.pipes.transpose, [
      { b: 0, buf: k, size: S * DIM * 4 },
      { b: 3, buf: this.bufs.kt, size: S * DIM * 4 },
      { b: 4, buf: ut.buf, off: ut.off, size: 48 },
    ], Math.ceil(S * DIM / 256), 1, S * DIM * 8);

    // attention_stream_qt: QT=8 queries per workgroup (slices stay
    // QT-aligned so tiles never straddle a slice boundary).
    const QT = 8;
    const perQuery = HEADS * 4 * HEAD_DIM * S;
    const maxQ = Math.max(64, Math.floor(this.ctx.budget / perQuery / QT) * QT);
    for (let q0 = 0; q0 < S; q0 += maxQ) {
      const n = Math.min(maxQ, S - q0);
      const u = this.uni([HEADS, HEAD_DIM, S, S, q0]);
      this.ctx.dispatch(this.pipes.attn, [
        { b: 0, buf: q, size: S * DIM * 4 },
        { b: 1, buf: this.bufs.kt, size: S * DIM * 4 },
        { b: 2, buf: this.bufs.v, size: S * DIM * 4 },
        { b: 3, buf: this.bufs.attn, size: S * DIM * 4 },
        { b: 4, buf: u.buf, off: u.off, size: 48 },
      ], Math.ceil(n / QT), HEADS, n * perQuery);
    }
  }

  private swiglu(rows: number) {
    const u = this.uni([rows, MLP_OUT, 0, 0]);
    this.ctx.dispatch(this.pipes.swiglu, [
      { b: 0, buf: this.bufs.mlp, size: rows * MLP_IN * 4 },
      { b: 2, buf: this.bufs.gate, size: rows * MLP_OUT * 4 },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(MLP_OUT / 256), rows, rows * MLP_IN * 4);
  }

  private siluVec(src: GPUBuffer, dst: GPUBuffer, n: number) {
    const u = this.uni([n, 0, ['f32', 0], 0, 0, 0]);
    this.ctx.dispatch(this.pipes.silu, [
      { b: 0, buf: src, size: n * 4 },
      { b: 1, buf: dst, size: n * 4 },
      { b: 2, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(n / 256), 1, n * 4);
  }

  private concatCols(src: GPUBuffer, srcW: number, dstColOff: number, rows: number) {
    const u = this.uni([rows, srcW, dstColOff, CAT_DIM]);
    this.ctx.dispatch(this.pipes.concat, [
      { b: 0, buf: src, size: rows * srcW * 4 },
      { b: 2, buf: this.bufs.cat, size: rows * CAT_DIM * 4 },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(srcW / 256), rows, rows * srcW * 4);
  }

  // ── blocks ───────────────────────────────────────────────────────────

  /** Double block; reads and (after two gated residuals) rewrites `cur`. */
  private doubleBlock(i: number, cur: GPUBuffer, nxt: GPUBuffer) {
    const p = `transformer_blocks.${i}.`;
    const b = this.bufs;
    const T = FLUX2_TXT_LEN, I = this.imgTokens;
    const imgOff = T * DIM * 4;

    // MSA: LN -> modulate (per stream) -> qkv (per stream) -> qk-norm -> rope -> SDPA
    this.ln(cur, 0, this.S, b.normedA, 0);
    this.modulate(b.normedA, 0, T, b.normedB, MOD_TXT + 0, MOD_TXT + DIM);
    this.modulate(b.normedA, imgOff, I, b.normedB, MOD_IMG + 0, MOD_IMG + DIM);
    this.gemm(b.normedB, 0, this.mat(p + 'attn.add_q_proj.weight'), 0, DIM, b.q, 0, T);
    this.gemm(b.normedB, 0, this.mat(p + 'attn.add_k_proj.weight'), 0, DIM, b.k, 0, T);
    this.gemm(b.normedB, 0, this.mat(p + 'attn.add_v_proj.weight'), 0, DIM, b.v, 0, T);
    this.gemm(b.normedB, imgOff, this.mat(p + 'attn.to_q.weight'), 0, DIM, b.q, imgOff, I);
    this.gemm(b.normedB, imgOff, this.mat(p + 'attn.to_k.weight'), 0, DIM, b.k, imgOff, I);
    this.gemm(b.normedB, imgOff, this.mat(p + 'attn.to_v.weight'), 0, DIM, b.v, imgOff, I);
    this.qkNorm(b.q, 0, T * HEADS, b.normedB, this.vec(p + 'attn.norm_added_q.weight'));
    this.qkNorm(b.q, imgOff, I * HEADS, b.normedB, this.vec(p + 'attn.norm_q.weight'));
    this.qkNorm(b.k, 0, T * HEADS, b.normedA, this.vec(p + 'attn.norm_added_k.weight'));
    this.qkNorm(b.k, imgOff, I * HEADS, b.normedA, this.vec(p + 'attn.norm_k.weight'));
    this.rope(b.normedB);
    this.rope(b.normedA);
    this.attention(b.normedB, b.normedA);
    this.gemm(b.attn, 0, this.mat(p + 'attn.to_add_out.weight'), 0, DIM, b.proj, 0, T);
    this.gemm(b.attn, imgOff, this.mat(p + 'attn.to_out.0.weight'), 0, DIM, b.proj, imgOff, I);
    this.gateAdd(cur, b.proj, 0, T, nxt, MOD_TXT + 2 * DIM);
    this.gateAdd(cur, b.proj, imgOff, I, nxt, MOD_IMG + 2 * DIM);

    // FFN: LN -> modulate (mlp set) -> SwiGLU -> gated residual
    this.ln(nxt, 0, this.S, b.normedA, 0);
    this.modulate(b.normedA, 0, T, b.normedB, MOD_TXT + 3 * DIM, MOD_TXT + 4 * DIM);
    this.modulate(b.normedA, imgOff, I, b.normedB, MOD_IMG + 3 * DIM, MOD_IMG + 4 * DIM);
    this.gemm(b.normedB, 0, this.mat(p + 'ff_context.linear_in.weight'), 0, MLP_IN, b.mlp, 0, T);
    this.gemm(b.normedB, imgOff, this.mat(p + 'ff.linear_in.weight'), 0, MLP_IN, b.mlp, T * MLP_IN * 4, I);
    this.swiglu(this.S);
    this.gemm(b.gate, 0, this.mat(p + 'ff_context.linear_out.weight'), 0, DIM, b.proj, 0, T);
    this.gemm(b.gate, T * MLP_OUT * 4, this.mat(p + 'ff.linear_out.weight'), 0, DIM, b.proj, imgOff, I);
    this.gateAdd(nxt, b.proj, 0, T, cur, MOD_TXT + 5 * DIM);
    this.gateAdd(nxt, b.proj, imgOff, I, cur, MOD_IMG + 5 * DIM);
  }

  /** Single block on the joint [txt|img] stream; cur -> nxt. */
  private singleBlock(i: number, cur: GPUBuffer, nxt: GPUBuffer) {
    const p = `single_transformer_blocks.${i}.`;
    const b = this.bufs;
    const S = this.S;
    const qkv = this.mat(p + 'attn.to_qkv_mlp_proj.weight');

    this.ln(cur, 0, S, b.normedA, 0);
    this.modulate(b.normedA, 0, S, b.normedB, MOD_SINGLE + 0, MOD_SINGLE + DIM);
    this.gemm(b.normedB, 0, qkv, 0, DIM, b.q, 0, S);
    this.gemm(b.normedB, 0, qkv, DIM, DIM, b.k, 0, S);
    this.gemm(b.normedB, 0, qkv, 2 * DIM, DIM, b.v, 0, S);
    this.gemm(b.normedB, 0, qkv, 3 * DIM, MLP_IN, b.mlp, 0, S);
    this.qkNorm(b.q, 0, S * HEADS, b.normedB, this.vec(p + 'attn.norm_q.weight'));
    this.qkNorm(b.k, 0, S * HEADS, b.normedA, this.vec(p + 'attn.norm_k.weight'));
    this.rope(b.normedB);
    this.rope(b.normedA);
    this.attention(b.normedB, b.normedA);
    this.swiglu(S);
    this.concatCols(b.attn, DIM, 0, S);
    this.concatCols(b.gate, MLP_OUT, DIM, S);
    this.gemm(b.cat, 0, this.mat(p + 'attn.to_out.weight'), 0, DIM, b.proj, 0, S);
    this.gateAdd(cur, b.proj, 0, S, nxt, MOD_SINGLE + 2 * DIM);
  }

  // ── one denoise step ─────────────────────────────────────────────────

  /** timestep1000 = the scheduler timestep (sigma*1000, f32). Returns
   *  noise_pred [imgTokens, 128] plus any requested capture readbacks. */
  async step(
    latents: Float32Array,
    timestep1000: number,
    capture?: Set<string>,
  ): Promise<{ noisePred: Float32Array; caps: Map<string, Float32Array> }> {
    if (!this.promptEmbedsSet) throw new Error('[Flux2 DiT] setPromptEmbeds not called');
    const I = this.imgTokens, T = FLUX2_TXT_LEN, S = this.S;
    if (latents.length !== I * 128) throw new Error(`[Flux2 DiT] latents ${latents.length} != ${I * 128}`);
    const b = this.bufs;
    const caps = new Map<string, Float32Array>();
    const cap = async (name: string, buf: GPUBuffer, off: number, elems: number) => {
      if (capture?.has(name)) caps.set(name, await this.ctx.read(buf, off, elems));
    };

    this.arena.reset();
    this.device.queue.writeBuffer(b.lat, 0, latents.buffer as ArrayBuffer, latents.byteOffset, latents.byteLength);

    // temb sinusoid. Replicates get_timestep_embedding's f32 op chain exactly
    // (args reach ~1000 rad, where f32-vs-f64 rounding of exp/mul already
    // shifts cos by ~1e-4 — enough to matter for the parity gate).
    const t = F32(F32(timestep1000 / 1000) * 1000);
    const sinu = new Float32Array(256);
    for (let k = 0; k < 128; k++) {
      const expo = F32(F32(-Math.log(10000) * k) / 128);
      const arg = F32(t * F32(Math.exp(expo)));
      sinu[k] = Math.cos(arg);       // flip_sin_to_cos -> [cos | sin]
      sinu[128 + k] = Math.sin(arg);
    }
    this.device.queue.writeBuffer(b.sinusoid, 0, sinu);

    const te = 'time_guidance_embed.timestep_embedder.';
    this.gemm(b.sinusoid, 0, this.mat(te + 'linear_1.weight'), 0, DIM, b.t1, 0, 1);
    this.siluVec(b.t1, b.t2, DIM);
    this.gemm(b.t2, 0, this.mat(te + 'linear_2.weight'), 0, DIM, b.temb, 0, 1);
    this.siluVec(b.temb, b.tembSilu, DIM);
    this.gemm(b.tembSilu, 0, this.mat('double_stream_modulation_img.linear.weight'), 0, 18432, b.mod, MOD_IMG * 4, 1);
    this.gemm(b.tembSilu, 0, this.mat('double_stream_modulation_txt.linear.weight'), 0, 18432, b.mod, MOD_TXT * 4, 1);
    this.gemm(b.tembSilu, 0, this.mat('single_stream_modulation.linear.weight'), 0, 9216, b.mod, MOD_SINGLE * 4, 1);
    this.gemm(b.tembSilu, 0, this.mat('norm_out.linear.weight'), 0, 6144, b.mod, MOD_NORM_OUT * 4, 1);
    await cap('temb', b.temb, 0, DIM);
    await cap('mod_double_img', b.mod, MOD_IMG * 4, 18432);
    await cap('mod_double_txt', b.mod, MOD_TXT * 4, 18432);
    await cap('mod_single', b.mod, MOD_SINGLE * 4, 9216);

    // embedders -> joint stream [txt | img]
    const imgOff = T * DIM * 4;
    this.gemm(b.pe, 0, this.mat('context_embedder.weight'), 0, DIM, b.streamA, 0, T);
    this.gemm(b.lat, 0, this.mat('x_embedder.weight'), 0, DIM, b.streamA, imgOff, I);
    await cap('context_embedder', b.streamA, 0, T * DIM);
    await cap('x_embedder', b.streamA, imgOff, I * DIM);

    let cur = b.streamA, nxt = b.streamB;
    for (let i = 0; i < 5; i++) {
      this.doubleBlock(i, cur, nxt); // net result back in cur
      await cap(`double${i}.txt`, cur, 0, T * DIM);
      await cap(`double${i}.img`, cur, imgOff, I * DIM);
    }
    for (let i = 0; i < 20; i++) {
      this.singleBlock(i, cur, nxt);
      [cur, nxt] = [nxt, cur];
      await cap(`single${i}`, cur, 0, S * DIM);
    }

    // norm_out (AdaLayerNormContinuous: chunk order (scale, shift)) + proj_out
    this.ln(cur, imgOff, I, b.normedA, imgOff);
    this.modulate(b.normedA, imgOff, I, b.normedB, MOD_NORM_OUT + DIM, MOD_NORM_OUT + 0);
    await cap('norm_out', b.normedB, imgOff, I * DIM);
    this.gemm(b.normedB, imgOff, this.mat('proj_out.weight'), 0, 128, b.out, 0, I);

    const noisePred = await this.ctx.read(b.out, 0, I * 128);
    return { noisePred, caps };
  }

  private releaseActivations() {
    for (const k of Object.keys(this.bufs)) this.bufs[k].destroy();
    this.bufs = {};
  }

  destroy() {
    this.releaseActivations();
    this.arena.destroy();
  }
}
