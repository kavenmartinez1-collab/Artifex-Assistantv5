// FLUX.2-klein VAE decoder on WebGPU (AutoencoderKLFlux2, f32 / force_upcast).
//
// Replicates diffusers 0.39 vae.py Decoder + pipeline latent un-packing:
//   CPU: packed (tokens,128) -> grid (128,g,g) -> bn de-normalize with
//        BatchNorm2d RUNNING stats (x*sqrt(var+1e-4)+mean, affine=False)
//        -> 2x2 unpatchify -> (32, 2g, 2g)
//   GPU: post_quant_conv 1x1 -> conv_in 3x3 (32->512)
//        -> mid: ResnetBlock2D, single-head Attention (dim 512, GN(32),
//           q/k/v/out biased, scale 512^-0.5, residual), ResnetBlock2D
//        -> 4 UpDecoderBlock2D (3 resnets; nearest-2x upsample + conv3x3 on
//           blocks 0..2; channels 512,512,256,128)
//        -> GroupNorm + SiLU + conv_out 3x3 (128 -> 3 RGB, [-1,1])
//   ResnetBlock2D: GN -> SiLU -> conv1 -> GN -> SiLU -> conv2, 1x1
//   conv_shortcut on channel change, output_scale_factor 1, eps 1e-6.
//
// All weights bf16 -> f32 on CPU at load (bf16 exact in f32: both sides then
// do f32 math on identical weights). TDR budgeting mirrors the DiT: shared
// cost-weighted flush; conv3x3 sliced by output-channel bands (64-aligned so
// weight/bias/out byte offsets stay 256-aligned), attention query-sliced.
//
// Device must be created with maxBufferSize/maxStorageBufferBindingSize
// raised (scratch reaches 256ch * px^2 * 4 = 268 MB at 512px).

import conv2dWGSL from '../shaders/conv2d.wgsl?raw';
import groupNormNchwWGSL from '../shaders/group_norm_nchw.wgsl?raw';
import elementwiseWGSL from '../shaders/elementwise.wgsl?raw';
import attnStreamWGSL from '../shaders/attention_stream.wgsl?raw';
import { parseHeader, parseHeaderLength } from '../model/safetensors';
import { fetchRange } from '../model/hf-hub';
import { UniArena, Ctx, type Bind } from './flux2-transformer';

const GN_EPS = 1e-6;
export const BN_EPS = 1e-4;
const F32 = Math.fround;

// ── weights ──────────────────────────────────────────────────────────────

export interface Flux2VaeWeights {
  bufs: Map<string, GPUBuffer>; // f32; decoder.*+post_quant_conv.* or encoder.*+quant_conv.*
  bnMean: Float32Array;         // (128,) CPU — latent (de-)normalization
  bnVar: Float32Array;
  totalBytes: number;
  destroy(): void;
}

function bf16ToF32(raw: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(raw);
  const out = new Float32Array(u16.length);
  const view = new DataView(out.buffer);
  for (let i = 0; i < u16.length; i++) view.setUint32(i * 4, u16[i] << 16, true);
  return out;
}

/** Loads one VAE half + bn stats: part 'decoder' = decoder.* +
 *  post_quant_conv.* (t2i and the edit finish), part 'encoder' = encoder.* +
 *  quant_conv.* (edit reference conditioning). */
export async function loadFlux2Vae(
  device: GPUDevice,
  url: string,
  onProgress?: (loadedBytes: number, totalBytes: number, name: string) => void,
  part: 'decoder' | 'encoder' = 'decoder',
): Promise<Flux2VaeWeights> {
  const first8 = await fetchRange(url, 0, 8);
  if (first8.byteLength !== 8) throw new Error(`[Flux2 VAE] header-length read: ${first8.byteLength}`);
  const headerLen = parseHeaderLength(first8);
  const headerBytes = await fetchRange(url, 0, 8 + headerLen);
  if (headerBytes.byteLength !== 8 + headerLen) {
    throw new Error(`[Flux2 VAE] header read: ${headerBytes.byteLength} vs ${8 + headerLen}`);
  }
  const header = parseHeader(headerBytes);
  const dataStart = header.headerByteLength;

  const want = (name: string) =>
    (part === 'decoder'
      ? name.startsWith('decoder.') || name.startsWith('post_quant_conv.')
      : name.startsWith('encoder.') || name.startsWith('quant_conv.')) ||
    name === 'bn.running_mean' || name === 'bn.running_var';

  let totalBytes = 0;
  for (const [name, t] of header.tensors) if (want(name)) totalBytes += t.byteLength;

  const bufs = new Map<string, GPUBuffer>();
  let bnMean: Float32Array | null = null;
  let bnVar: Float32Array | null = null;
  let loaded = 0;

  for (const [name, t] of header.tensors) {
    if (!want(name)) continue;
    if (t.dtype !== 'BF16') throw new Error(`[Flux2 VAE] ${name}: unexpected dtype ${t.dtype}`);
    const raw = await fetchRange(url, dataStart + t.dataOffsets[0], dataStart + t.dataOffsets[1]);
    if (raw.byteLength !== t.byteLength) {
      throw new Error(`[Flux2 VAE] ${name}: fetched ${raw.byteLength} bytes, want ${t.byteLength}`);
    }
    const f = bf16ToF32(raw);
    if (name === 'bn.running_mean') { bnMean = f; }
    else if (name === 'bn.running_var') { bnVar = f; }
    else {
      const buffer = device.createBuffer({
        size: f.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, f.buffer as ArrayBuffer, f.byteOffset, f.byteLength);
      bufs.set(name, buffer);
    }
    loaded += t.byteLength;
    onProgress?.(loaded, totalBytes, name);
  }
  if (!bnMean || !bnVar) throw new Error('[Flux2 VAE] bn running stats missing');

  return {
    bufs, bnMean, bnVar, totalBytes,
    destroy() { for (const b of bufs.values()) b.destroy(); bufs.clear(); },
  };
}

// ── CPU latent un-packing (pipeline __call__ + unpatchify_latents) ───────

/** packed (tokens,128) row-major -> bn de-normalized, 2x2-unpatchified
 *  (32, 2*gridH, 2*gridW). Packed channel = c4*4 + py*2 + px. */
export function unpackLatents(
  packed: Float32Array, gridH: number, gridW: number,
  bnMean: Float32Array, bnVar: Float32Array,
): Float32Array {
  const H = 2 * gridH, W = 2 * gridW;
  const out = new Float32Array(32 * H * W);
  const std = new Float32Array(128);
  for (let c = 0; c < 128; c++) std[c] = F32(Math.sqrt(F32(bnVar[c] + BN_EPS)));
  for (let c4 = 0; c4 < 32; c4++) {
    for (let py = 0; py < 2; py++) {
      for (let px = 0; px < 2; px++) {
        const ch = c4 * 4 + py * 2 + px;
        for (let y = 0; y < gridH; y++) {
          for (let x = 0; x < gridW; x++) {
            const v = F32(F32(packed[(y * gridW + x) * 128 + ch] * std[ch]) + bnMean[ch]);
            out[c4 * H * W + (2 * y + py) * W + 2 * x + px] = v;
          }
        }
      }
    }
  }
  return out;
}

/** Inverse companion of unpackLatents for the edit path
 *  (_encode_vae_image): latent mode (32, l, l) -> 2x2 patchify -> bn
 *  NORMALIZE ((x-mean)/sqrt(var+eps)) -> packed (l/2*l/2, 128) row-major.
 *  Packed channel = c4*4 + py*2 + px. */
export function packRefLatents(
  mode: Float32Array, l: number,
  bnMean: Float32Array, bnVar: Float32Array,
): Float32Array {
  if (mode.length !== 32 * l * l) throw new Error(`[Flux2 VAE] mode ${mode.length} != ${32 * l * l}`);
  const g = l / 2;
  const out = new Float32Array(g * g * 128);
  const std = new Float32Array(128);
  for (let c = 0; c < 128; c++) std[c] = F32(Math.sqrt(F32(bnVar[c] + BN_EPS)));
  for (let c4 = 0; c4 < 32; c4++) {
    for (let py = 0; py < 2; py++) {
      for (let px = 0; px < 2; px++) {
        const ch = c4 * 4 + py * 2 + px;
        for (let y = 0; y < g; y++) {
          for (let x = 0; x < g; x++) {
            const v = F32(F32(mode[c4 * l * l + (2 * y + py) * l + 2 * x + px] - bnMean[ch]) / std[ch]);
            out[(y * g + x) * 128 + ch] = v;
          }
        }
      }
    }
  }
  return out;
}

// ── decoder + encoder ────────────────────────────────────────────────────

interface UpBlock { cin: number; cout: number; upsample: boolean }
const UP_BLOCKS: UpBlock[] = [
  { cin: 512, cout: 512, upsample: true },
  { cin: 512, cout: 512, upsample: true },
  { cin: 512, cout: 256, upsample: true },
  { cin: 256, cout: 128, upsample: false },
];

// Encoder DownEncoderBlock2D chain (2 resnets each; stride-2 downsampler with
// asymmetric (0,1,0,1) pad after blocks 0..2). block_out_channels (128,256,512,512).
interface DownBlock { cin: number; cout: number; down: boolean }
const DOWN_BLOCKS: DownBlock[] = [
  { cin: 128, cout: 128, down: true },
  { cin: 128, cout: 256, down: true },
  { cin: 256, cout: 512, down: true },
  { cin: 512, cout: 512, down: false },
];

export class Flux2VaeDecoder {
  private pipes!: Record<string, GPUComputePipeline>;
  private arena: UniArena;
  private ctx: Ctx;
  private bufs: Record<string, GPUBuffer> = {};
  private latSize = 0; // l (latent H = W); buffers sized for it

  constructor(
    private device: GPUDevice,
    private weights: Flux2VaeWeights,
    flopBudget = 0.35e12,
  ) {
    this.arena = new UniArena(device);
    this.ctx = new Ctx(device, flopBudget);
    const mk = (code: string, entry: string) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: entry },
    });
    this.pipes = {
      conv3: mk(conv2dWGSL, 'conv2d_3x3'),
      convS2: mk(conv2dWGSL, 'conv2d_3x3_s2'),
      conv1: mk(conv2dWGSL, 'conv2d_1x1'),
      upsample: mk(conv2dWGSL, 'upsample_nearest_2x'),
      gn: mk(groupNormNchwWGSL, 'group_norm_nchw'),
      silu: mk(elementwiseWGSL, 'silu'),
      add: mk(elementwiseWGSL, 'add'),
      attn: mk(attnStreamWGSL, 'attention_stream'),
      transpose: mk(attnStreamWGSL, 'transpose_khds'),
    };
  }

  private w(name: string): GPUBuffer {
    const b = this.weights.bufs.get(name);
    if (!b) throw new Error(`[Flux2 VAE] missing weight ${name}`);
    return b;
  }

  private uni(words: (number | ['f32', number])[]): { buf: GPUBuffer; off: number } {
    const bytes = new ArrayBuffer(Math.max(16, words.length * 4));
    const dv = new DataView(bytes);
    words.forEach((v, i) => {
      if (Array.isArray(v)) dv.setFloat32(i * 4, v[1], true);
      else dv.setUint32(i * 4, v >>> 0, true);
    });
    return this.arena.alloc(bytes);
  }

  /** Allocate the scratch set for latent size l (idempotent per size). */
  private setup(l: number) {
    if (this.latSize === l) return;
    for (const b of Object.values(this.bufs)) b.destroy();
    this.bufs = {};
    const d = this.device;
    const px = 8 * l;
    const peak = 256 * px * px; // up_blocks[2] upsample output (largest tensor)
    const alloc = (elems: number) => d.createBuffer({
      size: elems * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    this.bufs.s0 = alloc(peak);
    this.bufs.s1 = alloc(peak);
    this.bufs.a = alloc(peak);
    this.bufs.b = alloc(peak);
    this.bufs.c = alloc(peak);
    // mid attention scratch, [hw, 512] / [512, hw] at latent resolution
    const attnElems = 512 * l * l;
    this.bufs.q = alloc(attnElems);
    this.bufs.kt = alloc(attnElems);
    this.bufs.v = alloc(attnElems);
    this.bufs.ao = alloc(attnElems);
    this.latSize = l;
  }

  // ── dispatch helpers (all NCHW batch-1, dims explicit) ─────────────────

  /** conv2d_3x3/3x3_s2/1x1, output-channel-band sliced to the flop budget.
   *  Band starts are 64-multiples: weight off = oc0*cin*k2*4 (k2*4 in
   *  {36,4}, 64*36=2304 and 64*4=256 are 256-multiples), bias off = oc0*4,
   *  out off = oc0*outHw*4 — all storage-offset aligned. h/w are INPUT dims;
   *  convS2 halves them at the output. */
  private conv(
    entry: 'conv3' | 'convS2' | 'conv1', name: string,
    src: GPUBuffer, dst: GPUBuffer, cin: number, cout: number, h: number, w: number,
  ) {
    const outHw = entry === 'convS2' ? (h / 2) * (w / 2) : h * w;
    const k2 = entry === 'conv1' ? 1 : 9;
    const flopPerC = 2 * k2 * cin * outHw;
    let band = cout;
    if (flopPerC * cout > this.ctx.budget) {
      band = Math.max(64, Math.floor(this.ctx.budget / flopPerC / 64) * 64);
    }
    const wt = this.w(`${name}.weight`), bias = this.w(`${name}.bias`);
    for (let oc0 = 0; oc0 < cout; oc0 += band) {
      const bc = Math.min(band, cout - oc0);
      const u = this.uni([cin, bc, h, w]);
      this.ctx.dispatch(this.pipes[entry], [
        { b: 0, buf: src },
        { b: 1, buf: wt, off: oc0 * cin * k2 * 4, size: bc * cin * k2 * 4 },
        { b: 2, buf: bias, off: oc0 * 4, size: bc * 4 },
        { b: 3, buf: dst, off: oc0 * outHw * 4, size: bc * outHw * 4 },
        { b: 4, buf: u.buf, off: u.off, size: 48 },
      ], Math.ceil(outHw / 256), bc, flopPerC * bc);
    }
  }

  private gn(name: string, src: GPUBuffer, dst: GPUBuffer, c: number, hw: number) {
    const u = this.uni([c, 32, hw, ['f32', GN_EPS]]);
    this.ctx.dispatch(this.pipes.gn, [
      { b: 0, buf: src, size: c * hw * 4 },
      { b: 1, buf: dst, size: c * hw * 4 },
      { b: 2, buf: this.w(`${name}.weight`) },
      { b: 3, buf: this.w(`${name}.bias`) },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], 32, 1, 10 * c * hw);
  }

  // elementwise.wgsl indexes with gid.x only, and 512px activations (up to
  // 256*512*512 = 67M elements) exceed the 65535 workgroups-per-dimension
  // limit; chunk via 1024-aligned buffer offsets (65535*256 elems per slice).
  private static readonly EW_CHUNK = 65535 * 256;

  private silu(src: GPUBuffer, dst: GPUBuffer, n: number) {
    for (let i = 0; i < n; i += Flux2VaeDecoder.EW_CHUNK) {
      const c = Math.min(Flux2VaeDecoder.EW_CHUNK, n - i);
      const u = this.uni([c, 0, ['f32', 0], 0, 0, 0]);
      this.ctx.dispatch(this.pipes.silu, [
        { b: 0, buf: src, off: i * 4, size: c * 4 },
        { b: 1, buf: dst, off: i * 4, size: c * 4 },
        { b: 2, buf: u.buf, off: u.off, size: 48 },
      ], Math.ceil(c / 256), 1, 4 * c);
    }
  }

  private add(a: GPUBuffer, b: GPUBuffer, dst: GPUBuffer, n: number) {
    for (let i = 0; i < n; i += Flux2VaeDecoder.EW_CHUNK) {
      const c = Math.min(Flux2VaeDecoder.EW_CHUNK, n - i);
      const u = this.uni([c, 0, ['f32', 0], 0, 0, 0]);
      this.ctx.dispatch(this.pipes.add, [
        { b: 0, buf: a, off: i * 4, size: c * 4 },
        { b: 3, buf: b, off: i * 4, size: c * 4 },
        { b: 1, buf: dst, off: i * 4, size: c * 4 },
        { b: 2, buf: u.buf, off: u.off, size: 48 },
      ], Math.ceil(c / 256), 1, c);
    }
  }

  private upsample(src: GPUBuffer, dst: GPUBuffer, c: number, h: number, w: number) {
    const u = this.uni([c, 0, h, w]);
    this.ctx.dispatch(this.pipes.upsample, [
      { b: 0, buf: src, size: c * h * w * 4 },
      // Wt@1 / Bias@2 are unreferenced by this entry -> dropped from the
      // 'auto' layout; binding them would be a validation error.
      { b: 3, buf: dst, size: c * 4 * h * w * 4 },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(4 * h * w / 256), c, 4 * c * h * w);
  }

  /** [rows, cols] row-major -> [cols, rows] via transpose_khds (H=1). */
  private transpose(src: GPUBuffer, dst: GPUBuffer, rows: number, cols: number) {
    const u = this.uni([1, cols, 0, rows, 0]); // heads=1, head_dim=cols, seq_kv=rows
    this.ctx.dispatch(this.pipes.transpose, [
      { b: 0, buf: src, size: rows * cols * 4 },
      { b: 3, buf: dst, size: rows * cols * 4 },
      { b: 4, buf: u.buf, off: u.off, size: 48 },
    ], Math.ceil(rows * cols / 256), 1, rows * cols);
  }

  /** ResnetBlock2D: src -> dst (distinct buffers). */
  private resnet(name: string, src: GPUBuffer, dst: GPUBuffer, cin: number, cout: number, h: number, w: number) {
    const { a, b, c } = this.bufs;
    const hw = h * w;
    this.gn(`${name}.norm1`, src, a, cin, hw);
    this.silu(a, b, cin * hw);
    this.conv('conv3', `${name}.conv1`, b, a, cin, cout, h, w);
    this.gn(`${name}.norm2`, a, b, cout, hw);
    this.silu(b, a, cout * hw);
    this.conv('conv3', `${name}.conv2`, a, b, cout, cout, h, w);
    if (cin !== cout) {
      this.conv('conv1', `${name}.conv_shortcut`, src, c, cin, cout, h, w);
      this.add(b, c, dst, cout * hw);
    } else {
      this.add(b, src, dst, cout * hw);
    }
  }

  /** Mid-block Attention (heads=1, dim 512, residual): src -> dst. */
  private attnBlock(src: GPUBuffer, dst: GPUBuffer, hw: number,
                    p = 'decoder.mid_block.attentions.0') {
    const { a, b, q, kt, v, ao } = this.bufs;
    this.gn(`${p}.group_norm`, src, a, 512, hw);
    // to_q/k/v as 1x1 convs over the [512, hw] grid; K lands directly in the
    // [D, S] transposed layout attention_stream wants, q/v get transposed to
    // token-major [hw, 512].
    this.conv('conv1', `${p}.to_q`, a, b, 512, 512, hw, 1);
    this.transpose(b, q, 512, hw);
    this.conv('conv1', `${p}.to_v`, a, b, 512, 512, hw, 1);
    this.transpose(b, v, 512, hw);
    this.conv('conv1', `${p}.to_k`, a, kt, 512, 512, hw, 1);
    // query-sliced SDPA
    const flopPerQ = 2 * hw * 512 * 2;
    const qSlice = Math.max(1, Math.min(hw, Math.floor(this.ctx.budget / flopPerQ)));
    for (let q0 = 0; q0 < hw; q0 += qSlice) {
      const n = Math.min(qSlice, hw - q0);
      const u = this.uni([1, 512, hw, hw, q0]);
      this.ctx.dispatch(this.pipes.attn, [
        { b: 0, buf: q, size: hw * 512 * 4 },
        { b: 1, buf: kt, size: hw * 512 * 4 },
        { b: 2, buf: v, size: hw * 512 * 4 },
        { b: 3, buf: ao, size: hw * 512 * 4 },
        { b: 4, buf: u.buf, off: u.off, size: 48 },
      ], n, 1, flopPerQ * n);
    }
    this.transpose(ao, b, hw, 512);
    this.conv('conv1', `${p}.to_out.0`, b, a, 512, 512, hw, 1);
    this.add(a, src, dst, 512 * hw);
  }

  /** Standalone mid-attention hook for the parity gate: x is the Attention
   *  module input (512, l, l) NCHW; returns the module output. */
  async midAttn(x: Float32Array, l: number): Promise<Float32Array> {
    this.setup(l);
    this.arena.reset();
    const hw = l * l;
    if (x.length !== 512 * hw) throw new Error(`[Flux2 VAE] midAttn input ${x.length} != ${512 * hw}`);
    this.device.queue.writeBuffer(this.bufs.s0, 0, x.buffer as ArrayBuffer, x.byteOffset, x.byteLength);
    this.attnBlock(this.bufs.s0, this.bufs.s1, hw);
    return this.ctx.read(this.bufs.s1, 0, 512 * hw);
  }

  /** Decode unpatchified latents (32, l, l) -> pixels (3, 8l, 8l) in [-1,1]. */
  async decode(latents: Float32Array, l: number, onProgress?: (stage: string) => void): Promise<Float32Array> {
    this.setup(l);
    this.arena.reset();
    if (latents.length !== 32 * l * l) {
      throw new Error(`[Flux2 VAE] latents ${latents.length} != ${32 * l * l}`);
    }
    const d = this.device;
    let cur = this.bufs.s0, nxt = this.bufs.s1;
    const swap = () => { const t = cur; cur = nxt; nxt = t; };
    d.queue.writeBuffer(cur, 0, latents.buffer as ArrayBuffer, latents.byteOffset, latents.byteLength);

    onProgress?.('post_quant_conv');
    this.conv('conv1', 'post_quant_conv', cur, nxt, 32, 32, l, l); swap();
    this.conv('conv3', 'decoder.conv_in', cur, nxt, 32, 512, l, l); swap();

    onProgress?.('mid_block');
    this.resnet('decoder.mid_block.resnets.0', cur, nxt, 512, 512, l, l); swap();
    this.attnBlock(cur, nxt, l * l); swap();
    this.resnet('decoder.mid_block.resnets.1', cur, nxt, 512, 512, l, l); swap();

    let h = l, w = l;
    for (let i = 0; i < 4; i++) {
      onProgress?.(`up_block ${i}`);
      const blk = UP_BLOCKS[i];
      for (let r = 0; r < 3; r++) {
        const cin = r === 0 ? blk.cin : blk.cout;
        this.resnet(`decoder.up_blocks.${i}.resnets.${r}`, cur, nxt, cin, blk.cout, h, w);
        swap();
      }
      if (blk.upsample) {
        this.upsample(cur, this.bufs.a, blk.cout, h, w);
        h *= 2; w *= 2;
        this.conv('conv3', `decoder.up_blocks.${i}.upsamplers.0.conv`, this.bufs.a, cur, blk.cout, blk.cout, h, w);
      }
    }

    onProgress?.('conv_out');
    this.gn('decoder.conv_norm_out', cur, this.bufs.a, 128, h * w);
    this.silu(this.bufs.a, this.bufs.b, 128 * h * w);
    this.conv('conv3', 'decoder.conv_out', this.bufs.b, nxt, 128, 3, h, w);
    return this.ctx.read(nxt, 0, 3 * h * w);
  }

  /** Encode a preprocessed image (3, px, px) in [-1,1] -> latent mode
   *  (32, l, l), l = px/8. Requires the 'encoder' weight part. The posterior
   *  mode/mean = first 32 of the 64 moment channels (sample_mode="argmax" in
   *  the pipeline — deterministic, no RNG). Square-only in v1. */
  async encode(image: Float32Array, px: number, onProgress?: (stage: string) => void): Promise<Float32Array> {
    if (px % 16 !== 0) throw new Error(`[Flux2 VAE] encode px ${px} not a multiple of 16`);
    if (image.length !== 3 * px * px) {
      throw new Error(`[Flux2 VAE] image ${image.length} != ${3 * px * px}`);
    }
    const l = px / 8;
    this.setup(l); // peak 256*(8l)^2 covers the encoder max 128*px^2
    this.arena.reset();
    let cur = this.bufs.s0, nxt = this.bufs.s1;
    const swap = () => { const t = cur; cur = nxt; nxt = t; };
    this.device.queue.writeBuffer(cur, 0, image.buffer as ArrayBuffer, image.byteOffset, image.byteLength);

    onProgress?.('conv_in');
    this.conv('conv3', 'encoder.conv_in', cur, nxt, 3, 128, px, px); swap();

    let h = px, w = px;
    for (let i = 0; i < 4; i++) {
      onProgress?.(`down_block ${i}`);
      const blk = DOWN_BLOCKS[i];
      for (let r = 0; r < 2; r++) {
        const cin = r === 0 ? blk.cin : blk.cout;
        this.resnet(`encoder.down_blocks.${i}.resnets.${r}`, cur, nxt, cin, blk.cout, h, w);
        swap();
      }
      if (blk.down) {
        this.conv('convS2', `encoder.down_blocks.${i}.downsamplers.0.conv`, cur, nxt, blk.cout, blk.cout, h, w);
        swap();
        h /= 2; w /= 2;
      }
    }

    onProgress?.('mid_block'); // h = w = l here
    this.resnet('encoder.mid_block.resnets.0', cur, nxt, 512, 512, h, w); swap();
    this.attnBlock(cur, nxt, h * w, 'encoder.mid_block.attentions.0'); swap();
    this.resnet('encoder.mid_block.resnets.1', cur, nxt, 512, 512, h, w); swap();

    onProgress?.('conv_out');
    this.gn('encoder.conv_norm_out', cur, this.bufs.a, 512, h * w);
    this.silu(this.bufs.a, this.bufs.b, 512 * h * w);
    this.conv('conv3', 'encoder.conv_out', this.bufs.b, cur, 512, 64, h, w);
    this.conv('conv1', 'quant_conv', cur, nxt, 64, 64, h, w);
    // mode = mean channels: first 32 of the 64-channel moments (NCHW)
    return this.ctx.read(nxt, 0, 32 * l * l);
  }

  destroy() {
    for (const b of Object.values(this.bufs)) b.destroy();
    this.bufs = {};
    this.latSize = 0;
    this.arena.destroy();
  }
}
