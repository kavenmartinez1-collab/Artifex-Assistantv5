// FLUX.2-klein text-to-image runtime: the full browser pipeline behind the
// image-gen UI. v2 (Phase 7) is ALL-RESIDENT on the 12 GB target — the first
// generation loads everything once, every later prompt starts denoising
// immediately:
//
//   resident: TE  Qwen3-4B Q4_K_M GGUF          ~2.4 GB
//             DiT Q8_0 .artq + dequant scratch  ~4.2 GB (per-GEMM GPU
//                 dequant → bf16 scratch ring feeding the bf16 GEMMs)
//             VAE decoder f32                   ~0.34 GB
//             ───────────────────────────────── ~7 GB total
//   transient: VAE ENCODER f32 0.34 GB per edit — ref image → packed
//              conditioning tokens (T=10 ids) → freed. noise_pred is
//              sliced back to the gen tokens.
//
// Every stage is the exact code path proven by the parity gates (Phase 2/3/4
// f32/bf16; P7.2 Q8 DiT; P7.3 Q4 TE); only the seeded noise (rng.ts) is new —
// parity runs inject torch noise. releaseFlux2Resident() frees the ~7 GB on
// unload/model-switch.

import { resolveFileUrl } from '../model/hf-hub';
import { loadFlux2TextEncoder, type Flux2TextEncoder } from './flux2-te';
import { embedFlux2Prompt } from './text-embedder';
import { loadFlux2DitQ8, type Flux2DitWeightsQ8 } from './flux2-loader';
import { Flux2Pipeline, type Flux2RefGrid } from './flux2-pipeline';
import {
  loadFlux2Vae, unpackLatents, packRefLatents, Flux2VaeDecoder,
  type Flux2VaeWeights,
} from './vae';
import { randn } from './rng';

export interface Flux2ImageOptions {
  prompt: string;
  px: number;              // output edge in pixels (multiple of 16)
  seed: number;
  numSteps?: number;       // klein default 4
  /** Edit mode: reference image (3, refPx, refPx) NCHW in [-1,1] (use
   *  preprocessRefImage). VAE-encoded to constant conditioning tokens with
   *  T=10 position ids; the denoise loop is otherwise unchanged. */
  refImage?: { data: Float32Array; px: number };
  /** stage: 'te' | 'dit' | 'vae'; detail is a human-readable progress line. */
  onProgress?: (stage: string, detail: string) => void;
}

export interface Flux2ImageResult {
  rgba: Uint8ClampedArray<ArrayBuffer>; // [px*px*4] straight-alpha, ready for ImageData
  width: number;
  height: number;
  timings: { teMs: number; ditMs: number; vaeMs: number; encMs: number };
}

/** Decode + cover-crop a reference image to (3, px, px) NCHW in [-1,1].
 *  v1 is square-only: the short side fills px, the rest center-crops
 *  (pipeline resize_mode="crop"; canvas resampling stands in for LANCZOS). */
export async function preprocessRefImage(blob: Blob, px: number): Promise<Float32Array> {
  const bmp = await createImageBitmap(blob);
  const canvas = new OffscreenCanvas(px, px);
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  const s = Math.max(px / bmp.width, px / bmp.height);
  const dw = bmp.width * s, dh = bmp.height * s;
  ctx.drawImage(bmp, (px - dw) / 2, (px - dh) / 2, dw, dh);
  bmp.close();
  const { data } = ctx.getImageData(0, 0, px, px);
  const hw = px * px;
  const out = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    out[i] = data[i * 4] / 127.5 - 1;
    out[hw + i] = data[i * 4 + 1] / 127.5 - 1;
    out[2 * hw + i] = data[i * 4 + 2] / 127.5 - 1;
  }
  return out;
}

export const FLUX2_REPO = 'local/flux.2-klein-4b';
export const FLUX2_TE_REPO = 'local/flux2-te-qwen3-4b-q4_k_m';
export const FLUX2_TE_GGUF = 'flux2-te-qwen3-4b-q4_k_m.gguf';
const FLUX2_DIT_ARTQ = 'transformer/diffusion_pytorch_model.q8_0.artq';

// TE forward is cheap once resident, but the embed cache still skips ~2 s
// on the dominant "reroll the seed" flow.
let cachedPrompt: string | null = null;
let cachedEmbeds: Float32Array | null = null;

interface Flux2Resident {
  device: GPUDevice;
  te: Flux2TextEncoder;
  ditWeights: Flux2DitWeightsQ8;
  pipeline: Flux2Pipeline;
  vae: Flux2VaeWeights; // decoder half
}
let resident: Flux2Resident | null = null;

/** Frees the ~7 GB resident set (TE+DiT+VAE). Call on unload/model-switch. */
export function releaseFlux2Resident(): void {
  if (!resident) return;
  resident.pipeline.destroy();
  resident.ditWeights.destroy();
  resident.te.destroy();
  resident.vae.destroy();
  resident = null;
  cachedPrompt = null;
  cachedEmbeds = null;
}

async function acquireResident(
  device: GPUDevice,
  progress: (stage: string, detail: string) => void,
): Promise<Flux2Resident> {
  if (resident?.device === device) return resident;
  releaseFlux2Resident(); // stale device (context loss / re-init)
  progress('load', 'loading text encoder (Qwen3-4B Q4_K_M, resident)...');
  const te = await loadFlux2TextEncoder(device, FLUX2_TE_REPO, FLUX2_TE_GGUF,
    (m, f) => progress('load', f !== undefined ? `${m} (${Math.round(f * 100)}%)` : m));
  const ditUrl = resolveFileUrl(FLUX2_REPO, FLUX2_DIT_ARTQ);
  const ditWeights = await loadFlux2DitQ8(device, ditUrl, (loaded, total) =>
    progress('load', `loading DiT Q8 weights (resident)... ${Math.round((loaded / total) * 100)}%`));
  const pipeline = new Flux2Pipeline(device, ditWeights);
  progress('load', 'loading VAE decoder (resident)...');
  const vaeUrl = resolveFileUrl(FLUX2_REPO, 'vae/diffusion_pytorch_model.safetensors');
  const vae = await loadFlux2Vae(device, vaeUrl);
  resident = { device, te, ditWeights, pipeline, vae };
  return resident;
}

export async function generateFlux2Image(
  device: GPUDevice,
  opts: Flux2ImageOptions,
): Promise<Flux2ImageResult> {
  const { prompt, px, seed } = opts;
  if (px % 16 !== 0) throw new Error(`[Flux2] px must be a multiple of 16, got ${px}`);
  const grid = px / 16; // latent token grid edge
  const progress = opts.onProgress ?? (() => {});
  const vaeUrl = resolveFileUrl(FLUX2_REPO, 'vae/diffusion_pytorch_model.safetensors');
  const r = await acquireResident(device, progress);

  // ── 0) Edit reference: VAE-encode + pack (the encoder half stays
  //       transient — loaded per edit, freed right after) ────────────────
  let refLatents: Float32Array | undefined;
  let refs: Flux2RefGrid[] | undefined;
  let encMs = 0;
  if (opts.refImage) {
    const tEnc = performance.now();
    progress('vae', 'loading VAE encoder (edit reference)...');
    const enc = await loadFlux2Vae(device, vaeUrl, undefined, 'encoder');
    try {
      const dec = new Flux2VaeDecoder(device, enc);
      try {
        progress('vae', 'encoding reference image...');
        const mode = await dec.encode(opts.refImage.data, opts.refImage.px);
        refLatents = packRefLatents(mode, opts.refImage.px / 8, enc.bnMean, enc.bnVar);
      } finally {
        dec.destroy();
      }
    } finally {
      enc.destroy();
    }
    const rg = opts.refImage.px / 16;
    refs = [{ h: rg, w: rg, t: 10 }]; // _prepare_image_ids: T = 10+10*i
    encMs = performance.now() - tEnc;
  }

  // ── 1) Text encoder (resident) ───────────────────────────────────────
  let teMs = 0;
  let promptEmbeds: Float32Array;
  if (cachedPrompt === prompt && cachedEmbeds) {
    promptEmbeds = cachedEmbeds;
    progress('te', 'prompt embedding reused (cached)');
  } else {
    const t0 = performance.now();
    progress('te', 'encoding prompt...');
    const emb = await embedFlux2Prompt(r.te.engine, r.te.tokenizer, prompt);
    promptEmbeds = emb.promptEmbeds;
    cachedPrompt = prompt;
    cachedEmbeds = promptEmbeds;
    teMs = performance.now() - t0;
  }

  // ── 2) DiT denoise (resident Q8 + dequant scratch) ───────────────────
  const t1 = performance.now();
  const numSteps = opts.numSteps ?? 4;
  const noise = randn(grid * grid * 128, seed);
  const res = await r.pipeline.generate({
    promptEmbeds, noise, gridH: grid, gridW: grid, numSteps,
    refLatents, refs,
    onProgress: (i, n) => progress('dit', `denoise step ${i + 1}/${n}...`),
  });
  const latents = res.latents;
  const ditMs = performance.now() - t1;

  // ── 3) VAE decode (resident weights, per-call decoder) ───────────────
  const t2 = performance.now();
  const unpacked = unpackLatents(latents, grid, grid, r.vae.bnMean, r.vae.bnVar);
  const dec = new Flux2VaeDecoder(device, r.vae);
  let pixels: Float32Array;
  try {
    pixels = await dec.decode(unpacked, grid * 2, (s) => progress('vae', `${s}...`));
  } finally {
    dec.destroy();
  }
  const vaeMs = performance.now() - t2;

  // NCHW [-1,1] → RGBA display units.
  const hw = px * px;
  const rgba = new Uint8ClampedArray(hw * 4);
  for (let i = 0; i < hw; i++) {
    rgba[i * 4] = (pixels[i] + 1) * 127.5;
    rgba[i * 4 + 1] = (pixels[hw + i] + 1) * 127.5;
    rgba[i * 4 + 2] = (pixels[2 * hw + i] + 1) * 127.5;
    rgba[i * 4 + 3] = 255;
  }
  return { rgba, width: px, height: px, timings: { teMs, ditMs, vaeMs, encMs } };
}
