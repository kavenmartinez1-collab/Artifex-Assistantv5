// FLUX.2-klein text-to-image runtime: the full browser pipeline behind the
// image-gen UI. Correctness-first v1 with a strictly sequential VRAM
// lifecycle on the 12 GB target — each stage frees its weights before the
// next loads (TE Q8 4.3 GB, DiT bf16 7.2 GB, VAE f32 0.34 GB never coexist):
//
//   1. TE:  Qwen3-4B Q8_0 GGUF → hidden taps 9/18/27 → prompt embeds (15.7 MB
//           CPU) → free. A one-entry cache skips this stage when only the
//           seed changes (the common "reroll" flow).
//   2. DiT: bf16 safetensors → 4-step FlowMatch Euler denoise → packed
//           latents (CPU) → free.
//   3. VAE: bn de-normalize + unpatchify → f32 decoder → RGB pixels → free.
//
// Every stage is the exact code path proven by the Phase 2/3/4 parity gates;
// only the seeded noise (rng.ts) is new — parity runs inject torch noise.

import { resolveFileUrl } from '../model/hf-hub';
import { loadFlux2TextEncoder } from './flux2-te';
import { embedFlux2Prompt } from './text-embedder';
import { loadFlux2Dit } from './flux2-loader';
import { Flux2Pipeline } from './flux2-pipeline';
import { loadFlux2Vae, unpackLatents, Flux2VaeDecoder } from './vae';
import { randn } from './rng';

export interface Flux2ImageOptions {
  prompt: string;
  px: number;              // output edge in pixels (multiple of 16)
  seed: number;
  numSteps?: number;       // klein default 4
  /** stage: 'te' | 'dit' | 'vae'; detail is a human-readable progress line. */
  onProgress?: (stage: string, detail: string) => void;
}

export interface Flux2ImageResult {
  rgba: Uint8ClampedArray<ArrayBuffer>; // [px*px*4] straight-alpha, ready for ImageData
  width: number;
  height: number;
  timings: { teMs: number; ditMs: number; vaeMs: number };
}

export const FLUX2_REPO = 'local/flux.2-klein-4b';
export const FLUX2_TE_REPO = 'local/flux2-te-qwen3-4b-q8_0';
export const FLUX2_TE_GGUF = 'flux2-te-qwen3-4b-q8_0.gguf';

// Re-embedding costs a full TE load+free cycle (~1 min); rerolling the seed
// on the same prompt is the dominant UX loop, so keep the last embedding.
let cachedPrompt: string | null = null;
let cachedEmbeds: Float32Array | null = null;

export async function generateFlux2Image(
  device: GPUDevice,
  opts: Flux2ImageOptions,
): Promise<Flux2ImageResult> {
  const { prompt, px, seed } = opts;
  if (px % 16 !== 0) throw new Error(`[Flux2] px must be a multiple of 16, got ${px}`);
  const grid = px / 16; // latent token grid edge
  const progress = opts.onProgress ?? (() => {});

  // ── 1) Text encoder ──────────────────────────────────────────────────
  let teMs = 0;
  let promptEmbeds: Float32Array;
  if (cachedPrompt === prompt && cachedEmbeds) {
    promptEmbeds = cachedEmbeds;
    progress('te', 'prompt embedding reused (cached)');
  } else {
    const t0 = performance.now();
    progress('te', 'loading text encoder (Qwen3-4B Q8)...');
    const te = await loadFlux2TextEncoder(device, FLUX2_TE_REPO, FLUX2_TE_GGUF,
      (m, f) => progress('te', f !== undefined ? `${m} (${Math.round(f * 100)}%)` : m));
    try {
      progress('te', 'encoding prompt...');
      const emb = await embedFlux2Prompt(te.engine, te.tokenizer, prompt);
      promptEmbeds = emb.promptEmbeds;
    } finally {
      te.destroy();
    }
    cachedPrompt = prompt;
    cachedEmbeds = promptEmbeds;
    teMs = performance.now() - t0;
  }

  // ── 2) DiT denoise ───────────────────────────────────────────────────
  const t1 = performance.now();
  progress('dit', 'loading DiT (7.2 GB bf16)...');
  const ditUrl = resolveFileUrl(FLUX2_REPO, 'transformer/diffusion_pytorch_model.safetensors');
  const ditWeights = await loadFlux2Dit(device, ditUrl, (loaded, total) =>
    progress('dit', `loading DiT weights... ${Math.round((loaded / total) * 100)}%`));
  let latents: Float32Array;
  try {
    const pipeline = new Flux2Pipeline(device, ditWeights);
    try {
      const numSteps = opts.numSteps ?? 4;
      const noise = randn(grid * grid * 128, seed);
      const res = await pipeline.generate({
        promptEmbeds, noise, gridH: grid, gridW: grid, numSteps,
        onProgress: (i, n) => progress('dit', `denoise step ${i + 1}/${n}...`),
      });
      latents = res.latents;
    } finally {
      pipeline.destroy();
    }
  } finally {
    ditWeights.destroy();
  }
  const ditMs = performance.now() - t1;

  // ── 3) VAE decode ────────────────────────────────────────────────────
  const t2 = performance.now();
  progress('vae', 'loading VAE...');
  const vaeUrl = resolveFileUrl(FLUX2_REPO, 'vae/diffusion_pytorch_model.safetensors');
  const vae = await loadFlux2Vae(device, vaeUrl);
  let pixels: Float32Array;
  try {
    const unpacked = unpackLatents(latents, grid, grid, vae.bnMean, vae.bnVar);
    const dec = new Flux2VaeDecoder(device, vae);
    try {
      pixels = await dec.decode(unpacked, grid * 2, (s) => progress('vae', `${s}...`));
    } finally {
      dec.destroy();
    }
  } finally {
    vae.destroy();
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
  return { rgba, width: px, height: px, timings: { teMs, ditMs, vaeMs } };
}
