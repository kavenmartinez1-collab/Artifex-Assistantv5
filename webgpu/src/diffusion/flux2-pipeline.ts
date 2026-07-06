// FLUX.2-klein text-to-image orchestrator (Phase 3: DiT denoise loop only;
// VAE decode lands in Phase 4).
//
// Replicates pipeline_flux2_klein.py:
//   - noise is sampled directly in packed latent space (1, 128, H/16, W/16)
//     and packed to (tokens, 128); tokens = (px/16)^2
//   - txt position ids (0,0,0,L), img ids (0,y,x,0) row-major; joint RoPE
//     tables built CPU-side in f64 (theta 2000, axes [32,32,32,32],
//     repeat_interleave_real) — validated vs fixture at ~2e-8 in Phase 1
//   - FlowMatchEulerDiscrete with empirical-mu exponential shift, 4 steps
//   - klein is guidance-distilled: no CFG anywhere

import { Flux2Transformer, FLUX2_TXT_LEN } from './flux2-transformer';
import type { Flux2DitWeights } from './flux2-loader';
import { flux2Schedule, eulerStep } from './scheduler';

const AXES = [32, 32, 32, 32];
const THETA = 2000;

/** Joint [txt(512) | img(gridH*gridW)] RoPE cos/sin tables, [S, 128] f32. */
export function flux2RopeTables(gridH: number, gridW: number): { cos: Float32Array; sin: Float32Array } {
  const S = FLUX2_TXT_LEN + gridH * gridW;
  const D = 128;
  const cos = new Float32Array(S * D), sin = new Float32Array(S * D);
  const ids = new Int32Array(S * 4);
  for (let l = 0; l < FLUX2_TXT_LEN; l++) ids[l * 4 + 3] = l;          // (0,0,0,L)
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const r = FLUX2_TXT_LEN + y * gridW + x;                          // (0,y,x,0)
      ids[r * 4 + 1] = y;
      ids[r * 4 + 2] = x;
    }
  }
  for (let s = 0; s < S; s++) {
    let off = 0;
    for (let a = 0; a < AXES.length; a++) {
      const dim = AXES[a];
      const pos = ids[s * 4 + a];
      for (let k = 0; k < dim / 2; k++) {
        const ang = pos / Math.pow(THETA, (2 * k) / dim);
        const c = Math.cos(ang), si = Math.sin(ang);
        const i = s * D + off + 2 * k;
        cos[i] = c; cos[i + 1] = c;
        sin[i] = si; sin[i + 1] = si;
      }
      off += dim;
    }
  }
  return { cos, sin };
}

export interface Flux2GenerateOpts {
  promptEmbeds: Float32Array;   // [512, 7680]
  noise: Float32Array;          // [gridH*gridW, 128] packed latent-space noise
  gridH: number;                // px / 16
  gridW: number;
  numSteps?: number;            // klein default 4
  /** Transformer capture names for step 0 (parity gates). */
  captureStep0?: Set<string>;
  /** Called after each Euler update with the CURRENT latents (copy). */
  onLatents?: (stepIdx: number, latents: Float32Array, noisePred: Float32Array) => void;
  onProgress?: (stepIdx: number, numSteps: number) => void;
}

export class Flux2Pipeline {
  readonly transformer: Flux2Transformer;
  private setupTokens = -1;

  constructor(device: GPUDevice, weights: Flux2DitWeights, flopBudget?: number) {
    this.transformer = new Flux2Transformer(device, weights, flopBudget);
  }

  /** Run the denoise loop; returns final packed latents [tokens, 128] and
   *  any step-0 capture readbacks. */
  async generate(opts: Flux2GenerateOpts): Promise<{
    latents: Float32Array;
    caps: Map<string, Float32Array>;
    sigmas: Float64Array;
  }> {
    const { gridH, gridW } = opts;
    const tokens = gridH * gridW;
    const numSteps = opts.numSteps ?? 4;
    if (opts.noise.length !== tokens * 128) {
      throw new Error(`[Flux2] noise length ${opts.noise.length} != ${tokens * 128}`);
    }
    if (this.setupTokens !== tokens) {
      const { cos, sin } = flux2RopeTables(gridH, gridW);
      this.transformer.setup(tokens, cos, sin);
      this.setupTokens = tokens;
    }
    this.transformer.setPromptEmbeds(opts.promptEmbeds);

    const sched = flux2Schedule(tokens, numSteps);
    const latents = new Float32Array(opts.noise); // work in place
    let caps = new Map<string, Float32Array>();

    for (let i = 0; i < numSteps; i++) {
      opts.onProgress?.(i, numSteps);
      const res = await this.transformer.step(
        latents, sched.timesteps[i], i === 0 ? opts.captureStep0 : undefined,
      );
      if (i === 0) caps = res.caps;
      eulerStep(latents, res.noisePred, sched.sigmas[i], sched.sigmas[i + 1]);
      opts.onLatents?.(i, latents.slice(), res.noisePred);
    }
    return { latents, caps, sigmas: sched.sigmas };
  }

  destroy() {
    this.transformer.destroy();
  }
}
