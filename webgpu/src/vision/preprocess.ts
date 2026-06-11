/**
 * Image preprocessing — browser-side twin of the HF image processors.
 *
 * Produces the patch matrix the vision tower consumes:
 *   patches[numPatches][inChannels * temporalPatch * patch * patch]
 *
 * Qwen-style ('smart' resize): dimensions snap to multiples of
 * patch*merge, total area clamped to [minPixels, maxPixels] preserving
 * aspect ratio. Patch ORDER groups each spatial-merge window contiguously
 * (matches Qwen2/3-VL's processor reshape) and each patch vector is laid
 * out channel-major then temporal: [c][t][ph][pw] — the flattened conv3d
 * weight layout. Parity-checked against the Python processor in M1
 * validation.
 *
 * Gemma-style ('fixed'): bilinear resize to an exact size, plain row-major
 * patch order, temporalPatch 1.
 */

import type { VisionDescriptor } from './vision-descriptor';

export interface PreprocessedImage {
  /** numPatches × patchDim f32 (patchDim = C * T * P * P). */
  patches: Float32Array;
  /** Patch grid (pre-merge). */
  gridH: number;
  gridW: number;
  /** Text-side placeholder count this image occupies (post-merge). */
  numTokens: number;
}

/** Qwen smart-resize: snap to factor multiples, clamp area, keep aspect. */
export function smartResize(
  width: number, height: number,
  factor: number, minPixels: number, maxPixels: number,
): { width: number; height: number } {
  if (width <= 0 || height <= 0) throw new Error(`smartResize: bad input ${width}x${height}`);
  const round = (v: number) => Math.round(v / factor) * factor;
  const floor = (v: number) => Math.floor(v / factor) * factor;
  const ceil = (v: number) => Math.ceil(v / factor) * factor;

  let h = Math.max(factor, round(height));
  let w = Math.max(factor, round(width));
  if (h * w > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.max(factor, floor(height / beta));
    w = Math.max(factor, floor(width / beta));
  } else if (h * w < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    h = ceil(height * beta);
    w = ceil(width * beta);
  }
  return { width: w, height: h };
}

async function toBitmap(source: ImageBitmap | Blob | HTMLImageElement): Promise<ImageBitmap> {
  if (typeof ImageBitmap !== 'undefined' && source instanceof ImageBitmap) return source;
  return createImageBitmap(source as Blob | HTMLImageElement);
}

/** Draw at target size and return normalized CHW float data. */
function rasterize(
  bitmap: ImageBitmap, width: number, height: number,
  mean: [number, number, number], std: [number, number, number],
): Float32Array {
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(bitmap, 0, 0, width, height);
  const rgba = ctx.getImageData(0, 0, width, height).data;

  // CHW, normalized: (x/255 - mean) / std
  const chw = new Float32Array(3 * height * width);
  const plane = height * width;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const p = y * width + x;
      chw[p] = (rgba[i] / 255 - mean[0]) / std[0];
      chw[plane + p] = (rgba[i + 1] / 255 - mean[1]) / std[1];
      chw[2 * plane + p] = (rgba[i + 2] / 255 - mean[2]) / std[2];
    }
  }
  return chw;
}

/**
 * Preprocess one image per the descriptor. Returns the patch matrix plus
 * the grid geometry the encoder and prompt builder both need.
 */
export async function preprocessImage(
  source: ImageBitmap | Blob | HTMLImageElement,
  desc: VisionDescriptor,
): Promise<PreprocessedImage> {
  const bitmap = await toBitmap(source);
  const P = desc.patchSize;
  const T = desc.temporalPatchSize;
  const C = desc.inChannels;
  const merge = desc.projector.kind === 'qwen_merger' ? desc.projector.spatialMergeSize : 1;

  let W: number, H: number;
  if (desc.preprocess.resize.kind === 'smart') {
    const r = desc.preprocess.resize;
    // The ViT attention kernel reads at most 3840 patches per image
    // (attention.wgsl workgroup scores array) — clamp the area budget so
    // smart-resize can never exceed it, whatever the checkpoint configures.
    const kernelMaxPixels = 3840 * P * P;
    const maxPx = Math.min(r.maxPixels, kernelMaxPixels);
    ({ width: W, height: H } = smartResize(bitmap.width, bitmap.height, r.factor, r.minPixels, maxPx));
  } else {
    W = desc.preprocess.resize.width;
    H = desc.preprocess.resize.height;
  }

  const chw = rasterize(bitmap, W, H, desc.preprocess.imageMean, desc.preprocess.imageStd);
  const gridH = Math.floor(H / P);
  const gridW = Math.floor(W / P);
  const numPatches = gridH * gridW;
  const patchDim = C * T * P * P;
  const patches = new Float32Array(numPatches * patchDim);
  const plane = H * W;

  // Patch output order: merge-window-grouped for Qwen (gh-block, gw-block,
  // then the merge×merge window row-major), plain row-major when merge=1.
  let outIdx = 0;
  const mh = Math.floor(gridH / merge);
  const mw = Math.floor(gridW / merge);
  const writePatch = (gy: number, gx: number) => {
    const base = outIdx * patchDim;
    outIdx++;
    // [c][t][ph][pw] — images duplicate their frame across T
    for (let c = 0; c < C; c++) {
      for (let t = 0; t < T; t++) {
        for (let py = 0; py < P; py++) {
          const srcRow = c * plane + (gy * P + py) * W + gx * P;
          const dst = base + ((c * T + t) * P + py) * P;
          for (let px = 0; px < P; px++) {
            patches[dst + px] = chw[srcRow + px];
          }
        }
      }
    }
  };

  if (merge > 1) {
    for (let by = 0; by < mh; by++) {
      for (let bx = 0; bx < mw; bx++) {
        for (let wy = 0; wy < merge; wy++) {
          for (let wx = 0; wx < merge; wx++) {
            writePatch(by * merge + wy, bx * merge + wx);
          }
        }
      }
    }
  } else {
    for (let gy = 0; gy < gridH; gy++) {
      for (let gx = 0; gx < gridW; gx++) writePatch(gy, gx);
    }
  }

  // Tokens per image: qwen merger divides by merge², gemma's pooler by
  // poolKernel² — unless the family pins a fixed count.
  const tokenDiv = desc.projector.kind === 'qwen_merger' ? merge : (desc.gemma?.poolKernel ?? 1);
  const numTokens = desc.placeholder.fixedTokens
    ?? Math.floor(numPatches / (tokenDiv * tokenDiv));

  return { patches, gridH, gridW, numTokens };
}
