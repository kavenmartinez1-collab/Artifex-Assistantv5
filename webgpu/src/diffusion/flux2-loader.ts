// FLUX.2-klein DiT weight loader: safetensors → GPU, v1 (raw bf16 resident).
//
// The 7.75 GB transformer/diffusion_pytorch_model.safetensors is Range-fetched
// tensor by tensor and uploaded as-is:
//   - 2-D weights ([N, K] bf16 row-major, little-endian): the raw bytes ARE
//     the [N, K/2] u32 layout matmul_bt_bf16 expects (even k in the low 16
//     bits of each u32), so they upload without any repacking.
//   - 1-D weights (the [128] QK RMSNorm affines): CPU-converted bf16 → f32
//     for rmsnorm.wgsl's f32 weight binding.
//
// Every fetched range is length-asserted (ArrayBuffer.slice / short-read
// truncation is silent and has bitten this codebase before).

import { parseHeader, parseHeaderLength } from '../model/safetensors';
import { fetchRange } from '../model/hf-hub';

export interface Flux2MatWeight {
  buffer: GPUBuffer; // [N, K/2] u32 (raw bf16 pairs)
  n: number;         // output features (rows of the safetensors [N, K])
  k: number;         // input features
}

export interface Flux2DitWeights {
  mats: Map<string, Flux2MatWeight>;
  vecs: Map<string, GPUBuffer>; // f32, [128] norm weights
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

export async function loadFlux2Dit(
  device: GPUDevice,
  url: string,
  onProgress?: (loadedBytes: number, totalBytes: number, name: string) => void,
): Promise<Flux2DitWeights> {
  const first8 = await fetchRange(url, 0, 8);
  if (first8.byteLength !== 8) throw new Error(`[Flux2 loader] header-length read: ${first8.byteLength} bytes`);
  const headerLen = parseHeaderLength(first8);
  const headerBytes = await fetchRange(url, 0, 8 + headerLen);
  if (headerBytes.byteLength !== 8 + headerLen) {
    throw new Error(`[Flux2 loader] header read: ${headerBytes.byteLength} vs ${8 + headerLen}`);
  }
  const header = parseHeader(headerBytes);
  const dataStart = header.headerByteLength;

  let totalBytes = 0;
  for (const t of header.tensors.values()) totalBytes += t.byteLength;

  const mats = new Map<string, Flux2MatWeight>();
  const vecs = new Map<string, GPUBuffer>();
  let loaded = 0;

  for (const [name, t] of header.tensors) {
    if (t.dtype !== 'BF16') throw new Error(`[Flux2 loader] ${name}: unexpected dtype ${t.dtype}`);
    const raw = await fetchRange(url, dataStart + t.dataOffsets[0], dataStart + t.dataOffsets[1]);
    if (raw.byteLength !== t.byteLength) {
      throw new Error(`[Flux2 loader] ${name}: fetched ${raw.byteLength} bytes, want ${t.byteLength}`);
    }
    if (t.shape.length === 2) {
      const [n, k] = t.shape;
      if (k % 2 !== 0) throw new Error(`[Flux2 loader] ${name}: odd K ${k}`);
      const buffer = device.createBuffer({
        size: raw.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, raw);
      mats.set(name, { buffer, n, k });
    } else if (t.shape.length === 1) {
      const f = bf16ToF32(raw);
      const buffer = device.createBuffer({
        size: f.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, f.buffer as ArrayBuffer, f.byteOffset, f.byteLength);
      vecs.set(name, buffer);
    } else {
      throw new Error(`[Flux2 loader] ${name}: unexpected rank ${t.shape.length}`);
    }
    loaded += t.byteLength;
    onProgress?.(loaded, totalBytes, name);
  }

  return {
    mats,
    vecs,
    totalBytes,
    destroy() {
      for (const m of mats.values()) m.buffer.destroy();
      for (const v of vecs.values()) v.destroy();
      mats.clear();
      vecs.clear();
    },
  };
}
