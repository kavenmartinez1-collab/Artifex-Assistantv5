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

// ── Phase 7: Q8_0 all-resident container (.artq) ─────────────────────────
// Written by scripts/convert-flux2-dit-q8.py:
//   'ARTQ' | u32 version | u64 jsonLen | JSON (space-padded) | data
// Per q8_0 tensor the i8 quants and f16 scales are deinterleaved; both live
// in ONE GPU buffer per tensor (quants at word 0, scales at sWordOff) so the
// dequant kernel binds a single storage buffer.

export interface Flux2Q8Mat {
  buffer: GPUBuffer; // [ i8 quants N*K | pad | f16 scales N*K/32 ]
  n: number;
  k: number;
  sWordOff: number;  // scale section offset in u32 words
}

export interface Flux2DitWeightsQ8 {
  q8mats: Map<string, Flux2Q8Mat>;
  vecs: Map<string, GPUBuffer>; // f32 norm vectors
  totalBytes: number;
  maxMatElems: number;          // max N*K — sizes the dequant scratch ring
  destroy(): void;
}

interface ArtqTensor {
  name: string;
  shape: number[];
  dtype: 'q8_0' | 'f32';
  offset: number;
  bytes: number;
  scaleOffset?: number;
  scaleBytes?: number;
}

export async function loadFlux2DitQ8(
  device: GPUDevice,
  url: string,
  onProgress?: (loadedBytes: number, totalBytes: number, name: string) => void,
): Promise<Flux2DitWeightsQ8> {
  const head = await fetchRange(url, 0, 16);
  if (head.byteLength !== 16) throw new Error(`[Flux2 Q8] header read: ${head.byteLength} bytes`);
  const dv = new DataView(head);
  if (dv.getUint32(0, true) !== 0x51545241) throw new Error('[Flux2 Q8] bad magic (want ARTQ)');
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`[Flux2 Q8] unsupported version ${version}`);
  const jsonLen = Number(dv.getBigUint64(8, true));
  const jsonBytes = await fetchRange(url, 16, 16 + jsonLen);
  if (jsonBytes.byteLength !== jsonLen) {
    throw new Error(`[Flux2 Q8] json read: ${jsonBytes.byteLength} vs ${jsonLen}`);
  }
  const tensors = (JSON.parse(new TextDecoder().decode(jsonBytes)) as
    { tensors: ArtqTensor[] }).tensors;
  const dataStart = 16 + jsonLen;

  let totalBytes = 0;
  for (const t of tensors) totalBytes += t.bytes + (t.scaleBytes ?? 0);

  const q8mats = new Map<string, Flux2Q8Mat>();
  const vecs = new Map<string, GPUBuffer>();
  let maxMatElems = 0;
  let loaded = 0;

  for (const t of tensors) {
    if (t.dtype === 'q8_0') {
      const [n, k] = t.shape;
      if (n * k !== t.bytes) throw new Error(`[Flux2 Q8] ${t.name}: ${n}x${k} != ${t.bytes} quant bytes`);
      if (t.scaleOffset! < t.offset + t.bytes || t.scaleOffset! - t.offset - t.bytes >= 64) {
        throw new Error(`[Flux2 Q8] ${t.name}: scales not adjacent to quants`);
      }
      if ((n * k) / 32 * 2 !== t.scaleBytes) {
        throw new Error(`[Flux2 Q8] ${t.name}: scaleBytes ${t.scaleBytes} != ${(n * k) / 16}`);
      }
      const end = t.scaleOffset! + t.scaleBytes!;
      const raw = await fetchRange(url, dataStart + t.offset, dataStart + end);
      if (raw.byteLength !== end - t.offset) {
        throw new Error(`[Flux2 Q8] ${t.name}: fetched ${raw.byteLength}, want ${end - t.offset}`);
      }
      const buffer = device.createBuffer({
        size: Math.ceil(raw.byteLength / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const aligned = raw.byteLength - (raw.byteLength % 4);
      device.queue.writeBuffer(buffer, 0, raw, 0, aligned);
      if (raw.byteLength % 4) { // ragged f16 tail: pad the final word
        const tail = new Uint8Array(4);
        tail.set(new Uint8Array(raw, aligned));
        device.queue.writeBuffer(buffer, aligned, tail);
      }
      q8mats.set(t.name, { buffer, n, k, sWordOff: (t.scaleOffset! - t.offset) / 4 });
      maxMatElems = Math.max(maxMatElems, n * k);
      loaded += raw.byteLength;
    } else {
      const raw = await fetchRange(url, dataStart + t.offset, dataStart + t.offset + t.bytes);
      if (raw.byteLength !== t.bytes) {
        throw new Error(`[Flux2 Q8] ${t.name}: fetched ${raw.byteLength}, want ${t.bytes}`);
      }
      const buffer = device.createBuffer({
        size: raw.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, raw);
      vecs.set(t.name, buffer);
      loaded += raw.byteLength;
    }
    onProgress?.(loaded, totalBytes, t.name);
  }

  return {
    q8mats,
    vecs,
    totalBytes,
    maxMatElems,
    destroy() {
      for (const m of q8mats.values()) m.buffer.destroy();
      for (const v of vecs.values()) v.destroy();
      q8mats.clear();
      vecs.clear();
    },
  };
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
