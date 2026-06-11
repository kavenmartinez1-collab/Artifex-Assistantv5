/**
 * Vision weight loader — fetches ONLY the vision-tower tensors the text
 * loaders deliberately skip, on demand (first image attach), and uploads
 * them as f32 GPU buffers.
 *
 * v1 keeps everything f32 for parity-first correctness: the tower runs once
 * per image (not per token), so bandwidth is not the bottleneck. ~1.2 GB
 * for Qwen3-VL-4B's 300M-param tower.
 *
 * Safetensors (HF) source only for now — GGUF mmproj / Ollama projector
 * blobs land in M2 via the same VisionWeights shape.
 */

import { listModelFiles, fetchRange, resolveFileUrl } from '../model/hf-hub';
import { parseHeader, parseHeaderLength, tensorToFloat32, type TensorInfo } from '../model/safetensors';
import { createStorageBuffer } from '../engine/buffers';
import { createVisionLocator, type VisionLocator, type VisionRole } from './vision-locator';
import type { VisionDescriptor } from './vision-descriptor';

export interface VisionBlockWeights {
  ln1: GPUBuffer; ln1Bias: GPUBuffer;
  qkv: GPUBuffer; qkvBias: GPUBuffer;        // fused [3H, H] — or null path uses q/k/v
  q?: GPUBuffer; k?: GPUBuffer; v?: GPUBuffer;
  qBias?: GPUBuffer; kBias?: GPUBuffer; vBias?: GPUBuffer;
  attnOut: GPUBuffer; attnOutBias: GPUBuffer;
  ln2: GPUBuffer; ln2Bias: GPUBuffer;
  fc1: GPUBuffer; fc1Bias: GPUBuffer;
  fc2: GPUBuffer; fc2Bias: GPUBuffer;
}

export interface VisionMergerWeights {
  norm: GPUBuffer; normBias: GPUBuffer;
  fc1: GPUBuffer; fc1Bias: GPUBuffer;
  fc2: GPUBuffer; fc2Bias: GPUBuffer;
}

export interface VisionWeights {
  patchEmbed: GPUBuffer;        // [hidden, C*T*P*P] flattened conv
  patchEmbedBias: GPUBuffer;    // [hidden]
  /** Learned position table kept CPU-side — interpolated per image grid. */
  posEmbedCPU: Float32Array | null;   // [count, hidden]
  blocks: VisionBlockWeights[];
  merger: VisionMergerWeights;
  deepstack: VisionMergerWeights[];   // one per deepstackIndexes entry
  totalGPUBytes: number;
  destroy(): void;
}

/** Load vision tensors for an HF-safetensors repo. */
export async function loadVisionWeights(
  device: GPUDevice,
  repo: string,
  desc: VisionDescriptor,
  onProgress?: (msg: string) => void,
): Promise<VisionWeights> {
  // ── Map vision tensors → (file, TensorInfo, headerLen) ──────────────
  const files = (await listModelFiles(repo)).filter(f => f.path.endsWith('.safetensors'));
  interface Located { file: string; info: TensorInfo; headerLen: number }
  const tensorMap = new Map<string, Located>();
  for (const f of files) {
    const url = resolveFileUrl(repo, f.path);
    const first8 = await fetchRange(url, 0, 8);
    const headerLen = parseHeaderLength(first8);
    // parseHeader expects the buffer INCLUDING the 8-byte length prefix
    const headerBytes = await fetchRange(url, 0, 8 + headerLen);
    const header = parseHeader(headerBytes);
    for (const [name, info] of header.tensors) {
      if (name.startsWith('model.visual.') || name.startsWith('visual.')) {
        tensorMap.set(name, { file: f.path, info, headerLen });
      }
    }
  }
  if (tensorMap.size === 0) {
    throw new Error(`No vision tensors found in ${repo} — is this a multimodal checkpoint?`);
  }

  const locator: VisionLocator = createVisionLocator(tensorMap.keys(), 'hf');
  let totalGPUBytes = 0;
  const allBuffers: GPUBuffer[] = [];

  async function fetchF32(name: string): Promise<Float32Array> {
    const loc = tensorMap.get(name);
    if (!loc) throw new Error(`[Vision] tensor missing: ${name}`);
    const url = resolveFileUrl(repo, loc.file);
    const start = 8 + loc.headerLen + loc.info.dataOffsets[0];
    const end = 8 + loc.headerLen + loc.info.dataOffsets[1];
    const raw = await fetchRange(url, start, end);
    return tensorToFloat32(raw, loc.info.dtype);
  }

  async function gpu(role: VisionRole, blockIdx?: number, dsIdx?: number): Promise<GPUBuffer> {
    const name = locator.locate(role, blockIdx, dsIdx);
    if (!name) {
      throw new Error(`[Vision] required role "${role}"${blockIdx !== undefined ? ` (block ${blockIdx})` : ''} not present`);
    }
    const f32 = await fetchF32(name);
    const buf = createStorageBuffer(device, f32, f32.byteLength, `vis-${role}-${blockIdx ?? dsIdx ?? ''}`);
    totalGPUBytes += f32.byteLength;
    allBuffers.push(buf);
    return buf;
  }

  onProgress?.(`Loading vision tower: ${tensorMap.size} tensors...`);

  const patchEmbed = await gpu('patchEmbed');
  const patchEmbedBias = await gpu('patchEmbedBias');

  const posName = locator.locate('posEmbed');
  const posEmbedCPU = posName ? await fetchF32(posName) : null;

  const blocks: VisionBlockWeights[] = [];
  for (let i = 0; i < desc.depth; i++) {
    onProgress?.(`Vision block ${i + 1}/${desc.depth}...`);
    blocks.push({
      ln1: await gpu('norm1', i), ln1Bias: await gpu('norm1Bias', i),
      qkv: await gpu('qkv', i), qkvBias: await gpu('qkvBias', i),
      attnOut: await gpu('attnOut', i), attnOutBias: await gpu('attnOutBias', i),
      ln2: await gpu('norm2', i), ln2Bias: await gpu('norm2Bias', i),
      fc1: await gpu('mlpFc1', i), fc1Bias: await gpu('mlpFc1Bias', i),
      fc2: await gpu('mlpFc2', i), fc2Bias: await gpu('mlpFc2Bias', i),
    });
  }

  const merger: VisionMergerWeights = {
    norm: await gpu('mergerNorm'), normBias: await gpu('mergerNormBias'),
    fc1: await gpu('mergerFc1'), fc1Bias: await gpu('mergerFc1Bias'),
    fc2: await gpu('mergerFc2'), fc2Bias: await gpu('mergerFc2Bias'),
  };

  const deepstack: VisionMergerWeights[] = [];
  for (let d = 0; d < desc.deepstackIndexes.length; d++) {
    deepstack.push({
      norm: await gpu('dsNorm', undefined, d), normBias: await gpu('dsNormBias', undefined, d),
      fc1: await gpu('dsFc1', undefined, d), fc1Bias: await gpu('dsFc1Bias', undefined, d),
      fc2: await gpu('dsFc2', undefined, d), fc2Bias: await gpu('dsFc2Bias', undefined, d),
    });
  }

  onProgress?.(`Vision tower loaded: ${(totalGPUBytes / 1e9).toFixed(2)} GB GPU`);

  return {
    patchEmbed, patchEmbedBias, posEmbedCPU,
    blocks, merger, deepstack, totalGPUBytes,
    destroy() { for (const b of allBuffers) b.destroy(); },
  };
}
