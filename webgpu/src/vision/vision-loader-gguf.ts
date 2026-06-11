/**
 * GGUF vision weight loader — reads a Qwen3-VL-style tower from a GGUF
 * mmproj file (clip.cpp conventions: v.* tower, mm.* projector) and builds
 * the same VisionWeights shape the safetensors loader produces, so one
 * encoder serves both sources.
 *
 * qwen3vl_merger mapping (verified against a real mmproj header):
 *   patchEmbed   ← v.patch_embd.weight (+ optional .weight.1 temporal slice,
 *                  summed — image frames are duplicated, x·W0+x·W1 = x·(W0+W1))
 *   posEmbedCPU  ← v.position_embd.weight
 *   block i      ← v.blk.{i}.{ln1,attn_qkv,attn_out,ln2,ffn_up,ffn_down}
 *   merger       ← norm: v.post_ln, fc1: mm.0, fc2: mm.2
 *
 * All tensors dequant to f32 on CPU (F16/F32/BF16 in practice for mmproj
 * files; k-quants handled by dequantGGML if a quantized mmproj shows up).
 */

import { fetchRange, resolveFileUrl } from '../model/hf-hub';
import { parseGGUF, type GGUFFile, type GGUFTensorInfo } from '../model/gguf';
import { dequantGGML } from '../model/gguf-dequant';
import { createStorageBuffer } from '../engine/buffers';
import type { VisionDescriptor } from './vision-descriptor';
import type { VisionWeights, VisionBlockWeights, VisionMergerWeights } from './vision-loader';

// ── Gemma 4 tower weights (separate shape from the qwen-clip tower) ─────

/** Gemma4ClippableLinear calibration ranges — clamp x before the matmul and
 *  the output after. Values are tiny f32[1] tensors, read CPU-side. */
export interface ClampRange { inLo: number; inHi: number; outLo: number; outHi: number }

export interface GemmaBlockWeights {
  ln1: GPUBuffer;
  q: GPUBuffer; k: GPUBuffer; v: GPUBuffer;
  qClamp?: ClampRange; kClamp?: ClampRange; vClamp?: ClampRange;
  qNorm: GPUBuffer; kNorm: GPUBuffer;        // per-head RMS [head_dim]
  attnOut: GPUBuffer; attnOutClamp?: ClampRange;
  attnPostNorm: GPUBuffer;
  ln2: GPUBuffer;
  gate: GPUBuffer; gateClamp?: ClampRange;
  up: GPUBuffer; upClamp?: ClampRange;
  down: GPUBuffer; downClamp?: ClampRange;
  ffnPostNorm: GPUBuffer;
}

export interface GemmaVisionWeights {
  variant: 'gemma4';
  patchEmbed: GPUBuffer;          // [hidden, C·P·P] flattened conv, no bias
  /** Factorized position tables, CPU-side: x then y, each [posSize, hidden]. */
  posTableX: Float32Array;
  posTableY: Float32Array;
  blocks: GemmaBlockWeights[];
  inputProjection: GPUBuffer;     // [textHidden, hidden]
  totalGPUBytes: number;
  destroy(): void;
}

export async function loadVisionWeightsGGUF(
  device: GPUDevice,
  repo: string,
  filename: string,
  desc: VisionDescriptor,
  onProgress?: (msg: string) => void,
): Promise<VisionWeights> {
  const url = resolveFileUrl(repo, filename);
  const file = await parseGGUF((s, e) => fetchRange(url, s, e));

  if (desc.deepstackIndexes.length > 0) {
    throw new Error('[Vision] GGUF mmproj with deepstack taps not yet mapped — file an issue with the tensor names');
  }

  let totalGPUBytes = 0;
  const allBuffers: GPUBuffer[] = [];

  function info(name: string): GGUFTensorInfo {
    const t = file.tensors.get(name);
    if (!t) throw new Error(`[Vision] mmproj missing tensor: ${name}`);
    return t;
  }

  async function fetchF32(name: string): Promise<Float32Array> {
    const t = info(name);
    const raw = await fetchRange(url, t.offset, t.offset + t.byteLength);
    const n = t.ne.reduce((a, b) => a * b, 1);
    return dequantGGML(t.ggmlType, raw, n);
  }

  function upload(f32: Float32Array, label: string): GPUBuffer {
    const buf = createStorageBuffer(device, f32, f32.byteLength, `vis-gg-${label}`);
    totalGPUBytes += f32.byteLength;
    allBuffers.push(buf);
    return buf;
  }

  async function gpu(name: string, label: string): Promise<GPUBuffer> {
    return upload(await fetchF32(name), label);
  }

  onProgress?.(`Loading vision tower (GGUF): ${filename}...`);

  // Patch embed — sum the temporal kernel slices when present
  const pe0 = await fetchF32('v.patch_embd.weight');
  if (file.tensors.has('v.patch_embd.weight.1')) {
    const pe1 = await fetchF32('v.patch_embd.weight.1');
    for (let i = 0; i < pe0.length; i++) pe0[i] += pe1[i];
  }
  const patchEmbed = upload(pe0, 'patch-embd');
  const patchEmbedBias = await gpu('v.patch_embd.bias', 'patch-bias');

  const posEmbedCPU = file.tensors.has('v.position_embd.weight')
    ? await fetchF32('v.position_embd.weight')
    : null;

  const blocks: VisionBlockWeights[] = [];
  for (let i = 0; i < desc.depth; i++) {
    onProgress?.(`Vision block ${i + 1}/${desc.depth} (GGUF)...`);
    const b = (suffix: string, label: string) => gpu(`v.blk.${i}.${suffix}`, `${label}-${i}`);
    blocks.push({
      ln1: await b('ln1.weight', 'ln1'), ln1Bias: await b('ln1.bias', 'ln1b'),
      qkv: await b('attn_qkv.weight', 'qkv'), qkvBias: await b('attn_qkv.bias', 'qkvb'),
      attnOut: await b('attn_out.weight', 'ao'), attnOutBias: await b('attn_out.bias', 'aob'),
      ln2: await b('ln2.weight', 'ln2'), ln2Bias: await b('ln2.bias', 'ln2b'),
      fc1: await b('ffn_up.weight', 'fc1'), fc1Bias: await b('ffn_up.bias', 'fc1b'),
      fc2: await b('ffn_down.weight', 'fc2'), fc2Bias: await b('ffn_down.bias', 'fc2b'),
    });
  }

  const merger: VisionMergerWeights = {
    norm: await gpu('v.post_ln.weight', 'm-norm'),
    normBias: await gpu('v.post_ln.bias', 'm-normb'),
    fc1: await gpu('mm.0.weight', 'm-fc1'),
    fc1Bias: await gpu('mm.0.bias', 'm-fc1b'),
    fc2: await gpu('mm.2.weight', 'm-fc2'),
    fc2Bias: await gpu('mm.2.bias', 'm-fc2b'),
  };

  onProgress?.(`Vision tower loaded: ${(totalGPUBytes / 1e9).toFixed(2)} GB GPU`);

  return {
    patchEmbed, patchEmbedBias, posEmbedCPU,
    blocks, merger, deepstack: [], totalGPUBytes,
    destroy() { for (const b of allBuffers) b.destroy(); },
  };
}

/** Load the Gemma 4 tower from an Ollama-packed GGUF (v. and mm. tensors inline). */
export async function loadGemmaVisionWeightsGGUF(
  device: GPUDevice,
  repo: string,
  filename: string,
  desc: VisionDescriptor,
  onProgress?: (msg: string) => void,
): Promise<GemmaVisionWeights> {
  const url = resolveFileUrl(repo, filename);
  const file: GGUFFile = await parseGGUF((s, e) => fetchRange(url, s, e));

  let totalGPUBytes = 0;
  const allBuffers: GPUBuffer[] = [];

  function info(name: string): GGUFTensorInfo {
    const t = file.tensors.get(name);
    if (!t) throw new Error(`[Vision] gemma blob missing tensor: ${name}`);
    return t;
  }
  async function fetchF32(name: string): Promise<Float32Array> {
    const t = info(name);
    const raw = await fetchRange(url, t.offset, t.offset + t.byteLength);
    const n = t.ne.reduce((a, b) => a * b, 1);
    return dequantGGML(t.ggmlType, raw, n);
  }
  function upload(f32: Float32Array, label: string): GPUBuffer {
    const buf = createStorageBuffer(device, f32, f32.byteLength, `vis-gm-${label}`);
    totalGPUBytes += f32.byteLength;
    allBuffers.push(buf);
    return buf;
  }
  async function gpu(name: string, label: string): Promise<GPUBuffer> {
    return upload(await fetchF32(name), label);
  }
  /** Read the four Gemma4ClippableLinear scalars; undefined when absent. */
  async function clampOf(base: string): Promise<ClampRange | undefined> {
    if (!file.tensors.has(`${base}.input_min`)) return undefined;
    const one = async (s: string) => (await fetchF32(`${base}.${s}`))[0];
    return {
      inLo: await one('input_min'), inHi: await one('input_max'),
      outLo: await one('output_min'), outHi: await one('output_max'),
    };
  }

  onProgress?.(`Loading Gemma vision tower: ${filename}...`);

  const patchEmbed = await gpu('v.patch_embd.weight', 'patch');

  // v.position_embd.weight ne [hidden, posSize, 2] → two tables, x then y
  const posT = info('v.position_embd.weight');
  const posAll = await fetchF32('v.position_embd.weight');
  const posSize = posT.ne[1];
  const H = posT.ne[0];
  const posTableX = posAll.subarray(0, posSize * H);
  const posTableY = posAll.subarray(posSize * H, 2 * posSize * H);

  const blocks: GemmaBlockWeights[] = [];
  for (let i = 0; i < desc.depth; i++) {
    onProgress?.(`Gemma vision block ${i + 1}/${desc.depth}...`);
    const base = `v.blk.${i}`;
    blocks.push({
      ln1: await gpu(`${base}.ln1.weight`, `ln1-${i}`),
      q: await gpu(`${base}.attn_q.weight`, `q-${i}`),
      k: await gpu(`${base}.attn_k.weight`, `k-${i}`),
      v: await gpu(`${base}.attn_v.weight`, `v-${i}`),
      qClamp: await clampOf(`${base}.attn_q`),
      kClamp: await clampOf(`${base}.attn_k`),
      vClamp: await clampOf(`${base}.attn_v`),
      qNorm: await gpu(`${base}.attn_q_norm.weight`, `qn-${i}`),
      kNorm: await gpu(`${base}.attn_k_norm.weight`, `kn-${i}`),
      attnOut: await gpu(`${base}.attn_out.weight`, `ao-${i}`),
      attnOutClamp: await clampOf(`${base}.attn_out`),
      attnPostNorm: await gpu(`${base}.attn_post_norm.weight`, `apn-${i}`),
      ln2: await gpu(`${base}.ln2.weight`, `ln2-${i}`),
      gate: await gpu(`${base}.ffn_gate.weight`, `g-${i}`),
      gateClamp: await clampOf(`${base}.ffn_gate`),
      up: await gpu(`${base}.ffn_up.weight`, `u-${i}`),
      upClamp: await clampOf(`${base}.ffn_up`),
      down: await gpu(`${base}.ffn_down.weight`, `d-${i}`),
      downClamp: await clampOf(`${base}.ffn_down`),
      ffnPostNorm: await gpu(`${base}.ffn_post_norm.weight`, `fpn-${i}`),
    });
  }

  const inputProjection = await gpu('mm.input_projection.weight', 'proj');

  onProgress?.(`Gemma vision tower loaded: ${(totalGPUBytes / 1e9).toFixed(2)} GB GPU`);

  return {
    variant: 'gemma4',
    patchEmbed, posTableX, posTableY, blocks, inputProjection, totalGPUBytes,
    destroy() { for (const b of allBuffers) b.destroy(); },
  };
}
