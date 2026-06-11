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
import { parseGGUF, type GGUFTensorInfo } from '../model/gguf';
import { dequantGGML } from '../model/gguf-dequant';
import { createStorageBuffer } from '../engine/buffers';
import type { VisionDescriptor } from './vision-descriptor';
import type { VisionWeights, VisionBlockWeights, VisionMergerWeights } from './vision-loader';

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
