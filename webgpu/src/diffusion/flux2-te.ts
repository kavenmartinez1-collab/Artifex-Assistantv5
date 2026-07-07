// Programmatic loader for the FLUX.2-klein text encoder (Qwen3-4B Q8_0 GGUF).
//
// A trimmed-down copy of main.ts buildGGUFSession's GGUF→engine bridge for
// the plain dense-attention path only — the klein TE is a vanilla Qwen3
// (no MoE, no PLE, no hybrid layers, no vision), so the exotic branches are
// asserted away instead of duplicated. Used by the image-gen pipeline, which
// loads the TE, taps hidden states 9/18/27, and frees it before the 7.2 GB
// DiT comes in (both do not fit in 12 GB together).

import { loadGGUFModel } from '../model/gguf-loader';
import { descriptorFromGGUF } from '../model/model-descriptor';
import { ggufArchitecture } from '../model/gguf';
import { createGGUFLocator, type TensorRole } from '../model/tensor-locator';
import { createForwardPassEngine, type ForwardPassEngine } from '../engine/forward-pass';
import { createTokenizer } from '../model/tokenizer';

export interface Flux2TextEncoder {
  engine: ForwardPassEngine;
  tokenizer: { encode(text: string): number[] };
  totalGPUBytes: number;
  destroy(): void;
}

export async function loadFlux2TextEncoder(
  device: GPUDevice,
  repo: string,
  ggufFile: string,
  onProgress?: (message: string, frac?: number) => void,
): Promise<Flux2TextEncoder> {
  const model = await loadGGUFModel(device, repo, ggufFile, (p) => {
    onProgress?.(p.message, p.overallProgress);
  });

  const config = descriptorFromGGUF(model.file);
  if (config.isHybrid || config.perLayerEmbed || config.layers.some((d: any) => d.moe)) {
    throw new Error('[Flux2 TE] expected a plain dense Qwen3 GGUF');
  }

  const loc = createGGUFLocator(model.file.tensors, ggufArchitecture(model.file));
  const requireBuf = (role: TensorRole, l?: number): GPUBuffer => {
    const n = loc.locate(role, l);
    const t = n ? model.tensors.get(n) : undefined;
    if (!t) throw new Error(`[Flux2 TE] missing tensor for role "${role}"${l !== undefined ? ` (layer ${l})` : ''}`);
    return t.buffer;
  };
  const roleBuf = (role: TensorRole, l?: number): GPUBuffer | undefined => {
    const n = loc.locate(role, l);
    return n ? model.tensors.get(n)?.buffer : undefined;
  };
  const assignProj = (lw: any, slot: string, role: TensorRole, l: number) => {
    const n = loc.locate(role, l);
    const t = n ? model.tensors.get(n) : undefined;
    if (!t) throw new Error(`[Flux2 TE] missing tensor for role "${role}" (layer ${l})`);
    if (t.isQuantized) lw[`${slot}_gg`] = { data: t.buffer, ggmlType: t.ggmlType };
    else lw[slot] = t.buffer;
  };

  const embedCpu = model.cpuTensors.get('token_embd.weight');
  if (!embedCpu) throw new Error('[Flux2 TE] token_embd.weight missing from CPU store');
  const finalNorm = requireBuf('finalNorm');
  const lmHeadName = loc.locate('lmHead')!;
  const lmHeadT = model.tensors.get(lmHeadName);
  if (!lmHeadT) throw new Error(`[Flux2 TE] lm_head tensor "${lmHeadName}" not on GPU`);
  const global: any = {
    embedTokens: finalNorm, // dummy — embedGG row-gather path is used
    finalNorm,
    lmHead: lmHeadT.isQuantized ? finalNorm : lmHeadT.buffer,
    embedGG: { data: embedCpu.data, ggmlType: embedCpu.ggmlType, rowBytes: embedCpu.rowBytes },
  };
  if (lmHeadT.isQuantized) {
    global.lmHeadGG = { data: lmHeadT.buffer, ggmlType: lmHeadT.ggmlType };
  }

  const layers: any[] = [];
  for (let l = 0; l < config.numLayers; l++) {
    const lw: any = {
      inputNorm: requireBuf('inputNorm', l),
      postAttnNorm: requireBuf('postAttnNorm', l),
    };
    assignProj(lw, 'qProj', 'qProj', l);
    assignProj(lw, 'kProj', 'kProj', l);
    assignProj(lw, 'vProj', 'vProj', l);
    assignProj(lw, 'oProj', 'oProj', l);
    lw.qNorm = roleBuf('qNorm', l);
    lw.kNorm = roleBuf('kNorm', l);
    assignProj(lw, 'gateProj', 'gateProj', l);
    assignProj(lw, 'upProj', 'upProj', l);
    assignProj(lw, 'downProj', 'downProj', l);
    layers.push(lw);
  }

  const engine = createForwardPassEngine(device, config, { global, layers });
  const tokenizer = await createTokenizer({ modelId: repo });
  if (tokenizer.vocabSize && Math.abs(tokenizer.vocabSize - config.vocabSize) > 1024) {
    throw new Error(`[Flux2 TE] tokenizer vocab ${tokenizer.vocabSize} vs model ${config.vocabSize}`);
  }

  return {
    engine,
    tokenizer,
    totalGPUBytes: model.totalGPUBytes,
    destroy() {
      for (const t of model.tensors.values()) t.buffer.destroy();
      model.tensors.clear();
    },
  };
}
