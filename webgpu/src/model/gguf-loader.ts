/**
 * GGUF model loader — streams tensors from a .gguf file into GPU buffers.
 *
 * Placement:
 *   - token_embd.weight → CPU RAM (raw blocks; forward pass row-gathers +
 *     dequants per token — embedCPU precedent, saves ~0.5-2 GB VRAM)
 *   - F32 tensors (norms, biases, ssm_a, conv1d, dt_bias) → f32 buffers
 *   - F16/BF16 tensors → dequant to f32 at load (small tensors only)
 *   - Q8_0/Q4_K/Q5_K/Q6_K matrices → repackGGUFForGPU → u32 buffers,
 *     dispatched via shaders/matmul_gguf.wgsl
 *
 * Tied embeddings (no output.weight): token_embd is kept on CPU for embed
 * AND uploaded repacked to GPU so the lm_head matmul has a buffer.
 *
 * Tensor data is fetched per-tensor over HTTP Range (dev-server / HF CDN),
 * chunked at 128 MB with byte-count assertions — never a whole-file buffer.
 */

import {
  parseGGUF,
  ggmlTypeTraits,
  archKV,
  GGML_TYPES,
  type GGUFFile,
  type GGUFTensorInfo,
} from './gguf';
import { repackGGUFForGPU, dequantF16, dequantBF16, GGUF_GPU_LAYOUT } from './gguf-dequant';
import { fetchRange, resolveFileUrl } from './hf-hub';

// ── Types ──────────────────────────────────────────────────────────────

export interface GGUFTensorGPU {
  name: string;
  buffer: GPUBuffer;
  /** HF/torch convention: [out, in] for matrices. */
  shape: number[];
  /** GGML type name ('F32', 'Q4_K', ...). */
  dtype: string;
  ggmlType: number;
  byteLength: number;
  elementCount: number;
  /** True for k-quant block data (dispatch via matmul_gguf). */
  isQuantized: boolean;
}

export interface GGUFTensorCPU {
  data: Uint8Array;
  ggmlType: number;
  ne: number[];
  /** Bytes per tensor row (ne[0] elements) in raw GGUF block layout. */
  rowBytes: number;
}

export interface LoadedGGUFModel {
  repo: string;
  file: GGUFFile;
  /** Resolved file URL — MoE workers Range-fetch expert slabs from it. */
  url: string;
  tensors: Map<string, GGUFTensorGPU>;
  cpuTensors: Map<string, GGUFTensorCPU>;
  /**
   * MoE expert tensors (ffn_{gate,up,down}_exps) — NOT fetched/uploaded here.
   * Raw GGUF tensor infos (abs offset/byteLength/ne) for the CPU worker fleet.
   */
  expertTensors: Map<string, GGUFTensorInfo>;
  expertBytes: number;
  totalGPUBytes: number;
  tensorCount: number;
  loadTimeMs: number;
}

export interface GGUFLoadProgress {
  message: string;
  overallProgress?: number;
}

const QUANT_GPU_TYPES = new Set<number>([
  GGML_TYPES.Q8_0, GGML_TYPES.Q4_K, GGML_TYPES.Q5_K, GGML_TYPES.Q6_K,
]);

const MAX_CHUNK = 128 * 1024 * 1024;

// MoE expert placement (Phase C): expert slabs stay in the file, fetched by
// the CPU worker fleet; the per-layer shared-expert router gate vector stays
// on CPU (scalar sigmoid gate is computed in JS).
const EXPERT_RE = /^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$/;
const FUSED_EXPERT_RE = /^blk\.\d+\.ffn_gate_up_exps\.weight$/;
const SHEXP_GATE_RE = /^blk\.\d+\.ffn_gate_inp_shexp\.weight$/;
const BLK_RE = /^blk\.(\d+)\./;
const DEAD_KV_RE = /^blk\.(\d+)\.(attn_k|attn_v|attn_k_norm)\.weight$/;

// ── Helpers ────────────────────────────────────────────────────────────

/** Planned GPU bytes for a tensor after repack/dequant (for the VRAM gate). */
function plannedGPUBytes(t: GGUFTensorInfo): number {
  if (QUANT_GPU_TYPES.has(t.ggmlType)) {
    const layout = GGUF_GPU_LAYOUT[t.ggmlType];
    return (t.elementCount / layout.blockElems) * layout.strideU32 * 4;
  }
  if (t.ggmlType === GGML_TYPES.F16 || t.ggmlType === GGML_TYPES.BF16) {
    return t.elementCount * 4; // dequanted to f32
  }
  return t.byteLength; // F32 raw
}

/** Fetch a tensor's bytes in ≤128 MB chunks, asserting every chunk size. */
async function fetchTensorBytes(url: string, offset: number, byteLength: number): Promise<Uint8Array> {
  const out = new Uint8Array(byteLength);
  let got = 0;
  while (got < byteLength) {
    const n = Math.min(MAX_CHUNK, byteLength - got);
    const part = await fetchRange(url, offset + got, offset + got + n);
    if (part.byteLength !== n) {
      throw new Error(`[GGUF] Short range read at ${offset + got}: got ${part.byteLength}, want ${n}`);
    }
    out.set(new Uint8Array(part), got);
    got += n;
  }
  return out;
}

function uploadToGPU(
  device: GPUDevice,
  data: Uint8Array | Uint32Array | Float32Array,
  label: string,
): GPUBuffer {
  const size = Math.ceil(data.byteLength / 4) * 4;
  const buf = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
  });
  // writeBuffer size must be 4-aligned; all our payloads are (f32/u32 data,
  // and raw F32 tensors are element-multiples of 4 bytes).
  if (data.byteLength % 4 !== 0) {
    throw new Error(`[GGUF] Upload "${label}": byteLength ${data.byteLength} not 4-aligned`);
  }
  device.queue.writeBuffer(buf, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
  return buf;
}

// ── Loader ─────────────────────────────────────────────────────────────

export async function loadGGUFModel(
  device: GPUDevice,
  repo: string,
  filename: string,
  onProgress?: (p: GGUFLoadProgress) => void,
  opts?: { vramBudgetBytes?: number; ramBudgetBytes?: number },
): Promise<LoadedGGUFModel> {
  const t0 = performance.now();
  const url = resolveFileUrl(repo, filename);

  onProgress?.({ message: `Parsing GGUF header: ${filename}` });
  const file = await parseGGUF((s, e) => fetchRange(url, s, e));

  const tied = !file.tensors.has('output.weight');
  // token_embd stays in RAM for CPU row-gather; if tied it ALSO goes to GPU
  // (it doubles as lm_head there). Gemma 4: the PLE table (2+ GB) is also a
  // per-token row-gather → CPU; rope_freqs is read by the descriptor bridge
  // on CPU (applyRopeFreqFactors), never dispatched.
  const cpuNames = new Set([
    'token_embd.weight',
    'per_layer_token_embd.weight',
    'rope_freqs.weight',
  ]);
  const alsoGPU = tied ? new Set(['token_embd.weight']) : new Set<string>();

  // MTP guard: some GGUFs carry next-token-prediction blocks at blk.L,
  // L ≥ block_count — the forward pass never runs them, skip entirely.
  const numLayers = archKV<number>(file, 'block_count');
  const isMTP = (name: string): boolean => {
    const m = BLK_RE.exec(name);
    return m !== null && Number(m[1]) >= numLayers;
  };
  const isCPUOnly = (name: string): boolean => cpuNames.has(name) || SHEXP_GATE_RE.test(name);

  // KV-sharing guard (gemma4): layers ≥ kvFromStart read another layer's KV
  // cache; their own k/v projection weights ship in the GGUF but are dead.
  const sharedKvLayers = archKV<number>(file, 'attention.shared_kv_layers', 0);
  const kvFromStart = numLayers - sharedKvLayers;
  const isDeadKV = (name: string): boolean => {
    if (sharedKvLayers <= 0) return false;
    const m = DEAD_KV_RE.exec(name);
    return m !== null && Number(m[1]) >= kvFromStart;
  };

  // ── VRAM + RAM gates: refuse before downloading anything ──
  let gpuBytesPlanned = 0;
  let cpuBytesPlanned = 0;
  let expertBytes = 0;
  let totalFileBytes = 0;
  for (const t of file.tensors.values()) {
    if (FUSED_EXPERT_RE.test(t.name)) {
      throw new Error(`[GGUF] "${t.name}": fused gate+up expert tensors unsupported — expected split ffn_gate_exps/ffn_up_exps`);
    }
    if (isMTP(t.name) || isDeadKV(t.name)) continue;
    if (EXPERT_RE.test(t.name)) {
      expertBytes += t.byteLength;
      continue; // CPU worker fleet, never downloaded here
    }
    totalFileBytes += t.byteLength;
    if (isCPUOnly(t.name)) {
      cpuBytesPlanned += t.byteLength; // raw blocks kept in RAM
      if (!alsoGPU.has(t.name)) continue;
    }
    gpuBytesPlanned += plannedGPUBytes(t);
  }
  const budget = opts?.vramBudgetBytes ?? 6.5e9;
  if (gpuBytesPlanned > budget) {
    throw new Error(
      `Model needs ~${(gpuBytesPlanned / 1e9).toFixed(1)} GB VRAM for weights but only `
      + `~${(budget / 1e9).toFixed(1)} GB is available on this GPU — refusing to load. `
      + `(${file.tensorCount} tensors, ${(totalFileBytes / 1e9).toFixed(1)} GB on disk)`,
    );
  }
  const ramBudget = opts?.ramBudgetBytes ?? 24e9;
  if (expertBytes + cpuBytesPlanned > ramBudget) {
    throw new Error(
      `CPU-resident weights (MoE experts ~${(expertBytes / 1e9).toFixed(1)} GB + `
      + `embeddings/PLE ~${(cpuBytesPlanned / 1e9).toFixed(1)} GB) exceed the RAM budget of `
      + `~${(ramBudget / 1e9).toFixed(1)} GB — refusing to load. Close other applications or `
      + `use a smaller quantization of this model.`,
    );
  }

  const tensors = new Map<string, GGUFTensorGPU>();
  const cpuTensors = new Map<string, GGUFTensorCPU>();
  const expertTensors = new Map<string, GGUFTensorInfo>();
  let totalGPUBytes = 0;
  let doneBytes = 0;
  let idx = 0;

  for (const t of file.tensors.values()) {
    idx++;
    if (isMTP(t.name) || isDeadKV(t.name)) continue;
    if (EXPERT_RE.test(t.name)) {
      expertTensors.set(t.name, t); // worker fleet fetches these itself
      continue;
    }
    onProgress?.({
      message: `[${idx}/${file.tensorCount}] ${t.name} (${t.typeName}, ${(t.byteLength / 1e6).toFixed(1)} MB)`,
      overallProgress: doneBytes / totalFileBytes,
    });

    const raw = await fetchTensorBytes(url, t.offset, t.byteLength);

    if (isCPUOnly(t.name)) {
      const { blockSize, typeSize } = ggmlTypeTraits(t.ggmlType);
      cpuTensors.set(t.name, {
        data: raw,
        ggmlType: t.ggmlType,
        ne: t.ne,
        rowBytes: (t.ne[0] / blockSize) * typeSize,
      });
    }

    if (!isCPUOnly(t.name) || alsoGPU.has(t.name)) {
      let buf: GPUBuffer;
      let isQuantized = false;
      if (t.ggmlType === GGML_TYPES.F32) {
        let payload: Uint8Array | Float32Array = raw;
        if (/^blk\.\d+\.ssm_a$/.test(t.name)) {
          // llama.cpp's converter stores ssm_a = -exp(A_log) (Qwen3NextModel.
          // modify_tensors); the engine's decay kernel computes exp(-exp(a)*dt)
          // and expects a = A_log. Invert: A_log = log(-ssm_a).
          const f = new Float32Array(raw.buffer, raw.byteOffset, t.elementCount);
          const inv = new Float32Array(t.elementCount);
          for (let i = 0; i < inv.length; i++) {
            if (!(f[i] < 0)) throw new Error(`[GGUF] ${t.name}[${i}] = ${f[i]} — expected negative (-exp(A_log))`);
            inv[i] = Math.log(-f[i]);
          }
          payload = inv;
        }
        buf = uploadToGPU(device, payload, t.name);
      } else if (t.ggmlType === GGML_TYPES.F16) {
        buf = uploadToGPU(device, dequantF16(raw, t.elementCount), t.name);
      } else if (t.ggmlType === GGML_TYPES.BF16) {
        buf = uploadToGPU(device, dequantBF16(raw, t.elementCount), t.name);
      } else if (QUANT_GPU_TYPES.has(t.ggmlType)) {
        buf = uploadToGPU(device, repackGGUFForGPU(t.ggmlType, raw, t.elementCount), t.name);
        isQuantized = true;
      } else {
        throw new Error(`[GGUF] Tensor "${t.name}": unsupported ggml type ${t.typeName} (${t.ggmlType})`);
      }
      tensors.set(t.name, {
        name: t.name,
        buffer: buf,
        shape: t.shape,
        dtype: t.typeName,
        ggmlType: t.ggmlType,
        byteLength: buf.size,
        elementCount: t.elementCount,
        isQuantized,
      });
      totalGPUBytes += buf.size;
    }

    doneBytes += t.byteLength;
  }

  await device.queue.onSubmittedWorkDone();
  const loadTimeMs = performance.now() - t0;
  console.log(
    `[GGUF-Loader] ${repo}/${filename}: ${tensors.size} GPU tensors `
    + `(${(totalGPUBytes / 1e9).toFixed(2)} GB), ${cpuTensors.size} CPU tensors, `
    + `${expertTensors.size} expert tensors deferred (${(expertBytes / 1e9).toFixed(2)} GB for worker fleet), `
    + `${(loadTimeMs / 1000).toFixed(1)}s${tied ? ' (tied embeddings)' : ''}`,
  );

  return {
    repo,
    file,
    url,
    tensors,
    cpuTensors,
    expertTensors,
    expertBytes,
    totalGPUBytes,
    tensorCount: file.tensorCount,
    loadTimeMs,
  };
}
