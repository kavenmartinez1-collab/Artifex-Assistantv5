/**
 * GGUF model loader — streams tensors from a .gguf file into GPU buffers.
 *
 * Placement:
 *   - token_embd.weight → CPU RAM (raw blocks; forward pass row-gathers +
 *     dequants per token — embedCPU precedent, saves ~0.5-2 GB VRAM)
 *   - F32 tensors (norms, biases, ssm_a, conv1d, dt_bias) → f32 buffers
 *   - F16/BF16 tensors → dequant to f32 at load (small tensors only)
 *   - k-quant/legacy/IQ4 matrices (Q2_K…Q6_K, Q4_0/Q5_0/Q8_0, IQ4_NL/IQ4_XS)
 *     → repackGGUFForGPU → u32 buffers, dispatched via shaders/matmul_gguf.wgsl
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
  /** Raw blocks. Empty (length 0) when the tensor is sharded into `parts`. */
  data: Uint8Array;
  /**
   * Row-aligned shards for tensors whose byteLength exceeds the JS
   * ArrayBuffer cap (2^31-1) — e.g. Gemma 4's 2.31 GB PLE table.
   * Row r lives in parts[floor(r / rowsPerPart)] at (r % rowsPerPart) * rowBytes.
   */
  parts?: Uint8Array[];
  rowsPerPart?: number;
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
  /**
   * Multi-token-prediction (MTP) head tensors (blk.L, L ≥ block_count) —
   * uploaded to GPU but kept OUT of the trunk `tensors` map so the decode
   * forward (layers 0..block_count-1) never touches them. The speculative
   * MTP drafter pulls its weights from here.
   */
  mtpTensors: Map<string, GGUFTensorGPU>;
  totalGPUBytes: number;
  tensorCount: number;
  loadTimeMs: number;
}

export interface GGUFLoadProgress {
  message: string;
  overallProgress?: number;
}

const QUANT_GPU_TYPES = new Set<number>([
  GGML_TYPES.Q4_0, GGML_TYPES.Q5_0, GGML_TYPES.Q2_K, GGML_TYPES.Q3_K,
  GGML_TYPES.Q8_0, GGML_TYPES.Q4_K, GGML_TYPES.Q5_K, GGML_TYPES.Q6_K,
  GGML_TYPES.IQ2_XXS, GGML_TYPES.IQ4_NL, GGML_TYPES.IQ4_XS,
  GGML_TYPES.IQ3_XXS, GGML_TYPES.IQ3_S, GGML_TYPES.IQ2_S,
]);

const MAX_CHUNK = 128 * 1024 * 1024;
/** Shard CPU-resident tensors above this size (JS ArrayBuffer cap is 2^31-1). */
const MAX_CPU_PART = 1 << 30;

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

  // MTP guard: Qwen3.6 GGUFs carry `nextn_predict_layers` next-token-prediction
  // blocks at the END of the block range (indices >= block_count - nextn). The
  // trunk forward only runs layers 0..numLayers-1 (matching model-descriptor.ts),
  // so those blocks are dead weight unless an MTP drafter consumes them. Use the
  // SAME trunk count the engine uses so isMTP actually catches them — checking
  // against block_count alone never matched (the MTP index is < block_count).
  const numLayers = archKV<number>(file, 'block_count') - archKV<number>(file, 'nextn_predict_layers', 0);
  const isMTP = (name: string): boolean => {
    const m = BLK_RE.exec(name);
    return m !== null && Number(m[1]) >= numLayers;
  };
  // The MTP / next-token-prediction head (blk.L, L >= block_count) is loaded
  // for a future speculative MTP drafter, but NO forward / decode / draft path
  // reads it today — drafts come from the n-gram drafter (generate.ts). On a
  // VRAM-tight card it is pure dead weight (~0.5 GB for the 27B), so skip it by
  // default. Opt in with ?mtpHead=1 to load it into mtpTensors (e.g. to verify
  // VRAM fit / wire up an MTP drafter); loading it alone does NOT speed decode.
  const LOAD_MTP_HEAD = typeof window !== 'undefined'
    && new URLSearchParams(window.location.search).get('mtpHead') === '1';
  const isCPUOnly = (name: string): boolean => cpuNames.has(name) || SHEXP_GATE_RE.test(name);

  // Vision tower guard: multimodal GGUFs (Ollama-packed Gemma 4, Qwen-VL)
  // carry the ViT under 'v.' and the multimodal projector under 'mm.'.
  // Text generation never dispatches them — skip the VRAM until the vision
  // path (M2) loads them deliberately.
  const isVision = (name: string): boolean => name.startsWith('v.') || name.startsWith('mm.');

  // KV-sharing guard (gemma4): layers ≥ kvFromStart read another layer's KV
  // cache; their own k/v projection weights ship in the GGUF but are dead.
  const sharedKvLayers = archKV<number>(file, 'attention.shared_kv_layers', 0);
  const kvFromStart = numLayers - sharedKvLayers;
  const isDeadKV = (name: string): boolean => {
    if (sharedKvLayers <= 0) return false;
    const m = DEAD_KV_RE.exec(name);
    return m !== null && Number(m[1]) >= kvFromStart;
  };

  // ── Pre-flight quant check — fail fast with guidance, before any download ──
  // The GPU path dequantizes the k-quants (Q2_K…Q6_K), legacy Q4_0/Q5_0/Q8_0,
  // the IQ4 codebook pair (IQ4_NL/IQ4_XS) and the IQ2_XXS/IQ2_S/IQ3_XXS/IQ3_S
  // grid-codebook quants natively. The remaining grid-codebook IQ quants
  // (IQ1/IQ2_XS) and Q4_1/Q5_1 aren't wired, so a model in those would fail
  // tensor-by-tensor mid-load. Catch it up front and tell the user what to
  // download instead.
  const SUPPORTED_GPU = new Set<number>([
    GGML_TYPES.F32, GGML_TYPES.F16, GGML_TYPES.BF16,
    GGML_TYPES.Q4_0, GGML_TYPES.Q5_0, GGML_TYPES.Q2_K, GGML_TYPES.Q3_K,
    GGML_TYPES.Q8_0, GGML_TYPES.Q4_K, GGML_TYPES.Q5_K, GGML_TYPES.Q6_K,
    GGML_TYPES.IQ2_XXS, GGML_TYPES.IQ4_NL, GGML_TYPES.IQ4_XS,
    GGML_TYPES.IQ3_XXS, GGML_TYPES.IQ3_S, GGML_TYPES.IQ2_S,
  ]);
  const unsupportedTypes = new Map<string, number>();
  for (const t of file.tensors.values()) {
    if (isVision(t.name) || isDeadKV(t.name) || EXPERT_RE.test(t.name)) continue;
    if (!SUPPORTED_GPU.has(t.ggmlType)) {
      unsupportedTypes.set(t.typeName, (unsupportedTypes.get(t.typeName) ?? 0) + 1);
    }
  }
  if (unsupportedTypes.size > 0) {
    const list = [...unsupportedTypes.entries()].map(([n, c]) => `${n} (${c} tensors)`).join(', ');
    throw new Error(
      `This GGUF uses quantization the WebGPU engine can't run yet: ${list}. `
      + `Supported: Q2_K, Q3_K, Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS (and F16/F32/BF16). `
      + `Download a K-quant or IQ build instead — e.g. a *-Q4_K_M.gguf or *-IQ4_XS.gguf. `
      + `(IQ1/IQ2_XS grid-codebook quants and Q4_1/Q5_1 aren't supported.)`);
  }

  // ── VRAM + RAM gates: refuse before downloading anything ──
  let gpuBytesPlanned = 0;
  let cpuBytesPlanned = 0;
  let expertBytes = 0;
  let totalFileBytes = 0;
  let visionBytesSkipped = 0;
  for (const t of file.tensors.values()) {
    if (FUSED_EXPERT_RE.test(t.name)) {
      throw new Error(`[GGUF] "${t.name}": fused gate+up expert tensors unsupported — expected split ffn_gate_exps/ffn_up_exps`);
    }
    if (isVision(t.name)) { visionBytesSkipped += t.byteLength; continue; }
    if (isDeadKV(t.name)) continue;
    if (EXPERT_RE.test(t.name)) {
      expertBytes += t.byteLength;
      continue; // CPU worker fleet, never downloaded here
    }
    if (!LOAD_MTP_HEAD && isMTP(t.name)) continue; // dead weight: no path reads it
    totalFileBytes += t.byteLength;
    if (isCPUOnly(t.name)) {
      cpuBytesPlanned += t.byteLength; // raw blocks kept in RAM
      if (!alsoGPU.has(t.name)) continue;
    }
    gpuBytesPlanned += plannedGPUBytes(t);
  }
  // ── Runtime GPU headroom: KV cache + compute buffers ──
  // The gate used to check weights only, so a model could pass, load, say
  // "ready", then OOM on the FIRST forward when the KV cache and activation
  // buffers allocate. Estimate that headroom and include it, so the gate
  // either refuses clearly at load or the model actually runs.
  const MAX_SEQ = 2048;        // KV clamp (MAX_ATTN_SEQ_LEN)
  const headCount = archKV<number>(file, 'attention.head_count', 8);
  const hiddenSize = archKV<number>(file, 'embedding_length', headCount * 128);
  const isHybrid = archKV<number>(file, 'ssm.conv_kernel', 0) > 0;
  // Mirror forward-pass.ts: hybrid models size activation buffers to 64 (their
  // prefill chunk is 16), dense to 512. Keep this estimate in lockstep.
  const MAX_PREFILL = isHybrid ? 64 : 512;
  const fullAttnInterval = archKV<number>(file, 'full_attention_interval', 1);
  let nKV = archKV<number | number[]>(file, 'attention.head_count_kv', headCount);
  if (Array.isArray(nKV)) { const nz = nKV.filter(v => v > 0); nKV = nz.length ? Math.max(...nz) : headCount; }
  const kLen = archKV<number>(file, 'attention.key_length', Math.floor(hiddenSize / headCount));
  const vLen = archKV<number>(file, 'attention.value_length', kLen);
  // Count layers that allocate their OWN KV cache: attention (non-linear)
  // layers, minus the trailing shared-KV layers that reuse another's.
  let attnLayers = 0;
  for (let i = 0; i < numLayers; i++) {
    if (!isHybrid || (i + 1) % fullAttnInterval === 0) attnLayers++;
  }
  const kvCacheLayers = Math.max(0, attnLayers - sharedKvLayers);
  const kvBytes = kvCacheLayers * MAX_SEQ * (nKV as number) * (kLen + vLen) * 4;
  const vocab = (file.tensors.get('token_embd.weight')?.shape?.[0]) ?? 152064;
  const computeBytes = MAX_PREFILL * hiddenSize * 4 * 12 + vocab * 4;  // activations + logits
  const runtimeBytes = kvBytes + computeBytes;

  const budget = opts?.vramBudgetBytes ?? 6.5e9;
  if (gpuBytesPlanned + runtimeBytes > budget) {
    throw new Error(
      `Model needs ~${(gpuBytesPlanned / 1e9).toFixed(1)} GB for weights + `
      + `~${(runtimeBytes / 1e9).toFixed(1)} GB for the KV cache and compute buffers `
      + `(~${((gpuBytesPlanned + runtimeBytes) / 1e9).toFixed(1)} GB total), but only `
      + `~${(budget / 1e9).toFixed(1)} GB is free on this GPU — refusing to load. `
      + `Close other GPU apps to free VRAM, or use a smaller model/quant. `
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
  const mtpTensors = new Map<string, GGUFTensorGPU>();
  const cpuTensors = new Map<string, GGUFTensorCPU>();
  const expertTensors = new Map<string, GGUFTensorInfo>();
  let totalGPUBytes = 0;
  let doneBytes = 0;
  let idx = 0;

  if (visionBytesSkipped > 0) {
    console.log(`[GGUF] Skipping vision tower: ${(visionBytesSkipped / 1e9).toFixed(2)} GB (text-only load)`);
  }

  for (const t of file.tensors.values()) {
    idx++;
    if (isVision(t.name) || isDeadKV(t.name)) continue;
    if (EXPERT_RE.test(t.name)) {
      expertTensors.set(t.name, t); // worker fleet fetches these itself
      continue;
    }
    if (!LOAD_MTP_HEAD && isMTP(t.name)) continue; // dead weight: skip GPU upload
    onProgress?.({
      message: `[${idx}/${file.tensorCount}] ${t.name} (${t.typeName}, ${(t.byteLength / 1e6).toFixed(1)} MB)`,
      overallProgress: doneBytes / totalFileBytes,
    });

    // Oversized CPU-only tensors (> ArrayBuffer cap, e.g. Gemma 4's 2.31 GB
    // PLE table): fetch into row-aligned ≤1 GiB shards instead of one buffer.
    if (isCPUOnly(t.name) && !alsoGPU.has(t.name) && t.byteLength > MAX_CPU_PART) {
      const { blockSize, typeSize } = ggmlTypeTraits(t.ggmlType);
      const rowBytes = (t.ne[0] / blockSize) * typeSize;
      const totalRows = t.byteLength / rowBytes;
      if (!Number.isInteger(totalRows)) {
        throw new Error(`[GGUF] "${t.name}": byteLength ${t.byteLength} not a multiple of rowBytes ${rowBytes}`);
      }
      const rowsPerPart = Math.floor(MAX_CPU_PART / rowBytes);
      const parts: Uint8Array[] = [];
      for (let r0 = 0; r0 < totalRows; r0 += rowsPerPart) {
        const rows = Math.min(rowsPerPart, totalRows - r0);
        parts.push(await fetchTensorBytes(url, t.offset + r0 * rowBytes, rows * rowBytes));
      }
      cpuTensors.set(t.name, {
        data: new Uint8Array(0),
        parts,
        rowsPerPart,
        ggmlType: t.ggmlType,
        ne: t.ne,
        rowBytes,
      });
      doneBytes += t.byteLength;
      continue;
    }

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
      (isMTP(t.name) ? mtpTensors : tensors).set(t.name, {
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
    + `(${(totalGPUBytes / 1e9).toFixed(2)} GB)`
    + (mtpTensors.size > 0 ? ` + ${mtpTensors.size} MTP-head tensors` : '')
    + `, ${cpuTensors.size} CPU tensors, `
    + `${expertTensors.size} expert tensors deferred (${(expertBytes / 1e9).toFixed(2)} GB for worker fleet), `
    + `${(loadTimeMs / 1000).toFixed(1)}s${tied ? ' (tied embeddings)' : ''}`,
  );

  // GEMV lever 4 Phase 0: per-quant-type inventory on the console (the
  // per-tensor type only goes to the progress callback, invisible to
  // headless drivers), plus which layers carry off-majority types — needed
  // to correlate per-type GPU timing (gguf_* categories) with what's loaded.
  const byType = new Map<string, { n: number; mb: number }>();
  for (const t of tensors.values()) {
    const e = byType.get(t.dtype) ?? { n: 0, mb: 0 };
    e.n++; e.mb += t.byteLength / 1e6;
    byType.set(t.dtype, e);
  }
  console.log('[GGUF-Loader] GPU tensors by type: '
    + [...byType.entries()].sort((a, b) => b[1].mb - a[1].mb)
      .map(([k, v]) => `${k} x${v.n} (${v.mb.toFixed(0)} MB)`).join(', '));
  const kinds = new Map<string, Map<string, number[]>>();
  for (const t of tensors.values()) {
    const m = /^blk\.(\d+)\.(.+)$/.exec(t.name);
    if (!m) continue;
    const k = kinds.get(m[2]) ?? new Map<string, number[]>();
    kinds.set(m[2], k);
    let arr = k.get(t.dtype);
    if (!arr) { arr = []; k.set(t.dtype, arr); }
    arr.push(+m[1]);
  }
  for (const [kind, types] of kinds) {
    if (types.size < 2) continue;
    console.log(`[GGUF-Loader] mixed types for ${kind}: `
      + [...types.entries()].map(([ty, ls]) => {
        ls.sort((a, b) => a - b);
        const span = ls.length <= 12 ? ls.join(',') : `${ls.length} layers L${ls[0]}..L${ls[ls.length - 1]}`;
        return `${ty}@[${span}]`;
      }).join(' | '));
  }

  return {
    repo,
    file,
    url,
    tensors,
    mtpTensors,
    cpuTensors,
    expertTensors,
    expertBytes,
    totalGPUBytes,
    tensorCount: file.tensorCount,
    loadTimeMs,
  };
}
