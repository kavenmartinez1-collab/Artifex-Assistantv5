/**
 * Weight Loader — Downloads SafeTensors from HuggingFace and uploads to GPU buffers.
 *
 * Flow:
 *   1. Discover shards from HuggingFace repo
 *   2. For each shard: download → parse SafeTensors header → extract tensors
 *   3. Convert tensors to float32 (from F16/BF16) and upload to GPU storage buffers
 *   4. Cache downloaded files in browser for instant reload
 *
 * The 2GB WebGPU buffer limit means each tensor gets its own GPU buffer.
 * For a 9B model with ~200 tensors, this creates ~200 buffers.
 */

import {
  parseHeader, extractTensorData, tensorToFloat32, tensorToTypedArray,
  formatBytes, summarizeHeader,
  type SafeTensorsHeader, type TensorInfo,
} from './safetensors';

import {
  discoverShards, downloadFile, downloadShardHeader, fetchModelConfig, fetchRange,
  type ShardInfo, type DownloadProgress, type HFModelConfig,
} from './hf-hub';

import { getCache, putCache, hasCache } from './cache';
import { reportMetric } from '../utils/metrics';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GPUTensor {
  name: string;
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;        // original dtype (F16, BF16, I32, etc.)
  byteLength: number;   // size on GPU
  elementCount: number;
  /** True for GPTQ packed weights (.qweight, .qzeros, .scales) */
  isQuantized?: boolean;
}

export interface LoadedModel {
  repo: string;
  config: HFModelConfig;
  tensors: Map<string, GPUTensor>;
  totalGPUBytes: number;
  tensorCount: number;
  loadTimeMs: number;
}

export interface LoadProgress {
  phase: 'discovering' | 'downloading' | 'uploading' | 'complete' | 'error';
  message: string;
  shard?: number;
  totalShards?: number;
  shardProgress?: number;   // 0-1 for current shard
  overallProgress?: number; // 0-1 for total
  tensorsLoaded?: number;
  totalTensors?: number;
}

type ProgressCallback = (progress: LoadProgress) => void;

// ─── Weight Loader ───────────────────────────────────────────────────────────

/**
 * Load a model's weights from HuggingFace into GPU buffers.
 *
 * @param device - WebGPU device to create buffers on
 * @param repo - HuggingFace repo ID (e.g. "Qwen/Qwen3.5-0.6B")
 * @param onProgress - Progress callback for UI updates
 * @returns LoadedModel with all tensors as GPU buffers
 */
export async function loadModel(
  device: GPUDevice,
  repo: string,
  onProgress?: ProgressCallback,
  /** Keep BF16 tensors in native format (halves VRAM, uses BF16 matmul kernels) */
  keepBF16 = false,
): Promise<LoadedModel> {
  const startTime = performance.now();

  const progress = (p: Partial<LoadProgress>) => {
    if (onProgress) {
      onProgress({
        phase: 'downloading',
        message: '',
        ...p,
      } as LoadProgress);
    }
  };

  // ── Step 1: Discover model ──────────────────────────────────────────

  progress({ phase: 'discovering', message: `Fetching model info for ${repo}...` });

  const [config, shards] = await Promise.all([
    fetchModelConfig(repo),
    discoverShards(repo),
  ]);

  const totalDownloadSize = shards.reduce((s, sh) => s + sh.size, 0);

  progress({
    phase: 'discovering',
    message: `Found ${shards.length} shard(s), ${formatBytes(totalDownloadSize)} total`,
    totalShards: shards.length,
  });

  await reportMetric('model-discover', {
    repo,
    shards: shards.length,
    totalSize: totalDownloadSize,
    modelType: config.model_type,
    hiddenSize: config.hidden_size,
    numLayers: config.num_hidden_layers,
  });

  // ── Step 2: Download and parse each shard ───────────────────────────

  const allTensors = new Map<string, GPUTensor>();
  let totalGPUBytes = 0;
  let tensorsProcessed = 0;
  let bytesDownloadedTotal = 0;

  for (let shardIdx = 0; shardIdx < shards.length; shardIdx++) {
    const shard = shards[shardIdx];
    const cacheKey = `${repo}/${shard.filename}`;

    progress({
      phase: 'downloading',
      message: `Shard ${shardIdx + 1}/${shards.length}: ${shard.filename}`,
      shard: shardIdx + 1,
      totalShards: shards.length,
      shardProgress: 0,
      overallProgress: bytesDownloadedTotal / totalDownloadSize,
    });

    // ── Step 3: Download header first (small — a few KB) ──────────────

    const headerData = await downloadShardHeader(shard.url);
    const header = parseHeader(headerData);
    console.log(`[WeightLoader] Shard ${shardIdx + 1}: ${summarizeHeader(header)}`);

    // Decide: download whole file (small shards) or stream per-tensor (large shards)
    const MAX_FULL_DOWNLOAD = 2 * 1024 * 1024 * 1024; // 2 GB threshold
    const useStreaming = shard.size > MAX_FULL_DOWNLOAD;

    if (useStreaming) {
      console.log(`[WeightLoader] Shard ${shardIdx + 1}: parallel chunked download for ${header.tensors.size} tensors (${formatBytes(shard.size)})`);
    }

    // For small shards, download the whole file at once (faster, cacheable)
    let shardData: ArrayBuffer | null = null;
    // For large shards, download in ~512MB chunks with parallel prefetch
    const CHUNK_SIZE = 512 * 1024 * 1024; // 512 MB
    const PARALLEL_CHUNKS = 4; // download 4 chunks at once
    const chunkCache = new Map<number, ArrayBuffer | Promise<ArrayBuffer>>();
    const totalChunks = Math.ceil(shard.size / CHUNK_SIZE);

    // Helper: fetch a chunk (checks browser cache first, then downloads)
    async function fetchChunk(idx: number): Promise<ArrayBuffer> {
      const chunkKey = `${cacheKey}/chunk-${idx}`;
      // Try browser cache first
      const cachedChunk = await getCache(chunkKey);
      if (cachedChunk) return cachedChunk;
      // Download and cache for next time
      const start = idx * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, shard.size);
      const data = await fetchRange(shard.url, start, end);
      putCache(chunkKey, data).catch(() => {}); // fire-and-forget cache write
      return data;
    }

    // Helper: get a chunk (from cache or download), returns ArrayBuffer
    // Evicts failed prefetch promises so they can be re-attempted
    async function getChunk(idx: number): Promise<ArrayBuffer> {
      if (idx >= totalChunks) return new ArrayBuffer(0);
      const cached = chunkCache.get(idx);
      if (cached instanceof ArrayBuffer) return cached;
      if (cached instanceof Promise) {
        try {
          return await cached;
        } catch {
          // Prefetch failed after retries — evict and re-attempt below
          chunkCache.delete(idx);
        }
      }
      // Not in cache or evicted — start fresh download
      const promise = fetchChunk(idx);
      chunkCache.set(idx, promise);
      const data = await promise;
      chunkCache.set(idx, data);
      return data;
    }

    // Pre-fetch first N chunks in parallel for streaming mode
    if (useStreaming) {
      for (let i = 0; i < Math.min(PARALLEL_CHUNKS, totalChunks); i++) {
        const promise = fetchChunk(i);
        chunkCache.set(i, promise);
        promise.then(
          data => { chunkCache.set(i, data); },
          () => { chunkCache.delete(i); }, // evict on failure — getChunk will retry
        );
      }
      progress({ phase: 'downloading',
        message: `Shard ${shardIdx + 1}: downloading ${Math.min(PARALLEL_CHUNKS, totalChunks)} chunks in parallel...`,
        shard: shardIdx + 1, totalShards: shards.length, shardProgress: 0 });
    }

    if (!useStreaming) {
      const cached = await hasCache(cacheKey);
      if (cached) {
        progress({ phase: 'downloading', message: `Shard ${shardIdx + 1}: loading from cache...`,
          shard: shardIdx + 1, totalShards: shards.length, shardProgress: 1,
          overallProgress: (bytesDownloadedTotal + shard.size) / totalDownloadSize });
        shardData = await getCache(cacheKey) as ArrayBuffer;
      } else {
        shardData = await downloadFile(shard.url, (downloaded, total) => {
          progress({ phase: 'downloading',
            message: `Shard ${shardIdx + 1}: ${formatBytes(downloaded)} / ${formatBytes(total)}`,
            shard: shardIdx + 1, totalShards: shards.length,
            shardProgress: total > 0 ? downloaded / total : 0,
            overallProgress: (bytesDownloadedTotal + downloaded) / totalDownloadSize });
        });
        await putCache(cacheKey, shardData);
      }
    }

    bytesDownloadedTotal += shard.size;

    // ── Step 4: Extract/download tensors and upload to GPU ────────────

    const t_upload_start = performance.now();

    progress({ phase: 'uploading',
      message: `Uploading ${header.tensors.size} tensors to GPU...`,
      shard: shardIdx + 1, totalShards: shards.length,
      tensorsLoaded: tensorsProcessed });

    let tensorIdx = 0;
    for (const [name, tensorInfo] of header.tensors) {
      // Skip g_idx tensors — unused by our matmul_q4 shader
      if (name.endsWith('.g_idx')) {
        tensorIdx++;
        continue;
      }
      // Get raw tensor bytes — either from full shard or via chunked range requests
      let rawData: ArrayBuffer;
      if (shardData) {
        rawData = extractTensorData(shardData, tensorInfo, header.headerByteLength);
      } else {
        // Chunked streaming: download 512MB chunks on demand, reuse for adjacent tensors
        const dataStart = header.headerByteLength + tensorInfo.dataOffsets[0];
        const dataEnd = header.headerByteLength + tensorInfo.dataOffsets[1];
        const tensorSize = dataEnd - dataStart;

        if (tensorSize > CHUNK_SIZE) {
          rawData = await fetchRange(shard.url, dataStart, dataEnd);
        } else {
          const chunkIdx = Math.floor(dataStart / CHUNK_SIZE);

          // Prefetch upcoming chunks in parallel
          for (let ahead = 1; ahead <= PARALLEL_CHUNKS; ahead++) {
            const futureIdx = chunkIdx + ahead;
            if (futureIdx < totalChunks && !chunkCache.has(futureIdx)) {
              const p = fetchChunk(futureIdx);
              chunkCache.set(futureIdx, p);
              p.then(
                data => { chunkCache.set(futureIdx, data); },
                () => { chunkCache.delete(futureIdx); },
              );
            }
          }

          // Get current chunk (may already be downloaded via prefetch)
          const chunk = await getChunk(chunkIdx);

          progress({ phase: 'downloading',
            message: `Shard ${shardIdx + 1}: chunk ${chunkIdx + 1}/${totalChunks}, tensor ${tensorIdx}/${header.tensors.size}`,
            shard: shardIdx + 1, totalShards: shards.length,
            shardProgress: (chunkIdx + 1) / totalChunks,
            overallProgress: bytesDownloadedTotal / totalDownloadSize });

          // Free old chunks to limit memory (~1 GB max)
          for (const [k, v] of chunkCache) {
            if (k < chunkIdx - 1 && v instanceof ArrayBuffer) chunkCache.delete(k);
          }

          const offsetInChunk = dataStart - chunkIdx * CHUNK_SIZE;
          rawData = chunk.slice(offsetInChunk, offsetInChunk + tensorSize);
        }
        tensorIdx++;
      }

      // GPTQ tensors stay in native format for GPU-side dequantization
      const isGPTQ = name.endsWith('.qweight') || name.endsWith('.qzeros')
        || name.endsWith('.scales') || name.endsWith('.g_idx');

      let gpuData: ArrayBufferView;
      const f32Size = tensorInfo.elementCount * 4;
      const exceedsBufferLimit = f32Size > 1.9 * 1024 * 1024 * 1024;

      if (isGPTQ) {
        gpuData = tensorToTypedArray(rawData, tensorInfo.dtype);
      } else if (keepBF16 && tensorInfo.dtype === 'BF16' && tensorInfo.byteLength > 1024 * 1024) {
        // Keep large BF16 tensors (projections) in native format — halves VRAM
        // Small tensors (norms, biases) always convert to f32 for compatibility
        gpuData = tensorToTypedArray(rawData, tensorInfo.dtype);
      } else if (exceedsBufferLimit && (tensorInfo.dtype === 'F16' || tensorInfo.dtype === 'BF16')) {
        console.log(`[WeightLoader] Keeping ${name} at ${tensorInfo.dtype} (f32 would be ${(f32Size / (1024**3)).toFixed(1)} GB)`);
        gpuData = tensorToTypedArray(rawData, tensorInfo.dtype);
      } else {
        gpuData = tensorToFloat32(rawData, tensorInfo.dtype);
      }

      // Create GPU buffer and upload
      const gpuBuffer = device.createBuffer({
        size: gpuData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: name,
        mappedAtCreation: true,
      });

      new Uint8Array(gpuBuffer.getMappedRange()).set(
        new Uint8Array(gpuData.buffer, gpuData.byteOffset, gpuData.byteLength)
      );
      gpuBuffer.unmap();

      allTensors.set(name, {
        name, buffer: gpuBuffer, shape: tensorInfo.shape,
        dtype: tensorInfo.dtype, byteLength: gpuData.byteLength,
        elementCount: tensorInfo.elementCount, isQuantized: isGPTQ,
      });

      totalGPUBytes += gpuData.byteLength;
      tensorsProcessed++;
    }

    const t_upload_end = performance.now();
    console.log(`[Perf] Shard ${shardIdx + 1}/${shards.length}: process+upload ${(t_upload_end - t_upload_start).toFixed(0)}ms (${header.tensors.size} tensors)`);
  }

  const loadTimeMs = performance.now() - startTime;

  progress({
    phase: 'complete',
    message: `Loaded ${tensorsProcessed} tensors (${formatBytes(totalGPUBytes)} GPU memory) in ${(loadTimeMs / 1000).toFixed(1)}s`,
    overallProgress: 1,
    tensorsLoaded: tensorsProcessed,
    totalTensors: tensorsProcessed,
  });

  await reportMetric('model-loaded', {
    repo,
    tensorCount: tensorsProcessed,
    gpuBytes: totalGPUBytes,
    gpuMB: Math.round(totalGPUBytes / (1024 * 1024)),
    loadTimeMs: Math.round(loadTimeMs),
    fromCache: false, // TODO: track cache hits
  });

  return {
    repo,
    config,
    tensors: allTensors,
    totalGPUBytes,
    tensorCount: tensorsProcessed,
    loadTimeMs,
  };
}

/**
 * CPU-side GPTQ dequantization — converts INT4 packed weights back to f32.
 * Used for mixed-precision: dequant SSM-critical weights to full precision
 * while keeping attention/FFN weights in INT4 for VRAM savings.
 *
 * @returns Float32Array of shape [N, K] (HF weight format: [out_features, in_features])
 */
export function dequantGPTQ(
  qweight: Uint32Array | Int32Array,
  scales: Uint16Array,
  qzeros: Uint32Array | Int32Array,
  K: number,
  N: number,
  groupSize: number,
): Float32Array {
  const result = new Float32Array(N * K);
  const scalesF32 = new Float32Array(scales.length);

  // Convert F16 scales to F32
  for (let i = 0; i < scales.length; i++) {
    const bits = scales[i];
    const buf = new ArrayBuffer(2);
    new Uint16Array(buf)[0] = bits;
    scalesF32[i] = new Float64Array(new Float32Array([new DataView(buf).getFloat16(0, true)])).length > 0
      ? (() => {
          // Manual F16 decode
          const sign = (bits >> 15) & 1;
          const exp = (bits >> 10) & 0x1F;
          const frac = bits & 0x3FF;
          if (exp === 0) return frac === 0 ? 0 : (frac / 1024) * Math.pow(2, -14) * (sign ? -1 : 1);
          if (exp === 31) return sign ? -Infinity : Infinity;
          return (1 + frac / 1024) * Math.pow(2, exp - 15) * (sign ? -1 : 1);
        })()
      : 0;
  }

  // Manual F16 decode (simpler version)
  function f16ToF32(bits: number): number {
    const sign = (bits >> 15) & 1;
    const exp = (bits >> 10) & 0x1F;
    const frac = bits & 0x3FF;
    if (exp === 0) return frac === 0 ? 0 : (frac / 1024) * Math.pow(2, -14) * (sign ? -1 : 1);
    if (exp === 31) return sign ? -Infinity : Infinity;
    return (1 + frac / 1024) * Math.pow(2, exp - 15) * (sign ? -1 : 1);
  }

  const numGroups = Math.ceil(K / groupSize);
  const zerosPerRow = Math.ceil(N / 8);

  for (let k = 0; k < K; k++) {
    const group = Math.floor(k / groupSize);
    const packedRow = Math.floor(k / 8);
    const nibbleInPacked = k % 8;

    for (let n = 0; n < N; n++) {
      // Extract 4-bit weight
      const packed = qweight[packedRow * N + n];
      const q4 = (packed >> (nibbleInPacked * 4)) & 0xF;

      // Get scale (F16 stored as u16)
      const scaleIdx = group * N + n;
      const scale = f16ToF32(scales[scaleIdx]);

      // Get zero point (packed INT4)
      const zeroPacked = qzeros[group * zerosPerRow + Math.floor(n / 8)];
      const zeroNibble = n % 8;
      const zero = (zeroPacked >> (zeroNibble * 4)) & 0xF;

      // Dequant: (q4 - zero) * scale
      result[n * K + k] = (q4 - zero) * scale;
    }
  }

  console.log(`[Dequant] GPTQ → f32: [${N}, ${K}] (${(result.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  return result;
}

/**
 * Unload a model — destroy all GPU buffers and free VRAM.
 */
export function unloadModel(model: LoadedModel): void {
  for (const tensor of model.tensors.values()) {
    tensor.buffer.destroy();
  }
  model.tensors.clear();
  console.log(`[WeightLoader] Unloaded ${model.repo} — freed ${formatBytes(model.totalGPUBytes)}`);
}

/**
 * Preview a model without downloading weights.
 * Fetches only headers (a few KB per shard) to report tensor info.
 */
export async function previewModel(
  repo: string,
  onProgress?: ProgressCallback,
): Promise<{ config: HFModelConfig; tensorCount: number; totalBytes: number; dtypes: string[] }> {
  const progress = (msg: string) => {
    if (onProgress) {
      onProgress({ phase: 'discovering', message: msg });
    }
  };

  progress(`Fetching ${repo} info...`);
  const [config, shards] = await Promise.all([
    fetchModelConfig(repo),
    discoverShards(repo),
  ]);

  let tensorCount = 0;
  let totalBytes = 0;
  const dtypes = new Set<string>();

  for (let i = 0; i < shards.length; i++) {
    progress(`Reading shard ${i + 1}/${shards.length} header...`);
    const headerData = await downloadShardHeader(shards[i].url);
    const header = parseHeader(headerData);

    for (const t of header.tensors.values()) {
      tensorCount++;
      totalBytes += t.byteLength;
      dtypes.add(t.dtype);
    }
  }

  return { config, tensorCount, totalBytes, dtypes: [...dtypes] };
}
