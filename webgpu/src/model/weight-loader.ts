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
import { readBuffer } from '../engine/buffers';

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
  /**
   * For GPTQ `*.g_idx` tensors only: true iff the values are trivially
   * `floor(k / group_size)` for some detected group_size. Many GPTQ exports
   * ship an identity g_idx for tooling compatibility even when they didn't
   * reorder columns (desc_act=false). When this is true the GEMV fast-path
   * can skip the per-K g_idx VRAM read and compute group_id in registers.
   */
  isTrivialGIdx?: boolean;
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

/**
 * Detect whether a GPTQ `g_idx` tensor is the trivial identity mapping
 * (i.e. g_idx[k] == floor(k / group_size) for some power-of-two group_size).
 *
 * The group_size is recovered empirically: it's the first k where g_idx[k]
 * transitions from 0 → 1. Once known we verify every remaining entry.
 * Returns false immediately if any divergence is found.
 *
 * Accepts a raw ArrayBuffer holding i32 little-endian values (safetensors
 * always stores I32 as LE on disk).
 */
function isTrivialGIdxBuffer(raw: ArrayBuffer, length: number): boolean {
  if (length < 2) return false;
  const view = new Int32Array(raw, 0, length);
  if (view[0] !== 0) return false;

  // Find first index where value changes — that's the group_size.
  let gs = -1;
  for (let k = 1; k < length; k++) {
    if (view[k] !== 0) {
      if (view[k] !== 1) return false; // non-monotone or jumped past 1
      gs = k;
      break;
    }
  }
  if (gs <= 0) return false; // all zeros (degenerate) — treat as non-trivial
  // Group size must divide length (GPTQ pads K to a multiple of gs).
  if (length % gs !== 0) return false;

  // Verify the remainder matches floor(k / gs). Tight loop — ~16 KB tensor
  // typical for K=4096, so this is negligible (<1ms).
  for (let k = 0; k < length; k++) {
    if (view[k] !== Math.floor(k / gs)) return false;
  }
  return true;
}

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
  /** GPTQ weight bases to dequant to BF16 (for mixed-precision SSM). E.g., 'model.layers.0.linear_attn.in_proj_qkv' */
  dequantToBF16?: Set<string>,
  /** Max extra VRAM bytes dequant may add (BF16 overhead minus Q4 replaced). 0 = no dequant budget. */
  maxDequantOverhead = 0,
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

  // Mixed-precision GPTQ: accumulate raw CPU data for triplets to dequant to BF16
  const dequantAccum = new Map<string, {
    qweight?: { data: ArrayBuffer; shape: number[] };
    scales?: { data: ArrayBuffer; shape: number[] };
    qzeros?: { data: ArrayBuffer; shape: number[] };
  }>();
  const dequantSet = dequantToBF16 ?? new Set<string>();
  let dequantDisabled = false;
  let dequantedCount = 0;
  if (dequantSet.size > 0) {
    console.log(`[WeightLoader] Mixed-precision: ${dequantSet.size} projections will be dequanted to BF16`);
  }
  let bytesDownloadedTotal = 0;

  const FLUSH_THRESHOLD = 512 * 1024 * 1024; // 512 MB
  let bytesSinceFlush = 0;
  let dequantOverheadUsed = 0;

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
      if (cachedChunk) {
        console.log(`[WeightLoader] Chunk ${idx}: cache hit (${(cachedChunk.byteLength / 1024 / 1024).toFixed(0)} MB)`);
        return cachedChunk;
      }
      // Download and cache for next time
      const start = idx * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, shard.size);
      console.log(`[WeightLoader] Chunk ${idx}: downloading bytes ${start}-${end} from ${shard.filename}`);
      try {
        const data = await fetchRange(shard.url, start, end);
        console.log(`[WeightLoader] Chunk ${idx}: downloaded ${(data.byteLength / 1024 / 1024).toFixed(0)} MB`);
        putCache(chunkKey, data).catch(() => {}); // fire-and-forget cache write
        return data;
      } catch (err) {
        console.error(`[WeightLoader] Chunk ${idx} FAILED:`, err);
        throw err;
      }
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
      // Vision tower / MTP guard: multimodal checkpoints (Qwen-VL) carry the
      // ViT under model.visual.* and Qwen3.6 ships mtp.* next-token blocks —
      // text generation never dispatches either. Skip the VRAM until the
      // vision path (M2) loads them deliberately.
      if (name.startsWith('model.visual.') || name.startsWith('visual.')
          || name.startsWith('vision_tower.') || name.startsWith('mtp.')) {
        continue;
      }
      // g_idx tensors are now loaded for actorder GPTQ support
      // Get raw tensor bytes — either from full shard or via chunked range requests
      let rawData: ArrayBuffer;
      if (shardData) {
        rawData = extractTensorData(shardData, tensorInfo, header.headerByteLength);
      } else {
        // Chunked streaming: download 512MB chunks on demand, reuse for adjacent tensors
        const dataStart = header.headerByteLength + tensorInfo.dataOffsets[0];
        const dataEnd = header.headerByteLength + tensorInfo.dataOffsets[1];
        const tensorSize = dataEnd - dataStart;

        // Detect oversized BF16/F16 tensors that can't fit in a single GPU buffer.
        // Split into multiple GPU buffers (lossless) — each under maxBufferSize.
        const isEmbedHere = name.endsWith('embed_tokens.weight');
        const isLmHeadHere = name === 'lm_head.weight';
        const isOversized = (isEmbedHere || isLmHeadHere)
          && (tensorInfo.dtype === 'BF16' || tensorInfo.dtype === 'F16')
          && tensorSize > device.limits.maxBufferSize;

        if (isOversized && isEmbedHere) {
          // ── CPU-side BF16 embedding (lossless, saves VRAM) ─────────────
          // Embed lookup is O(H) per token — trivial on CPU. Keeping it in
          // JS memory saves ~2.4 GB VRAM for models with 248K vocab.
          const [vocabSize, hiddenSize] = tensorInfo.shape;
          const rowBytes = hiddenSize * 2;
          const maxPartBytes = 1024 * 1024 * 1024; // 1 GB per JS array
          const rowsPerPart = Math.floor(maxPartBytes / rowBytes);
          const numParts = Math.ceil(vocabSize / rowsPerPart);
          const splitPoint = Math.min(rowsPerPart, vocabSize);
          const baseName = name.replace('.weight', '');

          console.log(
            `[WeightLoader] CPU-side BF16 embed: ${name} [${vocabSize}, ${hiddenSize}] ${tensorInfo.dtype} `
            + `→ ${numParts} JS arrays, splitPoint=${splitPoint} (saves ${(tensorSize / 1024 / 1024).toFixed(0)} MB VRAM)`
          );

          const cpuParts: Uint8Array[] = [];
          let rowsLoaded = 0;
          for (let part = 0; part < numParts; part++) {
            const partRows = Math.min(rowsPerPart, vocabSize - rowsLoaded);
            const partBytes = partRows * rowBytes;
            const partStart = dataStart + rowsLoaded * rowBytes;
            const partEnd = partStart + partBytes;

            const partData = new Uint8Array(partBytes);
            let written = 0;
            for (let off = partStart; off < partEnd; off += CHUNK_SIZE) {
              const end = Math.min(off + CHUNK_SIZE, partEnd);
              const chunk = await fetchRange(shard.url, off, end);
              partData.set(new Uint8Array(chunk), written);
              written += chunk.byteLength;
            }
            cpuParts.push(partData);
            rowsLoaded += partRows;

            progress({
              phase: 'uploading',
              message: `Loading embed to CPU: part ${part + 1}/${numParts} (${(partBytes / 1024 / 1024).toFixed(0)} MB)`,
              shard: shardIdx + 1, totalShards: shards.length,
              shardProgress: rowsLoaded / vocabSize,
            });
          }

          allTensors.set(`${baseName}.cpu_embed`, {
            name: `${baseName}.cpu_embed`,
            buffer: device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM, label: 'cpu-embed-marker' }),
            shape: [vocabSize, hiddenSize],
            dtype: tensorInfo.dtype,
            byteLength: 0, elementCount: 0,
            cpuEmbedData: { parts: cpuParts, splitPoint, isBF16: tensorInfo.dtype === 'BF16' },
          } as any);

          tensorsProcessed++;
          console.log(
            `[WeightLoader] CPU embed done: ${baseName} — ${numParts} parts, `
            + `${rowsLoaded} rows in JS memory (0 MB GPU)`
          );
          tensorIdx++;
          continue;
        } else if (isOversized && isLmHeadHere) {
          // ── CPU-side BF16 lm_head (saves VRAM) ────────────────────────
          // When total model weights exceed Chrome's GPU allocation limit,
          // keeping lm_head on CPU saves ~2.4 GB. The forward pass reads
          // the hidden state from GPU and computes logits via CPU matmul.
          const [vocabSize, hiddenSize] = tensorInfo.shape;
          const rowBytes = hiddenSize * 2;
          const maxPartBytes = 1024 * 1024 * 1024;
          const rowsPerPart = Math.floor(maxPartBytes / rowBytes);
          const numParts = Math.ceil(vocabSize / rowsPerPart);
          const splitPoint = Math.min(rowsPerPart, vocabSize);
          const baseName = name.replace('.weight', '');

          console.log(
            `[WeightLoader] CPU-side BF16 lm_head: ${name} [${vocabSize}, ${hiddenSize}] ${tensorInfo.dtype} `
            + `→ ${numParts} JS arrays, splitPoint=${splitPoint} (saves ${(tensorSize / 1024 / 1024).toFixed(0)} MB VRAM)`
          );

          const cpuParts: Uint8Array[] = [];
          let rowsLoaded = 0;
          for (let part = 0; part < numParts; part++) {
            const partRows = Math.min(rowsPerPart, vocabSize - rowsLoaded);
            const partBytes = partRows * rowBytes;
            const partStart = dataStart + rowsLoaded * rowBytes;
            const partEnd = partStart + partBytes;

            const partData = new Uint8Array(partBytes);
            let written = 0;
            for (let off = partStart; off < partEnd; off += CHUNK_SIZE) {
              const end = Math.min(off + CHUNK_SIZE, partEnd);
              const chunk = await fetchRange(shard.url, off, end);
              partData.set(new Uint8Array(chunk), written);
              written += chunk.byteLength;
            }
            cpuParts.push(partData);
            rowsLoaded += partRows;

            progress({
              phase: 'uploading',
              message: `Loading lm_head to CPU: part ${part + 1}/${numParts} (${(partBytes / 1024 / 1024).toFixed(0)} MB)`,
              shard: shardIdx + 1, totalShards: shards.length,
              shardProgress: rowsLoaded / vocabSize,
            });
          }

          allTensors.set(`${baseName}.cpu_lm_head`, {
            name: `${baseName}.cpu_lm_head`,
            buffer: device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM, label: 'cpu-lm-head-marker' }),
            shape: [vocabSize, hiddenSize],
            dtype: tensorInfo.dtype,
            byteLength: 0, elementCount: 0,
            cpuLmHeadData: { parts: cpuParts, splitPoint, isBF16: tensorInfo.dtype === 'BF16' },
          } as any);

          tensorsProcessed++;
          console.log(
            `[WeightLoader] CPU lm_head done: ${baseName} — ${numParts} parts, `
            + `${rowsLoaded} rows in JS memory (0 MB GPU)`
          );
          tensorIdx++;
          continue;
        } else if (tensorSize > CHUNK_SIZE) {
          // Large tensor: fetch in chunks and stitch (single fetch would OOM the browser)
          console.log(`[WeightLoader] Large tensor ${name}: ${(tensorSize / 1024 / 1024).toFixed(0)} MB — chunked fetch`);
          const stitched = new Uint8Array(tensorSize);
          let written = 0;
          for (let off = dataStart; off < dataEnd; off += CHUNK_SIZE) {
            const chunkEnd = Math.min(off + CHUNK_SIZE, dataEnd);
            const chunk = await fetchRange(shard.url, off, chunkEnd);
            stitched.set(new Uint8Array(chunk), written);
            written += chunk.byteLength;
          }
          rawData = stitched.buffer;
        } else {
          const firstChunkIdx = Math.floor(dataStart / CHUNK_SIZE);
          const lastChunkIdx = Math.floor((dataEnd - 1) / CHUNK_SIZE);

          // Prefetch upcoming chunks in parallel (ahead of the last chunk we need)
          for (let ahead = 1; ahead <= PARALLEL_CHUNKS; ahead++) {
            const futureIdx = lastChunkIdx + ahead;
            if (futureIdx < totalChunks && !chunkCache.has(futureIdx)) {
              const p = fetchChunk(futureIdx);
              chunkCache.set(futureIdx, p);
              p.then(
                data => { chunkCache.set(futureIdx, data); },
                () => { chunkCache.delete(futureIdx); },
              );
            }
          }

          progress({ phase: 'downloading',
            message: `Shard ${shardIdx + 1}: chunk ${firstChunkIdx + 1}/${totalChunks}, tensor ${tensorIdx}/${header.tensors.size}`,
            shard: shardIdx + 1, totalShards: shards.length,
            shardProgress: (firstChunkIdx + 1) / totalChunks,
            overallProgress: bytesDownloadedTotal / totalDownloadSize });

          if (firstChunkIdx === lastChunkIdx) {
            // Fast path: tensor fits within a single chunk
            const chunk = await getChunk(firstChunkIdx);
            const offsetInChunk = dataStart - firstChunkIdx * CHUNK_SIZE;
            rawData = chunk.slice(offsetInChunk, offsetInChunk + tensorSize);
          } else {
            // Cross-chunk path: tensor straddles one or more chunk boundaries.
            // Stitch bytes from each chunk it spans. Without this, chunk.slice()
            // silently truncates at the chunk end — producing undersized buffers
            // and silently-wrong outputs (see L21 audit investigation).
            const stitched = new Uint8Array(tensorSize);
            let written = 0;
            for (let idx = firstChunkIdx; idx <= lastChunkIdx; idx++) {
              const chunk = await getChunk(idx);
              const chunkStart = idx * CHUNK_SIZE;
              const readStart = Math.max(dataStart, chunkStart) - chunkStart;
              const readEnd = Math.min(dataEnd, chunkStart + chunk.byteLength) - chunkStart;
              const part = new Uint8Array(chunk, readStart, readEnd - readStart);
              stitched.set(part, written);
              written += part.byteLength;
            }
            if (written !== tensorSize) {
              throw new Error(
                `Chunk stitching shortfall for ${name}: stitched ${written} of ${tensorSize} bytes`
              );
            }
            rawData = stitched.buffer;
          }

          // Free old chunks to limit memory (~1 GB max). Keep previous chunk in
          // case the next tensor also spans a boundary backwards.
          for (const [k, v] of chunkCache) {
            if (k < firstChunkIdx - 1 && v instanceof ArrayBuffer) chunkCache.delete(k);
          }
        }
        tensorIdx++;
      }

      // GPTQ/INT8/E8 tensors stay in native format for GPU-side dequantization
      const isGPTQ = name.endsWith('.qweight') || name.endsWith('.qzeros')
        || name.endsWith('.scales') || name.endsWith('.g_idx')
        || name.endsWith('.qweight_q8') || name.endsWith('.qzeros_q8')
        || name.endsWith('.scales_q8') || name.endsWith('.g_idx_q8')
        || name.endsWith('.e8_indices') || name.endsWith('.e8_scales')
        || name.endsWith('.e8_offsets');

      // INT8→F32 CPU dequant: intercept Q8 triplets and convert to regular f32 weights
      // Disabled by default: f32 expansion doubles VRAM. Enable for debugging Q8 shader.
      if (false && isGPTQ) {
        const q8Suffix = name.endsWith('.qweight_q8') ? '.qweight_q8'
          : name.endsWith('.scales_q8') ? '.scales_q8'
          : name.endsWith('.qzeros_q8') ? '.qzeros_q8'
          : name.endsWith('.g_idx_q8') ? '.g_idx_q8' : '';
        if (q8Suffix) {
          const base = name.slice(0, -q8Suffix.length);
          const entry = (dequantAccum as any).__q8__ ?? new Map();
          (dequantAccum as any).__q8__ = entry;
          const parts = entry.get(base) ?? {};
          const part = q8Suffix.slice(1);
          parts[part] = { data: rawData.slice(0), shape: tensorInfo.shape };
          entry.set(base, parts);

          // Check if Q8 triplet is complete (qweight + scales + qzeros; g_idx optional)
          if (parts.qweight_q8 && parts.scales_q8 && parts.qzeros_q8) {
            const qw = new Int32Array(parts.qweight_q8.data);
            const sc = new Uint16Array(parts.scales_q8.data);
            const qz = new Int32Array(parts.qzeros_q8.data);
            const gi = parts.g_idx_q8 ? new Uint32Array(parts.g_idx_q8.data) : null;
            const N_q8 = parts.qweight_q8.shape[1];
            const K_q8 = parts.qweight_q8.shape[0] * 4;
            const gs_q8 = config.quantization_config?.group_size ?? 128;

            const f32Data = dequantQ8toF32(qw, sc, qz, gi, K_q8, N_q8, gs_q8);

            const gpuBuffer = device.createBuffer({
              size: f32Data.byteLength,
              usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
              label: `${base}.weight`,
            });
            device.queue.writeBuffer(gpuBuffer, 0, f32Data.buffer as ArrayBuffer, f32Data.byteOffset, f32Data.byteLength);

            allTensors.set(`${base}.weight`, {
              name: `${base}.weight`, buffer: gpuBuffer, shape: [N_q8, K_q8],
              dtype: 'F32', byteLength: f32Data.byteLength,
              elementCount: N_q8 * K_q8, isQuantized: false,
            });
            totalGPUBytes += f32Data.byteLength;
            bytesSinceFlush += f32Data.byteLength;
            tensorsProcessed++;
            entry.delete(base);

            if (bytesSinceFlush >= FLUSH_THRESHOLD) {
              device.queue.submit([]);
              await device.queue.onSubmittedWorkDone();
              bytesSinceFlush = 0;
            }
          }
          continue; // Skip normal GPU upload for Q8 tensors
        }
      }

      // Mixed-precision: intercept GPTQ INT4 triplets that should be dequanted to BF16.
      // If GPU OOM hits, disable dequant and fall back to INT4 for remaining layers.
      if (isGPTQ && dequantSet.size > 0 && !dequantDisabled) {
        const suffix = name.endsWith('.qweight') ? '.qweight'
          : name.endsWith('.scales') ? '.scales'
          : name.endsWith('.qzeros') ? '.qzeros' : '';
        if (suffix) {
          const base = name.slice(0, -suffix.length);
          if (dequantSet.has(base)) {
            const entry = dequantAccum.get(base) ?? {};
            const part = suffix.slice(1) as 'qweight' | 'scales' | 'qzeros';
            entry[part] = { data: rawData.slice(0), shape: tensorInfo.shape };
            dequantAccum.set(base, entry);

            if (entry.qweight && entry.scales && entry.qzeros) {
              const qw = new Int32Array(entry.qweight.data);
              const sc = new Uint16Array(entry.scales.data);
              const qz = new Int32Array(entry.qzeros.data);
              const N = entry.qweight.shape[1];
              const K = entry.qweight.shape[0] * 8;
              const gs = config.quantization_config?.group_size ?? 128;

              const bf16Size = N * K * 2;
              const q4Size = entry.qweight!.data.byteLength + entry.scales!.data.byteLength + entry.qzeros!.data.byteLength;
              const nextOverhead = bf16Size - q4Size;
              if (!dequantDisabled && dequantOverheadUsed + nextOverhead > maxDequantOverhead) {
                console.warn(
                  `[WeightLoader] Dequant overhead cap at ${base} (used: ${(dequantOverheadUsed / 1024 / 1024).toFixed(0)} MB, `
                  + `cap: ${(maxDequantOverhead / 1024 / 1024).toFixed(0)} MB) — stopping dequant (${dequantedCount} projections done)`
                );
                dequantDisabled = true;
              }

              let dequantedOk = false;
              if (!dequantDisabled) {
                try {
                  const bf16Data = dequantGPTQtoBF16(qw, sc, qz, K, N, gs);

                  device.pushErrorScope('out-of-memory');
                  const gpuBuffer = device.createBuffer({
                    size: bf16Data.byteLength,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                    label: `${base}.weight`,
                  });
                  device.queue.writeBuffer(gpuBuffer, 0, bf16Data.buffer as ArrayBuffer, bf16Data.byteOffset, bf16Data.byteLength);
                  device.queue.submit([]);
                  await device.queue.onSubmittedWorkDone();
                  const oomError = await device.popErrorScope();

                  if (oomError) {
                    gpuBuffer.destroy();
                    console.warn(
                      `[WeightLoader] GPU OOM at ${base} (GPU: ${(totalGPUBytes / 1024 / 1024).toFixed(0)} MB)`
                      + ` — stopping dequant (${dequantedCount} projections done)`
                    );
                    dequantDisabled = true;
                  } else {
                    allTensors.set(`${base}.weight`, {
                      name: `${base}.weight`, buffer: gpuBuffer, shape: [N, K],
                      dtype: 'BF16', byteLength: bf16Data.byteLength,
                      elementCount: N * K, isQuantized: false,
                    });
                    totalGPUBytes += bf16Data.byteLength;
                    bytesSinceFlush += bf16Data.byteLength;
                    dequantOverheadUsed += nextOverhead;
                    tensorsProcessed++;
                    dequantedCount++;
                    dequantAccum.delete(base);
                    dequantedOk = true;

                    if (bytesSinceFlush >= FLUSH_THRESHOLD) {
                      device.queue.submit([]);
                      await device.queue.onSubmittedWorkDone();
                      bytesSinceFlush = 0;
                    }
                  }
                } catch (e) {
                  console.warn(
                    `[WeightLoader] SSM dequant error at ${base} (GPU: ${(totalGPUBytes / 1024 / 1024).toFixed(0)} MB)`
                    + ` — falling back to INT4 (${dequantedCount} projections dequanted)`
                  );
                  dequantDisabled = true;
                }
              }

              if (!dequantedOk) {
                const fallbackParts: Array<{ suffix: string; data: ArrayBuffer; shape: number[]; dtype: string }> = [
                  { suffix: '.qweight', data: entry.qweight!.data, shape: entry.qweight!.shape, dtype: 'I32' },
                  { suffix: '.scales', data: entry.scales!.data, shape: entry.scales!.shape, dtype: 'F16' },
                  { suffix: '.qzeros', data: entry.qzeros!.data, shape: entry.qzeros!.shape, dtype: 'I32' },
                ];
                for (const fp of fallbackParts) {
                  const fbName = `${base}${fp.suffix}`;
                  const fbData = new Uint8Array(fp.data);
                  const fbBuf = device.createBuffer({
                    size: fbData.byteLength,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                    label: fbName,
                  });
                  device.queue.writeBuffer(fbBuf, 0, fbData.buffer as ArrayBuffer, fbData.byteOffset, fbData.byteLength);
                  allTensors.set(fbName, {
                    name: fbName, buffer: fbBuf, shape: fp.shape, dtype: fp.dtype,
                    byteLength: fbData.byteLength,
                    elementCount: fp.shape.reduce((a: number, b: number) => a * b, 1),
                    isQuantized: true,
                  });
                  totalGPUBytes += fbData.byteLength;
                }
                tensorsProcessed++;
                dequantAccum.delete(base);
                for (const [accBase, accEntry] of dequantAccum) {
                  for (const [accSuffix, accData] of Object.entries(accEntry) as Array<[string, { data: ArrayBuffer; shape: number[] }]>) {
                    const accName = `${accBase}.${accSuffix}`;
                    const accBytes = new Uint8Array(accData.data);
                    const accBuf = device.createBuffer({
                      size: accBytes.byteLength,
                      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
                      label: accName,
                    });
                    device.queue.writeBuffer(accBuf, 0, accBytes.buffer as ArrayBuffer, accBytes.byteOffset, accBytes.byteLength);
                    allTensors.set(accName, {
                      name: accName, buffer: accBuf, shape: accData.shape,
                      dtype: accSuffix === 'scales' ? 'F16' : 'I32',
                      byteLength: accBytes.byteLength,
                      elementCount: accData.shape.reduce((a: number, b: number) => a * b, 1),
                      isQuantized: true,
                    });
                    totalGPUBytes += accBytes.byteLength;
                  }
                }
                dequantAccum.clear();
              }
            }
            continue;
          }
        }
      }

      let gpuData: ArrayBufferView;
      let gpuDtype = tensorInfo.dtype; // Track actual format on GPU (may differ from original)
      const f32Size = tensorInfo.elementCount * 4;
      const exceedsBufferLimit = f32Size > 1.9 * 1024 * 1024 * 1024;
      const maxBufferSize = device.limits.maxBufferSize;

      // Runtime handling for oversized BF16/F16 embedding/lm_head tensors.
      // embed_tokens → CPU (trivial lookup, saves ~2.4 GB VRAM)
      // lm_head → split BF16 GPU buffers (needs GPU matmul, lossless)
      const isEmbed = name.endsWith('embed_tokens.weight');
      const isLmHead = name === 'lm_head.weight';
      const needsOversizedHandling = (isEmbed || isLmHead)
        && (tensorInfo.dtype === 'BF16' || tensorInfo.dtype === 'F16')
        && tensorInfo.byteLength > maxBufferSize;

      if (needsOversizedHandling && isEmbed) {
        const [vocabSize, hiddenSize] = tensorInfo.shape;
        const rowBytes = hiddenSize * 2;
        const maxPartBytes = 1024 * 1024 * 1024;
        const rowsPerPart = Math.floor(maxPartBytes / rowBytes);
        const numParts = Math.ceil(vocabSize / rowsPerPart);
        const splitPoint = Math.min(rowsPerPart, vocabSize);
        const baseName = name.replace('.weight', '');

        console.log(
          `[WeightLoader] In-memory CPU embed: ${name} [${vocabSize}, ${hiddenSize}] `
          + `→ ${numParts} JS arrays, splitPoint=${splitPoint} (saves ${(tensorInfo.byteLength / 1024 / 1024).toFixed(0)} MB VRAM)`
        );

        const raw = new Uint8Array(rawData);
        const cpuParts: Uint8Array[] = [];
        for (let part = 0; part < numParts; part++) {
          const startRow = part * rowsPerPart;
          const endRow = Math.min(startRow + rowsPerPart, vocabSize);
          const partBytes = (endRow - startRow) * rowBytes;
          cpuParts.push(raw.slice(startRow * rowBytes, startRow * rowBytes + partBytes));
        }

        allTensors.set(`${baseName}.cpu_embed`, {
          name: `${baseName}.cpu_embed`,
          buffer: device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM, label: 'cpu-embed-marker' }),
          shape: [vocabSize, hiddenSize],
          dtype: tensorInfo.dtype,
          byteLength: 0, elementCount: 0,
          cpuEmbedData: { parts: cpuParts, splitPoint, isBF16: tensorInfo.dtype === 'BF16' },
        } as any);

        tensorsProcessed++;
        console.log(`[WeightLoader] CPU embed done: ${baseName} — ${numParts} parts, ${vocabSize} rows in JS memory (0 MB GPU)`);
        continue;
      }

      if (needsOversizedHandling && isLmHead) {
        const [vocabSize, hiddenSize] = tensorInfo.shape;
        const rowBytes = hiddenSize * 2;
        const maxPartBytes = 1024 * 1024 * 1024;
        const rowsPerPart = Math.floor(maxPartBytes / rowBytes);
        const numParts = Math.ceil(vocabSize / rowsPerPart);
        const splitPoint = Math.min(rowsPerPart, vocabSize);
        const baseName = name.replace('.weight', '');

        console.log(
          `[WeightLoader] In-memory CPU lm_head: ${name} [${vocabSize}, ${hiddenSize}] `
          + `→ ${numParts} JS arrays, splitPoint=${splitPoint} (saves ${(tensorInfo.byteLength / 1024 / 1024).toFixed(0)} MB VRAM)`
        );

        const raw = new Uint8Array(rawData);
        const cpuParts: Uint8Array[] = [];
        for (let part = 0; part < numParts; part++) {
          const startRow = part * rowsPerPart;
          const endRow = Math.min(startRow + rowsPerPart, vocabSize);
          const partBytes = (endRow - startRow) * rowBytes;
          cpuParts.push(raw.slice(startRow * rowBytes, startRow * rowBytes + partBytes));
        }

        allTensors.set(`${baseName}.cpu_lm_head`, {
          name: `${baseName}.cpu_lm_head`,
          buffer: device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM, label: 'cpu-lm-head-marker' }),
          shape: [vocabSize, hiddenSize],
          dtype: tensorInfo.dtype,
          byteLength: 0, elementCount: 0,
          cpuLmHeadData: { parts: cpuParts, splitPoint, isBF16: tensorInfo.dtype === 'BF16' },
        } as any);

        tensorsProcessed++;
        console.log(`[WeightLoader] CPU lm_head done: ${baseName} — ${numParts} parts, ${vocabSize} rows in JS memory (0 MB GPU)`);
        continue;
      }

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
        gpuDtype = 'F32';
      }

      // Skip tensors that exceed WebGPU max buffer size (typically 2 GB)
      if (gpuData.byteLength > maxBufferSize) {
        console.warn(`[WeightLoader] SKIPPING ${name}: ${(gpuData.byteLength / 1024 / 1024).toFixed(0)} MB exceeds GPU max buffer (${(maxBufferSize / 1024 / 1024).toFixed(0)} MB)`);
        tensorsProcessed++;
        continue;
      }

      // Create GPU buffer and upload via writeBuffer (Dawn manages staging
      // internally via a ring buffer, avoiding the 2x peak memory of mappedAtCreation)
      const gpuBuffer = device.createBuffer({
        size: gpuData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: name,
      });
      device.queue.writeBuffer(gpuBuffer, 0, gpuData.buffer as ArrayBuffer, gpuData.byteOffset, gpuData.byteLength);

      // ── VRAM audit (debug) ───────────────────────────────────────────
      // When __DEBUG_VRAM_AUDIT_PREFIXES__ is set on globalThis, read back the
      // just-uploaded GPU buffer and byte-compare it to the CPU source bytes.
      // Detects upload corruption, buffer-limit truncation, or byte-alignment bugs.
      // One-shot diagnostic for narrowing per-tensor engine bugs.
      const vramAuditPrefixes: string[] =
        (globalThis as any).__DEBUG_VRAM_AUDIT_PREFIXES__ ?? [];
      if (vramAuditPrefixes.length > 0 && vramAuditPrefixes.some(p => name.startsWith(p))) {
        const diskBytes = new Uint8Array(gpuData.buffer, gpuData.byteOffset, gpuData.byteLength);
        const vramRaw = await readBuffer(device, gpuBuffer, gpuData.byteLength);
        const vramBytes = new Uint8Array(vramRaw);
        let mismatchCount = 0;
        let firstMismatch = -1;
        const n = Math.min(diskBytes.length, vramBytes.length);
        for (let i = 0; i < n; i++) {
          if (diskBytes[i] !== vramBytes[i]) {
            if (firstMismatch === -1) firstMismatch = i;
            mismatchCount++;
          }
        }
        const hexSlice = (arr: Uint8Array, start: number, len: number) =>
          Array.from(arr.slice(start, start + len))
            .map(b => b.toString(16).padStart(2, '0')).join(' ');
        const status = mismatchCount === 0 ? 'OK' : 'MISMATCH';
        console.log(
          `[VRAM-AUDIT ${status}] ${name} (${gpuDtype}) ` +
          `disk=${diskBytes.length}B vram=${vramBytes.length}B ` +
          `mismatches=${mismatchCount}/${n} firstAt=${firstMismatch}`
        );
        console.log(`  disk[0:16] = ${hexSlice(diskBytes, 0, 16)}`);
        console.log(`  vram[0:16] = ${hexSlice(vramBytes, 0, 16)}`);
        if (mismatchCount > 0) {
          console.log(`  disk@first = ${hexSlice(diskBytes, Math.max(0, firstMismatch - 4), 16)}`);
          console.log(`  vram@first = ${hexSlice(vramBytes, Math.max(0, firstMismatch - 4), 16)}`);
        }
      }

      // For GPTQ g_idx tensors, probe whether the values are trivially
      // floor(k / group_size). If so, the engine can use the fast GEMV path.
      let isTrivialGIdx: boolean | undefined;
      if (name.endsWith('.g_idx') || name.endsWith('.g_idx_q8')) {
        isTrivialGIdx = isTrivialGIdxBuffer(rawData, tensorInfo.elementCount);
      }

      allTensors.set(name, {
        name, buffer: gpuBuffer, shape: tensorInfo.shape,
        dtype: gpuDtype, byteLength: gpuData.byteLength,
        elementCount: tensorInfo.elementCount, isQuantized: isGPTQ,
        isTrivialGIdx,
      });

      totalGPUBytes += gpuData.byteLength;
      bytesSinceFlush += gpuData.byteLength;
      tensorsProcessed++;

      if (bytesSinceFlush >= FLUSH_THRESHOLD) {
        device.queue.submit([]);
        await device.queue.onSubmittedWorkDone();
        bytesSinceFlush = 0;
      }
    }

    // Final flush after each shard
    device.queue.submit([]);
    await device.queue.onSubmittedWorkDone();
    bytesSinceFlush = 0;

    const t_upload_end = performance.now();
    console.log(
      `[Perf] Shard ${shardIdx + 1}/${shards.length}: process+upload ${(t_upload_end - t_upload_start).toFixed(0)}ms `
      + `(${header.tensors.size} tensors, GPU total: ${(totalGPUBytes / 1024 / 1024).toFixed(0)} MB)`
    );
  }

  if (dequantedCount > 0) {
    const totalSSM = dequantSet.size > 0 ? Math.floor(dequantSet.size / 10) : 0;
    console.log(
      `[WeightLoader] SSM dequant: ${dequantedCount} projections → BF16`
      + (dequantDisabled ? ` (stopped early — GPU OOM after ${(totalGPUBytes / 1024 / 1024).toFixed(0)} MB)` : '')
    );
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
// Shared scratch buffers for bitcast conversions (avoids per-call allocation)
const _cvtBuf = new ArrayBuffer(4);
const _cvtU32 = new Uint32Array(_cvtBuf);
const _cvtF32 = new Float32Array(_cvtBuf);

/** Decode BF16 u16 bits to f32 number (used by runtime quantizer) */
function bf16ToF32Bits(bits: number): number {
  _cvtU32[0] = bits << 16;
  return _cvtF32[0];
}

/** Decode F16 u16 bits to f32 number (used by runtime quantizer) */
function f16ToF32Bits(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const frac = bits & 0x3FF;
  if (exp === 0) return frac === 0 ? 0 : (frac / 1024) * Math.pow(2, -14) * (sign ? -1 : 1);
  if (exp === 31) return sign ? -Infinity : Infinity;
  return (1 + frac / 1024) * Math.pow(2, exp - 15) * (sign ? -1 : 1);
}

/** Encode f32 to F16 u16 bits (for scales in runtime quantizer) */
function f32ToF16Bits(val: number): number {
  _cvtF32[0] = val;
  const f32Bits = _cvtU32[0];
  const sign = (f32Bits >> 31) & 1;
  let exp = ((f32Bits >> 23) & 0xFF) - 127 + 15;
  let frac = (f32Bits >> 13) & 0x3FF;
  if (exp <= 0) { exp = 0; frac = 0; }
  if (exp >= 31) { exp = 31; frac = 0; }
  return (sign << 15) | (exp << 10) | frac;
}

/** Decode IEEE 754 half-precision (F16) u16 bits to f32 number */
function f16ToF32(bits: number): number {
  const sign = (bits >> 15) & 1;
  const exp = (bits >> 10) & 0x1F;
  const frac = bits & 0x3FF;
  if (exp === 0) return frac === 0 ? 0 : (frac / 1024) * Math.pow(2, -14) * (sign ? -1 : 1);
  if (exp === 31) return sign ? -Infinity : Infinity;
  return (1 + frac / 1024) * Math.pow(2, exp - 15) * (sign ? -1 : 1);
}

/**
 * CPU-side GPTQ dequantization — converts INT4 packed weights to BF16.
 * Used for mixed-precision: dequant SSM-critical weights to BF16
 * while keeping attention/FFN weights in INT4 for VRAM savings.
 *
 * Output is a Uint16Array of BF16 values in [N, K] row-major order.
 * When uploaded to GPU and read as array<u32>, pairs of BF16 values
 * naturally pack into u32 words matching the matmul_bt_bf16 shader layout.
 *
 * @returns Uint16Array of shape [N, K] as BF16 values
 */
export function dequantGPTQtoBF16(
  qweight: Uint32Array | Int32Array,
  scales: Uint16Array,
  qzeros: Uint32Array | Int32Array,
  K: number,
  N: number,
  groupSize: number,
): Uint16Array {
  const result = new Uint16Array(N * K);
  // Temp buffer for f32→bf16 conversion (reuse to avoid allocations)
  const f32Buf = new ArrayBuffer(4);
  const f32View = new DataView(f32Buf);

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

      // Dequant: (q4 - zero) * scale → f32 → bf16
      const val = (q4 - zero) * scale;
      f32View.setFloat32(0, val, true);
      result[n * K + k] = f32View.getUint32(0, true) >>> 16; // BF16 = upper 16 bits
    }
  }

  console.log(`[Dequant] GPTQ → BF16: [${N}, ${K}] (${(result.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  return result;
}

/**
 * CPU-side GPTQ dequantization — converts INT4 packed weights to f32.
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
  const zerosPerRow = Math.ceil(N / 8);

  for (let k = 0; k < K; k++) {
    const group = Math.floor(k / groupSize);
    const packedRow = Math.floor(k / 8);
    const nibbleInPacked = k % 8;

    for (let n = 0; n < N; n++) {
      const packed = qweight[packedRow * N + n];
      const q4 = (packed >> (nibbleInPacked * 4)) & 0xF;

      const scaleIdx = group * N + n;
      const scale = f16ToF32(scales[scaleIdx]);

      const zeroPacked = qzeros[group * zerosPerRow + Math.floor(n / 8)];
      const zeroNibble = n % 8;
      const zero = (zeroPacked >> (zeroNibble * 4)) & 0xF;

      result[n * K + k] = (q4 - zero) * scale;
    }
  }

  console.log(`[Dequant] GPTQ → f32: [${N}, ${K}] (${(result.byteLength / 1024 / 1024).toFixed(1)} MB)`);
  return result;
}

/**
 * CPU-side INT8 dequantization — converts INT8 packed weights to f32.
 * Layout: qweight_q8 is [K/4, N] (4 INT8 per i32), scales_q8 is [numGroups, N] F16,
 *         qzeros_q8 is [numGroups, N/4] (4 INT8 per i32), g_idx_q8 is [K] u32.
 * @returns Float32Array of shape [N, K] (HF weight format: [out_features, in_features])
 */
export function dequantQ8toF32(
  qweight: Int32Array,
  scales: Uint16Array,
  qzeros: Int32Array,
  gIdx: Uint32Array | null,
  K: number,
  N: number,
  groupSize: number,
): Float32Array {
  const result = new Float32Array(N * K);
  const zerosStride = Math.ceil(N / 4);

  for (let k = 0; k < K; k++) {
    const groupId = gIdx ? gIdx[k] : Math.floor(k / groupSize);
    const packedRow = Math.floor(k / 4);
    const byteInPacked = k % 4;

    for (let n = 0; n < N; n++) {
      // Extract 8-bit weight
      const packed = qweight[packedRow * N + n];
      const q8 = (packed >> (byteInPacked * 8)) & 0xFF;

      // Get scale (F16 stored as u16)
      const scale = f16ToF32(scales[groupId * N + n]);

      // Get zero point (packed INT8)
      const zeroPacked = qzeros[groupId * zerosStride + Math.floor(n / 4)];
      const zeroByte = n % 4;
      const zero = (zeroPacked >> (zeroByte * 8)) & 0xFF;

      result[n * K + k] = (q8 - zero) * scale;
    }
  }

  console.log(`[Dequant] Q8 → f32: [${N}, ${K}] (${(result.byteLength / 1024 / 1024).toFixed(1)} MB)`);
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
