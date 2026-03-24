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
  parseHeader, extractTensorData, tensorToFloat32,
  formatBytes, summarizeHeader,
  type SafeTensorsHeader, type TensorInfo,
} from './safetensors';

import {
  discoverShards, downloadFile, downloadShardHeader, fetchModelConfig,
  type ShardInfo, type DownloadProgress, type HFModelConfig,
} from './hf-hub';

import { getCache, putCache, hasCache } from './cache';
import { reportMetric } from '../utils/metrics';

// ─── Types ───────────────────────────────────────────────────────────────────

export interface GPUTensor {
  name: string;
  buffer: GPUBuffer;
  shape: number[];
  dtype: string;        // original dtype (F16, BF16, etc.)
  byteLength: number;   // size on GPU (always F32 = elementCount × 4)
  elementCount: number;
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

    // Check browser cache first
    let shardData: ArrayBuffer;
    const cached = await hasCache(cacheKey);

    if (cached) {
      progress({
        phase: 'downloading',
        message: `Shard ${shardIdx + 1}: loading from cache...`,
        shard: shardIdx + 1,
        totalShards: shards.length,
        shardProgress: 1,
        overallProgress: (bytesDownloadedTotal + shard.size) / totalDownloadSize,
      });
      shardData = await getCache(cacheKey) as ArrayBuffer;
    } else {
      // Download with progress
      shardData = await downloadFile(shard.url, (downloaded, total) => {
        progress({
          phase: 'downloading',
          message: `Shard ${shardIdx + 1}: ${formatBytes(downloaded)} / ${formatBytes(total)}`,
          shard: shardIdx + 1,
          totalShards: shards.length,
          shardProgress: total > 0 ? downloaded / total : 0,
          overallProgress: (bytesDownloadedTotal + downloaded) / totalDownloadSize,
        });
      });

      // Cache for next time
      await putCache(cacheKey, shardData);
    }

    bytesDownloadedTotal += shard.size;

    // ── Step 3: Parse SafeTensors header ────────────────────────────────

    const header = parseHeader(shardData);
    console.log(`[WeightLoader] Shard ${shardIdx + 1}: ${summarizeHeader(header)}`);

    // ── Step 4: Extract tensors and upload to GPU ───────────────────────

    progress({
      phase: 'uploading',
      message: `Uploading ${header.tensors.size} tensors to GPU...`,
      shard: shardIdx + 1,
      totalShards: shards.length,
      tensorsLoaded: tensorsProcessed,
    });

    for (const [name, tensorInfo] of header.tensors) {
      // Extract raw tensor bytes from the shard
      const rawData = extractTensorData(shardData, tensorInfo, header.headerByteLength);

      // Convert to float32 for GPU compute
      const float32Data = tensorToFloat32(rawData, tensorInfo.dtype);

      // Create GPU buffer and upload
      const gpuBuffer = device.createBuffer({
        size: float32Data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: name,
        mappedAtCreation: true,
      });

      new Float32Array(gpuBuffer.getMappedRange()).set(float32Data);
      gpuBuffer.unmap();

      allTensors.set(name, {
        name,
        buffer: gpuBuffer,
        shape: tensorInfo.shape,
        dtype: tensorInfo.dtype,
        byteLength: float32Data.byteLength,
        elementCount: tensorInfo.elementCount,
      });

      totalGPUBytes += float32Data.byteLength;
      tensorsProcessed++;
    }
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
