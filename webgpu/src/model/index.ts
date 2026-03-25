/**
 * Model Loading — Public API
 *
 * Re-exports the model loading pipeline:
 *   safetensors.ts  — Parse SafeTensors binary format
 *   hf-hub.ts       — HuggingFace Hub API client
 *   weight-loader.ts — Orchestrate download → parse → GPU upload
 *   cache.ts        — Browser-side caching for model weights
 */

export {
  // SafeTensors parser
  parseHeader, parseHeaderLength, extractTensorData,
  tensorToFloat32, dtypeSize, formatBytes, summarizeHeader,
  type TensorInfo, type SafeTensorDtype, type SafeTensorsHeader,
} from './safetensors';

export {
  // HuggingFace Hub
  setAuthToken, listModelFiles, fetchModelConfig, fetchShardIndex,
  discoverShards, fetchRange, downloadFile, downloadShardHeader,
  getModelInfo,
  type HFModelFile, type HFModelConfig, type ShardInfo, type DownloadProgress,
} from './hf-hub';

export {
  // Weight Loader
  loadModel, unloadModel, previewModel,
  type GPUTensor, type LoadedModel, type LoadProgress,
} from './weight-loader';

export {
  // Cache
  hasCache, getCache, putCache, removeCache, clearCache, getCacheStats,
} from './cache';

export {
  // Model Config
  parseModelConfig, getWeightNameMap, resolveLayerWeightName,
  getAllWeightNames, estimateVRAM,
  type ModelConfig, type WeightNameMap,
} from './model-config';

export {
  // Tokenizer
  createTokenizer, applyChatTemplate,
  type Tokenizer, type TokenizerConfig,
} from './tokenizer';

export {
  // TurboQuant KV Cache Compression
  initTurboQuant, buildCodebook,
  generateRotationMatrix, generateJLMatrix,
  cpuEncode, cpuDecode, computeMSE, computeRelativeMSE,
  type TurboQuantConfig, type TurboQuantBuffers, type TurboQuantCodebook,
} from './turboquant';
