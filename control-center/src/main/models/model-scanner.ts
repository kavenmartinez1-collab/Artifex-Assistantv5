import * as fs from 'fs';
import * as path from 'path';

export interface ModelInfo {
  name: string;           // directory name
  path: string;           // full path
  sizeBytes: number;      // total on disk
  sizeFormatted: string;  // "5.74 GB"
  modelType: string;      // from config
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  isQuantized: boolean;
  quantBits: number | null;
  quantMethod: string | null;
  quantDetail: string;    // "GPTQ INT4", "BF16", "Mixed INT4" etc.
  groupSize: number | null;
  mixedPrecision: boolean;
  shardCount: number;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(2)} ${units[i]}`;
}

function getDirSizeSync(dirPath: string): number {
  let total = 0;
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isFile()) {
      try {
        const stat = fs.statSync(fullPath);
        total += stat.size;
      } catch {
        // Skip files we can't stat
      }
    } else if (entry.isDirectory()) {
      total += getDirSizeSync(fullPath);
    }
  }
  return total;
}

function countShards(dirPath: string): number {
  try {
    const files = fs.readdirSync(dirPath);
    return files.filter(f => f.endsWith('.safetensors')).length;
  } catch {
    return 0;
  }
}

function buildQuantDetail(isQuantized: boolean, quantMethod: string | null, bits: number | null, mixedPrecision: boolean): string {
  if (!isQuantized) return 'BF16';

  const method = (quantMethod || 'quant').toUpperCase();
  const bitStr = bits ? `INT${bits}` : 'INT4';

  if (mixedPrecision) {
    return `Mixed ${method} ${bitStr}`;
  }
  return `${method} ${bitStr}`;
}

function parseConfig(configPath: string): {
  modelType: string;
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  isQuantized: boolean;
  quantBits: number | null;
  quantMethod: string | null;
  groupSize: number | null;
  mixedPrecision: boolean;
} {
  const defaults = {
    modelType: 'unknown',
    hiddenSize: 0,
    numLayers: 0,
    vocabSize: 0,
    isQuantized: false,
    quantBits: null as number | null,
    quantMethod: null as string | null,
    groupSize: null as number | null,
    mixedPrecision: false,
  };

  try {
    const raw = fs.readFileSync(configPath, 'utf-8');
    const config = JSON.parse(raw);

    // Model type at top level
    defaults.modelType = config.model_type || 'unknown';

    // Text config holds the architecture details for Qwen3.5 and similar
    const textCfg = config.text_config || config;
    defaults.hiddenSize = textCfg.hidden_size || config.hidden_size || 0;
    defaults.numLayers = textCfg.num_hidden_layers || config.num_hidden_layers || 0;
    defaults.vocabSize = textCfg.vocab_size || config.vocab_size || 0;

    // Quantization config
    const qcfg = config.quantization_config;
    if (qcfg) {
      defaults.isQuantized = true;
      defaults.quantBits = qcfg.bits ?? null;
      defaults.quantMethod = qcfg.quant_method ?? null;
      defaults.groupSize = qcfg.group_size ?? null;
      defaults.mixedPrecision = qcfg.mixed_precision === true;
    }
  } catch {
    // Config missing or malformed — return defaults
  }

  return defaults;
}

export async function scanModels(projectRoot: string): Promise<ModelInfo[]> {
  const modelsDir = path.join(projectRoot, 'models');

  if (!fs.existsSync(modelsDir)) {
    return [];
  }

  const entries = fs.readdirSync(modelsDir, { withFileTypes: true });
  const models: ModelInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    // Skip hidden directories like .cache
    if (entry.name.startsWith('.')) continue;

    const modelPath = path.join(modelsDir, entry.name);
    const configPath = path.join(modelPath, 'config.json');

    const cfg = parseConfig(configPath);
    const sizeBytes = getDirSizeSync(modelPath);
    const shardCount = countShards(modelPath);

    models.push({
      name: entry.name,
      path: modelPath,
      sizeBytes,
      sizeFormatted: formatBytes(sizeBytes),
      modelType: cfg.modelType,
      hiddenSize: cfg.hiddenSize,
      numLayers: cfg.numLayers,
      vocabSize: cfg.vocabSize,
      isQuantized: cfg.isQuantized,
      quantBits: cfg.quantBits,
      quantMethod: cfg.quantMethod,
      quantDetail: buildQuantDetail(cfg.isQuantized, cfg.quantMethod, cfg.quantBits, cfg.mixedPrecision),
      groupSize: cfg.groupSize,
      mixedPrecision: cfg.mixedPrecision,
      shardCount,
    });
  }

  // Sort by size descending (largest first)
  models.sort((a, b) => b.sizeBytes - a.sizeBytes);

  return models;
}

export async function deleteModel(modelPath: string): Promise<void> {
  // Safety check: resolve to absolute and verify it's inside a models/ directory
  const resolved = path.resolve(modelPath);
  const parentDir = path.basename(path.dirname(resolved));

  if (parentDir !== 'models') {
    throw new Error(`Safety check failed: "${resolved}" is not inside a models/ directory`);
  }

  if (!fs.existsSync(resolved)) {
    throw new Error(`Model directory not found: "${resolved}"`);
  }

  const stat = fs.statSync(resolved);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: "${resolved}"`);
  }

  // Recursive delete
  fs.rmSync(resolved, { recursive: true, force: true });
}
