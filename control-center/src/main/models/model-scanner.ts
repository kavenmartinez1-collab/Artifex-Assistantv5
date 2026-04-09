import * as fsp from 'fs/promises';
import * as http from 'http';
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
  source: 'transformers'; // distinguishes from Ollama models in the UI
}

export interface OllamaModelInfo {
  name: string;            // tag, e.g. "qwen3-vl:8b-instruct"
  sizeBytes: number;
  sizeFormatted: string;
  family: string;          // e.g. "qwen3vl"
  paramSize: string;       // e.g. "8.5B"
  quantLevel: string;      // e.g. "Q4_K_M"
  format: string;          // e.g. "gguf"
  modifiedAt: string;
  source: 'ollama';
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const value = bytes / Math.pow(1024, i);
  return `${value.toFixed(2)} ${units[i]}`;
}

async function getDirSize(dirPath: string): Promise<number> {
  let total = 0;
  const entries = await fsp.readdir(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isFile()) {
      try {
        const stat = await fsp.stat(fullPath);
        total += stat.size;
      } catch {
        // Skip files we can't stat
      }
    } else if (entry.isDirectory()) {
      total += await getDirSize(fullPath);
    }
  }
  return total;
}

async function countShards(dirPath: string): Promise<number> {
  try {
    const files = await fsp.readdir(dirPath);
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

async function parseConfig(configPath: string): Promise<{
  modelType: string;
  hiddenSize: number;
  numLayers: number;
  vocabSize: number;
  isQuantized: boolean;
  quantBits: number | null;
  quantMethod: string | null;
  groupSize: number | null;
  mixedPrecision: boolean;
}> {
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
    const raw = await fsp.readFile(configPath, 'utf-8');
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

  try {
    await fsp.access(modelsDir);
  } catch {
    return [];
  }

  const entries = await fsp.readdir(modelsDir, { withFileTypes: true });
  const models: ModelInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    // Skip hidden directories like .cache
    if (entry.name.startsWith('.')) continue;

    const modelPath = path.join(modelsDir, entry.name);
    const configPath = path.join(modelPath, 'config.json');

    const cfg = await parseConfig(configPath);
    const sizeBytes = await getDirSize(modelPath);
    const shardCount = await countShards(modelPath);

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
      source: 'transformers',
    });
  }

  // Sort by size descending (largest first)
  models.sort((a, b) => b.sizeBytes - a.sizeBytes);

  return models;
}

/**
 * Discover models installed in a running Ollama server.
 *
 * Path-independent: queries Ollama's HTTP API on localhost:11434, so it
 * works regardless of where Ollama stores its blobs on disk. Returns an
 * empty array if Ollama isn't reachable.
 */
export async function scanOllamaModels(): Promise<OllamaModelInfo[]> {
  const raw = await ollamaHttpGet('/api/tags').catch(() => null);
  if (!raw) return [];

  let parsed: { models?: Array<Record<string, unknown>> };
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }

  const models: OllamaModelInfo[] = [];
  for (const m of parsed.models || []) {
    const name = String(m.name || '');
    if (!name) continue;
    const details = (m.details as Record<string, unknown>) || {};
    const sizeBytes = Number(m.size || 0);
    models.push({
      name,
      sizeBytes,
      sizeFormatted: formatBytes(sizeBytes),
      family: String(details.family || 'unknown'),
      paramSize: String(details.parameter_size || ''),
      quantLevel: String(details.quantization_level || ''),
      format: String(details.format || 'gguf'),
      modifiedAt: String(m.modified_at || ''),
      source: 'ollama',
    });
  }

  // Sort by size descending (largest first), matching the Transformers list
  models.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return models;
}

/** Minimal HTTP GET against the local Ollama server. Rejects on any error. */
function ollamaHttpGet(pathname: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const req = http.get(
      { host: '127.0.0.1', port: 11434, path: pathname, timeout: 3000 },
      (res) => {
        if (res.statusCode !== 200) {
          res.resume();
          reject(new Error(`Ollama returned HTTP ${res.statusCode}`));
          return;
        }
        let body = '';
        res.setEncoding('utf-8');
        res.on('data', (chunk) => { body += chunk; });
        res.on('end', () => resolve(body));
      },
    );
    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy(new Error('Ollama request timed out'));
    });
  });
}

/**
 * Delete an Ollama model via the local server's /api/delete endpoint.
 * Tag name only (e.g. "qwen3-vl:8b-instruct"), no filesystem path.
 */
export async function deleteOllamaModel(name: string): Promise<void> {
  const body = JSON.stringify({ name });
  await new Promise<void>((resolve, reject) => {
    const req = http.request(
      {
        host: '127.0.0.1',
        port: 11434,
        path: '/api/delete',
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 10000,
      },
      (res) => {
        let respBody = '';
        res.setEncoding('utf-8');
        res.on('data', (chunk) => { respBody += chunk; });
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolve();
          } else {
            reject(new Error(`Ollama delete failed (HTTP ${res.statusCode}): ${respBody}`));
          }
        });
      },
    );
    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('Ollama delete timed out')));
    req.write(body);
    req.end();
  });
}

export async function deleteModel(modelPath: string): Promise<void> {
  // Safety check: resolve to absolute and verify it's inside a models/ directory
  const resolved = path.resolve(modelPath);
  const parentDir = path.basename(path.dirname(resolved));

  if (parentDir !== 'models') {
    throw new Error(`Safety check failed: "${resolved}" is not inside a models/ directory`);
  }

  try {
    await fsp.access(resolved);
  } catch {
    throw new Error(`Model directory not found: "${resolved}"`);
  }

  const stat = await fsp.stat(resolved);
  if (!stat.isDirectory()) {
    throw new Error(`Not a directory: "${resolved}"`);
  }

  // Recursive delete
  await fsp.rm(resolved, { recursive: true, force: true });
}
