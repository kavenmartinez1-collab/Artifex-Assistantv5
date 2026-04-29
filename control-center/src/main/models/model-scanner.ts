import * as fsp from 'fs/promises';
import * as http from 'http';
import * as path from 'path';

export interface ModelInfo {
  name: string;           // directory name
  path: string;           // full path
  sizeBytes: number;      // total on disk (model dir only, not cache)
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
  // NF4 cache state — set by the Python engine on first slow-path load.
  // Tied to a sibling directory `<name>-nf4-cached/`. The `.no_cache`
  // marker is the user-controllable opt-out (Control Center checkbox).
  cachePath: string | null;       // absolute path to cache dir, or null
  cacheSizeBytes: number;         // 0 if no cache present
  cacheSizeFormatted: string;     // "0 B" when none
  noCache: boolean;               // true when `.no_cache` marker present
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

export interface LlamaCppModelInfo {
  name: string;            // config key, e.g. "qwen3.6-27b-tq3_4s"
  ggufPath: string;        // absolute path to the .gguf file
  sizeBytes: number;       // GGUF size on disk; 0 if not stat'able
  sizeFormatted: string;
  port: number;
  numCtx: number | null;   // null when auto-sized from VRAM
  numGpuLayers: number;
  serverPath: string;      // resolved llama-server binary
  hasMmproj: boolean;      // true if `--mmproj <...>` is in extra_flags
  flagsSummary: string[];  // condensed extra_flags for display
  ggufMissing: boolean;    // true if the GGUF doesn't exist on disk
  source: 'llama_cpp';
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

const NF4_CACHE_SUFFIX = '-nf4-cached';
const NO_CACHE_MARKER = '.no_cache';

export async function scanModels(projectRoot: string): Promise<ModelInfo[]> {
  const modelsDir = path.join(projectRoot, 'models');

  try {
    await fsp.access(modelsDir);
  } catch {
    return [];
  }

  const entries = await fsp.readdir(modelsDir, { withFileTypes: true });
  // First pass: build a set of all directory names so we can look up
  // each model's `<name>-nf4-cached` sibling without re-stat'ing.
  const dirNames = new Set<string>();
  for (const entry of entries) {
    if (entry.isDirectory()) dirNames.add(entry.name);
  }

  const models: ModelInfo[] = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    // Skip hidden directories like .cache
    if (entry.name.startsWith('.')) continue;
    // Skip NF4 cache directories — they're not user-facing models, just
    // a speedup artifact owned by the Python engine. Surfacing them as
    // separate cards confuses users (they look like duplicate models).
    if (entry.name.endsWith(NF4_CACHE_SUFFIX)) continue;

    const modelPath = path.join(modelsDir, entry.name);
    const configPath = path.join(modelPath, 'config.json');

    const cfg = await parseConfig(configPath);
    const sizeBytes = await getDirSize(modelPath);
    const shardCount = await countShards(modelPath);

    // Cache state: a sibling `<name>-nf4-cached` may or may not exist.
    // The `.no_cache` marker lives in the source directory and gates
    // future cache writes by the Python engine.
    const cacheName = entry.name + NF4_CACHE_SUFFIX;
    let cachePath: string | null = null;
    let cacheSizeBytes = 0;
    if (dirNames.has(cacheName)) {
      cachePath = path.join(modelsDir, cacheName);
      cacheSizeBytes = await getDirSize(cachePath);
    }
    const noCache = await fileExists(path.join(modelPath, NO_CACHE_MARKER));

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
      cachePath,
      cacheSizeBytes,
      cacheSizeFormatted: formatBytes(cacheSizeBytes),
      noCache,
      source: 'transformers',
    });
  }

  // Sort by size descending (largest first)
  models.sort((a, b) => b.sizeBytes - a.sizeBytes);

  return models;
}

async function fileExists(p: string): Promise<boolean> {
  try {
    await fsp.access(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Toggle the `.no_cache` marker on a transformers model directory.
 *
 * When `enabled` is true: writes the marker AND deletes the existing
 * `<name>-nf4-cached` sibling (if any) so the user actually reclaims
 * disk. The Python engine will skip future cache writes for this model.
 *
 * When `enabled` is false: removes the marker. The cache will regenerate
 * on the next slow-path load.
 *
 * Returns the new cache state so the renderer can update the card
 * without re-scanning everything.
 */
export async function setNoCacheMarker(
  modelPath: string,
  enabled: boolean,
): Promise<{ noCache: boolean; cachePath: string | null; cacheSizeBytes: number; cacheSizeFormatted: string }> {
  // Safety: must be inside a models/ directory, must be a directory itself.
  const resolved = path.resolve(modelPath);
  const parentDir = path.basename(path.dirname(resolved));
  if (parentDir !== 'models') {
    throw new Error(`Safety check failed: "${resolved}" is not inside a models/ directory`);
  }
  const stat = await fsp.stat(resolved).catch(() => null);
  if (!stat || !stat.isDirectory()) {
    throw new Error(`Not a model directory: "${resolved}"`);
  }

  const markerPath = path.join(resolved, NO_CACHE_MARKER);
  const cachePath = resolved + NF4_CACHE_SUFFIX;

  if (enabled) {
    // Idempotent write — `wx` would error if it already existed, but
    // we want to be safe to call repeatedly without surprising the user.
    await fsp.writeFile(markerPath, '', 'utf-8');
    // Also reclaim disk by removing the existing cache directory.
    // rm with force:true is a no-op if the path doesn't exist.
    await fsp.rm(cachePath, { recursive: true, force: true });
    return {
      noCache: true,
      cachePath: null,
      cacheSizeBytes: 0,
      cacheSizeFormatted: formatBytes(0),
    };
  }

  // Disable: remove the marker only — leave the cache alone (it'll
  // be regenerated naturally on the next slow-path load).
  await fsp.rm(markerPath, { force: true });
  const exists = await fileExists(cachePath);
  const cacheSizeBytes = exists ? await getDirSize(cachePath) : 0;
  return {
    noCache: false,
    cachePath: exists ? cachePath : null,
    cacheSizeBytes,
    cacheSizeFormatted: formatBytes(cacheSizeBytes),
  };
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

/**
 * Discover models declared in `llama_cpp_config.json` at the project root.
 *
 * Unlike Ollama (HTTP daemon) or Transformers (HF directories under
 * `models/`), llama.cpp models are explicitly declared in the project's
 * config — paths can live anywhere on disk. We just stat each declared
 * GGUF and surface its launch settings. Returns [] if the config is
 * missing or unparseable.
 */
export async function scanLlamaCppModels(projectRoot: string): Promise<LlamaCppModelInfo[]> {
  const configPath = path.join(projectRoot, 'llama_cpp_config.json');
  let raw: string;
  try {
    raw = await fsp.readFile(configPath, 'utf-8');
  } catch {
    return [];
  }

  let cfg: {
    server_path?: string;
    default_port?: number;
    models?: Record<string, {
      path?: string;
      port?: number;
      num_ctx?: number;
      num_gpu_layers?: number;
      server_path?: string;
      extra_flags?: string[];
    }>;
  };
  try {
    cfg = JSON.parse(raw);
  } catch {
    return [];
  }

  const defaultServerPath = cfg.server_path || 'llama-server';
  const defaultPort = cfg.default_port || 8081;
  const models = cfg.models || {};
  const out: LlamaCppModelInfo[] = [];

  for (const [name, m] of Object.entries(models)) {
    const ggufPath = m.path || '';
    const flags = m.extra_flags || [];
    const hasMmproj = flags.some((f) => f === '--mmproj');
    let sizeBytes = 0;
    let ggufMissing = true;
    if (ggufPath) {
      try {
        const stat = await fsp.stat(ggufPath);
        sizeBytes = stat.size;
        ggufMissing = false;
      } catch {
        // Leave defaults — surfaced as `ggufMissing: true` in the card
      }
    }

    out.push({
      name,
      ggufPath,
      sizeBytes,
      sizeFormatted: formatBytes(sizeBytes),
      port: m.port || defaultPort,
      numCtx: typeof m.num_ctx === 'number' ? m.num_ctx : null,
      numGpuLayers: typeof m.num_gpu_layers === 'number' ? m.num_gpu_layers : 99,
      serverPath: m.server_path || defaultServerPath,
      hasMmproj,
      flagsSummary: condenseFlags(flags),
      ggufMissing,
      source: 'llama_cpp',
    });
  }

  // Sort by size descending; missing GGUFs (size 0) sink to the bottom
  // so the user notices them.
  out.sort((a, b) => b.sizeBytes - a.sizeBytes);
  return out;
}

/**
 * Compress llama-server's verbose `extra_flags` array into a short list of
 * human-readable badges for the model card. e.g.
 *   ["-fa", "on", "-ctk", "q8_0", "--jinja"] → ["FA on", "KV q8_0", "jinja"]
 */
function condenseFlags(flags: string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < flags.length; i++) {
    const f = flags[i];
    const next = flags[i + 1];
    if (f === '-fa' && next) {
      out.push(`FA ${next}`);
      i++;
    } else if (f === '-ctk' && next) {
      out.push(`KV ${next}`);
      i++;
      // Skip a matching -ctv so we don't double-report when both are equal
      if (flags[i + 1] === '-ctv' && flags[i + 2] === next) i += 2;
    } else if (f === '-ctv' && next) {
      out.push(`KV-V ${next}`);
      i++;
    } else if (f === '--jinja') {
      out.push('jinja');
    } else if (f === '--mmproj') {
      // mmproj surfaces separately as a vision badge; skip its arg here
      i++;
    } else if (f.startsWith('--') || f.startsWith('-')) {
      // Unknown flag — show as-is, swallow its arg if there's one
      out.push(f);
      if (next && !next.startsWith('-')) i++;
    }
  }
  return out;
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
