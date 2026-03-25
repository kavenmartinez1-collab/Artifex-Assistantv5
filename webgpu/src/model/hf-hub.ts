/**
 * HuggingFace Hub API Client
 *
 * Discovers model files, fetches configs, and downloads SafeTensors shards
 * via HTTP range requests. All requests go directly to huggingface.co CDN.
 *
 * No API key required for public models.
 */

// ─── Types ───────────────────────────────────────────────────────────────────

export interface HFModelFile {
  path: string;        // e.g. "model-00001-of-00004.safetensors"
  size: number;        // bytes
  type: 'file' | 'directory';
  lfs?: { size: number }; // large file storage metadata
}

export interface HFModelConfig {
  model_type: string;
  hidden_size: number;
  num_hidden_layers: number;
  num_attention_heads: number;
  num_key_value_heads: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  rms_norm_eps: number;
  rope_theta?: number;
  tie_word_embeddings?: boolean;
  torch_dtype?: string;
  [key: string]: any;
}

export interface ShardInfo {
  filename: string;
  url: string;
  size: number;
}

export interface DownloadProgress {
  shard: number;
  totalShards: number;
  shardFilename: string;
  bytesDownloaded: number;
  bytesTotal: number;
  overallProgress: number; // 0-1
}

// ─── Constants ───────────────────────────────────────────────────────────────

const HF_BASE = 'https://huggingface.co';
const HF_API = 'https://huggingface.co/api/models';

// ─── Auth Token ─────────────────────────────────────────────────────────────

let _authToken = '';

/**
 * Set the HuggingFace auth token for gated model access.
 * Call before any API requests. Empty string = no auth (public models only).
 */
export function setAuthToken(token: string): void {
  _authToken = token.trim();
}

/** Get auth headers (empty object if no token set). */
function authHeaders(): Record<string, string> {
  return _authToken ? { Authorization: `Bearer ${_authToken}` } : {};
}

// ─── Model Discovery ─────────────────────────────────────────────────────────

/**
 * List all files in a HuggingFace model repo.
 */
export async function listModelFiles(repo: string): Promise<HFModelFile[]> {
  const url = `${HF_API}/${repo}/tree/main`;
  const resp = await fetch(url, { headers: authHeaders() });

  if (!resp.ok) {
    if (resp.status === 404) {
      throw new Error(`Model not found: ${repo}`);
    }
    throw new Error(`HuggingFace API error ${resp.status}: ${await resp.text()}`);
  }

  return resp.json();
}

/**
 * Fetch the model's config.json.
 */
export async function fetchModelConfig(repo: string): Promise<HFModelConfig> {
  const url = `${HF_BASE}/${repo}/raw/main/config.json`;
  const resp = await fetch(url, { headers: authHeaders() });

  if (!resp.ok) {
    throw new Error(`Failed to fetch config.json for ${repo}: ${resp.status}`);
  }

  return resp.json();
}

/**
 * Fetch the safetensors index (model.safetensors.index.json) for multi-shard models.
 * Returns null if the model is single-shard.
 */
export async function fetchShardIndex(repo: string): Promise<Record<string, any> | null> {
  const url = `${HF_BASE}/${repo}/raw/main/model.safetensors.index.json`;
  const resp = await fetch(url, { headers: authHeaders() });

  if (!resp.ok) {
    // Single-shard model — no index file
    return null;
  }

  return resp.json();
}

/**
 * Discover all SafeTensors shard files for a model.
 * Handles both single-shard and multi-shard models.
 */
export async function discoverShards(repo: string): Promise<ShardInfo[]> {
  const files = await listModelFiles(repo);

  // Find safetensors files
  const stFiles = files.filter(f =>
    f.path.endsWith('.safetensors') && f.type === 'file'
  );

  if (stFiles.length === 0) {
    throw new Error(
      `No .safetensors files found in ${repo}. ` +
      `This model may use a different format (GGUF, PyTorch .bin, etc.)`
    );
  }

  // Sort shards by name (model-00001, model-00002, etc.)
  stFiles.sort((a, b) => a.path.localeCompare(b.path));

  return stFiles.map(f => ({
    filename: f.path,
    url: `${HF_BASE}/${repo}/resolve/main/${f.path}`,
    size: f.lfs?.size ?? f.size,
  }));
}

// ─── File Download ───────────────────────────────────────────────────────────

/**
 * Download a byte range from a URL.
 * Uses HTTP Range requests to avoid downloading the entire file at once.
 *
 * @param url - The file URL
 * @param start - Start byte (inclusive)
 * @param end - End byte (exclusive)
 * @returns ArrayBuffer of the requested range
 */
export async function fetchRange(
  url: string,
  start: number,
  end: number,
): Promise<ArrayBuffer> {
  const resp = await fetch(url, {
    headers: {
      ...authHeaders(),
      Range: `bytes=${start}-${end - 1}`, // HTTP Range is inclusive on both ends
    },
  });

  if (!resp.ok && resp.status !== 206) {
    throw new Error(`Range request failed: ${resp.status} ${resp.statusText}`);
  }

  return resp.arrayBuffer();
}

/**
 * Download a complete file with progress reporting.
 * Streams the download in chunks to report progress.
 *
 * @param url - The file URL
 * @param onProgress - Called with bytes downloaded so far and total
 * @returns Complete file as ArrayBuffer
 */
export async function downloadFile(
  url: string,
  onProgress?: (downloaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const resp = await fetch(url, { headers: authHeaders() });

  if (!resp.ok) {
    throw new Error(`Download failed: ${resp.status} ${resp.statusText}`);
  }

  const contentLength = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body?.getReader();

  if (!reader) {
    // Fallback: no streaming support
    return resp.arrayBuffer();
  }

  const chunks: Uint8Array[] = [];
  let downloaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    downloaded += value.byteLength;

    if (onProgress) {
      onProgress(downloaded, contentLength);
    }
  }

  // Concatenate chunks into a single ArrayBuffer
  const result = new Uint8Array(downloaded);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.byteLength;
  }

  return result.buffer;
}

/**
 * Download just the SafeTensors header from a shard.
 * Fetches the first 8 bytes to get header length, then fetches the full header.
 * This avoids downloading the entire multi-GB file just to read metadata.
 */
export async function downloadShardHeader(url: string): Promise<ArrayBuffer> {
  // Step 1: fetch first 8 bytes to get header length
  const lengthBytes = await fetchRange(url, 0, 8);
  const view = new DataView(lengthBytes);
  const headerLen = view.getUint32(0, true);

  // Step 2: fetch the full header (8 bytes + JSON)
  const headerBytes = await fetchRange(url, 0, 8 + headerLen);
  return headerBytes;
}

// ─── Model Info ──────────────────────────────────────────────────────────────

/**
 * Get a complete summary of a model: config, shards, total size.
 */
export async function getModelInfo(repo: string) {
  const [config, shards] = await Promise.all([
    fetchModelConfig(repo),
    discoverShards(repo),
  ]);

  const totalSize = shards.reduce((sum, s) => sum + s.size, 0);

  return {
    repo,
    config,
    shards,
    totalSize,
    shardCount: shards.length,
    modelType: config.model_type,
    hiddenSize: config.hidden_size,
    numLayers: config.num_hidden_layers,
    vocabSize: config.vocab_size,
  };
}
