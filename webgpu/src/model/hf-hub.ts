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

const REMOTE_BASE = 'https://huggingface.co';
const REMOTE_API = 'https://huggingface.co/api/models';

let HF_BASE = REMOTE_BASE;
let HF_API = REMOTE_API;
let _localBase = '';

/** Switch to local HF cache served by the dev server (50-100x faster than CDN) */
export function useLocalCache(devServerBase = '/api/hf-cache'): void {
  _localBase = devServerBase;
  HF_BASE = devServerBase;
  HF_API = devServerBase;
  console.log(`[HF Hub] Using local cache: ${devServerBase}`);
}

/** Switch back to HuggingFace CDN */
export function resetToRemote(): void {
  _localBase = '';
  HF_BASE = REMOTE_BASE;
  HF_API = REMOTE_API;
  console.log(`[HF Hub] Using remote CDN`);
}

/** True for repos that exist ONLY on this machine (local/ aliases, Ollama
 *  blobs). Falling back to the HF CDN for these is never meaningful — it
 *  turns a transient local error into a confusing remote 401/404. */
function isMachineLocalUrl(url: string): boolean {
  return _localBase !== ''
    && (url.startsWith(`${_localBase}/local/`) || url.startsWith(`${_localBase}/ollama/`));
}

/** Fetch with local-first, CDN-fallback. If local returns 404, retry from CDN —
 *  except for machine-local repos, which retry locally with backoff and never
 *  go remote (transient fs/socket hiccups under heavy streaming are real on
 *  Windows; a remote fallback for these only manufactures phantom 401/404s). */
async function fetchLocalFirst(localUrl: string, remoteUrl: string, init?: RequestInit): Promise<Response> {
  const rangeHeader = (init?.headers as Record<string, string>)?.Range ?? '';
  const shortName = localUrl.split('/').slice(-1)[0];
  if (_localBase) {
    const localOnly = isMachineLocalUrl(localUrl);

    if (localOnly) {
      const LOCAL_ATTEMPTS = 4;       // 1 try + 3 retries
      const BACKOFF_MS = [0, 300, 1000, 3000];
      let lastErr: unknown = null;
      let lastResp: Response | null = null;
      for (let attempt = 0; attempt < LOCAL_ATTEMPTS; attempt++) {
        if (BACKOFF_MS[attempt] > 0) await new Promise(r => setTimeout(r, BACKOFF_MS[attempt]));
        try {
          if (attempt > 0) console.warn(`[HF Hub] Local retry ${attempt}/${LOCAL_ATTEMPTS - 1}: ${shortName} ${rangeHeader}`);
          const resp = await fetch(localUrl, init);
          if (resp.ok || resp.status === 206) return resp;
          console.warn(`[HF Hub] Local non-OK: ${resp.status} for ${shortName}`);
          lastResp = resp;
          lastErr = null;
        } catch (err) {
          console.warn(`[HF Hub] Local fetch threw for ${shortName} ${rangeHeader}:`, err);
          lastErr = err;
          lastResp = null;
        }
      }
      if (lastResp) return lastResp;
      throw lastErr;
    }

    try {
      console.log(`[HF Hub] Local fetch: ${shortName} ${rangeHeader}`);
      const resp = await fetch(localUrl, init);
      if (resp.ok || resp.status === 206) return resp;
      console.warn(`[HF Hub] Local non-OK: ${resp.status} for ${shortName}`);
      if (resp.status === 404) {
        console.log(`[HF Hub] Local miss, falling back to CDN: ${remoteUrl.split('/').slice(-1)[0]}`);
        return fetchWithRetry(remoteUrl, init);
      }
    } catch (err) {
      console.error(`[HF Hub] Local fetch THREW for ${shortName} ${rangeHeader}:`, err);
      console.log(`[HF Hub] Falling back to CDN: ${remoteUrl}`);
      return fetchWithRetry(remoteUrl, init);
    }
  }
  return fetchWithRetry(localUrl, init);
}

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000; // 1s, 2s, 4s exponential backoff

// ─── Retry Logic ────────────────────────────────────────────────────────────

/** Returns true for errors that are worth retrying (transient failures). */
function isRetryable(error: unknown): boolean {
  if (error instanceof TypeError) {
    // Network failures (ERR_CONTENT_LENGTH_MISMATCH, DNS, timeout, etc.)
    return true;
  }
  if (error instanceof RetryableHTTPError) {
    return true;
  }
  return false;
}

class RetryableHTTPError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/**
 * Fetch wrapper with exponential backoff retry for transient failures.
 * Retries on: network errors, 429 (rate limit), 500-599 (server errors).
 * Does NOT retry on: 401, 403, 404, or other client errors.
 */
async function fetchWithRetry(
  input: RequestInfo,
  init?: RequestInit,
): Promise<Response> {
  let lastError: unknown;

  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    try {
      const resp = await fetch(input, init);

      // Don't retry client errors (except 429)
      if (!resp.ok && resp.status !== 206) {
        if (resp.status === 429 || resp.status >= 500) {
          throw new RetryableHTTPError(resp.status, `HTTP ${resp.status} ${resp.statusText}`);
        }
      }

      return resp;
    } catch (err) {
      lastError = err;

      if (attempt < MAX_RETRIES && isRetryable(err)) {
        const delay = BASE_DELAY_MS * Math.pow(2, attempt);
        console.warn(
          `[HF-Hub] Retry ${attempt + 1}/${MAX_RETRIES} after ${delay}ms — ${err instanceof Error ? err.message : err}`
        );
        await new Promise(resolve => setTimeout(resolve, delay));
        continue;
      }

      throw err;
    }
  }

  throw lastError;
}

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
  const localUrl = `${HF_API}/${repo}/tree/main`;
  const remoteUrl = `${REMOTE_API}/${repo}/tree/main`;
  const resp = await fetchLocalFirst(localUrl, remoteUrl, { headers: authHeaders() });

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
  const localUrl = `${HF_BASE}/${repo}/raw/main/config.json`;
  const remoteUrl = `${REMOTE_BASE}/${repo}/raw/main/config.json`;
  const resp = await fetchLocalFirst(localUrl, remoteUrl, { headers: authHeaders() });

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
  const localUrl = `${HF_BASE}/${repo}/raw/main/model.safetensors.index.json`;
  const remoteUrl = `${REMOTE_BASE}/${repo}/raw/main/model.safetensors.index.json`;
  const resp = await fetchLocalFirst(localUrl, remoteUrl, { headers: authHeaders() });

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

/** Resolve a repo file to its download URL (local dev-server or HF CDN). */
export function resolveFileUrl(repo: string, filename: string): string {
  return `${HF_BASE}/${repo}/resolve/main/${filename}`;
}

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
  // Build CDN fallback URL if using local cache
  const remoteUrl = _localBase && url.startsWith(_localBase)
    ? url.replace(_localBase, REMOTE_BASE)
    : url;
  const init = {
    headers: {
      ...authHeaders(),
      Range: `bytes=${start}-${end - 1}`, // HTTP Range is inclusive on both ends
    },
  };
  const resp = await fetchLocalFirst(url, remoteUrl, init);

  if (!resp.ok && resp.status !== 206) {
    let detail = '';
    try {
      const body = await resp.text();
      detail = body ? ` — ${body.slice(0, 200)}` : '';
    } catch { /* body unreadable */ }
    throw new Error(`Range request failed: ${resp.status} ${resp.statusText}${detail} (${url.split('/').slice(-1)[0]})`);
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
  const resp = await fetchWithRetry(url, { headers: authHeaders() });

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
