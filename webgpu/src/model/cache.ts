/**
 * Browser Cache Layer for Model Weights
 *
 * Uses the Cache API to persist downloaded SafeTensors shards across browser sessions.
 * When a user loads a model the second time, shards are read from the browser cache
 * instead of re-downloading from HuggingFace.
 *
 * The Cache API is:
 *   - Persistent (survives browser restart)
 *   - Large (no practical size limit — Chrome allows up to 80% of disk)
 *   - Async (non-blocking)
 *   - Available in all modern browsers
 */

const CACHE_NAME = 'artifex-model-cache-v1';

/**
 * Check if a key exists in the cache.
 */
export async function hasCache(key: string): Promise<boolean> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const resp = await cache.match(keyToUrl(key));
    return resp !== undefined;
  } catch {
    return false;
  }
}

/**
 * Get data from the cache.
 * Returns null if not found.
 */
export async function getCache(key: string): Promise<ArrayBuffer | null> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const resp = await cache.match(keyToUrl(key));
    if (!resp) return null;
    return resp.arrayBuffer();
  } catch {
    return null;
  }
}

/**
 * Store data in the cache.
 */
export async function putCache(key: string, data: ArrayBuffer): Promise<void> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const resp = new Response(data, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Length': data.byteLength.toString(),
        'X-Cached-At': new Date().toISOString(),
      },
    });
    await cache.put(keyToUrl(key), resp);
  } catch (e) {
    console.warn(`[Cache] Failed to cache ${key}:`, e);
  }
}

/**
 * Remove a specific key from the cache.
 */
export async function removeCache(key: string): Promise<boolean> {
  try {
    const cache = await caches.open(CACHE_NAME);
    return cache.delete(keyToUrl(key));
  } catch {
    return false;
  }
}

/**
 * Clear the entire model cache.
 */
export async function clearCache(): Promise<boolean> {
  try {
    return caches.delete(CACHE_NAME);
  } catch {
    return false;
  }
}

/**
 * Remove every cached shard belonging to one model repo.
 * Returns the number of entries deleted.
 */
export async function removeModelFromCache(repo: string): Promise<number> {
  let removed = 0;
  try {
    const cache = await caches.open(CACHE_NAME);
    for (const request of await cache.keys()) {
      const key = urlToKey(request.url);
      if (key === repo || key.startsWith(`${repo}/`)) {
        if (await cache.delete(request)) removed++;
      }
    }
  } catch {
    // Cache API not available
  }
  return removed;
}

/**
 * Get the total size of all cached items and list cached models.
 */
export async function getCacheStats(): Promise<{
  totalBytes: number;
  itemCount: number;
  models: Map<string, { shardCount: number; totalBytes: number }>;
}> {
  const models = new Map<string, { shardCount: number; totalBytes: number }>();
  let totalBytes = 0;
  let itemCount = 0;

  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();

    for (const request of keys) {
      const resp = await cache.match(request);
      if (!resp) continue;

      const size = parseInt(resp.headers.get('Content-Length') || '0', 10);
      totalBytes += size;
      itemCount++;

      // Extract model repo from the URL key. Keys are `${repo}/${filename}`
      // for whole-shard entries and `${repo}/${filename}/chunk-${i}` for
      // streamed shards — strip the chunk suffix before the filename.
      const key = urlToKey(request.url);
      const parts = key.split('/');
      if (parts.length > 2 && /^chunk-\d+$/.test(parts[parts.length - 1])) parts.pop();
      const repo = parts.slice(0, -1).join('/'); // "Qwen/Qwen3.5-0.6B"
      const existing = models.get(repo) || { shardCount: 0, totalBytes: 0 };
      existing.shardCount++;
      existing.totalBytes += size;
      models.set(repo, existing);
    }
  } catch {
    // Cache API not available
  }

  return { totalBytes, itemCount, models };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Convert a cache key to a fake URL (Cache API requires Request/URL objects).
 */
function keyToUrl(key: string): string {
  return `https://artifex-cache.local/${encodeURIComponent(key)}`;
}

/**
 * Convert a cache URL back to the original key.
 */
function urlToKey(url: string): string {
  const path = new URL(url).pathname.slice(1); // remove leading /
  return decodeURIComponent(path);
}
