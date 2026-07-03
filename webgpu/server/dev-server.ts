/**
 * Artifex WebGPU Dev Server
 *
 * Express server that:
 * - Receives POST /metrics from the browser (inference stats, WebGPU info, errors)
 * - Logs metrics to console and metrics.jsonl
 * - Provides a clean feedback loop for development
 */

import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import http from 'http';
import { execFile } from 'child_process';
import { fileURLToPath } from 'url';
import { OrchestrationHub } from './ws-hub.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const METRICS_FILE = path.join(__dirname, '..', 'metrics.jsonl');
const PORT = Number(process.env.PORT) || 3001;

const app = express();
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// ─── Metrics Endpoint ────────────────────────────────────────────────────────

interface MetricPayload {
  timestamp: string;
  event: string;
  data: Record<string, unknown>;
}

app.post('/metrics', (req, res) => {
  const payload: MetricPayload = {
    timestamp: new Date().toISOString(),
    event: req.body.event || 'unknown',
    data: req.body.data || req.body,
  };

  // Console output with color coding
  const eventColors: Record<string, string> = {
    'webgpu-init': '\x1b[36m',      // cyan
    'webgpu-error': '\x1b[31m',     // red
    'kernel-test': '\x1b[33m',      // yellow
    'inference-start': '\x1b[35m',  // magenta
    'inference-token': '\x1b[32m',  // green
    'inference-done': '\x1b[32m',   // green
    'model-load': '\x1b[34m',      // blue
    'perf': '\x1b[33m',            // yellow
  };
  const color = eventColors[payload.event] || '\x1b[37m';
  const reset = '\x1b[0m';

  console.log(`${color}[${payload.event}]${reset} ${JSON.stringify(payload.data, null, 0)}`);

  // Append to metrics file
  try {
    fs.appendFileSync(METRICS_FILE, JSON.stringify(payload) + '\n');
  } catch (err) {
    // Non-fatal: metrics file write failure shouldn't crash the server
  }

  res.json({ ok: true });
});

// ─── Health endpoint ─────────────────────────────────────────────────────────

app.get('/metrics/health', (_req, res) => {
  res.json({
    status: 'running',
    metricsFile: METRICS_FILE,
    uptime: process.uptime(),
  });
});

// ─── Recent metrics viewer ───────────────────────────────────────────────────

app.get('/metrics/recent', (_req, res) => {
  try {
    if (!fs.existsSync(METRICS_FILE)) {
      res.json({ metrics: [] });
      return;
    }
    const lines = fs.readFileSync(METRICS_FILE, 'utf-8').trim().split('\n');
    const recent = lines.slice(-50).map(line => {
      try { return JSON.parse(line); } catch { return null; }
    }).filter(Boolean);
    res.json({ metrics: recent });
  } catch {
    res.json({ metrics: [] });
  }
});

// ─── GPU info (VRAM auto-budget) ─────────────────────────────────────────────
// WebGPU never exposes VRAM size, but the dev server shares the machine with
// the driver. The browser picks the entry with displayActive (Chrome's GPU
// process runs on the display adapter) to size its load budget. Localhost
// only — this never leaves the machine. Non-NVIDIA boxes return [] and the
// loader falls back to its conservative default.

app.get('/api/gpu-info', (_req, res) => {
  execFile(
    'nvidia-smi',
    ['--query-gpu=name,memory.total,memory.free,display_active', '--format=csv,noheader,nounits'],
    { timeout: 5000 },
    (err, stdout) => {
      if (err) return res.json({ gpus: [] });
      const gpus = stdout.trim().split('\n').filter(Boolean).map(line => {
        const parts = line.split(',').map(s => s.trim());
        return {
          name: parts[0],
          totalMB: Number(parts[1]) || 0,
          freeMB: Number(parts[2]) || 0,
          displayActive: /enabled/i.test(parts[3] ?? ''),
        };
      });
      res.json({ gpus });
    },
  );
});

// ─── Web search (self-owned, keyless) ────────────────────────────────────────
// GET /api/search?q=... — the browser's web_search tool routes here so no
// third-party API key or account ever exists, client-side or otherwise.
// Engines:
//   - ARTIFEX_SEARXNG_URL env — a self-hosted SearXNG instance (JSON API),
//     for fully self-owned search infrastructure
//   - default — DuckDuckGo's HTML endpoint, parsed server-side (keyless)
// Privacy/safety shape: only the query string leaves the machine, sent
// directly to the engine; nothing is stored or logged beyond the console.
// The endpoint fetches exactly one hard-coded engine URL — client input is
// never used as a fetch target, so there is no SSRF surface toward LAN
// services. Localhost-bound like the rest of this server.

const SEARXNG_URL = (process.env.ARTIFEX_SEARXNG_URL ?? '').replace(/\/+$/, '');
const SEARCH_TIMEOUT_MS = 10_000;

function decodeEntities(s: string): string {
  return s
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&nbsp;/g, ' ');
}

function stripTags(s: string): string {
  return decodeEntities(s.replace(/<[^>]*>/g, '')).replace(/\s+/g, ' ').trim();
}

interface SearchResult { title: string; url: string; snippet: string }

async function searchSearxng(q: string, max: number): Promise<SearchResult[]> {
  const resp = await fetch(
    `${SEARXNG_URL}/search?q=${encodeURIComponent(q)}&format=json`,
    { signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS) });
  if (!resp.ok) throw new Error(`searxng returned ${resp.status}`);
  const data = await resp.json() as { results?: Array<{ title?: string; url?: string; content?: string }> };
  return (data.results ?? []).slice(0, max).map(r => ({
    title: r.title ?? '',
    url: r.url ?? '',
    snippet: (r.content ?? '').slice(0, 300),
  }));
}

async function searchDuckDuckGo(q: string, max: number): Promise<SearchResult[]> {
  const resp = await fetch(
    `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`,
    {
      headers: {
        // The HTML endpoint serves plain markup to browser UAs; the default
        // undici UA gets challenged.
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
        'Accept': 'text/html',
      },
      signal: AbortSignal.timeout(SEARCH_TIMEOUT_MS),
    });
  if (!resp.ok) throw new Error(`engine returned ${resp.status}`);
  const html = await resp.text();

  // Result anchors: <a class="result__a" href="//duckduckgo.com/l/?uddg=<url>...">Title</a>
  // paired in document order with <a class="result__snippet" ...>snippet</a>.
  const links = [...html.matchAll(/class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g)];
  const snippets = [...html.matchAll(/class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g)];
  const resolveUrl = (href: string): string => {
    const m = href.match(/[?&]uddg=([^&]+)/);
    if (m) { try { return decodeURIComponent(m[1]); } catch { /* keep raw */ } }
    return href.startsWith('//') ? `https:${href}` : href;
  };
  const results: SearchResult[] = [];
  for (let i = 0; i < links.length && results.length < max; i++) {
    const url = resolveUrl(decodeEntities(links[i][1]));
    if (/duckduckgo\.com\/y\.js/.test(url)) continue; // ad click-through
    results.push({
      title: stripTags(links[i][2]),
      url,
      snippet: stripTags(snippets[i]?.[1] ?? '').slice(0, 300),
    });
  }
  return results;
}

app.get('/api/search', async (req, res) => {
  const q = String(req.query.q ?? '').trim();
  if (!q) return res.status(400).json({ error: 'missing q parameter' });
  const max = Math.min(Math.max(Number(req.query.max) || 5, 1), 10);
  const engine = SEARXNG_URL ? 'searxng' : 'duckduckgo';
  try {
    const results = SEARXNG_URL
      ? await searchSearxng(q, max)
      : await searchDuckDuckGo(q, max);
    console.log(`\x1b[35m[Search]\x1b[0m ${engine}: "${q}" → ${results.length} results`);
    res.json({ engine, results });
  } catch (err) {
    console.warn(`\x1b[35m[Search]\x1b[0m ${engine} failed for "${q}": ${err}`);
    res.status(502).json({ error: `search failed: ${err instanceof Error ? err.message : err}` });
  }
});

// ─── Local HuggingFace Cache ─────────────────────────────────────────────────
// Serves SafeTensors files from the local HF cache, eliminating CDN downloads.
// The browser's hf-hub.ts switches its base URL to use these endpoints instead.

const HF_CACHE_DIR = process.env.HF_HOME
  ? path.join(process.env.HF_HOME, 'hub')
  : path.join(process.env.USERPROFILE || process.env.HOME || '', '.cache', 'huggingface', 'hub');

// Project's local models directory (for custom quantized models)
// Project model directories, in priority order. The repo-local `models/`
// works standalone; the parent `../models` keeps the in-Artifex layout
// working unchanged. Either can be overridden with ARTIFEX_PROJECT_MODELS.
const PROJECT_MODELS_DIRS: string[] = (
  process.env.ARTIFEX_PROJECT_MODELS
    ? process.env.ARTIFEX_PROJECT_MODELS.split(path.delimiter)
    : [path.resolve(__dirname, '..', 'models'), path.resolve(__dirname, '..', '..', 'models')]
).filter(d => { try { return fs.statSync(d).isDirectory(); } catch { return false; } });

// ─── Extra model directories (private to this machine) ──────────────────────
// Two config sources, both server-side only:
//   - ARTIFEX_MODEL_DIRS env var — ';'-separated directory list
//   - model-dirs.local.json (gitignored, next to package.json) — JSON string array
// Each directory is scanned for loose *.gguf files and for model subdirectories
// (config.json or *.gguf inside). Everything surfaces to the browser as
// 'local/<alias>' — absolute paths never reach the client, so no box/system
// info leaks into the UI, exports, or the public repo.

const EXTRA_DIRS_FILE = path.join(__dirname, '..', 'model-dirs.local.json');

function readExtraModelDirs(): string[] {
  const dirs: string[] = [];
  const env = process.env.ARTIFEX_MODEL_DIRS;
  if (env) dirs.push(...env.split(';').map(s => s.trim()).filter(Boolean));
  if (fs.existsSync(EXTRA_DIRS_FILE)) {
    try {
      const fromFile = JSON.parse(fs.readFileSync(EXTRA_DIRS_FILE, 'utf-8'));
      if (Array.isArray(fromFile)) dirs.push(...fromFile.filter((d): d is string => typeof d === 'string'));
      else console.warn(`[Models] ${path.basename(EXTRA_DIRS_FILE)} must be a JSON array of directory strings`);
    } catch (err) {
      // A malformed config should be LOUD — common mistake is unescaped
      // backslashes in Windows paths (use forward slashes or \\).
      console.warn(`[Models] Failed to parse ${path.basename(EXTRA_DIRS_FILE)}: ${err}`);
    }
  }
  const valid = dirs.filter(d => {
    try { return fs.statSync(d).isDirectory(); } catch { return false; }
  });
  for (const d of dirs) {
    if (!valid.includes(d)) console.warn(`[Models] Configured model dir not found: ${d}`);
  }
  return valid;
}

/** A resolved local model: a directory, optionally restricted to one file
 *  (loose .gguf files and Ollama blobs get their own alias). displayName
 *  is what the browser sees when the on-disk name is unfriendly (blobs). */
interface LocalModelDir { dir: string; only?: string; displayName?: string }

// ─── Ollama model store ──────────────────────────────────────────────────────
// Ollama blobs ARE GGUF files under sha256 names. Manifests map name:tag →
// model blob. Surfaced as 'ollama/<name>:<tag>' with a friendly .gguf
// displayName; the blob path stays server-side like everything else.

const OLLAMA_DIR = process.env.OLLAMA_MODELS
  || path.join(process.env.USERPROFILE || process.env.HOME || '', '.ollama', 'models');

function scanOllamaModels(): Map<string, LocalModelDir> {
  const map = new Map<string, LocalModelDir>();
  const manifestsRoot = path.join(OLLAMA_DIR, 'manifests');
  const blobsDir = path.join(OLLAMA_DIR, 'blobs');
  const walk = (dir: string, rel: string[]) => {
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      const p = path.join(dir, e.name);
      if (e.isDirectory()) {
        walk(p, [...rel, e.name]);
      } else if (e.isFile()) {
        // rel = [registry, namespace, ...nameParts], file name = tag
        try {
          const manifest = JSON.parse(fs.readFileSync(p, 'utf-8'));
          const layer = (manifest.layers ?? []).find(
            (l: any) => l.mediaType === 'application/vnd.ollama.image.model');
          if (!layer?.digest) continue;
          const blob = String(layer.digest).replace(':', '-');
          if (!fs.existsSync(path.join(blobsDir, blob))) continue;
          const namespace = rel[1] ?? 'library';
          const name = rel.slice(2).join('/');
          if (!name) continue;
          const alias = (namespace === 'library' ? name : `${namespace}.${name.replace(/\//g, '.')}`) + ':' + e.name;
          const displayName = `${namespace === 'library' ? '' : namespace + '-'}${name.replace(/\//g, '-')}-${e.name}.gguf`;
          if (!map.has(alias)) map.set(alias, { dir: blobsDir, only: blob, displayName });
        } catch {}
      }
    }
  };
  walk(manifestsRoot, []);
  return map;
}

/** Scan the extra dirs: alias → location. Project models/ wins name collisions. */
function scanExtraModels(): Map<string, LocalModelDir> {
  const map = new Map<string, LocalModelDir>();
  for (const dir of readExtraModelDirs()) {
    let entries: fs.Dirent[];
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      if (e.isFile() && e.name.toLowerCase().endsWith('.gguf')) {
        const alias = e.name.replace(/\.gguf$/i, '');
        if (!map.has(alias)) map.set(alias, { dir, only: e.name });
      } else if (e.isDirectory()) {
        const sub = path.join(dir, e.name);
        try {
          const files = fs.readdirSync(sub);
          const isModel = files.includes('config.json') || files.some(f => f.toLowerCase().endsWith('.gguf'));
          if (isModel && !map.has(e.name)) map.set(e.name, { dir: sub });
        } catch {}
      }
    }
  }
  return map;
}

// During a model load the browser issues hundreds of range requests; scanning
// the filesystem per request is wasteful AND fragile (a single transient fs
// error under heavy streaming I/O would 404 a model that exists). Cache the
// scans briefly.
const SCAN_TTL_MS = 5000;
let _extraScan: { at: number; map: Map<string, LocalModelDir> } | null = null;
let _ollamaScan: { at: number; map: Map<string, LocalModelDir> } | null = null;

function scanExtraModelsCached(): Map<string, LocalModelDir> {
  if (_extraScan && Date.now() - _extraScan.at < SCAN_TTL_MS) return _extraScan.map;
  const map = scanExtraModels();
  // Keep serving the previous non-empty scan if a rescan transiently comes
  // back empty (fs hiccup mid-load) — a model that existed 5s ago still does.
  if (map.size === 0 && _extraScan && _extraScan.map.size > 0) {
    _extraScan.at = Date.now();
    return _extraScan.map;
  }
  _extraScan = { at: Date.now(), map };
  return map;
}

function scanOllamaModelsCached(): Map<string, LocalModelDir> {
  if (_ollamaScan && Date.now() - _ollamaScan.at < SCAN_TTL_MS) return _ollamaScan.map;
  const map = scanOllamaModels();
  if (map.size === 0 && _ollamaScan && _ollamaScan.map.size > 0) {
    _ollamaScan.at = Date.now();
    return _ollamaScan.map;
  }
  _ollamaScan = { at: Date.now(), map };
  return map;
}

/** Resolve a repo ID to its local location (project models/ → extra dirs → Ollama → HF cache) */
function resolveSnapshot(repo: string): LocalModelDir | null {
  // 'local/<name>' maps to models/<name>/ or an extra-dir alias
  if (repo.startsWith('local/')) {
    const name = repo.slice(6); // strip "local/"
    for (const base of PROJECT_MODELS_DIRS) {
      const modelDir = path.join(base, name);
      if (fs.existsSync(modelDir)) return { dir: modelDir };
    }
    const hit = scanExtraModelsCached().get(name) ?? null;
    if (!hit) console.warn(`[Models] Unresolved local alias: ${repo}`);
    return hit;
  }

  // 'ollama/<name>:<tag>' maps to the model blob via the manifest
  if (repo.startsWith('ollama/')) {
    const hit = scanOllamaModelsCached().get(repo.slice(7)) ?? null;
    if (!hit) console.warn(`[Models] Unresolved ollama alias: ${repo}`);
    return hit;
  }

  // Check HF cache
  const dirName = `models--${repo.replace(/\//g, '--')}`;
  const refsPath = path.join(HF_CACHE_DIR, dirName, 'refs', 'main');
  try {
    const hash = fs.readFileSync(refsPath, 'utf-8').trim();
    const snapDir = path.join(HF_CACHE_DIR, dirName, 'snapshots', hash);
    if (fs.existsSync(snapDir)) return { dir: snapDir };
  } catch {}
  return null;
}

/** Files visible for a resolved model (respects loose-file restriction). */
function listModelFiles(loc: LocalModelDir): string[] {
  if (loc.only) return [loc.displayName ?? loc.only];
  try { return fs.readdirSync(loc.dir); } catch { return []; }
}

/** List all locally cached models (HF cache + project models/) */
app.get('/api/hf-cache/models', (_req, res) => {
  try {
    const models: Array<{ repo: string; files: string[]; totalSize: number }> = [];

    const isModelFile = (f: string) =>
      f.endsWith('.safetensors') || f.toLowerCase().endsWith('.gguf') || f === 'config.json';
    const sizeOf = (dir: string, files: string[]) => files.reduce((s, f) => {
      try { return s + fs.statSync(path.join(dir, f)).size; } catch { return s; }
    }, 0);
    const seen = new Set<string>();
    const push = (repo: string, dir: string, allFiles: string[]) => {
      if (seen.has(repo)) return;
      const files = allFiles.filter(isModelFile);
      if (files.length === 0) return;
      seen.add(repo);
      models.push({ repo, files, totalSize: sizeOf(dir, files) });
    };

    // Scan HF cache
    try {
      const entries = fs.readdirSync(HF_CACHE_DIR).filter(d => d.startsWith('models--'));
      for (const dir of entries) {
        const repo = dir.replace('models--', '').replace(/--/g, '/');
        const snap = resolveSnapshot(repo);
        if (snap) push(repo, snap.dir, listModelFiles(snap));
      }
    } catch {}

    // Scan project models/ directories (repo-local + optional parent)
    for (const base of PROJECT_MODELS_DIRS) {
      try {
        for (const dir of fs.readdirSync(base)) {
          const fullDir = path.join(base, dir);
          try { if (!fs.statSync(fullDir).isDirectory()) continue; } catch { continue; }
          push(`local/${dir}`, fullDir, fs.readdirSync(fullDir));
        }
      } catch {}
    }

    // Scan configured extra dirs (loose .gguf files + model subdirectories)
    // and the Ollama store. Only aliases are sent to the browser — never the
    // underlying paths.
    const pushLoc = (repo: string, loc: LocalModelDir) => {
      if (!loc.only) { push(repo, loc.dir, listModelFiles(loc)); return; }
      if (seen.has(repo)) return;
      try {
        const size = fs.statSync(path.join(loc.dir, loc.only)).size;
        seen.add(repo);
        models.push({ repo, files: [loc.displayName ?? loc.only], totalSize: size });
      } catch {}
    };
    for (const [alias, loc] of scanExtraModelsCached()) pushLoc(`local/${alias}`, loc);
    for (const [alias, loc] of scanOllamaModelsCached()) pushLoc(`ollama/${alias}`, loc);

    res.json(models);
  } catch (err) {
    // Do NOT echo directory paths to the client — machine info stays server-side.
    res.status(500).json({ error: 'Failed to read local model directories' });
  }
});

/** Emulate HF API tree listing */
app.get('/api/hf-cache/:org/:model/tree/main', (req, res) => {
  const repo = `${req.params.org}/${req.params.model}`;
  const snap = resolveSnapshot(repo);
  if (!snap) return res.status(404).json({ error: `Model not found in local cache: ${repo}` });

  // Single-file aliases (loose .gguf, Ollama blobs): stat the real file but
  // report the friendly name.
  if (snap.only) {
    try {
      const stat = fs.statSync(path.join(snap.dir, snap.only));
      return res.json([{ path: snap.displayName ?? snap.only, size: stat.size, type: 'file' as const }]);
    } catch {
      return res.json([]);
    }
  }
  const files = listModelFiles(snap).map(f => {
    try {
      const stat = fs.statSync(path.join(snap.dir, f));
      return { path: f, size: stat.size, type: 'file' as const };
    } catch { return null; }
  }).filter(Boolean);
  res.json(files);
});

/** Serve files with HTTP Range support (for chunked downloads) */
app.get('/api/hf-cache/:org/:model/:action(resolve|raw)/main/*', (req, res) => {
  const repo = `${req.params.org}/${req.params.model}`;
  const filename = (req.params as any)[0] as string; // wildcard capture after /main/
  const snap = resolveSnapshot(repo);
  if (!snap) return res.status(404).json({ error: `Model not found: ${repo}` });

  // Single-file aliases serve exactly one file (requested by real or display
  // name); everything else in the directory stays hidden.
  let realName = filename;
  if (snap.only) {
    if (filename !== snap.only && filename !== (snap.displayName ?? snap.only)) {
      return res.status(404).json({ error: `File not found: ${filename}` });
    }
    realName = snap.only;
  }
  // Containment: reject path traversal out of the model directory.
  const filePath = path.resolve(snap.dir, realName);
  if (!filePath.startsWith(path.resolve(snap.dir) + path.sep)) {
    return res.status(404).json({ error: `File not found: ${filename}` });
  }
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: `File not found: ${filename}` });

  const stat = fs.statSync(filePath);
  const fileSize = stat.size;

  // JSON files (config.json, index.json)
  if (filename.endsWith('.json')) {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Content-Length', fileSize);
    return fs.createReadStream(filePath).pipe(res);
  }

  // Binary files with Range support
  res.setHeader('Accept-Ranges', 'bytes');
  res.setHeader('Content-Type', 'application/octet-stream');

  // Strided gather (MoE row-split expert strips): returns `gatherCount`
  // chunks of `gatherChunk` bytes read at gatherStart + i*gatherStride,
  // concatenated. Lets a worker fetch its 1/N row-strip of all experts in
  // one round-trip per tensor instead of 256 tiny Range requests.
  if (req.query.gatherStart !== undefined) {
    const start = Number(req.query.gatherStart);
    const stride = Number(req.query.gatherStride);
    const chunk = Number(req.query.gatherChunk);
    const count = Number(req.query.gatherCount);
    const valid =
      [start, stride, chunk, count].every((v) => Number.isSafeInteger(v) && v >= 0)
      && chunk > 0 && count > 0 && stride >= chunk
      && chunk * count <= 1 << 30
      && start + (count - 1) * stride + chunk <= fileSize;
    if (!valid) return res.status(416).json({ error: 'bad gather params' });
    const total = chunk * count;
    (async () => {
      const fd = await fs.promises.open(filePath, 'r');
      try {
        const buf = Buffer.allocUnsafe(total);
        for (let i = 0; i < count; i++) {
          const { bytesRead } = await fd.read(buf, i * chunk, chunk, start + i * stride);
          if (bytesRead !== chunk) throw new Error(`short read: chunk ${i} got ${bytesRead}/${chunk}`);
        }
        res.setHeader('Content-Length', total);
        res.end(buf);
      } finally {
        await fd.close();
      }
    })().catch((err) => {
      if (!res.headersSent) res.status(500).json({ error: String(err) });
      else res.destroy();
    });
    return;
  }

  // A read stream that errors mid-flight (transient EBUSY/EIO under heavy
  // concurrent I/O, antivirus interference) must close the socket cleanly —
  // an unhandled stream error leaves the client request hanging.
  const streamWithErrorHandling = (stream: fs.ReadStream) => {
    stream.on('error', (err) => {
      console.error(`[Serve] read stream error for ${path.basename(filePath)}: ${err}`);
      res.destroy();
    });
    stream.pipe(res);
  };

  const rangeHeader = req.headers.range;
  if (rangeHeader) {
    const [startStr, endStr] = rangeHeader.replace('bytes=', '').split('-');
    const start = parseInt(startStr, 10);
    const end = endStr ? parseInt(endStr, 10) : fileSize - 1;
    const chunkSize = end - start + 1;

    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${fileSize}`,
      'Content-Length': chunkSize,
    });
    streamWithErrorHandling(fs.createReadStream(filePath, { start, end }));
  } else {
    res.setHeader('Content-Length', fileSize);
    streamWithErrorHandling(fs.createReadStream(filePath));
  }
});

// ─── Start ───────────────────────────────────────────────────────────────────

// ─── Start with WebSocket Hub ─────────────────────────────────────────────

const server = http.createServer(app);
const hub = new OrchestrationHub(server);

server.listen(PORT, '127.0.0.1', () => {
  // Model discovery summary — counts only, no paths (console may be shared in
  // screenshots/issues). Misconfigured dirs warn loudly in readExtraModelDirs.
  try {
    const extra = scanExtraModelsCached().size;
    const ollama = scanOllamaModelsCached().size;
    console.log(`\x1b[33m[Models]\x1b[0m extra-dir aliases: ${extra} | ollama models: ${ollama} (plus project models/ + HF cache)`);
  } catch {}
  console.log(`\x1b[36m╔════════════════════════════════════════════╗\x1b[0m`);
  console.log(`\x1b[36m║  Artifex WebGPU Dev Server                 ║\x1b[0m`);
  console.log(`\x1b[36m║  Metrics endpoint: http://localhost:${PORT}   ║\x1b[0m`);
  console.log(`\x1b[36m║  WebSocket hub:    ws://localhost:${PORT}/ws  ║\x1b[0m`);
  console.log(`\x1b[36m║  POST /metrics — receive browser metrics    ║\x1b[0m`);
  console.log(`\x1b[36m║  GET  /metrics/recent — view last 50        ║\x1b[0m`);
  console.log(`\x1b[36m╚════════════════════════════════════════════╝\x1b[0m`);
});
