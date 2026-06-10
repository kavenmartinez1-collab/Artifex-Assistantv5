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
import { fileURLToPath } from 'url';
import { OrchestrationHub } from './ws-hub.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const METRICS_FILE = path.join(__dirname, '..', 'metrics.jsonl');
const PORT = 3001;

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

// ─── Local HuggingFace Cache ─────────────────────────────────────────────────
// Serves SafeTensors files from the local HF cache, eliminating CDN downloads.
// The browser's hf-hub.ts switches its base URL to use these endpoints instead.

const HF_CACHE_DIR = process.env.HF_HOME
  ? path.join(process.env.HF_HOME, 'hub')
  : path.join(process.env.USERPROFILE || process.env.HOME || '', '.cache', 'huggingface', 'hub');

// Project's local models directory (for custom quantized models)
const LOCAL_MODELS_DIR = path.resolve(__dirname, '..', '..', 'models');

/** Resolve a repo ID to its local directory (checks project models/ first, then HF cache) */
function resolveSnapshot(repo: string): string | null {
  // Check project models/ directory first (e.g., "local/qwen3.5-9b-mixed-GPTQ-Int4")
  // Convention: "local/<dirname>" maps to models/<dirname>/
  if (repo.startsWith('local/')) {
    const dirName = repo.slice(6); // strip "local/"
    const modelDir = path.join(LOCAL_MODELS_DIR, dirName);
    if (fs.existsSync(modelDir)) return modelDir;
    return null;
  }

  // Check HF cache
  const dirName = `models--${repo.replace(/\//g, '--')}`;
  const refsPath = path.join(HF_CACHE_DIR, dirName, 'refs', 'main');
  try {
    const hash = fs.readFileSync(refsPath, 'utf-8').trim();
    const snapDir = path.join(HF_CACHE_DIR, dirName, 'snapshots', hash);
    if (fs.existsSync(snapDir)) return snapDir;
  } catch {}
  return null;
}

/** List all locally cached models (HF cache + project models/) */
app.get('/api/hf-cache/models', (_req, res) => {
  try {
    const models: Array<{ repo: string; files: string[]; totalSize: number }> = [];

    // Scan HF cache
    try {
      const entries = fs.readdirSync(HF_CACHE_DIR).filter(d => d.startsWith('models--'));
      for (const dir of entries) {
        const repo = dir.replace('models--', '').replace(/--/g, '/');
        const snap = resolveSnapshot(repo);
        if (!snap) continue;
        const files = fs.readdirSync(snap).filter(f => f.endsWith('.safetensors') || f.endsWith('.gguf') || f === 'config.json');
        const totalSize = files.reduce((s, f) => {
          try { return s + fs.statSync(path.join(snap, f)).size; } catch { return s; }
        }, 0);
        if (files.length > 0) models.push({ repo, files, totalSize });
      }
    } catch {}

    // Scan project models/ directory
    try {
      const localEntries = fs.readdirSync(LOCAL_MODELS_DIR);
      for (const dir of localEntries) {
        const fullDir = path.join(LOCAL_MODELS_DIR, dir);
        if (!fs.statSync(fullDir).isDirectory()) continue;
        // A model dir has config.json (safetensors) or .gguf files
        const allFiles = fs.readdirSync(fullDir);
        const hasConfig = fs.existsSync(path.join(fullDir, 'config.json'));
        const hasGguf = allFiles.some(f => f.endsWith('.gguf'));
        if (!hasConfig && !hasGguf) continue;
        const files = allFiles.filter(f => f.endsWith('.safetensors') || f.endsWith('.gguf') || f === 'config.json');
        const totalSize = files.reduce((s, f) => {
          try { return s + fs.statSync(path.join(fullDir, f)).size; } catch { return s; }
        }, 0);
        models.push({ repo: `local/${dir}`, files, totalSize });
      }
    } catch {}

    res.json(models);
  } catch (err) {
    res.status(500).json({ error: 'Failed to read cache', paths: [HF_CACHE_DIR, LOCAL_MODELS_DIR] });
  }
});

/** Emulate HF API tree listing */
app.get('/api/hf-cache/:org/:model/tree/main', (req, res) => {
  const repo = `${req.params.org}/${req.params.model}`;
  const snap = resolveSnapshot(repo);
  if (!snap) return res.status(404).json({ error: `Model not found in local cache: ${repo}` });

  const files = fs.readdirSync(snap).map(f => {
    try {
      const stat = fs.statSync(path.join(snap, f));
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

  const filePath = path.join(snap, filename);
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
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.setHeader('Content-Length', fileSize);
    fs.createReadStream(filePath).pipe(res);
  }
});

// ─── Start ───────────────────────────────────────────────────────────────────

// ─── Start with WebSocket Hub ─────────────────────────────────────────────

const server = http.createServer(app);
const hub = new OrchestrationHub(server);

server.listen(PORT, '127.0.0.1', () => {
  console.log(`\x1b[36m╔════════════════════════════════════════════╗\x1b[0m`);
  console.log(`\x1b[36m║  Artifex WebGPU Dev Server                 ║\x1b[0m`);
  console.log(`\x1b[36m║  Metrics endpoint: http://localhost:${PORT}   ║\x1b[0m`);
  console.log(`\x1b[36m║  WebSocket hub:    ws://localhost:${PORT}/ws  ║\x1b[0m`);
  console.log(`\x1b[36m║  POST /metrics — receive browser metrics    ║\x1b[0m`);
  console.log(`\x1b[36m║  GET  /metrics/recent — view last 50        ║\x1b[0m`);
  console.log(`\x1b[36m╚════════════════════════════════════════════╝\x1b[0m`);
});
