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
import { fileURLToPath } from 'url';

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

/** Resolve a repo ID to its local cache snapshot directory */
function resolveSnapshot(repo: string): string | null {
  const dirName = `models--${repo.replace(/\//g, '--')}`;
  const refsPath = path.join(HF_CACHE_DIR, dirName, 'refs', 'main');
  try {
    const hash = fs.readFileSync(refsPath, 'utf-8').trim();
    const snapDir = path.join(HF_CACHE_DIR, dirName, 'snapshots', hash);
    if (fs.existsSync(snapDir)) return snapDir;
  } catch {}
  return null;
}

/** List all locally cached models */
app.get('/api/hf-cache/models', (_req, res) => {
  try {
    const entries = fs.readdirSync(HF_CACHE_DIR).filter(d => d.startsWith('models--'));
    const models = entries.map(dir => {
      const repo = dir.replace('models--', '').replace(/--/g, '/');
      const snap = resolveSnapshot(repo);
      if (!snap) return null;
      const files = fs.readdirSync(snap).filter(f => f.endsWith('.safetensors') || f === 'config.json');
      const totalSize = files.reduce((s, f) => {
        try { return s + fs.statSync(path.join(snap, f)).size; } catch { return s; }
      }, 0);
      return { repo, files, totalSize };
    }).filter(Boolean);
    res.json(models);
  } catch (err) {
    res.status(500).json({ error: 'Failed to read HF cache', path: HF_CACHE_DIR });
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

app.listen(PORT, '127.0.0.1', () => {
  console.log(`\x1b[36m╔════════════════════════════════════════════╗\x1b[0m`);
  console.log(`\x1b[36m║  Artifex WebGPU Dev Server                 ║\x1b[0m`);
  console.log(`\x1b[36m║  Metrics endpoint: http://localhost:${PORT}   ║\x1b[0m`);
  console.log(`\x1b[36m║  POST /metrics — receive browser metrics    ║\x1b[0m`);
  console.log(`\x1b[36m║  GET  /metrics/recent — view last 50        ║\x1b[0m`);
  console.log(`\x1b[36m╚════════════════════════════════════════════╝\x1b[0m`);
});
