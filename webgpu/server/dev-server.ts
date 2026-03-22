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

// ─── Start ───────────────────────────────────────────────────────────────────

app.listen(PORT, '127.0.0.1', () => {
  console.log(`\x1b[36m╔════════════════════════════════════════════╗\x1b[0m`);
  console.log(`\x1b[36m║  Artifex WebGPU Dev Server                 ║\x1b[0m`);
  console.log(`\x1b[36m║  Metrics endpoint: http://localhost:${PORT}   ║\x1b[0m`);
  console.log(`\x1b[36m║  POST /metrics — receive browser metrics    ║\x1b[0m`);
  console.log(`\x1b[36m║  GET  /metrics/recent — view last 50        ║\x1b[0m`);
  console.log(`\x1b[36m╚════════════════════════════════════════════╝\x1b[0m`);
});
