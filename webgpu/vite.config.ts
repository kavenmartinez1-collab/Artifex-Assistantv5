import { defineConfig, type Plugin } from 'vite';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Vite plugin: Debug API for automated testing.
 *
 * POST /api/debug  — browser posts debug data (JSON body), written to debug-output.json
 * GET  /api/debug  — read last debug output
 * POST /api/test   — queue a test prompt (JSON body: {prompt: string})
 * GET  /api/test   — browser polls for pending test prompt
 */
function debugApiPlugin(): Plugin {
  const debugFile = path.resolve(__dirname, 'debug-output.json');
  let pendingTest: { prompt: string; temperature?: number } | null = null;

  return {
    name: 'debug-api',
    configureServer(server) {
      server.middlewares.use('/api/debug', (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
          req.on('end', () => {
            fs.writeFileSync(debugFile, body, 'utf-8');
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end('{"ok":true}');
          });
        } else {
          try {
            const data = fs.readFileSync(debugFile, 'utf-8');
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(data);
          } catch {
            res.writeHead(404, { 'Content-Type': 'application/json' });
            res.end('{"error":"no debug data yet"}');
          }
        }
      });

      server.middlewares.use('/api/test', (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          req.on('data', (chunk: Buffer) => { body += chunk.toString(); });
          req.on('end', () => {
            pendingTest = JSON.parse(body);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end('{"ok":true,"prompt":' + JSON.stringify(pendingTest?.prompt) + '}');
          });
        } else {
          // GET — browser polls for pending test
          if (pendingTest) {
            const t = pendingTest;
            pendingTest = null; // consume
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(t));
          } else {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end('null');
          }
        }
      });
    },
  };
}

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    target: 'esnext',
    rollupOptions: {
      input: {
        main: 'index.html',
        worker: 'worker.html',
      },
    },
  },
  plugins: [debugApiPlugin()],
  server: {
    host: '127.0.0.1',  // localhost only — no network exposure
    port: 5173,
    strictPort: true,
    open: true,
    proxy: {
      // Proxy metrics to the dev server
      '/metrics': 'http://127.0.0.1:3001',
      // Proxy local HF cache to the dev server (streams large weight shards)
      '/api/hf-cache': {
        target: 'http://127.0.0.1:3001',
        timeout: 600000,
        proxyTimeout: 600000,
        selfHandleResponse: false,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            proxyReq.setSocketKeepAlive(true);
          });
        },
      },
      // Proxy Artifex API calls
      '/v1': 'http://127.0.0.1:8000',
      // WebSocket proxy to orchestration hub
      '/ws': {
        target: 'http://127.0.0.1:3001',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
