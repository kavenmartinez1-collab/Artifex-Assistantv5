import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: 'public',
  build: {
    outDir: 'dist',
    target: 'esnext',
  },
  server: {
    host: '127.0.0.1',  // localhost only — no network exposure
    port: 5173,
    strictPort: true,
    open: true,
    proxy: {
      // Proxy metrics to the dev server
      '/metrics': 'http://127.0.0.1:3001',
      // Proxy Artifex API calls
      '/v1': 'http://127.0.0.1:8000',
    },
  },
});
