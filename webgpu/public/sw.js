/**
 * Artifex WebGPU — service worker for static/hosted builds.
 *
 * Two jobs:
 *  1. App-shell caching: hashed /assets/ bundles are cache-first (immutable);
 *     navigations and small statics are network-first with cache fallback, so
 *     the app opens offline after the first visit. Model weights are NOT
 *     handled here — the engine's own Cache API layer owns those, and this
 *     worker never intercepts Range requests or cross-origin fetches.
 *  2. Cross-origin isolation: static hosts (GitHub Pages) can't set response
 *     headers, and the MoE expert workers need SharedArrayBuffer, which needs
 *     COOP/COEP. Navigation responses get the same headers the dev server
 *     sends (COEP `credentialless`, so anonymous HF CDN fetches keep working).
 *     Isolation kicks in from the second load — the first visit installs the
 *     worker.
 *
 * Registered only in production builds (see main.ts) — the dev server sets
 * real headers and vite owns module serving there.
 */

const SHELL_CACHE = 'artifex-shell-v1';

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) {
      if (k.startsWith('artifex-shell-') && k !== SHELL_CACHE) await caches.delete(k);
    }
    await self.clients.claim();
  })());
});

/** Re-wrap a response with the cross-origin-isolation headers. */
function withCoiHeaders(resp) {
  if (resp.status === 0) return resp; // opaque — can't touch
  const headers = new Headers(resp.headers);
  headers.set('Cross-Origin-Opener-Policy', 'same-origin');
  headers.set('Cross-Origin-Embedder-Policy', 'credentialless');
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;        // HF CDN etc. — browser handles
  if (req.headers.has('range')) return;                    // weight streams — never intercept
  if (/^\/(api|metrics|ws)(\/|$)/.test(url.pathname)) return; // dev-server routes

  // Hashed immutable bundles: cache-first
  if (url.pathname.includes('/assets/')) {
    e.respondWith((async () => {
      const cached = await caches.match(req);
      if (cached) return cached;
      const resp = await fetch(req);
      if (resp.ok) (await caches.open(SHELL_CACHE)).put(req, resp.clone());
      return resp;
    })());
    return;
  }

  // Navigations + small statics: network-first, cache fallback, COI headers
  // on documents.
  e.respondWith((async () => {
    const isNav = req.mode === 'navigate';
    try {
      let resp = await fetch(req);
      if (resp.ok) (await caches.open(SHELL_CACHE)).put(req, resp.clone());
      if (isNav) resp = withCoiHeaders(resp);
      return resp;
    } catch {
      const cached = await caches.match(req);
      if (cached) return isNav ? withCoiHeaders(cached) : cached;
      return new Response('offline and not cached', { status: 503 });
    }
  })());
});
