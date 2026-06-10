/**
 * RAM smoke worker — allocates one Phase C expert-shard-sized wasm memory
 * and touches every OS page to force real commit (not just reservation).
 *
 * In: {cmd:'alloc', workerId, bytes}
 * Out: {cmd:'alloc-done', workerId, grewBytes, growMs, touchMs} or
 *      {cmd:'alloc-fail', workerId, grewBytes, error}
 * Memory stays alive until {cmd:'release'} so all 8 shards coexist.
 */

const PAGE = 65536; // wasm page
const GROW_STEP = 4096; // pages per grow = 256 MB

let mem: WebAssembly.Memory | null = null;

self.onmessage = (ev: MessageEvent) => {
  const msg = ev.data;
  if (msg.cmd === 'alloc') {
    const { workerId, bytes } = msg as { workerId: number; bytes: number };
    const targetPages = Math.ceil(bytes / PAGE);
    let grownPages = 0;
    try {
      const t0 = performance.now();
      mem = new WebAssembly.Memory({ initial: Math.min(GROW_STEP, targetPages), maximum: targetPages });
      grownPages = Math.min(GROW_STEP, targetPages);
      while (grownPages < targetPages) {
        const step = Math.min(GROW_STEP, targetPages - grownPages);
        mem.grow(step);
        grownPages += step;
      }
      const t1 = performance.now();
      // Touch one byte per 4 KiB OS page — forces actual commit.
      const u8 = new Uint8Array(mem.buffer);
      for (let i = 0; i < u8.length; i += 4096) u8[i] = 1;
      const t2 = performance.now();
      (self as unknown as Worker).postMessage({
        cmd: 'alloc-done',
        workerId,
        grewBytes: grownPages * PAGE,
        growMs: t1 - t0,
        touchMs: t2 - t1,
      });
    } catch (err) {
      (self as unknown as Worker).postMessage({
        cmd: 'alloc-fail',
        workerId,
        grewBytes: grownPages * PAGE,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  } else if (msg.cmd === 'release') {
    mem = null;
    (self as unknown as Worker).postMessage({ cmd: 'released' });
  }
};
