/**
 * WASM SIMD Q5_K GEMV throughput bench — Phase 0c.
 *
 * Spawns shard workers (1 then 8), each with a private slice of synthetic
 * Q5_K expert weights sized to defeat caches, and measures aggregate
 * dequant-dot memory throughput. Also measures SAB/Atomics worker wake
 * round-trip latency — Phase 0e (the Phase C control plane).
 */

export interface WasmBenchResult {
  workers: number;
  aggregateGBps: number;
  perWorkerGBps: number[];
  validationMaxRelErr: number;
  gemvsPerSec: number;
  matrixBytes: number;
}

export interface WakeBenchResult {
  roundTrips: number;
  meanUs: number;
  p95Us: number;
}

// Real expert gate/up shape for Qwen3.6-35B-A3B: [512 out, 2048 in] Q5_K
const ROWS = 512;
const COLS = 2048;

function spawnWorker(): Worker {
  return new Worker(new URL('./wasm-worker.ts', import.meta.url), { type: 'module' });
}

function once<T = any>(w: Worker): Promise<T> {
  return new Promise((resolve, reject) => {
    w.onmessage = (e) => {
      if (e.data.cmd === 'error') reject(new Error(e.data.message));
      else resolve(e.data);
    };
    w.onerror = (e) => reject(new Error(e.message));
  });
}

export async function benchWasmGemv(
  numWorkers: number,
  weightsMBPerWorker: number,
  runMs: number,
  onStatus?: (s: string) => void
): Promise<WasmBenchResult> {
  const workers: Worker[] = [];
  try {
    onStatus?.(`spawning ${numWorkers} worker(s), ${weightsMBPerWorker} MB each...`);
    for (let i = 0; i < numWorkers; i++) workers.push(spawnWorker());

    // Init (data generation + kernel self-validation) in parallel.
    const inits = await Promise.all(
      workers.map((w, i) => {
        const p = once(w);
        w.postMessage({ cmd: 'init', workerId: i, weightsMB: weightsMBPerWorker, rows: ROWS, cols: COLS });
        return p;
      })
    );
    const maxRelErr = Math.max(...inits.map((r: any) => r.maxRelErr));
    if (!inits.every((r: any) => r.ok)) {
      throw new Error(`kernel validation FAILED vs JS reference (maxRelErr=${maxRelErr})`);
    }
    onStatus?.(`validated vs JS reference (maxRelErr=${maxRelErr.toExponential(2)}); running ${runMs} ms...`);

    // Warmup
    await Promise.all(
      workers.map((w) => {
        const p = once(w);
        w.postMessage({ cmd: 'run', ms: 300 });
        return p;
      })
    );

    // Timed run — all workers concurrently.
    const t0 = performance.now();
    const runs = await Promise.all(
      workers.map((w) => {
        const p = once(w);
        w.postMessage({ cmd: 'run', ms: runMs });
        return p;
      })
    );
    const wallMs = performance.now() - t0;

    const totalBytes = runs.reduce((s: number, r: any) => s + r.bytes, 0);
    const totalGemvs = runs.reduce((s: number, r: any) => s + r.gemvs, 0);
    const matrixBytes = (COLS / 256) * 176 * ROWS;
    return {
      workers: numWorkers,
      aggregateGBps: totalBytes / 1e9 / (wallMs / 1000),
      perWorkerGBps: runs.map((r: any) => r.bytes / 1e9 / (r.elapsedMs / 1000)),
      validationMaxRelErr: maxRelErr,
      gemvsPerSec: totalGemvs / (wallMs / 1000),
      matrixBytes,
    };
  } finally {
    workers.forEach((w) => w.terminate());
  }
}

/** SAB/Atomics worker wake round-trip latency (requires crossOriginIsolated). */
export async function benchWorkerWake(roundTrips: number): Promise<WakeBenchResult> {
  if (!crossOriginIsolated) {
    throw new Error('not crossOriginIsolated — COOP/COEP headers missing, SharedArrayBuffer unavailable');
  }
  const w = spawnWorker();
  try {
    const sab = new SharedArrayBuffer(8);
    const ctl = new Int32Array(sab);
    w.postMessage({ cmd: 'wake', sab });
    await new Promise((r) => setTimeout(r, 100)); // let worker enter the wait loop

    const times: number[] = [];
    for (let gen = 1; gen <= roundTrips; gen++) {
      const t0 = performance.now();
      Atomics.store(ctl, 0, gen);
      Atomics.notify(ctl, 0);
      // Atomics.waitAsync is ES2024 — not in this project's ES2022 lib types yet
      const waitAsync = (Atomics as unknown as {
        waitAsync(ta: Int32Array, i: number, v: number): { async: boolean; value: Promise<string> | string };
      }).waitAsync;
      const res = waitAsync(ctl, 1, gen - 1);
      if (res.async) await res.value;
      times.push((performance.now() - t0) * 1000); // µs
    }
    Atomics.store(ctl, 0, -1);
    Atomics.notify(ctl, 0);

    times.sort((a, b) => a - b);
    return {
      roundTrips,
      meanUs: times.reduce((s, t) => s + t, 0) / times.length,
      p95Us: times[Math.floor(times.length * 0.95)],
    };
  } finally {
    w.terminate();
  }
}
