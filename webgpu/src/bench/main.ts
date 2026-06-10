/**
 * Phase 0 microbenchmark suite — MoE go/no-go for Qwen3.6-35B-A3B.
 *
 * Measures the numbers the CPU-WASM-expert design lives or dies by:
 *   (a) blocking 8 KB GPU readback latency (40/token, structural)
 *   (b) writeBuffer host→GPU bandwidth (prefill expert streaming)
 *   (c) WASM SIMD Q5_K GEMV aggregate throughput, 1 and 8 workers
 *   (e) SAB/Atomics worker wake round-trip (CPU expert control plane)
 *
 * Projects decode tok/s; gate: abort/redesign MoE if projection < 8 tok/s.
 * Results render in-page and POST to /api/debug for the dev loop.
 */

import { initWebGPU } from '../engine/gpu-device';
import { benchReadback, benchReadbackVariants, benchWriteBandwidth } from './gpu-bench';
import { benchWasmGemv, benchWorkerWake } from './wasm-bench';

// Qwen3.6-35B-A3B decode-token constants (verified from GGUF header)
const MOE_LAYERS = 40;
const EXPERT_BYTES_PER_TOKEN = 40 * 8 * (2 * 720896 + 860160); // ≈ 737 MB
const GPU_DENSE_MS_ESTIMATE = 12; // ~2.1 GB Q8_0 reads + dispatch overhead (refine in Phase C1)
const GATE_TOKS = 8;

const out = document.getElementById('results')!;
const statusEl = document.getElementById('status')!;

function log(html: string) {
  out.insertAdjacentHTML('beforeend', html);
}

function setStatus(s: string) {
  statusEl.textContent = s;
  console.log('[bench]', s);
}

function row(name: string, value: string, note = '') {
  return `<tr><td>${name}</td><td class="val">${value}</td><td class="note">${note}</td></tr>`;
}

const results: Record<string, unknown> = {
  startedAt: new Date().toISOString(),
  crossOriginIsolated,
};

async function postResults() {
  try {
    await fetch('/api/debug', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bench: results }, null, 2),
    });
  } catch {
    /* dev loop only */
  }
}

async function runAll() {
  out.innerHTML = '';
  const btn = document.getElementById('run') as HTMLButtonElement;
  btn.disabled = true;
  try {
    log(`<table><tbody id="tbody"></tbody></table>`);
    const tbody = document.getElementById('tbody')!;
    const add = (name: string, value: string, note = '') =>
      tbody.insertAdjacentHTML('beforeend', row(name, value, note));

    add('crossOriginIsolated', String(crossOriginIsolated), crossOriginIsolated ? '' : 'COOP/COEP missing — WASM multi-worker bench unavailable');

    // ── GPU benches ──
    setStatus('initializing WebGPU...');
    const gpu = await initWebGPU();
    add('GPU', gpu.adapterInfo.description || gpu.adapterInfo.vendor || 'unknown');

    setStatus('(a) readback latency...');
    const rb = await benchReadback(gpu.device, 200, setStatus);
    results.readback = rb;
    add(
      '(a) 8 KB blocking readback (reused staging)',
      `${rb.reused.meanMs.toFixed(3)} ms mean / ${rb.reused.p95Ms.toFixed(3)} ms p95`,
      `target < 0.8 ms; ${MOE_LAYERS}/token → ${(rb.reused.meanMs * MOE_LAYERS).toFixed(1)} ms/token`
    );
    add(
    '(a) same, fresh staging per call',
      `${rb.fresh.meanMs.toFixed(3)} ms mean / ${rb.fresh.p95Ms.toFixed(3)} ms p95`,
      'current buffers.ts readBuffer behavior'
    );

    setStatus('(a2) readback latency-reduction variants...');
    const rbv = await benchReadbackVariants(gpu.device, 200, setStatus);
    results.readbackVariants = rbv;
    let bestVariant = 'naive';
    for (const [name, s] of Object.entries(rbv)) {
      add(`(a2) readback: ${name}`, `${s.meanMs.toFixed(3)} ms mean / ${s.p95Ms.toFixed(3)} ms p95`);
      if (s.meanMs < rbv[bestVariant].meanMs) bestVariant = name;
    }

    setStatus('(b) writeBuffer bandwidth...');
    const wb = await benchWriteBandwidth(gpu.device, gpu.maxBufferSize, setStatus);
    results.writeBandwidth = wb;
    add('(b) writeBuffer, 0.7 MB chunks', `${wb.smallChunkGBps.toFixed(2)} GB/s`, 'per-expert granularity');
    add('(b) writeBuffer, single large write', `${wb.largeChunkGBps.toFixed(2)} GB/s`, `${wb.largeChunkMB} MB — prefill layer streaming`);

    // ── WASM benches ──
    let agg8 = 0;
    setStatus('(c) WASM Q5_K GEMV, 1 worker...');
    const w1 = await benchWasmGemv(1, 256, 2000, setStatus);
    results.wasm1 = w1;
    add(
      '(c) WASM Q5_K GEMV, 1 worker',
      `${w1.aggregateGBps.toFixed(2)} GB/s`,
      `kernel validated vs JS ref (maxRelErr ${w1.validationMaxRelErr.toExponential(1)})`
    );

    {
      // plain workers don't need isolation; only the wake bench does.
      // 2/4/8 scaling → per-worker bandwidth under contention, which decides
      // the Phase C straggler penalty (E[max experts on one worker] ≈ 2.3).
      for (const n of [2, 4, 8]) {
        setStatus(`(c) WASM Q5_K GEMV, ${n} workers...`);
        const wn = await benchWasmGemv(n, 256, n === 8 ? 3000 : 1500, setStatus);
        (results as Record<string, unknown>)[`wasm${n}`] = wn;
        if (n === 8) agg8 = wn.aggregateGBps;
        add(
          `(c) WASM Q5_K GEMV, ${n} workers`,
          `${wn.aggregateGBps.toFixed(2)} GB/s aggregate`,
          `${(wn.aggregateGBps / n).toFixed(2)} GB/s per worker${n === 8 ? ' — target ≥ 25 GB/s aggregate' : ''}`,
        );
      }
    }

    if (crossOriginIsolated) {
      setStatus('(e) worker wake latency...');
      const wake = await benchWorkerWake(1000);
      results.wake = wake;
      add('(e) SAB/Atomics wake round-trip', `${wake.meanUs.toFixed(1)} µs mean / ${wake.p95Us.toFixed(1)} µs p95`, 'CPU expert control plane');
    } else {
      add('(e) SAB/Atomics wake', 'SKIPPED', 'needs crossOriginIsolated');
    }

    // ── Projection ── (uses the best readback variant — that's what Phase C would implement)
    const bestReadbackMs = Math.min(rb.reused.meanMs, rbv[bestVariant].meanMs);
    const readbackMs = bestReadbackMs * MOE_LAYERS;
    const expertMs = agg8 > 0 ? (EXPERT_BYTES_PER_TOKEN / 1e9 / agg8) * 1000 : NaN;
    const tokenMs = readbackMs + expertMs + GPU_DENSE_MS_ESTIMATE + 1;
    const projTokS = 1000 / tokenMs;
    results.projection = {
      readbackVariantUsed: bestVariant,
      readbackMsPerToken: readbackMs,
      expertComputeMsPerToken: expertMs,
      gpuDenseMsEstimate: GPU_DENSE_MS_ESTIMATE,
      tokenMs,
      projectedTokS: projTokS,
      gate: GATE_TOKS,
      pass: projTokS >= GATE_TOKS,
    };
    add('── projected token time ──', `${tokenMs.toFixed(1)} ms`, `${readbackMs.toFixed(1)} sync + ${expertMs.toFixed(1)} experts + ${GPU_DENSE_MS_ESTIMATE} GPU(est) + 1 misc`);
    add(
      '── PROJECTED DECODE ──',
      `${projTokS.toFixed(1)} tok/s`,
      projTokS >= GATE_TOKS ? `PASS (gate ${GATE_TOKS} tok/s)` : `FAIL (gate ${GATE_TOKS} tok/s) — redesign needed`
    );

    results.finishedAt = new Date().toISOString();
    setStatus(`done — projected ${projTokS.toFixed(1)} tok/s (${projTokS >= GATE_TOKS ? 'PASS' : 'FAIL'})`);
    await postResults();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    results.error = msg;
    setStatus(`ERROR: ${msg}`);
    log(`<p class="err">${msg}</p>`);
    await postResults();
  } finally {
    btn.disabled = false;
  }
}

/**
 * C0 go/no-go: allocate 8 × Phase-C-shard-sized wasm memories in workers and
 * touch every OS page (~22.3 GB real commit, like 8 expert shards coexisting).
 * If Chrome can't hold this, Phase C is no-go.
 */
const RAM_WORKERS = 8;
const SHARD_BYTES = Math.ceil(22.25e9 / RAM_WORKERS); // 2.78 GB per worker

async function runRamSmoke() {
  out.innerHTML = '';
  const btns = [
    document.getElementById('run') as HTMLButtonElement,
    document.getElementById('ram-smoke') as HTMLButtonElement,
  ];
  btns.forEach((b) => (b.disabled = true));
  const workers: Worker[] = [];
  try {
    log(`<table><tbody id="tbody"></tbody></table>`);
    const tbody = document.getElementById('tbody')!;
    const add = (name: string, value: string, note = '') =>
      tbody.insertAdjacentHTML('beforeend', row(name, value, note));

    setStatus(`allocating ${RAM_WORKERS} × ${(SHARD_BYTES / 2 ** 30).toFixed(2)} GiB...`);
    const t0 = performance.now();

    type AllocResult =
      | { cmd: 'alloc-done'; workerId: number; grewBytes: number; growMs: number; touchMs: number }
      | { cmd: 'alloc-fail'; workerId: number; grewBytes: number; error: string };

    const settled = await Promise.all(
      Array.from({ length: RAM_WORKERS }, (_, i) => {
        const w = new Worker(new URL('./ram-smoke-worker.ts', import.meta.url), { type: 'module' });
        workers.push(w);
        return new Promise<AllocResult>((resolve, reject) => {
          w.onmessage = (ev: MessageEvent<AllocResult>) => {
            setStatus(`worker ${ev.data.workerId}: ${ev.data.cmd} (${(ev.data.grewBytes / 2 ** 30).toFixed(2)} GiB)`);
            resolve(ev.data);
          };
          w.onerror = (e) => reject(new Error(`worker ${i}: ${e.message}`));
          w.postMessage({ cmd: 'alloc', workerId: i, bytes: SHARD_BYTES });
        });
      })
    );
    const totalMs = performance.now() - t0;

    let committed = 0;
    let fails = 0;
    for (const r of settled) {
      committed += r.grewBytes;
      if (r.cmd === 'alloc-done') {
        add(`worker ${r.workerId}`, `${(r.grewBytes / 2 ** 30).toFixed(2)} GiB`, `grow ${r.growMs.toFixed(0)} ms, touch ${r.touchMs.toFixed(0)} ms`);
      } else {
        fails++;
        add(`worker ${r.workerId}`, `FAIL @ ${(r.grewBytes / 2 ** 30).toFixed(2)} GiB`, r.error);
      }
    }

    const pass = fails === 0;
    add(
      '── RAM SMOKE ──',
      `${(committed / 2 ** 30).toFixed(2)} GiB committed in ${(totalMs / 1000).toFixed(1)} s`,
      pass ? `PASS — all ${RAM_WORKERS} shards held simultaneously` : `FAIL — ${fails} worker(s) could not allocate; Phase C no-go`
    );
    results.ramSmoke = { workers: settled, committedBytes: committed, totalMs, pass };
    results.finishedAt = new Date().toISOString();
    setStatus(pass ? `RAM smoke PASS — ${(committed / 2 ** 30).toFixed(2)} GiB held; releasing...` : 'RAM smoke FAIL');
    await postResults();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    results.error = msg;
    setStatus(`ERROR: ${msg}`);
    log(`<p class="err">${msg}</p>`);
    await postResults();
  } finally {
    // Release the ~22 GB before re-enabling anything else.
    workers.forEach((w) => w.terminate());
    btns.forEach((b) => (b.disabled = false));
  }
}

document.getElementById('run')!.addEventListener('click', runAll);
document.getElementById('ram-smoke')!.addEventListener('click', runRamSmoke);
setStatus(`ready — crossOriginIsolated=${crossOriginIsolated}`);
