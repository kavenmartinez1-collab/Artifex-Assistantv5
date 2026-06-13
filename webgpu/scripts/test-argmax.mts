/**
 * GPU argmax kernel validation (argmax.wgsl) — JS port of the two-pass
 * (partial + final) reduction, checked against the CPU greedy sampler's
 * first-max-wins argmax (generate.ts sampleFromLogits, temperature === 0).
 * Kernel-audit rule: port must agree EXACTLY before the WGSL is wired in.
 *
 * Semantics under test: winner = maximum f32 value; ties broken by LOWEST
 * index. A sequential strict-`>` scan (the CPU sampler) picks exactly that
 * element, and the (max value, min index) pair is an associative/commutative
 * semilattice, so the parallel reduction order cannot change the result.
 *
 * Run: npx tsx scripts/test-argmax.mts
 */

const WG = 256;   // threads per workgroup (must mirror argmax.wgsl)
const NWG = 256;  // pass-1 workgroups     (must mirror argmax.wgsl)
const EMPTY = 0xFFFFFFFF;

// ── deterministic RNG ──────────────────────────────────────────────────
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// CPU reference: the greedy path in sampleFromLogits — strict `>` scan,
// first max wins (lowest index among ties).
function cpuArgmax(logits: Float32Array): number {
  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

// ── WGSL ports (must mirror argmax.wgsl exactly) ───────────────────────
type Pair = { v: number; i: number };          // i === EMPTY → no element

function better(v: number, i: number, bv: number, bi: number): boolean {
  return v > bv || (v === bv && i < bi);
}

function pick(a: Pair, b: Pair): Pair {
  if (a.i === EMPTY) return b;
  if (b.i === EMPTY) return a;
  return better(b.v, b.i, a.v, a.i) ? b : a;
}

/** Tree reduce over a WG-sized shared array — the s = WG/2, WG/4, ... loop. */
function treeReduce(sh: Pair[]): Pair {
  for (let s = sh.length >> 1; s > 0; s >>= 1) {
    for (let lid = 0; lid < s; lid++) {
      sh[lid] = pick(sh[lid], sh[lid + s]);
    }
  }
  return sh[0];
}

function gpuArgmaxPort(logits: Float32Array): number {
  const n = logits.length;
  const stride = WG * NWG;
  // Pass 1: argmax_partial — NWG workgroups, each thread strides the array.
  const partials: Pair[] = [];
  for (let w = 0; w < NWG; w++) {
    const sh: Pair[] = [];
    for (let l = 0; l < WG; l++) {
      let bv = 0, bi = EMPTY;
      for (let i = w * WG + l; i < n; i += stride) {
        const v = logits[i];
        if (bi === EMPTY || better(v, i, bv, bi)) { bv = v; bi = i; }
      }
      sh.push({ v: bv, i: bi });
    }
    partials.push(treeReduce(sh));
  }
  // Pass 2: argmax_final — one workgroup, one partial per thread.
  return treeReduce(partials).i;
}

/** Multi-row port: grid (NWG, rows) in pass 1, (1, rows) in pass 2.
 *  Row r reads logits[r*n + i]; partials live at [r*NWG + wg]; the final
 *  pass writes out_idx[r]. Indices are LOCAL to the row (0..n-1). rows=1
 *  must reduce to gpuArgmaxPort exactly. Must mirror argmax.wgsl. */
function gpuArgmaxPortRows(logits: Float32Array, n: number, rows: number): number[] {
  const stride = WG * NWG;
  const out: number[] = [];
  for (let r = 0; r < rows; r++) {
    const base = r * n;
    // Pass 1: argmax_partial — wid.y = r.
    const partials: Pair[] = [];
    for (let w = 0; w < NWG; w++) {
      const sh: Pair[] = [];
      for (let l = 0; l < WG; l++) {
        let bv = 0, bi = EMPTY;
        for (let i = w * WG + l; i < n; i += stride) {
          const v = logits[base + i];
          if (bi === EMPTY || better(v, i, bv, bi)) { bv = v; bi = i; }
        }
        sh.push({ v: bv, i: bi });
      }
      partials.push(treeReduce(sh));
    }
    // Pass 2: argmax_final — wid.y = r.
    out.push(treeReduce(partials).i);
  }
  return out;
}

// ── test driver ────────────────────────────────────────────────────────
let failures = 0;

function check(label: string, got: number, want: number) {
  if (got !== want) {
    failures++;
    console.log(`  FAIL ${label}: got ${got} want ${want}`);
  }
}

const rng = mulberry32(0xA56A11);
// Vocab-ish and awkward sizes: smaller than one WG, non-multiples of the
// thread grid, exactly the grid, and the real 27B/9B vocab sizes.
const SIZES = [1, 7, 255, 256, 257, 65536, 65537, 131071, 151936, 152064, 248320];

for (const n of SIZES) {
  // Case A: random finite logits (f32) — generic
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) a[i] = Math.fround((rng() - 0.5) * 40);
  check(`random n=${n}`, gpuArgmaxPort(a), cpuArgmax(a));

  // Case B: planted exact ties — duplicate the max at several indices,
  // including index 0 and the last element. Lowest index must win.
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) b[i] = Math.fround((rng() - 0.5) * 40);
  const maxV = Math.fround(50);
  const t0 = Math.floor(rng() * n);
  const t1 = Math.floor(rng() * n);
  b[t0] = maxV; b[t1] = maxV; b[n - 1] = maxV;
  check(`ties n=${n}`, gpuArgmaxPort(b), cpuArgmax(b));

  // Case C: all-equal (every index ties) — must return 0.
  const c = new Float32Array(n).fill(Math.fround(-1.5));
  check(`all-equal n=${n}`, gpuArgmaxPort(c), cpuArgmax(c));

  // Case D: strictly descending — max at 0 (exercises the empty-marker
  // ordering: high-index threads lose to thread 0's first element).
  const d = new Float32Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.fround(-i);
  check(`descending n=${n}`, gpuArgmaxPort(d), cpuArgmax(d));

  // Case E: strictly ascending — max at n-1 (winner in the last,
  // possibly partially-filled thread slot).
  const e = new Float32Array(n);
  for (let i = 0; i < n; i++) e[i] = Math.fround(i);
  check(`ascending n=${n}`, gpuArgmaxPort(e), cpuArgmax(e));

  console.log(`  ok   n=${n}: random/ties/all-equal/descending/ascending`);
}

// Case F: randomized tie fuzz at vocab size — 50 rounds, 2-8 planted ties.
for (let round = 0; round < 50; round++) {
  const n = 151936;
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = Math.fround((rng() - 0.5) * 40);
  const v = Math.fround(45 + rng() * 10);
  const k = 2 + Math.floor(rng() * 7);
  for (let j = 0; j < k; j++) f[Math.floor(rng() * n)] = v;
  check(`fuzz round=${round}`, gpuArgmaxPort(f), cpuArgmax(f));
}
console.log('  ok   tie fuzz: 50 rounds @ n=151936');

// ── Multi-row (spec decode verify) ─────────────────────────────────────
// Case G: rows=1 must agree with the single-row port AND the CPU sampler
// for every size above (the M=1 decode path must be untouched).
for (const n of SIZES) {
  const g = new Float32Array(n);
  for (let i = 0; i < n; i++) g[i] = Math.fround((rng() - 0.5) * 40);
  const single = gpuArgmaxPort(g);
  const multi = gpuArgmaxPortRows(g, n, 1);
  check(`rows=1 n=${n} (vs single)`, multi[0], single);
  check(`rows=1 n=${n} (vs cpu)`, multi[0], cpuArgmax(g));
}
console.log('  ok   rows=1 equivalence across all sizes');

// Case H: multi-row at vocab sizes — each row independent, with planted
// ties per row and a shared tie value ACROSS rows (row isolation check:
// a tie in row 2 must not leak into row 0's winner).
for (const n of [151936, 248320]) {
  for (const rows of [2, 3, 4, 8]) {
    const h = new Float32Array(rows * n);
    for (let i = 0; i < h.length; i++) h[i] = Math.fround((rng() - 0.5) * 40);
    const v = Math.fround(60);
    for (let r = 0; r < rows; r++) {
      h[r * n + Math.floor(rng() * n)] = v;           // per-row planted max
      if (r % 2 === 0) h[r * n + (n - 1)] = v;        // tie at row end
    }
    const got = gpuArgmaxPortRows(h, n, rows);
    for (let r = 0; r < rows; r++) {
      const want = cpuArgmax(h.subarray(r * n, (r + 1) * n) as Float32Array);
      check(`rows=${rows} n=${n} row=${r}`, got[r], want);
    }
  }
  console.log(`  ok   multi-row n=${n}: rows 2/3/4/8 with cross-row ties`);
}

// Case I: multi-row fuzz — 30 rounds, random rows 2..8, awkward n.
for (let round = 0; round < 30; round++) {
  const n = [257, 65537, 151936][Math.floor(rng() * 3)];
  const rows = 2 + Math.floor(rng() * 7);
  const f = new Float32Array(rows * n);
  for (let i = 0; i < f.length; i++) f[i] = Math.fround((rng() - 0.5) * 40);
  const v = Math.fround(45 + rng() * 10);
  for (let j = 0; j < 1 + Math.floor(rng() * 6); j++) {
    f[Math.floor(rng() * f.length)] = v;
  }
  const got = gpuArgmaxPortRows(f, n, rows);
  for (let r = 0; r < rows; r++) {
    const want = cpuArgmax(f.subarray(r * n, (r + 1) * n) as Float32Array);
    check(`multirow fuzz round=${round} row=${r}`, got[r], want);
  }
}
console.log('  ok   multi-row fuzz: 30 rounds');

if (failures > 0) {
  console.log(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nAll argmax checks passed.');
