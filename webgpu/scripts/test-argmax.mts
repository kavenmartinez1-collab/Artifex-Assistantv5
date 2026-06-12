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

if (failures > 0) {
  console.log(`\n${failures} FAILURE(S)`);
  process.exit(1);
}
console.log('\nAll argmax checks passed.');
