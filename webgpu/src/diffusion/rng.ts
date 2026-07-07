// Seeded standard-normal sampler for the FLUX.2 runtime path (UI seeds).
//
// mulberry32 uniform PRNG + Box-Muller. This does NOT reproduce torch.randn
// bit-for-bit — all parity gates inject torch-dumped noise; this generator
// only needs to be a deterministic, well-distributed N(0,1) source so a
// (prompt, seed) pair always re-creates the same image in this app.

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** n samples from N(0,1), deterministic per seed. */
export function randn(n: number, seed: number): Float32Array {
  const rand = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    // Box-Muller; u1 in (0,1] so log() is finite.
    const u1 = 1 - rand();
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    out[i] = r * Math.cos(2 * Math.PI * u2);
    if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * u2);
  }
  return out;
}
