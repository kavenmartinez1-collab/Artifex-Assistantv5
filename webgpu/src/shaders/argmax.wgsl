// GPU greedy argmax over the logits buffer — replaces the full-vocab
// readback (~600 KB) with a 4-byte index readback in the decode loop.
//
// Semantics: winner = maximum f32 value, ties broken by LOWEST index —
// exactly the CPU greedy sampler's sequential strict-`>` scan
// (generate.ts sampleFromLogits, temperature === 0). The (max value,
// min index) pair is an associative/commutative semilattice, so the
// parallel reduction order cannot change the result.
//
// Two passes:
//   argmax_partial — NWG=256 workgroups × WG=256 threads; each thread
//     strides the array (i = wid*WG + lid, step WG*NWG), keeps its best
//     (value, index) with index 0xFFFFFFFF as the "no element" marker,
//     then an LDS tree reduce writes one partial per workgroup.
//   argmax_final   — one workgroup reduces the 256 partials and writes
//     the winning index to out_idx[0].
//
// Partials are vec2<u32>: .x = bitcast<u32>(value), .y = index.
// JS parity port: scripts/test-argmax.mts (must mirror this file).

const WG: u32 = 256u;
const NWG: u32 = 256u;
const EMPTY: u32 = 0xFFFFFFFFu;

struct Params {
  n: u32,      // number of logits (vocabSize)
  _p0: u32,
  _p1: u32,
  _p2: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> out_idx: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> sh: array<vec2<u32>, WG>;

// Is (v, i) strictly better than (bv, bi)?  Max value, min index on ties.
fn better(v: f32, i: u32, bv: f32, bi: u32) -> bool {
  return v > bv || (v == bv && i < bi);
}

// Combine two candidates; .y == EMPTY means "no element".
fn pick(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
  if (a.y == EMPTY) { return b; }
  if (b.y == EMPTY) { return a; }
  if (better(bitcast<f32>(b.x), b.y, bitcast<f32>(a.x), a.y)) { return b; }
  return a;
}

@compute @workgroup_size(WG, 1, 1)
fn argmax_partial(@builtin(workgroup_id) wid: vec3u,
                  @builtin(local_invocation_id) lid: vec3u) {
  let n = params.n;
  let stride = WG * NWG;
  var bv: f32 = 0.0;
  var bi: u32 = EMPTY;
  for (var i = wid.x * WG + lid.x; i < n; i = i + stride) {
    let v = logits[i];
    if (bi == EMPTY || better(v, i, bv, bi)) { bv = v; bi = i; }
  }
  sh[lid.x] = vec2<u32>(bitcast<u32>(bv), bi);
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { sh[lid.x] = pick(sh[lid.x], sh[lid.x + s]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partials[wid.x] = sh[0]; }
}

@compute @workgroup_size(WG, 1, 1)
fn argmax_final(@builtin(local_invocation_id) lid: vec3u) {
  sh[lid.x] = partials[lid.x];
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { sh[lid.x] = pick(sh[lid.x], sh[lid.x + s]); }
    workgroupBarrier();
  }
  if (lid.x == 0u) { out_idx[0] = sh[0].y; }
}
