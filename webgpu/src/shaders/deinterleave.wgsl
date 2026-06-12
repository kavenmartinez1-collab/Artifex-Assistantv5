// Deinterleave — Qwen3.5 attnOutputGate Q-projection split
//
// The Q projection on gated-attention models outputs per-head interleaved
// blocks: row layout [Q_h0 (d_head floats), gate_h0 (d_head floats), Q_h1,
// gate_h1, ...] for each token. This kernel splits that into contiguous
// Q [seq, n_heads, d_head] and gate [seq, n_heads, d_head] streams in one
// dispatch — replacing the per-head CPU copy loop (2 copies × n_heads ×
// seq_len copyBufferToBuffer commands, each of which also breaks the fused
// compute pass in BatchedDispatcher).
//
// Pure data movement: bit-exact by construction.

struct Params {
  n: u32,       // elements per output stream = seq_len * n_heads * d_head
  d_head: u32,  // head dimension (block size of the interleave)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;        // [n*2] interleaved
@group(0) @binding(1) var<storage, read_write> out_q: array<f32>;    // [n]
@group(0) @binding(2) var<storage, read_write> out_gate: array<f32>; // [n]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn deinterleave_qgate(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let block = idx / params.d_head;  // s * n_heads + h
  let e = idx % params.d_head;
  let src = block * (params.d_head * 2u) + e;
  out_q[idx] = input[src];
  out_gate[idx] = input[src + params.d_head];
}
