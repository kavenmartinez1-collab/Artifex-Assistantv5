// Fast Walsh-Hadamard Transform (FWHT) — In-place, O(n log n)
//
// Applies the normalized Walsh-Hadamard transform to each row of the input.
// Used for QuIP#/QuaRot incoherence processing: rotates activations so that
// the quantized weight (stored in Hadamard-rotated space) can be used directly.
//
// The transform uses butterfly operations (add/subtract pairs) — no multiplications.
// After all log2(n) stages, divides by sqrt(n) for normalization.
//
// Layout: input[rows, cols] where cols must be a power of 2.
// Each workgroup processes one row.

struct Params {
  cols: u32,       // number of columns (must be power of 2)
  rows: u32,       // number of rows to process
  sign_seed: u32,  // seed for randomized sign flips (0 = no sign flips)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// Simple hash for deterministic per-element sign from seed
fn sign_flip(seed: u32, idx: u32) -> f32 {
  if (seed == 0u) { return 1.0; }
  // xorshift-based hash
  var h = seed ^ (idx * 2654435761u);
  h ^= h >> 16u;
  h *= 0x45d9f3bu;
  h ^= h >> 16u;
  return select(1.0, -1.0, (h & 1u) == 1u);
}

@compute @workgroup_size(256)
fn hadamard(@builtin(local_invocation_id) lid: vec3u,
            @builtin(workgroup_id) wid: vec3u) {
  let row = wid.x;
  if (row >= params.rows) { return; }

  let n = params.cols;
  let tid = lid.x;
  let row_offset = row * n;

  // Copy input to output with optional sign flips
  var i = tid;
  while (i < n) {
    let s = sign_flip(params.sign_seed, i);
    output[row_offset + i] = input[row_offset + i] * s;
    i += 256u;
  }
  storageBarrier();

  // Butterfly stages: log2(n) stages
  var half = 1u;
  while (half < n) {
    let stride = half * 2u;
    // Each thread handles one butterfly pair
    var j = tid;
    while (j < n / 2u) {
      let block = j / half;
      let pos = j % half;
      let idx_a = row_offset + block * stride + pos;
      let idx_b = idx_a + half;

      let a = output[idx_a];
      let b = output[idx_b];
      output[idx_a] = a + b;
      output[idx_b] = a - b;

      j += 256u;
    }
    storageBarrier();
    half *= 2u;
  }

  // Normalize by 1/sqrt(n)
  let inv_sqrt_n = 1.0 / sqrt(f32(n));
  i = tid;
  while (i < n) {
    output[row_offset + i] *= inv_sqrt_n;
    i += 256u;
  }
}
