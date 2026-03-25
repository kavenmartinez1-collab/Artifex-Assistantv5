// INT4 Dequantizing Matrix Multiplication — GPTQ Format
//
// C[M, N] = A[M, K] @ dequant(B_packed, scales, zeros)^T
//
// B_packed stores 8 INT4 weights per i32, layout [K/8, N] (GPTQ format).
// Scales are per-group FP16 stored as u32 pairs, layout [K/group_size, N].
// Zeros are per-group INT4 packed, layout [K/group_size, N/8].
//
// During tile loading, each element of B is unpacked from 4-bit nibble,
// dequantized using the group scale and zero point, then stored as f32
// in shared memory. The inner loop is identical to standard tiled matmul.
//
// Tiled 16x16 workgroups, same structure as matmul_bt.

const TILE: u32 = 16;

struct Params {
  M: u32,           // rows of A / rows of C
  N: u32,           // cols of output (out_features)
  K: u32,           // cols of A / in_features
  group_size: u32,  // quantization group size (typically 128)
}

// A: input activations [M, K] as f32
@group(0) @binding(0) var<storage, read> A: array<f32>;
// B_packed: INT4 packed weights [K/8, N] as i32 (8 weights per i32)
@group(0) @binding(1) var<storage, read> B_packed: array<i32>;
// C: output [M, N] as f32
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
// Params
@group(0) @binding(3) var<uniform> params: Params;
// Scales: per-group dequantization scales [num_groups, N] as f16 stored in u32
@group(0) @binding(4) var<storage, read> scales_raw: array<u32>;
// Zeros: per-group zero points [num_groups, N/8] as packed INT4 in i32
@group(0) @binding(5) var<storage, read> qzeros: array<i32>;

var<workgroup> tile_a: array<f32, 256>; // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

// Extract a 4-bit value from a packed i32 (8 nibbles per i32)
fn extract_q4(packed: i32, nibble_idx: u32) -> u32 {
  return u32((packed >> (nibble_idx * 4u)) & 0xF);
}

// Read f16 value from packed u32 array (2 f16 per u32)
fn read_f16_scale(idx: u32) -> f32 {
  let word = scales_raw[idx / 2u];
  let half_bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
  // Decode IEEE 754 half-precision (f16) to f32
  let sign = (half_bits >> 15u) & 1u;
  let exp = (half_bits >> 10u) & 0x1Fu;
  let frac = half_bits & 0x3FFu;
  if (exp == 0u) {
    // Subnormal or zero
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(frac) / 1024.0 * pow(2.0, -14.0);
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) {
    // Inf/NaN
    return select(1e30, -1e30, sign == 1u);
  }
  let f = (1.0 + f32(frac) / 1024.0) * pow(2.0, f32(exp) - 15.0);
  return select(f, -f, sign == 1u);
}

// Dequantize a single INT4 weight value
fn dequant_weight(q4_val: u32, scale: f32, zero: u32) -> f32 {
  // GPTQ dequant: (q4_val - zero_point) * scale
  return (f32(q4_val) - f32(zero)) * scale;
}

@compute @workgroup_size(16, 16)
fn matmul_bt_q4(@builtin(global_invocation_id) gid: vec3u,
                @builtin(local_invocation_id) lid: vec3u,
                @builtin(workgroup_id) wid: vec3u) {

  let row = wid.x * TILE + lid.x;  // M dimension (input sequence)
  let col = wid.y * TILE + lid.y;  // N dimension (output features)
  let local_idx = lid.x * TILE + lid.y;

  let M = params.M;
  let N = params.N;
  let K = params.K;
  let gs = params.group_size;
  let num_groups = (K + gs - 1u) / gs;

  var sum: f32 = 0.0;
  let num_tiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Load tile of A (same as standard matmul)
    let a_col = t * TILE + lid.y;
    if (row < M && a_col < K) {
      tile_a[local_idx] = A[row * K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    // Load tile of B^T with INT4 dequantization
    // B_packed layout (GPTQ): [K/8, N] — 8 INT4 weights packed per i32
    // We need B^T[k, n] = dequant(B_packed[k, n])
    let b_k = t * TILE + lid.x;  // k index into weight
    if (b_k < K && col < N) {
      // Extract the 4-bit weight value
      // GPTQ packs 8 weights along the K dimension: B_packed[k/8, n]
      let packed_idx = (b_k / 8u) * N + col;
      let nibble = b_k % 8u;
      let q4_val = extract_q4(B_packed[packed_idx], nibble);

      // Get group scale and zero point
      let group_id = b_k / gs;
      let scale = read_f16_scale(group_id * N + col);

      // Get zero point (also packed INT4: qzeros[group_id, col/8], nibble col%8)
      let zero_packed_idx = group_id * ((N + 7u) / 8u) + col / 8u;
      let zero_nibble = col % 8u;
      let zero = extract_q4(qzeros[zero_packed_idx], zero_nibble);

      tile_b[local_idx] = dequant_weight(q4_val, scale, zero);
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Standard tiled dot product
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      sum += tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
