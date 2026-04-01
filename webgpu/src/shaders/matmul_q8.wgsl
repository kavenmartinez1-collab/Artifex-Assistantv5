// INT8 Dequantizing Matrix Multiplication
//
// C[M, N] = A[M, K] @ dequant_q8(B_packed, scales, zeros)^T
//
// B_packed stores 4 INT8 weights per i32, layout [K/4, N].
// Scales are per-group FP16 stored as u32 pairs, layout [num_groups, N].
// Zeros are per-group INT8 packed, layout [num_groups, N/4] as i32 (4 per i32).
//
// INT8: 256 levels (0..255), per-group asymmetric quantization.
//   weight = scale * (q8_val - zero_point)
//
// Tiled 16x16 workgroups with Kahan compensation, same structure as matmul_q4.wgsl.

const TILE: u32 = 16;

struct Params {
  M: u32,           // rows of A / rows of C
  N: u32,           // cols of output (out_features)
  K: u32,           // cols of A / in_features
  group_size: u32,  // quantization group size (typically 128)
}

// A: input activations [M, K] as f32
@group(0) @binding(0) var<storage, read> A: array<f32>;
// B_packed: INT8 packed weights [K/4, N] as i32 (4 weights per i32)
@group(0) @binding(1) var<storage, read> B_packed: array<i32>;
// C: output [M, N] as f32
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
// Params
@group(0) @binding(3) var<uniform> params: Params;
// Scales: per-group dequantization scales [num_groups, N] as f16 stored in u32
@group(0) @binding(4) var<storage, read> scales_raw: array<u32>;
// Zeros: per-group zero points [num_groups, N/4] as packed INT8 in i32
@group(0) @binding(5) var<storage, read> qzeros: array<i32>;
// g_idx: per-column group mapping [K] as u32 (for actorder support)
@group(0) @binding(6) var<storage, read> g_idx: array<u32>;

var<workgroup> tile_a: array<f32, 256>; // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

// Extract an 8-bit value from a packed i32 (4 bytes per i32)
fn extract_q8(packed: i32, byte_idx: u32) -> u32 {
  return u32((packed >> (byte_idx * 8u)) & 0xFF);
}

// Read f16 value from packed u32 array (2 f16 per u32)
// Uses exact bitwise conversion — no pow() approximation.
fn read_f16_scale(idx: u32) -> f32 {
  let word = scales_raw[idx / 2u];
  let half_bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
  let sign = (half_bits >> 15u) & 1u;
  let exp = (half_bits >> 10u) & 0x1Fu;
  let frac = half_bits & 0x3FFu;
  if (exp == 0u) {
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(frac) * bitcast<f32>(0x33800000u);
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) {
    return select(1e30, -1e30, sign == 1u);
  }
  let f32_bits = (sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u);
  return bitcast<f32>(f32_bits);
}

// Dequantize a single INT8 weight value
fn dequant_weight(q8_val: u32, scale: f32, zero: u32) -> f32 {
  return (f32(q8_val) - f32(zero)) * scale;
}

@compute @workgroup_size(16, 16)
fn matmul_bt_q8(@builtin(global_invocation_id) gid: vec3u,
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
  var comp: f32 = 0.0;  // Kahan compensation
  let num_tiles = (K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Load tile of A (same as standard matmul)
    let a_col = t * TILE + lid.y;
    if (row < M && a_col < K) {
      tile_a[local_idx] = A[row * K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    // Load tile of B^T with INT8 dequantization
    // B_packed layout: [K/4, N] — 4 INT8 weights packed per i32
    let b_k = t * TILE + lid.x;  // k index into weight
    if (b_k < K && col < N) {
      // Extract the 8-bit weight value
      // Packing: B_packed[k/4, n], byte at k%4
      let packed_idx = (b_k / 4u) * N + col;
      let byte_pos = b_k % 4u;
      let q8_val = extract_q8(B_packed[packed_idx], byte_pos);

      // Get group scale and zero point (actorder-aware via g_idx)
      let group_id = g_idx[b_k];
      let scale = read_f16_scale(group_id * N + col);

      // Get zero point (packed INT8: qzeros[group_id, col/4], byte col%4)
      let zero_packed_idx = group_id * ((N + 3u) / 4u) + col / 4u;
      let zero_byte = col % 4u;
      let zero = extract_q8(qzeros[zero_packed_idx], zero_byte);

      tile_b[local_idx] = dequant_weight(q8_val, scale, zero);
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Kahan-compensated tiled dot product
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let product = tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
      let y = product - comp;
      let t_val = sum + y;
      comp = (t_val - sum) - y;
      sum = t_val;
    }

    workgroupBarrier();
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}
