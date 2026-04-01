// E8 Lattice 2-bit Dequantizing Matrix Multiplication
//
// C[M, N] = A[M, K] @ dequant_e8(B_indices, scales, offsets, codebook)^T
//
// B_indices stores 4 uint8 codebook indices per u32, layout [N, K/8/4].
// Each uint8 index maps to an 8-element vector in the E8 codebook [256, 8].
// Scales and offsets are per-group float16 packed in u32.
//
// Weight reconstruction:
//   For weight at position (n, k):
//     vec_idx = k / 8      (which 8-element E8 vector)
//     vec_off = k % 8      (position within vector)
//     cb_index = B_indices[n, vec_idx]  (uint8 codebook index)
//     group_id = k / group_size
//     weight = codebook[cb_index * 8 + vec_off] * scale[group_id, n] + offset[group_id, n]
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
// B_indices: packed uint8 codebook indices [N, K/8/4] as u32 (4 indices per u32)
@group(0) @binding(1) var<storage, read> B_indices: array<u32>;
// C: output [M, N] as f32
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
// Params
@group(0) @binding(3) var<uniform> params: Params;
// Scales: per-group dequantization scales [num_groups, N] as f16 stored in u32
@group(0) @binding(4) var<storage, read> scales_raw: array<u32>;
// Offsets: per-group dequantization offsets [num_groups, N] as f16 stored in u32
@group(0) @binding(5) var<storage, read> offsets_raw: array<u32>;
// Codebook: E8 lattice vectors [256 * 8] as f32
@group(0) @binding(6) var<storage, read> codebook: array<f32>;

var<workgroup> tile_a: array<f32, 256>; // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

// Extract uint8 index from packed u32 (4 bytes per u32)
fn extract_u8(packed: u32, byte_idx: u32) -> u32 {
  return (packed >> (byte_idx * 8u)) & 0xFFu;
}

// Decode f16 half_bits to f32 — shared logic for scales and offsets.
// Uses exact bitwise conversion, same as matmul_q4.wgsl read_f16_scale.
fn decode_f16(half_bits: u32) -> f32 {
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

// Read f16 scale at logical index idx (2 f16 packed per u32)
fn read_scale(idx: u32) -> f32 {
  let word = scales_raw[idx / 2u];
  let half_bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
  return decode_f16(half_bits);
}

// Read f16 offset at logical index idx (2 f16 packed per u32)
fn read_offset(idx: u32) -> f32 {
  let word = offsets_raw[idx / 2u];
  let half_bits = select(word & 0xFFFFu, word >> 16u, (idx & 1u) == 1u);
  return decode_f16(half_bits);
}

@compute @workgroup_size(16, 16)
fn matmul_e8(@builtin(global_invocation_id) gid: vec3u,
             @builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {

  let row = wid.x * TILE + lid.x;  // M dimension (input sequence)
  let col = wid.y * TILE + lid.y;  // N dimension (output features)
  let local_idx = lid.x * TILE + lid.y;

  let M = params.M;
  let N = params.N;
  let K = params.K;
  let gs = params.group_size;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;  // Kahan compensation
  let num_tiles = (K + TILE - 1u) / TILE;

  // Number of E8 vectors per row: K / 8, packed 4 per u32 => K / 32 u32s per row
  let vecs_per_row_packed = K / 32u;

  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    // Load tile of A (same as standard matmul)
    let a_col = t * TILE + lid.y;
    if (row < M && a_col < K) {
      tile_a[local_idx] = A[row * K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    // Load tile of B^T with E8 dequantization
    // B_indices layout: [N, K/8/4] — 4 uint8 indices packed per u32
    // Each index maps to an 8-element codebook vector
    let b_k = t * TILE + lid.x;  // k index into weight
    if (b_k < K && col < N) {
      // Which E8 vector does this k belong to?
      let vec_idx = b_k / 8u;          // which 8-element vector (0..K/8-1)
      let vec_offset = b_k % 8u;       // position within vector (0..7)

      // Get the codebook index: B_indices[col, vec_idx/4], byte vec_idx%4
      let packed_word_idx = col * vecs_per_row_packed + vec_idx / 4u;
      let byte_pos = vec_idx % 4u;
      let cb_index = extract_u8(B_indices[packed_word_idx], byte_pos);

      // Lookup codebook value for this position within the E8 vector
      let cb_val = codebook[cb_index * 8u + vec_offset];

      // Get per-group scale and offset
      // Layout: [num_groups, N] as f16 packed in u32
      let group_id = b_k / gs;
      let scale_idx = group_id * N + col;
      let scale = read_scale(scale_idx);
      let offset = read_offset(scale_idx);

      tile_b[local_idx] = cb_val * scale + offset;
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
