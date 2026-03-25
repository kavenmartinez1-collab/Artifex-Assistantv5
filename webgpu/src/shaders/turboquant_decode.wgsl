// TurboQuant Decode — KV Cache Decompression (Google, ICLR 2026)
//
// Reverses the TurboQuant encode:
//   1. Unpack quantized indices → look up centroids → dequantized vector
//   2. Unpack sign bits → inverse JL transform → correction vector
//   3. Add: reconstructed_rotated = dequantized + correction
//   4. Inverse rotation: x_hat = Pi^T · reconstructed_rotated
//   5. Rescale by stored norm
//
// One workgroup processes one vector. Workgroup size = 256.

struct Params {
  head_dim: u32,
  bits: u32,
  num_centroids: u32,
  num_thresholds: u32,
}

// Input: packed quantized indices (num_vectors × packed_words)
@group(0) @binding(0) var<storage, read> input_quantized: array<u32>;
// Input: QJL sign bits (num_vectors × sign_words)
@group(0) @binding(1) var<storage, read> input_sign_bits: array<u32>;
// Output: reconstructed f32 vectors (num_vectors × head_dim)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// Input: norms per vector (num_vectors × 1 f32)
@group(0) @binding(3) var<storage, read> input_norms: array<f32>;

// Rotation matrix Pi (d × d, column-major f32)
@group(1) @binding(0) var<storage, read> rotation_matrix: array<f32>;
// JL matrix S (d × d, column-major f32)
@group(1) @binding(1) var<storage, read> jl_matrix: array<f32>;
// Positive centroids (f32)
@group(1) @binding(2) var<storage, read> centroids: array<f32>;

@group(2) @binding(0) var<uniform> params: Params;

// Shared memory for the reconstructed vector in rotated space
var<workgroup> reconstructed: array<f32, 256>;

@compute @workgroup_size(256)
fn decode(@builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
  let vec_idx = wid.x;
  let tid = lid.x;
  let d = params.head_dim;
  let bits = params.bits;
  let num_pos = params.num_centroids;
  let inv_sqrt_d = 1.0 / sqrt(f32(d));

  let indices_per_u32 = 32u / bits;
  let packed_words = (d + indices_per_u32 - 1u) / indices_per_u32;
  let sign_words = (d + 31u) / 32u;
  let in_offset_quant = vec_idx * packed_words;
  let in_offset_sign = vec_idx * sign_words;
  let mask = (1u << bits) - 1u;

  // Read the norm for this vector
  let norm = input_norms[vec_idx];

  // ── Step 1: Unpack indices and dequantize (PolarQuant reconstruction) ─
  var i = tid;
  while (i < d) {
    let word_idx = i / indices_per_u32;
    let bit_offset = (i % indices_per_u32) * bits;
    let idx = (input_quantized[in_offset_quant + word_idx] >> bit_offset) & mask;

    let is_negative = idx >= num_pos;
    let bin = select(idx, idx - num_pos, is_negative);
    let sign_val = select(1.0, -1.0, is_negative);

    reconstructed[i] = sign_val * centroids[bin] * inv_sqrt_d;

    i = i + 256u;
  }
  workgroupBarrier();

  // ── Step 2: QJL correction ──────────────────────────────────────────
  // TODO: fix QJL scale factor — currently disabled to validate PolarQuant
  let sqrt_pi_over_2 = 1.2533141;  // sqrt(pi / 2)
  let jl_scale = 0.0;  // disabled: sqrt_pi_over_2 / f32(d);

  i = tid;
  while (i < d) {
    var sum: f32 = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
      let sw_idx = j / 32u;
      let bit_pos = j % 32u;
      let bit = (input_sign_bits[in_offset_sign + sw_idx] >> bit_pos) & 1u;
      let sign_val = select(-1.0, 1.0, bit == 1u);

      // S^T access: S[j * d + i]
      sum += jl_matrix[j * d + i] * sign_val;
    }

    reconstructed[i] = reconstructed[i] + sum * jl_scale;

    i = i + 256u;
  }
  workgroupBarrier();

  // ── Step 3: Inverse rotation + rescale by norm ──────────────────────
  // output = norm * Pi^T · reconstructed
  i = tid;
  while (i < d) {
    var sum: f32 = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
      sum += rotation_matrix[j * d + i] * reconstructed[j];
    }
    output[vec_idx * d + i] = sum * norm;

    i = i + 256u;
  }
}
