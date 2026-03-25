// TurboQuant Encode — KV Cache Compression (Google, ICLR 2026)
//
// Takes f32 KV vectors, compresses to (bits-1) + 1 bits per coordinate:
//   Stage 0: Compute L2 norm, normalize to unit vector
//   Stage 1 (PolarQuant): rotate → scalar quantize → pack indices
//   Stage 2 (QJL):        compute residual → JL project → sign bits
//
// One workgroup processes one vector (one row of the KV cache).
// Workgroup size = 256 threads, each thread handles ceil(d/256) coordinates.

struct Params {
  head_dim: u32,      // d (typically 64 or 128)
  bits: u32,          // quantization bits (1-4)
  num_centroids: u32, // number of positive centroids = 2^(bits-1)
  num_thresholds: u32,// number of thresholds = num_centroids - 1
  out_vec_offset: u32,// offset for output indexing (= position * num_kv_heads)
}

// Input: f32 vectors to quantize (num_vectors × head_dim)
@group(0) @binding(0) var<storage, read> input: array<f32>;
// Output: packed quantized indices (num_vectors × packed_words)
@group(0) @binding(1) var<storage, read_write> output_quantized: array<u32>;
// Output: QJL sign bits (num_vectors × sign_words)
@group(0) @binding(2) var<storage, read_write> output_sign_bits: array<u32>;
// Output: norms per vector (num_vectors × 1 f32)
@group(0) @binding(3) var<storage, read_write> output_norms: array<f32>;

// Rotation matrix Pi (d × d, column-major f32)
@group(1) @binding(0) var<storage, read> rotation_matrix: array<f32>;
// JL matrix S (d × d, column-major f32)
@group(1) @binding(1) var<storage, read> jl_matrix: array<f32>;
// Positive centroids (f32, length = num_centroids)
@group(1) @binding(2) var<storage, read> centroids: array<f32>;
// Decision thresholds (f32, length = num_thresholds)
@group(1) @binding(3) var<storage, read> thresholds: array<f32>;

@group(2) @binding(0) var<uniform> params: Params;

// Shared memory
var<workgroup> shmem: array<f32, 256>;
var<workgroup> rotated: array<f32, 256>;
var<workgroup> dequantized: array<f32, 256>;
var<workgroup> quant_indices: array<u32, 256>;

@compute @workgroup_size(256)
fn encode(@builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {
  let vec_idx = wid.x;           // local index within this dispatch
  let out_idx = params.out_vec_offset + vec_idx; // global index in compressed cache
  let tid = lid.x;
  let d = params.head_dim;
  let bits = params.bits;
  let num_pos = params.num_centroids;
  let num_thresh = params.num_thresholds;

  // ── Stage 0: Compute L2 norm via parallel reduction ─────────────────
  var partial_sq: f32 = 0.0;
  var i = tid;
  while (i < d) {
    let val = input[vec_idx * d + i];
    partial_sq += val * val;
    i = i + 256u;
  }
  shmem[tid] = partial_sq;
  workgroupBarrier();

  // Parallel reduction for sum of squares
  if (tid < 128u) { shmem[tid] = shmem[tid] + shmem[tid + 128u]; }
  workgroupBarrier();
  if (tid < 64u) { shmem[tid] = shmem[tid] + shmem[tid + 64u]; }
  workgroupBarrier();
  if (tid < 32u) { shmem[tid] = shmem[tid] + shmem[tid + 32u]; }
  workgroupBarrier();
  if (tid < 16u) { shmem[tid] = shmem[tid] + shmem[tid + 16u]; }
  workgroupBarrier();
  if (tid < 8u) { shmem[tid] = shmem[tid] + shmem[tid + 8u]; }
  workgroupBarrier();
  if (tid < 4u) { shmem[tid] = shmem[tid] + shmem[tid + 4u]; }
  workgroupBarrier();
  if (tid < 2u) { shmem[tid] = shmem[tid] + shmem[tid + 2u]; }
  workgroupBarrier();
  if (tid < 1u) { shmem[0] = shmem[0] + shmem[1]; }
  workgroupBarrier();

  let norm = sqrt(shmem[0]);
  let inv_norm = select(0.0, 1.0 / norm, norm > 1e-8);

  // Store norm for decode (at global cache position)
  if (tid == 0u) {
    output_norms[out_idx] = norm;
  }

  // ── Stage 1a: Rotate normalized vector ──────────────────────────────
  // rotated[i] = sum_j(Pi[i,j] * (input[j] / norm))
  let sqrt_d = sqrt(f32(d));
  let inv_sqrt_d = 1.0 / sqrt_d;

  i = tid;
  while (i < d) {
    var sum: f32 = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
      sum += rotation_matrix[i * d + j] * input[vec_idx * d + j] * inv_norm;
    }
    rotated[i] = sum;
    i = i + 256u;
  }
  workgroupBarrier();

  // ── Stage 1b: Scalar quantize each coordinate ───────────────────────
  i = tid;
  while (i < d) {
    let val = rotated[i] * sqrt_d;  // scale to ~N(0,1)
    let abs_val = abs(val);
    let sign_val = select(-1.0, 1.0, val >= 0.0);

    var bin = 0u;
    for (var t = 0u; t < num_thresh; t = t + 1u) {
      if (abs_val > thresholds[t]) {
        bin = t + 1u;
      }
    }

    let idx = select(num_pos + bin, bin, val >= 0.0);
    quant_indices[i] = idx;

    // Dequantize for residual
    dequantized[i] = sign_val * centroids[bin] * inv_sqrt_d;

    i = i + 256u;
  }
  workgroupBarrier();

  // ── Pack quantized indices into u32 words ───────────────────────────
  let indices_per_u32 = 32u / bits;
  let packed_words = (d + indices_per_u32 - 1u) / indices_per_u32;
  let out_offset_quant = out_idx * packed_words;

  var w = tid;
  while (w < packed_words) {
    var packed: u32 = 0u;
    let base_idx = w * indices_per_u32;
    for (var k = 0u; k < indices_per_u32; k = k + 1u) {
      let coord = base_idx + k;
      if (coord < d) {
        packed = packed | ((quant_indices[coord] & ((1u << bits) - 1u)) << (k * bits));
      }
    }
    output_quantized[out_offset_quant + w] = packed;
    w = w + 256u;
  }
  workgroupBarrier();

  // ── Stage 2: QJL residual correction ────────────────────────────────
  // Compute residual in rotated space (reuse shmem as residual storage)
  i = tid;
  while (i < d) {
    shmem[i] = rotated[i] - dequantized[i];
    i = i + 256u;
  }
  workgroupBarrier();

  // sign_bits[i] = sign(sum_j(S[i,j] * residual[j]))
  let sign_words = (d + 31u) / 32u;
  let out_offset_sign = out_idx * sign_words;

  var sw = tid;
  while (sw < sign_words) {
    var sign_word: u32 = 0u;
    let base_row = sw * 32u;

    for (var bit_pos = 0u; bit_pos < 32u; bit_pos = bit_pos + 1u) {
      let row = base_row + bit_pos;
      if (row < d) {
        var dot: f32 = 0.0;
        for (var j = 0u; j < d; j = j + 1u) {
          dot += jl_matrix[row * d + j] * shmem[j];
        }
        if (dot >= 0.0) {
          sign_word = sign_word | (1u << bit_pos);
        }
      }
    }

    output_sign_bits[out_offset_sign + sw] = sign_word;
    sw = sw + 256u;
  }
}
