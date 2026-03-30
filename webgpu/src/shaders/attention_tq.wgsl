// Asymmetric Attention with TurboQuant QJL Correction
//
// Modified attention kernel that adds the QJL inner product correction
// to attention scores, compensating for PolarQuant reconstruction error.
//
// For each cache position j:
//   score_j = (<q, k̂_PQ_j> + correction_j) * scale
//   correction_j = norm_j * residual_norm_j * sqrt(pi/2)/sqrt(d) * <S*Pi*q, sign_bits_j>
//
// The QJL correction is only applied to previously-cached positions
// (j < pos_offset). Current-token positions have exact K in the cache
// and don't need correction.
//
// This implements the unbiased inner product estimator from TurboQuant
// (Google, ICLR 2026, arXiv:2504.19874), achieving near-lossless attention
// scores despite 3-4 bit KV cache compression.
//
// Supports GQA, causal masking, single-token decode and prefill.
//
// Dispatch: one workgroup per (query_position, head) pair.

struct Params {
  num_heads: u32,       // total Q heads
  num_kv_heads: u32,    // KV heads (for GQA)
  head_dim: u32,        // dimension per head
  new_seq_len: u32,     // number of new query tokens (1 for decode)
  cache_len: u32,       // total KV cache length (including new tokens)
  scale: f32,           // 1.0 / sqrt(head_dim) — attention scale
  is_causal: u32,       // 1 = apply causal mask, 0 = no mask
  pos_offset: u32,      // position offset of first new token in the sequence
  qjl_constant: f32,    // sqrt(pi/2) / sqrt(head_dim)
  sign_words_per_vec: u32,  // ceil(head_dim / 32)
}

// Group 0: standard attention I/O
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;    // PolarQuant-decoded K
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;    // decoded V (or raw)
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Group 1: TurboQuant QJL correction data
@group(1) @binding(0) var<storage, read> sign_bits_k: array<u32>;      // [total_vecs * sign_words]
@group(1) @binding(1) var<storage, read> norms_k: array<f32>;          // [total_vecs] — ||k||
@group(1) @binding(2) var<storage, read> residual_norms_k: array<f32>; // [total_vecs] — ||r||
@group(1) @binding(3) var<storage, read> spi_matrix: array<f32>;       // [d * d] — S*Pi, row-major

// Shared memory layout:
//   sq[256]     = precomputed S*Pi*q (max 256-dim heads)
//   scores[3584] = attention scores (max cache_len = 3584)
//   shmem[256]  = reduction scratch
// Total: (256 + 3584 + 256) * 4 = 16384 bytes = 16 KB (WebGPU default limit)
var<workgroup> sq: array<f32, 256>;
var<workgroup> scores: array<f32, 3584>;
var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn attention_tq(@builtin(local_invocation_id) lid: vec3u,
                @builtin(workgroup_id) wid: vec3u) {
  let q_pos = wid.x;      // which query position (0..new_seq_len-1)
  let head = wid.y;        // which Q head (0..num_heads-1)
  let tid = lid.x;

  let d = params.head_dim;
  let cache_len = params.cache_len;
  let num_kv = params.num_kv_heads;
  let scale = params.scale;
  let qjl_c = params.qjl_constant;
  let sign_words = params.sign_words_per_vec;

  // GQA: map Q head to KV head
  let num_q_per_kv = params.num_heads / num_kv;
  let kv_head = head / num_q_per_kv;

  // Base offsets
  let q_offset = (q_pos * params.num_heads + head) * d;
  let out_offset = (q_pos * params.num_heads + head) * d;

  // ── Precompute sq = S*Pi*q in shared memory ───────────────────────
  // sq[i] = sum_j(spi_matrix[i*d + j] * q[q_offset + j])
  // One workgroup handles one (q_pos, head), so this is computed once.
  var i = tid;
  while (i < d) {
    var dot: f32 = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
      dot += spi_matrix[i * d + j] * q[q_offset + j];
    }
    sq[i] = dot;
    i = i + 256u;
  }
  workgroupBarrier();

  // ── Step 1: Compute corrected attention scores ────────────────────
  // For each cache position j:
  //   base = <q, k̂_PQ[j]>  (standard dot product on decoded K)
  //   For j < pos_offset (compressed cache): add QJL correction
  //   score[j] = (base + correction) * scale

  var j = tid;
  while (j < cache_len) {
    let k_offset = (j * num_kv + kv_head) * d;

    // Standard Q*K^T dot product
    var base_dot: f32 = 0.0;
    for (var k = 0u; k < d; k = k + 1u) {
      base_dot += q[q_offset + k] * k_cache[k_offset + k];
    }

    // QJL correction — only for previously-cached (compressed) positions.
    // Current-token positions (j >= pos_offset) have exact K, no correction needed.
    var correction: f32 = 0.0;
    if (j < params.pos_offset) {
      let vec_idx = j * num_kv + kv_head;
      let sign_base = vec_idx * sign_words;

      // <sq, sign_bits[j]>
      var correction_dot: f32 = 0.0;
      for (var k = 0u; k < d; k = k + 1u) {
        let sw_idx = k / 32u;
        let bit_pos = k % 32u;
        let bit = (sign_bits_k[sign_base + sw_idx] >> bit_pos) & 1u;
        let sign_val = select(-1.0, 1.0, bit == 1u);
        correction_dot += sq[k] * sign_val;
      }

      // correction = ||k|| * ||r|| * sqrt(pi/2)/sqrt(d) * <sq, sign>
      correction = norms_k[vec_idx] * residual_norms_k[vec_idx] * qjl_c * correction_dot;
    }

    scores[j] = (base_dot + correction) * scale;

    // Causal mask
    if (params.is_causal == 1u) {
      let q_abs_pos = params.pos_offset + q_pos;
      if (j > q_abs_pos) {
        scores[j] = -1e9;
      }
    }

    j = j + 256u;
  }
  workgroupBarrier();

  // ── Step 2: Softmax over scores ───────────────────────────────────
  // Find max for numerical stability
  var local_max: f32 = -1e9;
  j = tid;
  while (j < cache_len) {
    local_max = max(local_max, scores[j]);
    j = j + 256u;
  }
  shmem[tid] = local_max;
  workgroupBarrier();

  // Parallel reduction for max
  if (tid < 128u) { shmem[tid] = max(shmem[tid], shmem[tid + 128u]); }
  workgroupBarrier();
  if (tid < 64u) { shmem[tid] = max(shmem[tid], shmem[tid + 64u]); }
  workgroupBarrier();
  if (tid < 32u) { shmem[tid] = max(shmem[tid], shmem[tid + 32u]); }
  workgroupBarrier();
  if (tid < 16u) { shmem[tid] = max(shmem[tid], shmem[tid + 16u]); }
  workgroupBarrier();
  if (tid < 8u) { shmem[tid] = max(shmem[tid], shmem[tid + 8u]); }
  workgroupBarrier();
  if (tid < 4u) { shmem[tid] = max(shmem[tid], shmem[tid + 4u]); }
  workgroupBarrier();
  if (tid < 2u) { shmem[tid] = max(shmem[tid], shmem[tid + 2u]); }
  workgroupBarrier();
  if (tid < 1u) { shmem[0] = max(shmem[0], shmem[1]); }
  workgroupBarrier();
  let max_score = shmem[0];

  // Compute exp(score - max) and sum
  var local_sum: f32 = 0.0;
  j = tid;
  while (j < cache_len) {
    let e = exp(scores[j] - max_score);
    scores[j] = e;
    local_sum += e;
    j = j + 256u;
  }
  shmem[tid] = local_sum;
  workgroupBarrier();

  // Parallel reduction for sum
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
  let sum_exp = shmem[0];

  // Normalize
  let inv_sum = 1.0 / sum_exp;
  j = tid;
  while (j < cache_len) {
    scores[j] = scores[j] * inv_sum;
    j = j + 256u;
  }
  workgroupBarrier();

  // ── Step 3: Weighted sum of values ────────────────────────────────
  // output[q_pos, head, k] = sum_j(scores[j] * V[j, kv_head, k])
  var k = tid;
  while (k < d) {
    var weighted_sum: f32 = 0.0;
    for (var jj = 0u; jj < cache_len; jj = jj + 1u) {
      let v_offset = (jj * num_kv + kv_head) * d;
      weighted_sum += scores[jj] * v_cache[v_offset + k];
    }
    output[out_offset + k] = weighted_sum;
    k = k + 256u;
  }
}
