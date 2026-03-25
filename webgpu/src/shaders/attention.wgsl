// Multi-Head Attention with Grouped-Query Attention (GQA)
//
// Fused kernel: computes Q·K^T scoring, causal masking, softmax, and
// weighted V sum in a single dispatch. Avoids materializing the full
// [num_heads, seq_len, cache_len] attention matrix in global memory.
//
// Supports:
//   - Standard MHA (num_kv_heads == num_heads)
//   - GQA (num_kv_heads < num_heads, heads grouped)
//   - Single-token decode (new_seq_len = 1, no causal mask needed)
//   - Prefill (new_seq_len > 1, causal mask applied)
//
// Layout (all contiguous f32):
//   Q:      [new_seq_len, num_heads * head_dim]
//   K_cache: [cache_len, num_kv_heads * head_dim]
//   V_cache: [cache_len, num_kv_heads * head_dim]
//   output:  [new_seq_len, num_heads * head_dim]
//
// Dispatch: one workgroup per (query_position, head) pair.

struct Params {
  num_heads: u32,       // total Q heads
  num_kv_heads: u32,    // KV heads (for GQA)
  head_dim: u32,        // dimension per head
  new_seq_len: u32,     // number of new query tokens (1 for decode)
  cache_len: u32,       // total KV cache length (including new tokens)
  scale: f32,           // 1.0 / sqrt(head_dim)
  is_causal: u32,       // 1 = apply causal mask, 0 = no mask
  pos_offset: u32,      // position offset of first new token in the sequence
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for attention scores (one row of the attention matrix)
// Max cache_len supported in shared memory = 3840
// Total workgroup storage: 3840*4 + 256*4 = 16384 bytes (exactly at default limit)
var<workgroup> scores: array<f32, 3840>;
var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn attention(@builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let q_pos = wid.x;      // which query position (0..new_seq_len-1)
  let head = wid.y;        // which Q head (0..num_heads-1)
  let tid = lid.x;

  let d = params.head_dim;
  let cache_len = params.cache_len;
  let num_kv = params.num_kv_heads;
  let scale = params.scale;

  // GQA: map Q head to KV head
  let num_q_per_kv = params.num_heads / num_kv;
  let kv_head = head / num_q_per_kv;

  // Base offsets into Q and output arrays
  let q_offset = (q_pos * params.num_heads + head) * d;
  let out_offset = (q_pos * params.num_heads + head) * d;

  // ── Step 1: Compute attention scores ────────────────────────────────
  // score[j] = sum_k(Q[q_pos, head, k] * K[j, kv_head, k]) * scale
  var j = tid;
  while (j < cache_len) {
    let k_offset = (j * num_kv + kv_head) * d;
    var dot: f32 = 0.0;
    for (var k = 0u; k < d; k = k + 1u) {
      dot += q[q_offset + k] * k_cache[k_offset + k];
    }
    scores[j] = dot * scale;

    // Causal mask: if this query position can't attend to cache position j
    if (params.is_causal == 1u) {
      let q_abs_pos = params.pos_offset + q_pos;
      if (j > q_abs_pos) {
        scores[j] = -1e9;  // effectively -infinity for softmax
      }
    }

    j = j + 256u;
  }
  workgroupBarrier();

  // ── Step 2: Softmax over scores ─────────────────────────────────────
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

  // Normalize: scores[j] = exp(score_j - max) / sum_exp
  let inv_sum = 1.0 / sum_exp;
  j = tid;
  while (j < cache_len) {
    scores[j] = scores[j] * inv_sum;
    j = j + 256u;
  }
  workgroupBarrier();

  // ── Step 3: Weighted sum of values ──────────────────────────────────
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
