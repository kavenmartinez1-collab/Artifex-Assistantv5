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
  // Sliding-window attention (Gemma 4): 0 = disabled (full causal).
  // llama.cpp LLAMA_SWA_TYPE_STANDARD semantics: key j is masked when
  // q_pos - j >= window. Applied in BOTH prefill and decode (unlike the
  // causal mask, which only matters when new_seq_len > 1).
  window: u32,
}

// Softpick (rectified softmax, arXiv:2504.20966): USE_SOFTPICK=1 replaces
// the exp normalization with ReLU(e^(x-max) - e^(-max_clamped)) / sum(|...|).
// Allows true-zero attention weights, eliminating the attention-sink failure
// mode where softmax pins mass on one early token. Masked positions (score
// <= -1e8) are excluded from both numerator and denominator to avoid
// threshold pollution.
override USE_SOFTPICK: u32 = 0u;

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Shared memory for attention scores (one row of the attention matrix)
// Max cache_len supported in shared memory = 2048 (matches MAX_ATTN_SEQ_LEN)
// Total workgroup storage: 2048*4 + 256*4 = 9216 bytes (well under 16384 default limit)
var<workgroup> scores: array<f32, 2048>;
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
  // Uses Kahan summation for d=256+ head dimensions
  var j = tid;
  while (j < cache_len) {
    let k_offset = (j * num_kv + kv_head) * d;
    var dot: f32 = 0.0;
    var dot_comp: f32 = 0.0;
    for (var k = 0u; k < d; k = k + 1u) {
      let product = q[q_offset + k] * k_cache[k_offset + k];
      let y = product - dot_comp;
      let t_val = dot + y;
      dot_comp = (t_val - dot) - y;
      dot = t_val;
    }
    scores[j] = dot * scale;

    let q_abs_pos = params.pos_offset + q_pos;

    // Causal mask: if this query position can't attend to cache position j
    if (params.is_causal == 1u) {
      if (j > q_abs_pos) {
        scores[j] = -1e9;  // effectively -infinity for softmax
      }
    }

    // Sliding-window mask (active in decode too): mask j when
    // q_abs_pos - j >= window  ⇔  j + window <= q_abs_pos
    if (params.window > 0u && j + params.window <= q_abs_pos) {
      scores[j] = -1e9;
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

  // Compute numerator and denominator.
  // Standard softmax: numerator_j = exp(score_j - max), denom = sum(numerator_j).
  // Softpick:        numerator_j = ReLU(exp(score_j - max) - p),
  //                  denom      = sum |exp(score_j - max) - p|,
  //                  where p = exp(-max(max_score, 0)) is the rectification
  //                  threshold in shifted space (chosen so the numerator
  //                  becomes positive iff the original score > 0).
  let softpick_p = select(0.0, exp(-max(max_score, 0.0)), USE_SOFTPICK != 0u);
  var local_sum: f32 = 0.0;
  j = tid;
  while (j < cache_len) {
    let raw = scores[j];
    let ex = exp(raw - max_score);
    if (USE_SOFTPICK != 0u) {
      // Mask-aware: causal-masked entries have raw ≈ -1e9; skip them so
      // they don't each add `p` to the denominator.
      if (raw > -1.0e8) {
        let diff = ex - softpick_p;
        scores[j] = max(0.0, diff);
        local_sum += abs(diff);
      } else {
        scores[j] = 0.0;
      }
    } else {
      scores[j] = ex;
      local_sum += ex;
    }
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

  // Normalize: scores[j] = numerator / denom.
  // Softpick may produce denom = 0 (row has no above-threshold keys); use
  // epsilon to avoid NaN. Standard softmax sum_exp is strictly positive
  // (at least one entry = exp(0) = 1), so the epsilon is a no-op there.
  let inv_sum = 1.0 / max(sum_exp, 1.0e-20);
  j = tid;
  while (j < cache_len) {
    scores[j] = scores[j] * inv_sum;
    j = j + 256u;
  }
  workgroupBarrier();

  // ── Step 3: Weighted sum of values ──────────────────────────────────
  // output[q_pos, head, k] = sum_j(scores[j] * V[j, kv_head, k])
  // Kahan summation for cache_len elements
  var k = tid;
  while (k < d) {
    var weighted_sum: f32 = 0.0;
    var ws_comp: f32 = 0.0;
    for (var jj = 0u; jj < cache_len; jj = jj + 1u) {
      let v_offset = (jj * num_kv + kv_head) * d;
      let product = scores[jj] * v_cache[v_offset + k];
      let y = product - ws_comp;
      let t_val = weighted_sum + y;
      ws_comp = (t_val - weighted_sum) - y;
      weighted_sum = t_val;
    }
    output[out_offset + k] = weighted_sum;
    k = k + 256u;
  }
}
