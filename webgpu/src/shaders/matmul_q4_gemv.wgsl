// INT4 Dequantizing GEMV — GPTQ Format, specialized for M = 1 (decode)
//
// C[1, N] = A[1, K] @ dequant(B_packed, scales, zeros)^T
//
// This kernel is the decode-stage fast path for matmul_q4. The tiled 16x16
// kernel wastes 15/16 threads when M=1 (row >= M for all but lid.x=0). This
// GEMV uses one thread per output column, so every thread is useful.
//
// Bindings are IDENTICAL to matmul_bt_q4 so we can swap pipelines without
// rebuilding bind groups. Buffer layouts unchanged.
//
// Optimizations over the tiled kernel (at M=1):
//   * All 256 threads produce outputs (vs 16 of 256).
//   * A vector streamed into workgroup shared memory in 1024-element chunks.
//   * Per-group scale + zero cached in registers — re-decoded only when the
//     group_id changes (once per group_size K, typically 128). Saves ~127x
//     redundant f16 decodes and qzeros reads per output column.
//   * One B_packed read yields 8 nibbles processed inline with cached params.
//
// Kahan compensation retained — matches the tiled kernel's numerical behavior.

const WG_SIZE: u32 = 256;
const K_CHUNK: u32 = 1024;  // 4 KB workgroup memory for A chunks

// Pipeline-override constant: when 0, group_id is computed as k / group_size
// directly in registers, eliminating K in-loop VRAM reads of g_idx per output
// column. When 1, g_idx is consulted — required for GPTQ models quantized
// with actorder (desc_act=true). The engine compiles both variants and picks
// the right one per-tensor based on whether the model's g_idx was loaded from
// the file or synthesized as trivial k/group_size.
// u32 (not bool) because some WebGPU drivers historically had spottier bool-
// override support; 0/1 is universally safe.
override USE_ACTORDER: u32 = 0u;

struct Params {
  M: u32,           // must be 1 for this kernel
  N: u32,
  K: u32,
  group_size: u32,
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B_packed: array<i32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> scales_raw: array<u32>;
@group(0) @binding(5) var<storage, read> qzeros: array<i32>;
@group(0) @binding(6) var<storage, read> g_idx: array<u32>;

var<workgroup> a_chunk: array<f32, 1024>;

fn extract_q4(packed: i32, nibble_idx: u32) -> u32 {
  return u32((packed >> (nibble_idx * 4u)) & 0xF);
}

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

@compute @workgroup_size(256, 1, 1)
fn matmul_bt_q4_gemv(@builtin(local_invocation_id) lid: vec3u,
                     @builtin(workgroup_id) wid: vec3u) {
  let col = wid.x * WG_SIZE + lid.x;
  let N = params.N;
  let K = params.K;
  let tid = lid.x;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;  // Kahan compensation — matches tiled kernel
  var cur_group: u32 = 0xFFFFFFFFu;  // sentinel: no group loaded yet
  var cached_scale: f32 = 0.0;
  var cached_zero: u32 = 0u;

  let num_chunks = (K + K_CHUNK - 1u) / K_CHUNK;

  for (var chunk: u32 = 0u; chunk < num_chunks; chunk = chunk + 1u) {
    let k_base = chunk * K_CHUNK;
    let k_end_full = k_base + K_CHUNK;
    let k_end = select(k_end_full, K, k_end_full > K);
    let chunk_size = k_end - k_base;

    // Cooperative load of this chunk of A into workgroup memory.
    // All 256 threads participate, striding by WG_SIZE.
    for (var i: u32 = tid; i < chunk_size; i = i + WG_SIZE) {
      a_chunk[i] = A[k_base + i];
    }
    workgroupBarrier();

    if (col < N) {
      // Walk the chunk 8 K-values at a time (one packed i32 covers 8 nibbles).
      // K_CHUNK = 1024 is a multiple of 8; the tail chunk (if any) rounds down
      // — remainder (K not divisible by 8) is unreachable in practice because
      // GPTQ always pads K to a multiple of group_size, which is a multiple of 8.
      let p_start = k_base / 8u;
      let p_end = k_end / 8u;
      for (var p: u32 = p_start; p < p_end; p = p + 1u) {
        let packed = B_packed[p * N + col];
        for (var nib: u32 = 0u; nib < 8u; nib = nib + 1u) {
          let k = p * 8u + nib;
          // With actorder we need the reordered mapping from g_idx; without
          // it the mapping is trivially k / group_size and costs no memory.
          var group_id: u32;
          if (USE_ACTORDER != 0u) {
            group_id = g_idx[k];
          } else {
            group_id = k / params.group_size;
          }
          if (group_id != cur_group) {
            cur_group = group_id;
            cached_scale = read_f16_scale(group_id * N + col);
            let zero_packed_idx = group_id * ((N + 7u) / 8u) + col / 8u;
            let zero_nibble = col % 8u;
            cached_zero = extract_q4(qzeros[zero_packed_idx], zero_nibble);
          }
          let q4_val = extract_q4(packed, nib);
          let w = (f32(q4_val) - f32(cached_zero)) * cached_scale;
          let product = a_chunk[k - k_base] * w;
          // Kahan-compensated accumulation — matches matmul_q4.wgsl
          let y = product - comp;
          let t_val = sum + y;
          comp = (t_val - sum) - y;
          sum = t_val;
        }
      }
    }

    // Barrier before the next chunk's cooperative load.
    workgroupBarrier();
  }

  if (col < N) {
    C[col] = sum;
  }
}
