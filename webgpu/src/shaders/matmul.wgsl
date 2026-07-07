// Matrix Multiplication — Tiled implementation for WebGPU
// C[M, N] = A[M, K] @ B[K, N]
//
// Uses shared memory tiling for coalesced reads and data reuse.
// TILE_SIZE = 16 (16x16 tiles, 256 threads per workgroup)

const TILE: u32 = 16;

struct Params {
  M: u32,  // rows of A / rows of C
  N: u32,  // cols of B / cols of C
  K: u32,  // cols of A / rows of B
}

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tile_a: array<f32, 256>; // TILE * TILE
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) gid: vec3u,
          @builtin(local_invocation_id) lid: vec3u,
          @builtin(workgroup_id) wid: vec3u) {

  let row = wid.x * TILE + lid.x;
  let col = wid.y * TILE + lid.y;
  let local_idx = lid.x * TILE + lid.y;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  let num_tiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t++) {
    let a_col = t * TILE + lid.y;
    if (row < params.M && a_col < params.K) {
      tile_a[local_idx] = A[row * params.K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    let b_row = t * TILE + lid.x;
    if (b_row < params.K && col < params.N) {
      tile_b[local_idx] = B[b_row * params.N + col];
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k++) {
      let product = tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
      let y = product - comp;
      let t_val = sum + y;
      comp = (t_val - sum) - y;
      sum = t_val;
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    C[row * params.N + col] = sum;
  }
}

// ── B-transposed tiled matmul ────────────────────────────────────────────
// C[M, N] = A[M, K] @ B^T[K, N]  where B is stored as [N, K]
// Used for HuggingFace linear layers: weight shape = [out_features, in_features]
// So output = input[seq, in] @ weight^T = input[seq, K] @ weight[N, K]^T

@compute @workgroup_size(16, 16)
fn matmul_bt(@builtin(global_invocation_id) gid: vec3u,
             @builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {

  let row = wid.x * TILE + lid.x;
  let col = wid.y * TILE + lid.y;
  let local_idx = lid.x * TILE + lid.y;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  let num_tiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t++) {
    let a_col = t * TILE + lid.y;
    if (row < params.M && a_col < params.K) {
      tile_a[local_idx] = A[row * params.K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    let b_k = t * TILE + lid.x;
    if (b_k < params.K && col < params.N) {
      tile_b[local_idx] = B[col * params.K + b_k];
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k++) {
      let product = tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
      let y = product - comp;
      let t_val = sum + y;
      comp = (t_val - sum) - y;
      sum = t_val;
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    C[row * params.N + col] = sum;
  }
}

// ── B-transposed matmul with BF16 weight ────────────────────────────────
// Same as matmul_bt but B is stored as BF16 (u16 packed in u32 pairs).
// Used for lm_head when vocab embedding exceeds 2GB at f32.
// B_bf16 layout: [N, K/2] as u32, each u32 holds two bf16 values.

@group(0) @binding(5) var<storage, read> B_bf16: array<u32>;

// Convert BF16 (u16) to f32: shift left by 16 bits
fn bf16_to_f32(val: u32) -> f32 {
  return bitcast<f32>(val << 16u);
}

@compute @workgroup_size(16, 16)
fn matmul_bt_bf16(@builtin(global_invocation_id) gid: vec3u,
                  @builtin(local_invocation_id) lid: vec3u,
                  @builtin(workgroup_id) wid: vec3u) {

  let row = wid.x * TILE + lid.x;
  let col = wid.y * TILE + lid.y;
  let local_idx = lid.x * TILE + lid.y;

  var sum: f32 = 0.0;
  var comp: f32 = 0.0;
  let num_tiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t++) {
    let a_col = t * TILE + lid.y;
    if (row < params.M && a_col < params.K) {
      tile_a[local_idx] = A[row * params.K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    let b_k = t * TILE + lid.x;
    if (b_k < params.K && col < params.N) {
      let packed_idx = col * (params.K / 2u) + b_k / 2u;
      let packed = B_bf16[packed_idx];
      let bf16_val = select(packed >> 16u, packed & 0xFFFFu, (b_k % 2u) == 0u);
      tile_b[local_idx] = bf16_to_f32(bf16_val);
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k++) {
      let product = tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
      let y = product - comp;
      let t_val = sum + y;
      comp = (t_val - sum) - y;
      sum = t_val;
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    C[row * params.N + col] = sum;
  }
}

// ── Fast B-transposed bf16 GEMM (FLUX.2 DiT workhorse) ──────────────────
// Same bindings/params as matmul_bt_bf16, but tuned for large-M GEMMs where
// Kahan compensation is unnecessary (DiT block tolerance 5e-4; plain f32
// accumulation over K<=27648 lands ~1e-6 rel):
//   - 64x64 output tile per 16x16 workgroup: 4x4 outputs per thread
//     (strided by 16 so lanes stay coalesced on loads and stores)
//   - plain fma accumulation (no Kahan: 1 op/MAC instead of 4)
//   - bf16 weights fetched one u32 pair at a time (each packed word read
//     once, not twice as in matmul_bt_bf16)
//   - LDS tiles padded to stride 17 to avoid bank conflicts
// Dispatch grid: ceil(M/64) x ceil(N/64). K must be even (bf16 pairing).

const FT: u32 = 64u;  // output tile edge
const FK: u32 = 16u;  // K tile depth
const FP: u32 = 17u;  // padded LDS row stride

var<workgroup> fa: array<f32, 1088>; // [64][17] A tile (m-major)
var<workgroup> fb: array<f32, 1088>; // [64][17] B tile (n-major)

@compute @workgroup_size(16, 16)
fn matmul_bt_bf16_fast(@builtin(local_invocation_id) lid: vec3u,
                       @builtin(workgroup_id) wid: vec3u) {
  let m0 = wid.x * FT;
  let n0 = wid.y * FT;
  // local_invocation_index: lid.x is the fast (in-wave) dimension.
  let tid = lid.y * 16u + lid.x;

  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i++) { acc[i] = 0.0; }

  let num_tiles = (params.K + FK - 1u) / FK;
  for (var t = 0u; t < num_tiles; t++) {
    let k0 = t * FK;

    // A tile: 64 rows x 16 k = 1024 f32, 4 per thread (consecutive lanes
    // read consecutive k within a row).
    for (var i = 0u; i < 4u; i++) {
      let e = i * 256u + tid;
      let r = e / FK;
      let kk = e % FK;
      var v = 0.0;
      if (m0 + r < params.M && k0 + kk < params.K) {
        v = A[(m0 + r) * params.K + k0 + kk];
      }
      fa[r * FP + kk] = v;
    }

    // B tile: 64 n-rows x 8 packed u32 (16 bf16) = 512 u32, 2 per thread.
    for (var i = 0u; i < 2u; i++) {
      let e = i * 256u + tid;
      let nloc = e / 8u;
      let kp = e % 8u;
      let kk = k0 + kp * 2u;
      var lo = 0.0;
      var hi = 0.0;
      if (n0 + nloc < params.N && kk < params.K) {
        let packed = B_bf16[(n0 + nloc) * (params.K / 2u) + kk / 2u];
        lo = bf16_to_f32(packed & 0xFFFFu);
        hi = bf16_to_f32(packed >> 16u);
      }
      fb[nloc * FP + kp * 2u] = lo;
      fb[nloc * FP + kp * 2u + 1u] = hi;
    }

    workgroupBarrier();

    for (var k = 0u; k < FK; k++) {
      var av: array<f32, 4>;
      var bv: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) {
        av[i] = fa[(lid.y + i * 16u) * FP + k]; // broadcast across a wave
        bv[i] = fb[(lid.x + i * 16u) * FP + k]; // stride-17 => conflict-free
      }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
          acc[i * 4u + j] = fma(av[i], bv[j], acc[i * 4u + j]);
        }
      }
    }

    workgroupBarrier();
  }

  // Store: consecutive lanes (lid.x) write consecutive columns.
  for (var i = 0u; i < 4u; i++) {
    let row = m0 + lid.y + i * 16u;
    if (row >= params.M) { continue; }
    for (var j = 0u; j < 4u; j++) {
      let col = n0 + lid.x + j * 16u;
      if (col < params.N) {
        C[row * params.N + col] = acc[i * 4u + j];
      }
    }
  }
}

// ── Naive matmul (for correctness testing) ──────────────────────────────

@compute @workgroup_size(256)
fn matmul_naive(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let row = idx / params.N;
  let col = idx % params.N;

  if (row >= params.M || col >= params.N) { return; }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < params.K; k++) {
    sum += A[row * params.K + k] * B[k * params.N + col];
  }
  C[row * params.N + col] = sum;
}
