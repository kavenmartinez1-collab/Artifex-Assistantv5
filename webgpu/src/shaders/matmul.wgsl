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
  let num_tiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t++) {
    // Load tile of A into shared memory
    let a_col = t * TILE + lid.y;
    if (row < params.M && a_col < params.K) {
      tile_a[local_idx] = A[row * params.K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    // Load tile of B into shared memory
    let b_row = t * TILE + lid.x;
    if (b_row < params.K && col < params.N) {
      tile_b[local_idx] = B[b_row * params.N + col];
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product for this tile
    for (var k: u32 = 0u; k < TILE; k++) {
      sum += tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
    }

    workgroupBarrier();
  }

  // Write result
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
  let num_tiles = (params.K + TILE - 1u) / TILE;

  for (var t: u32 = 0u; t < num_tiles; t++) {
    let a_col = t * TILE + lid.y;
    if (row < params.M && a_col < params.K) {
      tile_a[local_idx] = A[row * params.K + a_col];
    } else {
      tile_a[local_idx] = 0.0;
    }

    // B^T[k, n] = B[n, k] = B[n * K + k]  (B stored as [N, K])
    let b_k = t * TILE + lid.x;
    if (b_k < params.K && col < params.N) {
      tile_b[local_idx] = B[col * params.K + b_k];
    } else {
      tile_b[local_idx] = 0.0;
    }

    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k++) {
      sum += tile_a[lid.x * TILE + k] * tile_b[k * TILE + lid.y];
    }

    workgroupBarrier();
  }

  if (row < params.M && col < params.N) {
    C[row * params.N + col] = sum;
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
