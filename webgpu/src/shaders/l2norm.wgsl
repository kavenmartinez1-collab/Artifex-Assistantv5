// L2 Normalize — normalize each head vector to unit length
// Used for Q and K in Gated DeltaNet (use_qk_l2norm_in_kernel=True)

struct Params {
  n: u32,         // total elements
  head_dim: u32,  // elements per head (for per-head normalization)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn l2_normalize(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }

  let hd = params.head_dim;
  let head = idx / hd;
  let base = head * hd;

  // Compute L2 norm of this head's vector
  var sum_sq: f32 = 0.0;
  for (var i = 0u; i < hd; i = i + 1u) {
    let v = input[base + i];
    sum_sq += v * v;
  }
  let inv_norm = 1.0 / max(sqrt(sum_sq), 1e-6);
  output[idx] = input[idx] * inv_norm;
}
