// Element-wise operations: SiLU, GELU, add, multiply

struct Params {
  n: u32,
  broadcast_b: u32, // if > 0, index input_b as input_b[idx % broadcast_b]
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

// SiLU (Swish): x * sigmoid(x) — used in Qwen3.5 SwiGLU FFN
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let x = input_a[idx];
  output[idx] = x / (1.0 + exp(-x));
}

// GELU (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let x = input_a[idx];
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  output[idx] = 0.5 * x * (1.0 + tanh(inner));
}

// ── Two-input operations ────────────────────────────────────────────────

@group(0) @binding(3) var<storage, read> input_b: array<f32>;

// Element-wise add: output = a + b (with optional broadcast on b)
// When broadcast_b > 0, input_b is indexed as input_b[idx % broadcast_b]
// This lets a bias vector [dim] be added to a batched tensor [seq * dim]
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let b_idx = select(idx, idx % params.broadcast_b, params.broadcast_b > 0u);
  output[idx] = input_a[idx] + input_b[b_idx];
}

// Element-wise multiply: output = a * b
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  output[idx] = input_a[idx] * input_b[idx];
}

// Fused multiply-add: output = a * b + output (accumulate)
@compute @workgroup_size(256)
fn fma(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  output[idx] = input_a[idx] * input_b[idx] + output[idx];
}

// ── Gated DeltaNet / Mamba-2 operations ──────────────────────────────────

// Gated SiLU: output = a * silu(b) — used for Mamba-2 output gating
@compute @workgroup_size(256)
fn gate_silu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let z = input_b[idx];
  let silu_z = z / (1.0 + exp(-z));
  output[idx] = input_a[idx] * silu_z;
}

// Softplus: output = log(1 + exp(a + b)) — used for SSM time delta
@compute @workgroup_size(256)
fn softplus(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let x = input_a[idx] + input_b[idx];
  // Numerically stable softplus: for large x, softplus(x) ≈ x
  output[idx] = select(log(1.0 + exp(x)), x, x > 20.0);
}

// Sigmoid: output = 1 / (1 + exp(-a)) — used for SSM beta gate
@compute @workgroup_size(256)
fn sigmoid_op(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  output[idx] = 1.0 / (1.0 + exp(-input_a[idx]));
}

// Decay: output = exp(-exp(a) * b) where a=A_log, b=dt
// Computes the SSM state decay factor. Always in (0, 1).
@compute @workgroup_size(256)
fn decay_compute(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let A = -exp(input_a[idx]);  // A_log → negative A
  let dt = input_b[idx];       // dt from softplus
  output[idx] = exp(A * dt);   // decay in (0, 1)
}

// L2 Normalize: normalize each head vector to unit length
// params.n = total elements, params.broadcast_b = head_dim (elements per head)
// One thread per element. Computes norm across the head, then divides.
@compute @workgroup_size(256)
fn l2_normalize(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let head_dim = params.broadcast_b; // reuse broadcast_b as head_dim
  let head = idx / head_dim;
  let base = head * head_dim;

  // Compute L2 norm of this head's vector
  var sum_sq: f32 = 0.0;
  for (var i = 0u; i < head_dim; i = i + 1u) {
    let v = input_a[base + i];
    sum_sq += v * v;
  }
  let inv_norm = 1.0 / max(sqrt(sum_sq), 1e-6);
  output[idx] = input_a[idx] * inv_norm;
}
