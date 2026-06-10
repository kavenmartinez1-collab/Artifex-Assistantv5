// Element-wise operations: SiLU, GELU, add, multiply

struct Params {
  n: u32,
  broadcast_b: u32, // if > 0, index input_b as input_b[idx % broadcast_b]
  scale: f32,       // softcap cap / addscale factor (uniform buffers are zero-filled by default)
  // Strided slice view of input_a (gate_gelu only; 0 = contiguous). Used by
  // the Gemma 4 PLE block to read layer l's [tokens, 256] slice out of the
  // token-major [tokens, n_layer, 256] combined per-layer input:
  //   a_idx = (idx / a_slice_len) * a_stride + a_off + (idx % a_slice_len)
  a_slice_len: u32,
  a_stride: u32,
  a_off: u32,
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

// Softcap (Gemma): output = c * tanh(x / c) — bounds logits to (-c, c)
@compute @workgroup_size(256)
fn softcap(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let c = params.scale;
  output[idx] = c * tanh(input_a[idx] / c);
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

// Element-wise multiply: output = a * b (with optional broadcast on b,
// e.g. a [1]-element layer_output_scale tensor scaling the whole stream)
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let b_idx = select(idx, idx % params.broadcast_b, params.broadcast_b > 0u);
  output[idx] = input_a[idx] * input_b[b_idx];
}

// Scaled add: output = (a + b) * scale — Gemma 4 PLE combine
// (per_layer_proj + inp_per_layer) * 1/sqrt(2)
@compute @workgroup_size(256)
fn addscale(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  let b_idx = select(idx, idx % params.broadcast_b, params.broadcast_b > 0u);
  output[idx] = (input_a[idx] + input_b[b_idx]) * params.scale;
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

// Gated GELU (tanh approx): output = a * gelu(b) — Gemma GeGLU FFN
// (a = up projection, b = gate projection; mirrors gate_silu's convention)
// Supports the strided a-slice view (a_slice_len > 0) for the PLE block.
@compute @workgroup_size(256)
fn gate_gelu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  if (idx >= params.n) { return; }
  var a_idx = idx;
  if (params.a_slice_len > 0u) {
    a_idx = (idx / params.a_slice_len) * params.a_stride + params.a_off + (idx % params.a_slice_len);
  }
  let x = input_b[idx];
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  output[idx] = input_a[a_idx] * 0.5 * x * (1.0 + tanh(inner));
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

// L2 Normalize: moved to separate shader (l2norm.wgsl) to avoid binding conflicts
