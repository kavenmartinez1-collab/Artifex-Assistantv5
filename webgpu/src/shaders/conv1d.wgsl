// Causal 1D Convolution for Gated DeltaNet / Mamba-2
//
// Applies a depthwise causal convolution with a short kernel (typically 4).
// For decode (seqLen=1): dot product of sliding window with conv kernel.
// The conv_state buffer holds the last (kernel_size - 1) input vectors.
//
// After convolution, the state is updated: shift left, append new input.
//
// Layout:
//   new_input:   [dim]                    (current token's projected values)
//   conv_state:  [kernel_size - 1, dim]   (sliding window of recent inputs)
//   conv_weight: [dim, kernel_size]       (depthwise conv kernel per channel)
//   output:      [dim]                    (convolved result)

struct Params {
  dim: u32,          // number of channels
  kernel_size: u32,  // convolution kernel size (e.g., 4)
}

@group(0) @binding(0) var<storage, read> new_input: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Convolve: output = conv(state ++ new_input, weight)
@compute @workgroup_size(256)
fn conv1d(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= params.dim) { return; }

  let ks = params.kernel_size;
  let state_len = ks - 1u;

  // Build window: [state[0,d], state[1,d], ..., state[ks-2,d], new_input[d]]
  // Dot product with conv_weight[d * ks + k]
  var sum: f32 = 0.0;
  for (var k = 0u; k < state_len; k = k + 1u) {
    sum += conv_state[k * params.dim + d] * conv_weight[d * ks + k];
  }
  // Last element of window is the new input
  sum += new_input[d] * conv_weight[d * ks + state_len];

  output[d] = sum;
}

// Fusion lever F4: conv1d + state update + SiLU in one dispatch.
// Replaces the conv1d → conv1d_update_state → silu 3-dispatch chain at
// decode. Each thread owns channel d end-to-end, so the read-before-write
// ordering within the thread reproduces the legacy sequence exactly:
//   sum   — identical loop/order to conv1d above
//   state — identical shift/append to conv1d_update_state below
//   silu  — expression identical to elementwise.wgsl silu (f32 store/load
//           roundtrip of `sum` in the legacy path is exact, so fusing the
//           SiLU onto the register value is bit-identical)
@compute @workgroup_size(256)
fn conv1d_silu_update(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= params.dim) { return; }

  let ks = params.kernel_size;
  let state_len = ks - 1u;

  var sum: f32 = 0.0;
  for (var k = 0u; k < state_len; k = k + 1u) {
    sum += conv_state[k * params.dim + d] * conv_weight[d * ks + k];
  }
  let x_new = new_input[d];
  sum += x_new * conv_weight[d * ks + state_len];

  // Shift state left by 1, append raw new input
  for (var k = 0u; k < state_len - 1u; k = k + 1u) {
    conv_state[k * params.dim + d] = conv_state[(k + 1u) * params.dim + d];
  }
  conv_state[(state_len - 1u) * params.dim + d] = x_new;

  output[d] = sum / (1.0 + exp(-sum));
}

// Update state: shift left by 1, write new_input at the end
@compute @workgroup_size(256)
fn conv1d_update_state(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= params.dim) { return; }

  let state_len = params.kernel_size - 1u;

  // Shift: state[k] = state[k+1] for k in [0, state_len-2]
  for (var k = 0u; k < state_len - 1u; k = k + 1u) {
    conv_state[k * params.dim + d] = conv_state[(k + 1u) * params.dim + d];
  }
  // Write new input at last position
  conv_state[(state_len - 1u) * params.dim + d] = new_input[d];
}
