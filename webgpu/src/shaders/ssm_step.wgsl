// SSM Step — Gated DeltaNet Single-Token Recurrence
//
// The core Mamba-2 / Gated DeltaNet state update for decode (seqLen=1).
//
// State shape: h[num_key_heads, key_head_dim, grouped_value_dim]
//   - grouped_value_dim = (num_value_heads / num_key_heads) * value_head_dim
//   - For Qwen3.5-9B: h[16, 128, 256] where 256 = 2 * 128
//
// Update rule (per key head kh, per key dim k, per grouped value dim v):
//   h[kh, k, v] = decay[kh, k] * h[kh, k, v] + beta[kh] * K[kh, k] * V_grouped[kh, v]
//
// Readout:
//   output[kh, v] = sum_k Q[kh, k] * h[kh, k, v]
//
// V grouping: V is [num_value_heads, value_head_dim] = [32, 128]
//   Reshaped to [num_key_heads, grouped_value_dim] = [16, 256]
//
// Dispatch: one workgroup per key_head. Each thread handles a slice of the v dimension.

struct Params {
  num_key_heads: u32,      // 16
  num_value_heads: u32,    // 32 (for reference, not directly used in kernel)
  key_head_dim: u32,       // 128
  grouped_value_dim: u32,  // 256 = (num_value_heads / num_key_heads) * value_head_dim
}

// Per-head vectors for the current token
@group(0) @binding(0) var<storage, read> Q: array<f32>;       // [num_key_heads * key_head_dim]
@group(0) @binding(1) var<storage, read> K: array<f32>;       // [num_key_heads * key_head_dim] (after conv1d + silu)
@group(0) @binding(2) var<storage, read> V: array<f32>;       // [num_value_heads * value_head_dim] = [grouped as num_key_heads * grouped_value_dim]
@group(0) @binding(3) var<storage, read> beta: array<f32>;    // [num_key_heads] (after sigmoid)
@group(0) @binding(4) var<storage, read> decay: array<f32>;   // [num_key_heads * key_head_dim]

// SSM hidden state (read-write, persists across tokens)
@group(1) @binding(0) var<storage, read_write> h: array<f32>; // [num_key_heads * key_head_dim * grouped_value_dim]

// Output
@group(1) @binding(1) var<storage, read_write> output: array<f32>; // [num_key_heads * grouped_value_dim] = [num_value_heads * value_head_dim]

@group(2) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(256)
fn ssm_step(@builtin(local_invocation_id) lid: vec3u,
            @builtin(workgroup_id) wid: vec3u) {
  let kh = wid.x;        // which key head (0..num_key_heads-1)
  let tid = lid.x;

  let kd = params.key_head_dim;
  let gvd = params.grouped_value_dim;
  let nkh = params.num_key_heads;

  // h layout: h[kh * kd * gvd + k * gvd + v]
  let h_base = kh * kd * gvd;
  let beta_val = beta[kh];

  // Step 1: Update h[kh, :, :] — decay and add outer product
  // For each v in the grouped value dimension (threads split across v):
  var v = tid;
  while (v < gvd) {
    // V is stored as [num_value_heads * value_head_dim] = [num_key_heads * grouped_value_dim]
    let v_val = V[kh * gvd + v];

    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * gvd + v;
      let d = decay[kh * kd + k];
      let k_val = K[kh * kd + k];

      // State update: h = decay * h + beta * K * V
      h[h_idx] = d * h[h_idx] + beta_val * k_val * v_val;
    }

    v = v + 256u;
  }
  workgroupBarrier();

  // Step 2: Readout — output[kh, v] = sum_k Q[kh, k] * h[kh, k, v]
  v = tid;
  while (v < gvd) {
    var dot: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      dot += Q[kh * kd + k] * h[h_base + k * gvd + v];
    }
    output[kh * gvd + v] = dot;

    v = v + 256u;
  }
}
