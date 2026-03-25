// SSM Step — Gated DeltaNet Single-Token Recurrence
//
// The core Mamba-2 / Gated DeltaNet state update for decode (seqLen=1).
//
// State shape: h[num_key_heads, key_head_dim, value_head_dim]
//   - h is a 3D tensor: each key head maintains a [k_dim, v_dim] matrix
//   - This matrix accumulates outer products of K and V vectors
//
// Update rule:
//   For each key head kh:
//     h[kh, :, :] = diag(decay[kh, :]) @ h[kh, :, :] + beta[kh] * K[kh, :] ⊗ V[:, :]
//   where decay = exp(A * dt), A = -exp(A_log), dt = softplus(alpha + dt_bias)
//
// Readout:
//   output[vh, vd] = sum_kh_in_group sum_kd Q[kh, kd] * h[kh, kd, vd]
//   (grouped: multiple value heads per key head group)
//
// Dispatch: one workgroup per (key_head, v_dim_tile).
// Each workgroup iterates over key_head_dim to update h and compute readout.

struct Params {
  num_key_heads: u32,    // 16
  num_value_heads: u32,  // 32
  key_head_dim: u32,     // 128
  value_head_dim: u32,   // 128
}

// Per-head vectors for the current token
@group(0) @binding(0) var<storage, read> Q: array<f32>;       // [num_key_heads * key_head_dim]
@group(0) @binding(1) var<storage, read> K: array<f32>;       // [num_key_heads * key_head_dim]
@group(0) @binding(2) var<storage, read> V: array<f32>;       // [num_value_heads * value_head_dim]
@group(0) @binding(3) var<storage, read> beta: array<f32>;    // [num_key_heads] update gate scalar
@group(0) @binding(4) var<storage, read> decay: array<f32>;   // [num_key_heads * key_head_dim]

// SSM hidden state (read-write, persists across tokens)
@group(1) @binding(0) var<storage, read_write> h: array<f32>; // [num_key_heads, key_head_dim, value_head_dim]

// Output
@group(1) @binding(1) var<storage, read_write> output: array<f32>; // [num_value_heads * value_head_dim]

@group(2) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(256)
fn ssm_step(@builtin(local_invocation_id) lid: vec3u,
            @builtin(workgroup_id) wid: vec3u) {
  let kh = wid.x;        // which key head (0..num_key_heads-1)
  let v_tile = wid.y;    // which value dim tile
  let tid = lid.x;

  let kd = params.key_head_dim;
  let vd = params.value_head_dim;
  let nkh = params.num_key_heads;
  let nvh = params.num_value_heads;

  // GQA-like grouping: value heads grouped under key heads
  let vh_per_kh = nvh / nkh;
  let vh_start = kh * vh_per_kh;

  // h layout: h[kh * kd * vd + k * vd + v]
  let h_base = kh * kd * vd;

  // Process a slice of value dimensions per workgroup
  // Each thread handles one or more (k, v) pairs

  // Step 1: Update h[kh, :, :] — decay and add outer product
  // For each k_dim, for each v_dim assigned to this thread:
  //   h[kh, k, v] = decay[kh, k] * h[kh, k, v] + beta[kh] * K[kh, k] * V_mapped[v]
  let beta_val = beta[kh];

  var v_idx = tid;
  while (v_idx < vd) {
    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * vd + v_idx;
      let d = decay[kh * kd + k];

      // Map v_idx to the correct value head
      // V layout: [num_value_heads * value_head_dim]
      // For the key head's group: V[(vh_start * vd)..((vh_start + vh_per_kh) * vd)]
      // Since we iterate v_idx over [0, vd), and vh_per_kh value heads share this key head,
      // we accumulate across value heads within the group
      let k_val = K[kh * kd + k];

      // Update: decay * old_h + beta * K * V
      // V is indexed per value head group
      let v_val = V[vh_start * vd + v_idx];
      h[h_idx] = d * h[h_idx] + beta_val * k_val * v_val;
    }
    v_idx = v_idx + 256u;
  }
  workgroupBarrier();

  // Step 2: Readout — output[v] = sum_k Q[kh, k] * h[kh, k, v]
  // Only the first value head in the group writes output
  v_idx = tid;
  while (v_idx < vd) {
    var dot: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      dot += Q[kh * kd + k] * h[h_base + k * vd + v_idx];
    }
    // Atomic add since multiple key heads in a group contribute to the same output
    // For now, use simple write (assuming 1:1 key-to-value head mapping or single dispatch)
    output[vh_start * vd + v_idx] = dot;

    v_idx = v_idx + 256u;
  }
}
