// SSM Step — Gated Delta Rule Single-Token Recurrence
//
// The DELTA RULE recurrence (not simple outer product!):
//   1. h = decay * h                              (decay state)
//   2. kv_mem = (h * K).sum(key_dim)               (predict V from state)
//   3. delta = (V - kv_mem) * beta                 (compute prediction error)
//   4. h = h + outer(K, delta)                     (update state with error)
//   5. output = (h * Q).sum(key_dim)               (readout)
//
// Reference: modeling_qwen3_5.py lines 426-437
//
// State shape: h[num_key_heads, key_head_dim, grouped_value_dim]
// Dispatch: one workgroup per key_head

struct Params {
  num_key_heads: u32,      // 16
  num_value_heads: u32,    // 32
  key_head_dim: u32,       // 128
  grouped_value_dim: u32,  // 256 = (num_value_heads / num_key_heads) * value_head_dim
  // V-head ordering of the V/beta/decay/output buffers:
  //   0 = grouped (HF safetensors): v-heads of k-head g are [g*r .. g*r+r-1]
  //   1 = tiled (GGUF, llama.cpp convert reorders): v-head (g, j) sits at j*nkh + g
  layout_tiled: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

// Per-head vectors for the current token (Q and K are L2-normalized and Q is scaled by 1/sqrt(kd))
@group(0) @binding(0) var<storage, read> Q: array<f32>;       // [num_key_heads * key_head_dim]
@group(0) @binding(1) var<storage, read> K: array<f32>;       // [num_key_heads * key_head_dim]
@group(0) @binding(2) var<storage, read> V: array<f32>;       // [num_value_heads * value_head_dim]
@group(0) @binding(3) var<storage, read> beta: array<f32>;    // [num_value_heads] (after sigmoid)
@group(0) @binding(4) var<storage, read> decay: array<f32>;   // [num_value_heads] (exp(g))

// SSM hidden state (read-write, persists across tokens)
@group(1) @binding(0) var<storage, read_write> h: array<f32>; // [num_key_heads * key_head_dim * grouped_value_dim]

// Output
@group(1) @binding(1) var<storage, read_write> output: array<f32>; // [num_key_heads * grouped_value_dim]

@group(2) @binding(0) var<uniform> params: Params;

// ── Fusion lever F5: beta/decay prologue ─────────────────────────────────
// ssm_step_fused replaces the sigmoid_op → softplus → decay_compute
// elementwise dispatches (3 per layer, [nvh]-sized) with a prologue: bindings
// 3/4 carry the RAW in_proj B/A projections instead of beta/decay, and the
// first nvh threads of every workgroup compute
//   beta  = sigmoid(B)                  (sigmoid_op, elementwise.wgsl)
//   dt    = softplus(A + dt_bias)       (softplus, incl. the x>20 branch)
//   decay = exp(-exp(A_log) * dt)       (decay_compute)
// into workgroup arrays. Expressions are verbatim from elementwise.wgsl; the
// legacy path's f32 buffer store/load roundtrips are exact, so the fused
// register-chained values are bit-identical. Each of the nkh workgroups
// redundantly computes all nvh values (nvh ≤ 64, negligible).

const MAX_NVH: u32 = 64u;
var<workgroup> wg_beta: array<f32, MAX_NVH>;
var<workgroup> wg_decay: array<f32, MAX_NVH>;

// F5 extra inputs (only referenced by ssm_step_fused)
@group(0) @binding(5) var<storage, read> dt_bias: array<f32>; // [num_value_heads]
@group(0) @binding(6) var<storage, read> a_log: array<f32>;   // [num_value_heads]

@compute @workgroup_size(256)
fn ssm_step(@builtin(local_invocation_id) lid: vec3u,
            @builtin(workgroup_id) wid: vec3u) {
  let kh = wid.x;        // which key head (0..num_key_heads-1)
  let tid = lid.x;

  let kd = params.key_head_dim;
  let gvd = params.grouped_value_dim;
  let nkh = params.num_key_heads;
  let nvh = params.num_value_heads;
  let vh_per_kh = nvh / nkh;
  let vhd = gvd / vh_per_kh; // value_head_dim (e.g., 128)

  let h_base = kh * kd * gvd;

  let tiled = params.layout_tiled == 1u;

  // Step 1: Decay state — h = decay * h
  // decay is per value head
  var v = tid;
  while (v < gvd) {
    let vh = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    let d = decay[vh];
    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * gvd + v;
      h[h_idx] = d * h[h_idx];
    }
    v = v + 256u;
  }
  workgroupBarrier();

  // Step 2: Compute kv_mem = (h * K).sum(key_dim) — predict V from state
  // kv_mem[v] = sum_k h[kh, k, v] * K[kh, k]
  v = tid;
  while (v < gvd) {
    var kv_mem: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      kv_mem += h[h_base + k * gvd + v] * K[kh * kd + k];
    }

    // Step 3: delta = (V - kv_mem) * beta
    let vh = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    let v_val = V[select(kh * gvd + v, vh * vhd + (v % vhd), tiled)];
    let beta_val = beta[vh];
    let delta = (v_val - kv_mem) * beta_val;

    // Step 4: h = h + outer(K, delta) — update state with delta
    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * gvd + v;
      h[h_idx] = h[h_idx] + K[kh * kd + k] * delta;
    }

    v = v + 256u;
  }
  workgroupBarrier();

  // Step 5: Readout — output[v] = (h * Q).sum(key_dim) * scale
  // scale = 1/sqrt(key_head_dim) — applied to Q as in the reference
  let scale = 1.0 / sqrt(f32(kd));
  v = tid;
  while (v < gvd) {
    var dot: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      dot += h[h_base + k * gvd + v] * Q[kh * kd + k];
    }
    // Tiled: write where the (tiled-ordered) z-gate and out_proj expect it.
    let vh_out = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    output[select(kh * gvd + v, vh_out * vhd + (v % vhd), tiled)] = dot * scale;
    v = v + 256u;
  }
}

// F5: identical recurrence to ssm_step, but binding 3 = RAW in_proj B and
// binding 4 = RAW in_proj A; beta/decay are computed in the prologue (see
// comment above wg_beta) and read from workgroup memory.
@compute @workgroup_size(256)
fn ssm_step_fused(@builtin(local_invocation_id) lid: vec3u,
                  @builtin(workgroup_id) wid: vec3u) {
  let kh = wid.x;        // which key head (0..num_key_heads-1)
  let tid = lid.x;

  let kd = params.key_head_dim;
  let gvd = params.grouped_value_dim;
  let nkh = params.num_key_heads;
  let nvh = params.num_value_heads;
  let vh_per_kh = nvh / nkh;
  let vhd = gvd / vh_per_kh; // value_head_dim (e.g., 128)

  let h_base = kh * kd * gvd;

  let tiled = params.layout_tiled == 1u;

  // Prologue: beta/dt/decay for all nvh value heads (expressions verbatim
  // from elementwise.wgsl sigmoid_op / softplus / decay_compute).
  if (tid < nvh) {
    wg_beta[tid] = 1.0 / (1.0 + exp(-beta[tid]));        // beta binding = raw B
    let x = decay[tid] + dt_bias[tid];                    // decay binding = raw A
    let dt = select(log(1.0 + exp(x)), x, x > 20.0);
    let A = -exp(a_log[tid]);
    wg_decay[tid] = exp(A * dt);
  }
  workgroupBarrier();

  // Step 1: Decay state — h = decay * h
  // decay is per value head
  var v = tid;
  while (v < gvd) {
    let vh = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    let d = wg_decay[vh];
    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * gvd + v;
      h[h_idx] = d * h[h_idx];
    }
    v = v + 256u;
  }
  workgroupBarrier();

  // Step 2: Compute kv_mem = (h * K).sum(key_dim) — predict V from state
  // kv_mem[v] = sum_k h[kh, k, v] * K[kh, k]
  v = tid;
  while (v < gvd) {
    var kv_mem: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      kv_mem += h[h_base + k * gvd + v] * K[kh * kd + k];
    }

    // Step 3: delta = (V - kv_mem) * beta
    let vh = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    let v_val = V[select(kh * gvd + v, vh * vhd + (v % vhd), tiled)];
    let beta_val = wg_beta[vh];
    let delta = (v_val - kv_mem) * beta_val;

    // Step 4: h = h + outer(K, delta) — update state with delta
    for (var k = 0u; k < kd; k = k + 1u) {
      let h_idx = h_base + k * gvd + v;
      h[h_idx] = h[h_idx] + K[kh * kd + k] * delta;
    }

    v = v + 256u;
  }
  workgroupBarrier();

  // Step 5: Readout — output[v] = (h * Q).sum(key_dim) * scale
  // scale = 1/sqrt(key_head_dim) — applied to Q as in the reference
  let scale = 1.0 / sqrt(f32(kd));
  v = tid;
  while (v < gvd) {
    var dot: f32 = 0.0;
    for (var k = 0u; k < kd; k = k + 1u) {
      dot += h[h_base + k * gvd + v] * Q[kh * kd + k];
    }
    // Tiled: write where the (tiled-ordered) z-gate and out_proj expect it.
    let vh_out = select(kh * vh_per_kh + v / vhd, (v / vhd) * nkh + kh, tiled);
    output[select(kh * gvd + v, vh_out * vhd + (v % vhd), tiled)] = dot * scale;
    v = v + 256u;
  }
}
