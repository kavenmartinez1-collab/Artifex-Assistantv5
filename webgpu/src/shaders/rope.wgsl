// Rotary Position Embeddings (RoPE) — rotate_half (Llama/Qwen) scheme
// Applies rotary embeddings to Q and K tensors for positional encoding.
// Qwen3.5 uses RoPE with theta = 10,000,000 and partial_rotary_factor = 0.25
//
// HF convention: rotate_half pairs element i with element i + rot_dim/2
//   q_rot[i]            = q[i]        * cos(theta) - q[i+rot/2] * sin(theta)
//   q_rot[i + rot/2]    = q[i+rot/2]  * cos(theta) + q[i]       * sin(theta)
// where theta = pos / (base ^ (2i / rot_dim)), and cos/sin are shared
// between the first-half element and its second-half partner.
// NOTE: This is NOT the interleaved (2i, 2i+1) GPT-NeoX scheme.

struct Params {
  seq_len: u32,     // number of positions to encode
  head_dim: u32,    // dimension per attention head
  num_heads: u32,   // number of heads to process
  pos_offset: u32,  // position offset (for KV-cache continuation)
  rope_base: f32,   // base frequency (10000.0 default)
  rotary_dim: u32,  // how many dims to rotate (0 = all, else partial RoPE)
  // Gemma 4 proportional RoPE: pairing still spans rot_dim (pair i with
  // i + rot_dim/2) and freqs use the rot_dim divisor, but only the first
  // `rotated_pairs` pairs rotate — the rest are identity. 0 = all pairs.
  rotated_pairs: u32,
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>; // in-place rotation
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  // rotary_dim: 0 means rotate all dims, >0 means only first rotary_dim dims
  let rot_dim = select(params.head_dim, params.rotary_dim, params.rotary_dim > 0u);
  let half_rot = rot_dim / 2u;
  let active_pairs = select(half_rot, min(params.rotated_pairs, half_rot), params.rotated_pairs > 0u);
  let total_pairs = params.seq_len * params.num_heads * active_pairs;

  if (idx >= total_pairs) { return; }

  // Decompose flat index into (seq_pos, head, pair_idx)
  let pair_idx = idx % active_pairs;
  let remaining = idx / active_pairs;
  let head = remaining % params.num_heads;
  let seq_pos = remaining / params.num_heads;

  let position = f32(seq_pos + params.pos_offset);
  let freq = 1.0 / pow(params.rope_base, f32(2u * pair_idx) / f32(rot_dim));
  let angle = position * freq;

  let cos_val = cos(angle);
  let sin_val = sin(angle);

  // rotate_half pairing: element i with element i + half_rot
  let base_idx = (seq_pos * params.num_heads + head) * params.head_dim;
  let i0 = base_idx + pair_idx;
  let i1 = base_idx + pair_idx + half_rot;

  let v0 = qk[i0];
  let v1 = qk[i1];

  qk[i0] = v0 * cos_val - v1 * sin_val;
  qk[i1] = v1 * cos_val + v0 * sin_val;
}
