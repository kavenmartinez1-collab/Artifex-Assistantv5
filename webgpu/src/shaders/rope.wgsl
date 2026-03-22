// Rotary Position Embeddings (RoPE)
// Applies rotary embeddings to Q and K tensors for positional encoding.
// Qwen3.5 uses RoPE with theta = 10000 (or extended theta for long context)
//
// For each pair of dimensions (2i, 2i+1):
//   q_rot[2i]   = q[2i] * cos(theta) - q[2i+1] * sin(theta)
//   q_rot[2i+1] = q[2i] * sin(theta) + q[2i+1] * cos(theta)
// where theta = pos / (base ^ (2i / d))

struct Params {
  seq_len: u32,     // number of positions to encode
  head_dim: u32,    // dimension per attention head
  num_heads: u32,   // number of heads to process
  pos_offset: u32,  // position offset (for KV-cache continuation)
  rope_base: f32,   // base frequency (10000.0 default)
}

@group(0) @binding(0) var<storage, read_write> qk: array<f32>; // in-place rotation
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn rope(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let half_dim = params.head_dim / 2u;
  let total_pairs = params.seq_len * params.num_heads * half_dim;

  if (idx >= total_pairs) { return; }

  // Decompose flat index into (seq_pos, head, pair_idx)
  let pair_idx = idx % half_dim;
  let remaining = idx / half_dim;
  let head = remaining % params.num_heads;
  let seq_pos = remaining / params.num_heads;

  let position = f32(seq_pos + params.pos_offset);
  let freq = 1.0 / pow(params.rope_base, f32(2u * pair_idx) / f32(params.head_dim));
  let angle = position * freq;

  let cos_val = cos(angle);
  let sin_val = sin(angle);

  // Compute buffer indices for the pair
  let base_idx = (seq_pos * params.num_heads + head) * params.head_dim;
  let i0 = base_idx + 2u * pair_idx;
  let i1 = base_idx + 2u * pair_idx + 1u;

  let v0 = qk[i0];
  let v1 = qk[i1];

  qk[i0] = v0 * cos_val - v1 * sin_val;
  qk[i1] = v0 * sin_val + v1 * cos_val;
}
