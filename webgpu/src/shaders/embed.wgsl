// Embedding Lookup — Gather rows from the embedding table by token ID.
//
// Each workgroup handles one token. Threads within the workgroup
// cooperatively copy the embedding vector (hidden_size elements).
//
// input:  token_ids[seq_len]           — u32 token IDs
// output: embeddings[seq_len × hidden] — f32 embedding vectors
// table:  embed_table[vocab × hidden]  — f32 embedding weights

struct Params {
  hidden_size: u32,
  seq_len: u32,
}

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> embed_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn embed(@builtin(local_invocation_id) lid: vec3u,
         @builtin(workgroup_id) wid: vec3u) {
  let token_idx = wid.x;  // which token in the sequence
  let tid = lid.x;
  let hidden = params.hidden_size;

  if (token_idx >= params.seq_len) {
    return;
  }

  let token_id = token_ids[token_idx];

  // Copy the embedding row: embed_table[token_id * hidden .. (token_id+1) * hidden]
  // to output[token_idx * hidden .. (token_idx+1) * hidden]
  var i = tid;
  while (i < hidden) {
    output[token_idx * hidden + i] = embed_table[token_id * hidden + i];
    i = i + 256u;
  }
}

// ── F16 Embedding Lookup ──────────────────────────────────────────────
// For large vocab models (Qwen3.5: 248K × 4096) where the f32 embedding
// exceeds the 2 GB WebGPU buffer limit. Reads F16 packed as u32 pairs
// and converts to f32 inline.
//
// embed_table_f16 stores 2 F16 values per u32 (little-endian).
// Element at index i is in word i/2, upper or lower 16 bits.

@group(0) @binding(0) var<storage, read> token_ids_f16: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_f16: array<f32>;
@group(0) @binding(2) var<storage, read> embed_table_f16: array<u32>;
@group(0) @binding(3) var<uniform> params_f16: Params;

// Decode IEEE 754 half-precision to f32
fn f16_to_f32(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let frac = bits & 0x3FFu;
  if (exp == 0u) {
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(frac) / 1024.0 * pow(2.0, -14.0);
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) {
    return select(1e30, -1e30, sign == 1u);
  }
  let f = (1.0 + f32(frac) / 1024.0) * pow(2.0, f32(exp) - 15.0);
  return select(f, -f, sign == 1u);
}

// Decode BF16 to f32 (just shift left 16 bits)
fn bf16_to_f32(bits: u32) -> f32 {
  return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn embed_f16(@builtin(local_invocation_id) lid: vec3u,
             @builtin(workgroup_id) wid: vec3u) {
  let token_idx = wid.x;
  let tid = lid.x;
  let hidden = params_f16.hidden_size;

  if (token_idx >= params_f16.seq_len) { return; }

  let token_id = token_ids_f16[token_idx];
  let row_start = token_id * hidden; // element index in f16 array

  var i = tid;
  while (i < hidden) {
    let elem_idx = row_start + i;
    let word_idx = elem_idx / 2u;
    let is_upper = (elem_idx & 1u) == 1u;
    let word = embed_table_f16[word_idx];
    let f16_bits = select(word & 0xFFFFu, word >> 16u, is_upper);
    output_f16[token_idx * hidden + i] = f16_to_f32(f16_bits);
    i = i + 256u;
  }
}
