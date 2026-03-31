// Embedding Lookup — Gather rows from the embedding table by token ID.
//
// Three variants:
//   embed       — f32 embedding table
//   embed_f16   — BF16/F16 packed embedding table
//   embed_q4    — GPTQ INT4 packed embedding (qweight + scales + qzeros)
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

// Decode IEEE 754 half-precision to f32 (exact bitwise construction)
fn f16_to_f32(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let frac = bits & 0x3FFu;
  if (exp == 0u) {
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    let f = f32(frac) * bitcast<f32>(0x33800000u); // 2^-24
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) {
    return select(1e30, -1e30, sign == 1u);
  }
  let f32_bits = (sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u);
  return bitcast<f32>(f32_bits);
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
    output_f16[token_idx * hidden + i] = bf16_to_f32(f16_bits);
    i = i + 256u;
  }
}

// ── GPTQ INT4 Embedding Lookup ──────────────────────────────────────
// Embedding stored as GPTQ: qweight[hidden/8, vocab] I32 (8 nibbles per i32),
// scales[hidden/group_size, vocab] F16, qzeros[hidden/group_size, vocab/8] I32.
// Dequant: val = (nibble - zero) * scale
//
// GPTQ packs weights column-major: qweight[k/8, n] has 8 k-values for column n.
// For embedding lookup, n=token_id, k=hidden dimension index.

struct Q4Params {
  hidden_size: u32,
  seq_len: u32,
  group_size: u32,
  vocab_size: u32,
}

@group(0) @binding(0) var<storage, read> token_ids_q4: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_q4: array<f32>;
@group(0) @binding(2) var<storage, read> qweight: array<i32>;    // [hidden/8, vocab]
@group(0) @binding(3) var<uniform> params_q4: Q4Params;
@group(0) @binding(4) var<storage, read> scales: array<u32>;     // F16 as u16 pairs in u32
@group(0) @binding(5) var<storage, read> qzeros: array<i32>;     // [hidden/gs, vocab/8]

fn f16_to_f32_embed(bits: u32) -> f32 {
  let sign = (bits >> 15u) & 1u;
  let exp = (bits >> 10u) & 0x1Fu;
  let frac = bits & 0x3FFu;
  if (exp == 0u) {
    if (frac == 0u) { return select(0.0, -0.0, sign == 1u); }
    // Subnormal f16: value = (-1)^sign * frac/1024 * 2^-14
    let f = f32(frac) * bitcast<f32>(0x33800000u); // 2^-24
    return select(f, -f, sign == 1u);
  }
  if (exp == 31u) { return select(1e30, -1e30, sign == 1u); }
  // Normal f16 -> f32: exact bitwise construction
  let f32_bits = (sign << 31u) | ((exp + 112u) << 23u) | (frac << 13u);
  return bitcast<f32>(f32_bits);
}

@compute @workgroup_size(256)
fn embed_q4(@builtin(local_invocation_id) lid: vec3u,
            @builtin(workgroup_id) wid: vec3u) {
  let token_idx = wid.x;
  let tid = lid.x;
  let hidden = params_q4.hidden_size;
  let gs = params_q4.group_size;
  let V = params_q4.vocab_size;

  if (token_idx >= params_q4.seq_len) { return; }

  let n = token_ids_q4[token_idx]; // token_id = column index in qweight

  var k = tid;
  while (k < hidden) {
    // Extract 4-bit weight from qweight[k/8, n]
    let packed_row = k / 8u;
    let nibble_idx = k % 8u;
    let packed = qweight[packed_row * V + n];
    let q4 = (u32(packed) >> (nibble_idx * 4u)) & 0xFu;

    // Get scale: scales[group, n] stored as F16
    let group = k / gs;
    let scale_flat = group * V + n;
    let scale_word = scales[scale_flat / 2u];
    let scale_bits = select(scale_word & 0xFFFFu, scale_word >> 16u, (scale_flat & 1u) == 1u);
    let scale = f16_to_f32_embed(scale_bits);

    // Get zero: qzeros[group, n/8] packed nibbles
    let zero_packed = qzeros[group * ((V + 7u) / 8u) + n / 8u];
    let zero_nibble = n % 8u;
    let zero = (u32(zero_packed) >> (zero_nibble * 4u)) & 0xFu;

    // Dequant
    output_q4[token_idx * hidden + k] = (f32(q4) - f32(zero)) * scale;

    k = k + 256u;
  }
}
