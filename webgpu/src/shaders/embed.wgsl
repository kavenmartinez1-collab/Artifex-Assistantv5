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
