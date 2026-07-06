// Streaming non-causal multi-head attention (online softmax).
//
// Built for FLUX.2 shapes that exceed attention.wgsl's fixed 2048-position
// scores array:
//   - DiT joint attention: up to 4608 positions, 24 heads x 128 dim
//   - VAE mid-block attention: up to 16384 positions, 1 head x 512 dim
//
// One workgroup per (query index, head). KV is streamed in WG-sized chunks
// with flash-attention style running max/sum rescaling, so memory use is
// independent of seq_kv.
//
// Layouts (row-major f32):
//   Q:   [seq_q,  num_heads * head_dim]
//   KT:  [num_heads, head_dim, seq_kv]   <-- TRANSPOSED (see below)
//   V:   [seq_kv, num_heads * head_dim]
//   Out: [seq_q,  num_heads * head_dim]
//
// K must be pre-transposed to [H, D, S]. In the score loop each of the 128
// threads owns one kv position j; with row-major K the threads would read
// addresses a full row apart (128-way memory divergence, measured 0.03
// TFLOPS on the 6700 XT). With KT, thread t reads KT[(h*D+d)*S + j] and
// adjacent threads hit adjacent addresses (coalesced). V needs no transpose:
// its accumulation is parallel over d, which is already coalesced.
//
// q_offset allows slicing the query range across submits (TDR budgeting):
// dispatch(gx = slice_len, gy = num_heads) with q_offset = slice start.

struct Params {
  num_heads: u32,
  head_dim: u32,   // <= D_MAX (512)
  seq_q: u32,
  seq_kv: u32,
  q_offset: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> KT: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const WG: u32 = 128u;
const D_MAX: u32 = 512u;

var<workgroup> q_sh: array<f32, D_MAX>;
var<workgroup> acc_sh: array<f32, D_MAX>;
var<workgroup> probs: array<f32, WG>;
var<workgroup> red: array<f32, WG>;
var<workgroup> m_run: f32;   // running max
var<workgroup> l_run: f32;   // running sum

@compute @workgroup_size(128)
fn attention_stream(@builtin(workgroup_id) wid: vec3u,
                    @builtin(local_invocation_id) lid: vec3u) {
  let qi = params.q_offset + wid.x;
  let h = wid.y;
  let t = lid.x;
  let D = params.head_dim;
  let H = params.num_heads;
  let scale = 1.0 / sqrt(f32(D));

  if (qi >= params.seq_q) { return; }

  let q_base = (qi * H + h) * D;
  for (var d = t; d < D; d += WG) {
    q_sh[d] = Q[q_base + d];
    acc_sh[d] = 0.0;
  }
  if (t == 0u) {
    m_run = -1e30;
    l_run = 0.0;
  }
  workgroupBarrier();

  let n_chunks = (params.seq_kv + WG - 1u) / WG;
  for (var c = 0u; c < n_chunks; c++) {
    let j = c * WG + t;

    // 1) score for this thread's kv position
    var s = -1e30;
    if (j < params.seq_kv) {
      var dot = 0.0;
      let kt_base = h * D * params.seq_kv + j;
      for (var d = 0u; d < D; d++) {
        dot += q_sh[d] * KT[kt_base + d * params.seq_kv];
      }
      s = dot * scale;
    }

    // 2) chunk max (workgroup reduction)
    red[t] = s;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
      if (t < stride) { red[t] = max(red[t], red[t + stride]); }
      workgroupBarrier();
    }
    let m_new = max(m_run, red[0]);

    // 3) exp probs + chunk sum
    var e = 0.0;
    if (j < params.seq_kv) { e = exp(s - m_new); }
    probs[t] = e;
    red[t] = e;
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride /= 2u) {
      if (t < stride) { red[t] += red[t + stride]; }
      workgroupBarrier();
    }
    let chunk_sum = red[0];
    let corr = exp(m_run - m_new);

    // 4) rescale accumulator and add this chunk's weighted V
    //    (threads own disjoint d indices)
    let lim = min(WG, params.seq_kv - c * WG);
    for (var d = t; d < D; d += WG) {
      var a = acc_sh[d] * corr;
      for (var jj = 0u; jj < lim; jj++) {
        a += probs[jj] * V[((c * WG + jj) * H + h) * D + d];
      }
      acc_sh[d] = a;
    }

    // 5) update running stats (barrier first: corr/m_new read m_run above)
    workgroupBarrier();
    if (t == 0u) {
      l_run = l_run * corr + chunk_sum;
      m_run = m_new;
    }
    workgroupBarrier();
  }

  let inv_l = 1.0 / l_run;
  for (var d = t; d < D; d += WG) {
    Out[q_base + d] = acc_sh[d] * inv_l;
  }
}
