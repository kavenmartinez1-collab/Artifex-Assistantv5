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

// ── Q-tiled streaming attention (DiT joint attention, head_dim <= 128) ──
// attention_stream is bandwidth-bound: every (query, head) workgroup
// re-streams the full per-head K and V (520 GB per 1024px DiT block,
// 1.16 s measured). This variant amortizes: one workgroup handles QT=8
// queries, each K/V element is loaded from global memory once per tile
// and reused across all 8 dot products in registers -> traffic / 8.
// Same bindings/params; dispatch gx = ceil(q_slice / QT), gy = num_heads.
// Softmax bookkeeping is done serially by threads 0..QT-1 over the chunk
// scores in LDS (2 barrier-phases/chunk instead of 8 tree reductions).

const QT: u32 = 8u;      // queries per workgroup
const DQ: u32 = 128u;    // max head_dim for this path (DiT: 128)

var<workgroup> qt_q: array<f32, 1024>;      // [QT][DQ] query tile
var<workgroup> qt_acc: array<f32, 1024>;    // [QT][DQ] output accumulators
var<workgroup> qt_probs: array<f32, 1024>;  // [QT][WG] scores -> probs
var<workgroup> qt_m: array<f32, QT>;        // running max
var<workgroup> qt_l: array<f32, QT>;        // running sum
var<workgroup> qt_mnew: array<f32, QT>;
var<workgroup> qt_corr: array<f32, QT>;

@compute @workgroup_size(128)
fn attention_stream_qt(@builtin(workgroup_id) wid: vec3u,
                       @builtin(local_invocation_id) lid: vec3u) {
  let q0 = params.q_offset + wid.x * QT;
  let h = wid.y;
  let t = lid.x;
  let D = params.head_dim; // must be <= DQ
  let H = params.num_heads;
  let scale = 1.0 / sqrt(f32(D));

  if (q0 >= params.seq_q) { return; }

  for (var qq = 0u; qq < QT; qq++) {
    let qi = q0 + qq;
    for (var d = t; d < D; d += WG) {
      qt_q[qq * DQ + d] = select(0.0, Q[(qi * H + h) * D + d], qi < params.seq_q);
      qt_acc[qq * DQ + d] = 0.0;
    }
  }
  if (t < QT) {
    qt_m[t] = -1e30;
    qt_l[t] = 0.0;
  }
  workgroupBarrier();

  let n_chunks = (params.seq_kv + WG - 1u) / WG;
  for (var c = 0u; c < n_chunks; c++) {
    let j = c * WG + t;
    let valid = j < params.seq_kv;

    // 1) this thread's kv position scored against all QT queries; each K
    //    element is read once and reused across the register dot array.
    var dots: array<f32, QT>;
    for (var qq = 0u; qq < QT; qq++) { dots[qq] = -1e30; }
    if (valid) {
      for (var qq = 0u; qq < QT; qq++) { dots[qq] = 0.0; }
      let kt_base = h * D * params.seq_kv + j;
      for (var d = 0u; d < D; d++) {
        let kd = KT[kt_base + d * params.seq_kv];
        for (var qq = 0u; qq < QT; qq++) {
          dots[qq] = fma(qt_q[qq * DQ + d], kd, dots[qq]);
        }
      }
      for (var qq = 0u; qq < QT; qq++) { dots[qq] *= scale; }
    }
    for (var qq = 0u; qq < QT; qq++) { qt_probs[qq * WG + t] = dots[qq]; }
    workgroupBarrier();

    // 2) per-query chunk max (serial scan by threads 0..QT-1)
    if (t < QT) {
      var m = qt_m[t];
      for (var jj = 0u; jj < WG; jj++) { m = max(m, qt_probs[t * WG + jj]); }
      qt_mnew[t] = m;
    }
    workgroupBarrier();

    // 3) exp probs (each thread still holds its raw scores in registers)
    for (var qq = 0u; qq < QT; qq++) {
      var e = 0.0;
      if (valid) { e = exp(dots[qq] - qt_mnew[qq]); }
      qt_probs[qq * WG + t] = e;
    }
    workgroupBarrier();

    // 4) per-query chunk sum + running-stat update
    if (t < QT) {
      var sum = 0.0;
      for (var jj = 0u; jj < WG; jj++) { sum += qt_probs[t * WG + jj]; }
      let corr = exp(qt_m[t] - qt_mnew[t]);
      qt_corr[t] = corr;
      qt_l[t] = qt_l[t] * corr + sum;
      qt_m[t] = qt_mnew[t];
    }
    workgroupBarrier();

    // 5) V accumulation: threads own disjoint d; each V element is read
    //    once and reused across all QT accumulators.
    let lim = min(WG, params.seq_kv - c * WG);
    for (var d = t; d < D; d += WG) {
      var a: array<f32, QT>;
      for (var qq = 0u; qq < QT; qq++) { a[qq] = qt_acc[qq * DQ + d] * qt_corr[qq]; }
      for (var jj = 0u; jj < lim; jj++) {
        let v = V[((c * WG + jj) * H + h) * D + d];
        for (var qq = 0u; qq < QT; qq++) {
          a[qq] = fma(qt_probs[qq * WG + jj], v, a[qq]);
        }
      }
      for (var qq = 0u; qq < QT; qq++) { qt_acc[qq * DQ + d] = a[qq]; }
    }
    workgroupBarrier(); // probs is rewritten next chunk
  }

  for (var qq = 0u; qq < QT; qq++) {
    let qi = q0 + qq;
    if (qi >= params.seq_q) { continue; }
    let inv_l = 1.0 / qt_l[qq];
    for (var d = t; d < D; d += WG) {
      Out[(qi * H + h) * D + d] = qt_acc[qq * DQ + d] * inv_l;
    }
  }
}

// ── transpose_khds: K [S, H*D] -> KT [H, D, S] ──────────────────────────
// Produces the transposed K layout attention_stream requires (see header).
// Reuses this file's bindings: source K in the Q slot (@0), dest in Out (@3).
// One thread per element over S*H*D. Dispatch: (ceil(S*H*D / 256), 1, 1)
// — at the max DiT joint seq (4608*3072/256 = 55296) this stays under the
// 65535 per-dimension workgroup limit.
@compute @workgroup_size(256)
fn transpose_khds(@builtin(global_invocation_id) gid: vec3u) {
  let D = params.head_dim;
  let H = params.num_heads;
  let S = params.seq_kv;
  let idx = gid.x;
  if (idx >= S * H * D) { return; }
  let s = idx / (H * D);
  let hd = idx % (H * D);
  let h = hd / D;
  let d = hd % D;
  Out[(h * D + d) * S + s] = Q[idx];
}
