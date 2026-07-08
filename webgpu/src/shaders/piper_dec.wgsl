// Piper HiFiGAN decoder kernels (channels-first [C, T] f32, batch 1).
//
// GPU port of decForward() in src/audio/piper.ts — validated against that CPU
// reference and the onnxruntime fixture waveform (relL2 <= 1e-3).
//
// Entries:
//   conv1d   - dense/dilated Conv1d [Cin,T] -> [Cout,Tout], weight [Cout,Cin,K]
//              flat, symmetric pad, stride, dilation. Bias always bound (a
//              zero buffer stands in for the bias-less conv_post).
//   convt1d  - ConvTranspose1d [Cin,Tin] -> [Cout,Tout] (gather form), weight
//              [Cin,Cout,K] flat, stride S, symmetric pad P, output_padding 0.
//   leaky    - Out[i] = x<0 ? x*slope : x   (slope 0.1 body, 0.01 final)
//   add_ip   - Out[i] += B[i]               (residual / MRF accumulate)
//   scale    - Out[i]  = X[i] * slope       (copy at slope=1, MRF mean at 1/3)
//   tanh_act - Out[i]  = tanh(X[i])
//
// One thread per output element. Conv dispatch: (ceil(Tout/256), Cout, 1) with
// thread.x = output time, wid.y = output channel. Elementwise: 1-D over N.
//
// Bindings (auto layout keeps only those an entry references):
//   0 X (input / A)   1 W (weight / B)   2 Bias   3 Out   4 params

struct Params {
  cin: u32,
  cout: u32,
  t_in: u32,
  t_out: u32,
  k: u32,
  pad: u32,
  dilation: u32,
  stride: u32,
  has_bias: u32,
  n: u32,
  slope: f32,
  _pad: u32,
}

@group(0) @binding(0) var<storage, read>       X: array<f32>;
@group(0) @binding(1) var<storage, read>       W: array<f32>;
@group(0) @binding(2) var<storage, read>       Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

@compute @workgroup_size(256)
fn conv1d(@builtin(workgroup_id) wid: vec3u,
          @builtin(local_invocation_id) lid: vec3u) {
  let to = wid.x * 256u + lid.x;
  if (to >= params.t_out) { return; }
  let co = wid.y;

  var acc = select(0.0, Bias[co], params.has_bias != 0u);
  let start = i32(to * params.stride) - i32(params.pad);
  let wbase = co * params.cin * params.k;
  for (var ci = 0u; ci < params.cin; ci++) {
    let xbase = ci * params.t_in;
    let wc = wbase + ci * params.k;
    for (var kk = 0u; kk < params.k; kk++) {
      let ti = start + i32(kk * params.dilation);
      if (ti >= 0 && ti < i32(params.t_in)) {
        acc += W[wc + kk] * X[xbase + u32(ti)];
      }
    }
  }
  Out[co * params.t_out + to] = acc;
}

@compute @workgroup_size(256)
fn convt1d(@builtin(workgroup_id) wid: vec3u,
           @builtin(local_invocation_id) lid: vec3u) {
  let to = wid.x * 256u + lid.x;
  if (to >= params.t_out) { return; }
  let co = wid.y;
  let S = i32(params.stride);

  var acc = select(0.0, Bias[co], params.has_bias != 0u);
  for (var ci = 0u; ci < params.cin; ci++) {
    let xbase = ci * params.t_in;
    let wc = (ci * params.cout + co) * params.k;
    for (var kk = 0u; kk < params.k; kk++) {
      let num = i32(to) + i32(params.pad) - i32(kk);   // ti*S = to + P - k
      if (num < 0 || (num % S) != 0) { continue; }
      let ti = num / S;
      if (ti >= 0 && ti < i32(params.t_in)) {
        acc += W[wc + kk] * X[xbase + u32(ti)];
      }
    }
  }
  Out[co * params.t_out + to] = acc;
}

@compute @workgroup_size(256)
fn leaky(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let v = X[i];
  Out[i] = select(v, v * params.slope, v < 0.0);
}

@compute @workgroup_size(256)
fn add_ip(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  Out[i] = Out[i] + W[i];
}

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  Out[i] = X[i] * params.slope;
}

@compute @workgroup_size(256)
fn tanh_act(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  Out[i] = tanh(X[i]);
}
