// 2D convolution kernels for the FLUX.2 VAE (NCHW f32, batch 1).
//
// Entries:
//   conv2d_3x3           - 3x3, stride 1, zero-pad 1 (same H, W)
//   conv2d_3x3_s2        - 3x3, stride 2, ASYMMETRIC zero-pad (0,1,0,1):
//                          diffusers Downsample2D(padding=0) F.pads right and
//                          bottom by 1, so [C, H, W] -> [C, H/2, W/2] (H, W even)
//   conv2d_1x1           - pointwise
//   upsample_nearest_2x  - [C, H, W] -> [C, 2H, 2W]
//
// All VAE convs carry bias, so Bias is a required binding for all conv
// entries. Weights are the PyTorch layout flattened row-major:
//   conv2d_3x3(_s2): [c_out, c_in, 3, 3]
//   conv2d_1x1:      [c_out, c_in]
//
// One thread per output element. Dispatch:
//   conv2d_3x3 / _1x1:   (ceil(H*W / 256),   c_out, 1)   H, W = input = output
//   conv2d_3x3_s2:       (ceil(H*W/4 / 256), c_out, 1)   H, W = INPUT dims
//   upsample_nearest_2x: (ceil(4*H*W / 256), c,     1)   H, W = INPUT dims

struct Params {
  c_in: u32,    // upsample: channel count
  c_out: u32,   // upsample: unused
  height: u32,  // input height
  width: u32,   // input width
}

@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Wt: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn conv2d_3x3(@builtin(workgroup_id) wid: vec3u,
              @builtin(local_invocation_id) lid: vec3u) {
  let hw = params.height * params.width;
  let pix = wid.x * 256u + lid.x;
  if (pix >= hw) { return; }
  let oc = wid.y;
  let y = i32(pix / params.width);
  let x = i32(pix % params.width);

  var acc = Bias[oc];
  for (var ic = 0u; ic < params.c_in; ic++) {
    let in_base = ic * hw;
    let w_base = (oc * params.c_in + ic) * 9u;
    for (var kh = 0u; kh < 3u; kh++) {
      let iy = y + i32(kh) - 1;
      if (iy < 0 || iy >= i32(params.height)) { continue; }
      for (var kw = 0u; kw < 3u; kw++) {
        let ix = x + i32(kw) - 1;
        if (ix < 0 || ix >= i32(params.width)) { continue; }
        acc += X[in_base + u32(iy) * params.width + u32(ix)] * Wt[w_base + kh * 3u + kw];
      }
    }
  }
  Out[oc * hw + pix] = acc;
}

@compute @workgroup_size(256)
fn conv2d_3x3_s2(@builtin(workgroup_id) wid: vec3u,
                 @builtin(local_invocation_id) lid: vec3u) {
  let oh = params.height / 2u;
  let ow = params.width / 2u;
  let out_hw = oh * ow;
  let pix = wid.x * 256u + lid.x;
  if (pix >= out_hw) { return; }
  let oc = wid.y;
  let oy = pix / ow;
  let ox = pix % ow;
  let in_hw = params.height * params.width;

  // window origin in input coords; only right/bottom are padded (0,1,0,1)
  var acc = Bias[oc];
  for (var ic = 0u; ic < params.c_in; ic++) {
    let in_base = ic * in_hw;
    let w_base = (oc * params.c_in + ic) * 9u;
    for (var kh = 0u; kh < 3u; kh++) {
      let iy = oy * 2u + kh;
      if (iy >= params.height) { continue; }
      for (var kw = 0u; kw < 3u; kw++) {
        let ix = ox * 2u + kw;
        if (ix >= params.width) { continue; }
        acc += X[in_base + iy * params.width + ix] * Wt[w_base + kh * 3u + kw];
      }
    }
  }
  Out[oc * out_hw + pix] = acc;
}

@compute @workgroup_size(256)
fn conv2d_1x1(@builtin(workgroup_id) wid: vec3u,
              @builtin(local_invocation_id) lid: vec3u) {
  let hw = params.height * params.width;
  let pix = wid.x * 256u + lid.x;
  if (pix >= hw) { return; }
  let oc = wid.y;

  var acc = Bias[oc];
  for (var ic = 0u; ic < params.c_in; ic++) {
    acc += X[ic * hw + pix] * Wt[oc * params.c_in + ic];
  }
  Out[oc * hw + pix] = acc;
}

@compute @workgroup_size(256)
fn upsample_nearest_2x(@builtin(workgroup_id) wid: vec3u,
                       @builtin(local_invocation_id) lid: vec3u) {
  let ow = params.width * 2u;
  let out_hw = params.height * params.width * 4u;
  let pix = wid.x * 256u + lid.x;
  if (pix >= out_hw) { return; }
  let c = wid.y;
  let oy = pix / ow;
  let ox = pix % ow;
  Out[c * out_hw + pix] =
    X[c * params.height * params.width + (oy / 2u) * params.width + (ox / 2u)];
}
