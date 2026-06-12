// One-shot Q2_K layout repack (lever C) — runs once at engine init per
// Q2_K weight tensor, producing the unit-contiguous layout consumed by
// matmul_gguf_q2_k_tiled_r.
//
// Source superblock (21 u32, llama.cpp raw 84 B): scales[16] @ words 0-3,
// qs[64] @ words 4-19 (two 32-byte groups; element l of sub-pair p lives
// in byte l of group p>>2 at shift plane 2*(p&3)), d/dmin f16 @ word 20.
//
// Destination keeps stride 21 and words 0-3 / 20 unchanged (zero VRAM
// growth); words 4-19 are permuted so unit p owns the contiguous pair
// [4+2p, 4+2p+1]. Per source word, plane t is compacted to one byte via
//   b = (w >> 2t) & 0x03030303;  y = b | (b >> 6);  z = y | (y >> 12);
//   byte = z & 0xFF  →  e0 | e1<<2 | e2<<4 | e3<<6
// so output word m (0,1) packs the compacted bytes of source words
// 4m..4m+3 of the group. Element l = w*4+k of the unit ends at bit
// (w&3)*8 + 2k of word (w<4 ? 0 : 1).
//
// JS parity port: repackQ2K in scripts/test-gemv-tiled.mts (gate 5).
// NOT in-place safe (each output pair reads 8 source words spread across
// the group) — src and dst must be distinct buffers.
//
// Grid: x = ceil(nSB / 64); one thread per superblock.

struct Params {
  nSB: u32,    // total superblocks in the tensor (rows × K/256)
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn repack_q2k(@builtin(global_invocation_id) gid: vec3u) {
  let b = gid.x;
  if (b >= params.nSB) { return; }
  let base = b * 21u;
  for (var i = 0u; i < 4u; i = i + 1u) { dst[base + i] = src[base + i]; }
  dst[base + 20u] = src[base + 20u];
  for (var p = 0u; p < 8u; p = p + 1u) {
    let g = p >> 2u;
    let t = p & 3u;
    let srcBase = base + 4u + g * 8u;
    for (var m = 0u; m < 2u; m = m + 1u) {
      var word: u32 = 0u;
      for (var j = 0u; j < 4u; j = j + 1u) {
        let bb = (src[srcBase + m * 4u + j] >> (2u * t)) & 0x03030303u;
        let y = bb | (bb >> 6u);
        let z = y | (y >> 12u);
        word = word | ((z & 0xFFu) << (j * 8u));
      }
      dst[base + 4u + p * 2u + m] = word;
    }
  }
}
