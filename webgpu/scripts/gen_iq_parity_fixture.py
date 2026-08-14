"""Generate a synthetic dequant parity fixture for the grid-codebook IQ quants.

Covers IQ3_XXS / IQ3_S / IQ2_S / IQ2_XXS / IQ2_XS / IQ1_M.

No real GGUF needed: we synthesize random blocks, force a sane f16 scale into
each block (so the reference doesn't produce NaN/Inf), and dequantize them with
the official `gguf` python package as the ground truth — an implementation
independent of both ggml-quants.c and our TS port. The TS side
(test-iq-parity.mts) runs dequantGGML on the same bytes and compares.

Every other bit in these formats is a legal code (grid and sign indices are
masked to their table size by construction), so random bytes are valid blocks.

Usage:
  ./venv/Scripts/python.exe webgpu/scripts/gen_iq_parity_fixture.py <out.json>
"""
import json
import sys

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

# id -> (name, block_size, type_size)
# Order matters: appending keeps earlier types' random bytes unchanged.
TYPES = {
    18: ("IQ3_XXS", 256, 98),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    29: ("IQ1_M", 256, 56),
}
N_BLOCKS = 4
SEED = 1234


def stamp_scale(raw, ttype, base, d_bytes):
    """Force a sane f16 superblock scale into the block starting at `base`.

    Most IQ types keep the f16 `d` in the first two bytes. IQ1_M has no `d`
    field at all: its f16 is assembled from the TOP NIBBLE of each of the four
    scale u16s at offset 48 (result bits 0-3 from s0, 4-7 from s1, 8-11 from
    s2, 12-15 from s3). The low 12 bits of each word are the 3-bit sub-scales
    and stay random.
    """
    if ttype != 29:
        raw[base + 0] = d_bytes[0]
        raw[base + 1] = d_bytes[1]
        return
    bits = int(np.frombuffer(d_bytes, dtype=np.uint16)[0])
    for k in range(4):
        off = base + 48 + 2 * k
        lo = int(raw[off]) | (int(raw[off + 1]) << 8)
        word = (lo & 0x0FFF) | (((bits >> (4 * k)) & 0xF) << 12)
        raw[off] = word & 0xFF
        raw[off + 1] = (word >> 8) & 0xFF


def main():
    out_path = sys.argv[1]
    rng = np.random.default_rng(SEED)
    # fixed positive f16 scale, little-endian bytes
    d_bytes = np.float16(0.0625).tobytes()

    samples = []
    for ttype, (name, bsize, tsize) in TYPES.items():
        raw = rng.integers(0, 256, size=N_BLOCKS * tsize, dtype=np.uint8)
        for b in range(N_BLOCKS):
            stamp_scale(raw, ttype, b * tsize, d_bytes)
        raw = bytes(raw)
        expected = dequantize(
            np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType(ttype)
        ).reshape(-1).astype(np.float32)
        assert len(expected) == N_BLOCKS * bsize, \
            f"{name}: {len(expected)} != {N_BLOCKS * bsize}"
        samples.append({
            "tensor": name,
            "ggml_type": ttype,
            "n_elements": N_BLOCKS * bsize,
            "bytes_hex": raw.hex(),
            "expected_f32": [float(x) for x in expected],
        })
        print(f"{name}: {N_BLOCKS} blocks, {len(expected)} values, "
              f"range [{expected.min():.4f}, {expected.max():.4f}]")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dequant_samples": samples}, f, indent=1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
