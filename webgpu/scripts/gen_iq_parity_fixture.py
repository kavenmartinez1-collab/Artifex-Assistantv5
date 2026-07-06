"""Generate a synthetic dequant parity fixture for IQ3_XXS / IQ3_S / IQ2_S.

No real GGUF needed: we synthesize random blocks, force a sane f16 scale at the
front of each block (so the reference doesn't produce NaN/Inf), and dequantize
them with the official `gguf` python package as the ground truth. The TS side
(test-iq-parity.mts) runs dequantGGML on the same bytes and compares.

Usage:
  ./venv/Scripts/python.exe webgpu/scripts/gen_iq_parity_fixture.py <out.json>
"""
import json
import sys

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

# id -> (name, block_size, type_size)
TYPES = {
    18: ("IQ3_XXS", 256, 98),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
}
N_BLOCKS = 4
SEED = 1234


def main():
    out_path = sys.argv[1]
    rng = np.random.default_rng(SEED)
    # fixed positive f16 scale, little-endian bytes
    d_bytes = np.float16(0.0625).tobytes()

    samples = []
    for ttype, (name, bsize, tsize) in TYPES.items():
        raw = rng.integers(0, 256, size=N_BLOCKS * tsize, dtype=np.uint8)
        for b in range(N_BLOCKS):
            raw[b * tsize + 0] = d_bytes[0]
            raw[b * tsize + 1] = d_bytes[1]
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
