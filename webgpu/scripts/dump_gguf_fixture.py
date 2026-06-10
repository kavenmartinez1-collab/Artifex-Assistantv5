"""Dump a GGUF header fixture + dequant reference samples as JSON.

The fixture is the ground truth for webgpu/src/model/gguf.ts (structure)
and gguf-dequant.ts (values). Uses the official `gguf` python package
(llama.cpp project) as the independent reference implementation.

Usage:
  ./venv/Scripts/python.exe webgpu/scripts/dump_gguf_fixture.py <model.gguf> <out.json>
"""
import json
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

# ggml types we ship CPU/WGSL dequant for (id -> [block_size, type_size])
SAMPLE_TYPES = {
    0: (1, 4),      # F32
    1: (1, 2),      # F16
    8: (32, 34),    # Q8_0
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    30: (1, 2),     # BF16
}
SAMPLE_BLOCKS = 4  # blocks per sample


def field_value(field):
    try:
        v = field.contents()
    except Exception:
        return {"__unreadable": True}
    if isinstance(v, list):
        head = v[:4]
        head = [h if isinstance(h, (int, float, str, bool)) else str(h) for h in head]
        return {"__array_len": len(v), "head": head}
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    return str(v)


def main():
    gguf_path, out_path = sys.argv[1], sys.argv[2]
    reader = GGUFReader(gguf_path)

    kv = {name: field_value(f) for name, f in reader.fields.items()}

    tensors = []
    samples = []
    sampled_types = set()
    for t in reader.tensors:
        ne = [int(d) for d in t.shape]  # innermost-first, as stored in file
        tensors.append({
            "name": t.name,
            "ne": ne,
            "ggml_type": int(t.tensor_type),
            "abs_offset": int(t.data_offset),
            "n_bytes": int(t.n_bytes),
            "n_elements": int(t.n_elements),
        })
        ttype = int(t.tensor_type)
        if ttype in SAMPLE_TYPES and ttype not in sampled_types:
            sampled_types.add(ttype)
            block_size, type_size = SAMPLE_TYPES[ttype]
            n_blocks = min(SAMPLE_BLOCKS, t.n_bytes // type_size)
            raw = bytes(t.data.reshape(-1).view(np.uint8)[: n_blocks * type_size])
            assert len(raw) == n_blocks * type_size, \
                f"short sample read: {len(raw)} != {n_blocks * type_size}"
            if ttype == 0:
                expected = np.frombuffer(raw, dtype=np.float32)
            elif ttype == 1:
                expected = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            elif ttype == 30:
                u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                expected = np.frombuffer((u16 << 16).astype(np.uint32).tobytes(), dtype=np.float32)
            else:
                expected = dequantize(
                    np.frombuffer(raw, dtype=np.uint8), t.tensor_type
                ).reshape(-1).astype(np.float32)
            samples.append({
                "tensor": t.name,
                "ggml_type": ttype,
                "n_elements": n_blocks * block_size,
                "bytes_hex": raw.hex(),
                "expected_f32": [float(x) for x in expected],
            })

    fixture = {
        "file": gguf_path.replace("\\", "/").split("/")[-1],
        "version": int(reader.fields["GGUF.version"].contents()),
        "tensor_count": len(reader.tensors),
        "alignment": int(reader.alignment),
        "data_offset": int(reader.data_offset),
        "kv": kv,
        "tensors": tensors,
        "dequant_samples": samples,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=1)
    print(f"Wrote {out_path}: {len(tensors)} tensors, {len(samples)} dequant samples "
          f"(types {sorted(sampled_types)}), data_offset={reader.data_offset}, "
          f"alignment={reader.alignment}")


if __name__ == "__main__":
    main()
