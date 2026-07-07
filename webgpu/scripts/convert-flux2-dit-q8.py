"""Phase 7 (P7.1): FLUX.2-klein DiT bf16 safetensors -> Q8_0 container (.artq).

Quantizes every 2-D weight to Q8_0 (per-32 blocks along K, f16 scale) and
keeps 1-D norm vectors as f32. Layout is GPU-friendly: per tensor the i8
quants and f16 scales are DEINTERLEAVED into two 4-byte-aligned sections so
the WGSL dequant kernel reads whole u32 words (GGUF's interleaved 34-byte
blocks straddle word boundaries).

Scale convention (matters for exact CPU/GPU parity): d = f16(amax/127) is
rounded to f16 FIRST, then q = clip(round(x / d_f16), -127, 127). Dequant is
x' = f32(d_f16) * q  — exact in f32 (11-bit * 7-bit mantissa < 24).

Container:
  magic 'ARTQ' | u32 version=1 | u64 jsonLen | JSON header | data
  JSON: { tensors: [ { name, shape, dtype: 'q8_0'|'f32',
                       offset, bytes,            # quants (i8) or raw f32
                       scaleOffset?, scaleBytes? # q8_0 only, f16
                     } ] }
  Offsets are relative to the data section start (= 16 + jsonLen; JSON is
  space-padded so the data section is 64-aligned), each tensor 64-aligned.

Run: ./venv/Scripts/python.exe webgpu/scripts/convert-flux2-dit-q8.py
"""
import json
import struct
import sys
import time

import numpy as np

SRC = 'models/flux.2-klein-4b/transformer/diffusion_pytorch_model.safetensors'
DST = 'models/flux.2-klein-4b/transformer/diffusion_pytorch_model.q8_0.artq'
ALIGN = 64


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    with open(SRC, 'rb') as f:
        (hlen,) = struct.unpack('<Q', f.read(8))
        header = json.loads(f.read(hlen))
    header.pop('__metadata__', None)
    data_start = 8 + hlen

    entries = []
    blobs = []  # (offset, bytes) in output order
    pos = 0

    def alloc(n: int) -> int:
        nonlocal pos
        pos = (pos + ALIGN - 1) // ALIGN * ALIGN
        off = pos
        pos += n
        return off

    t0 = time.time()
    worst = (0.0, '')
    src = open(SRC, 'rb')
    names = sorted(header)  # deterministic order
    for idx, name in enumerate(names):
        t = header[name]
        assert t['dtype'] == 'BF16', (name, t['dtype'])
        s0, s1 = t['data_offsets']
        src.seek(data_start + s0)
        raw = src.read(s1 - s0)
        assert len(raw) == s1 - s0, name
        x = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
        shape = t['shape']
        if len(shape) == 1:
            b = x.astype(np.float32).tobytes()
            entries.append({'name': name, 'shape': shape, 'dtype': 'f32',
                            'offset': alloc(len(b)), 'bytes': len(b)})
            blobs.append(b)
            continue
        n, k = shape
        assert k % 32 == 0, (name, shape)
        blk = x.reshape(n * k // 32, 32)
        amax = np.abs(blk).max(axis=1)
        d = (amax / 127.0).astype(np.float16)  # round scale to f16 FIRST
        df = d.astype(np.float32)
        inv = np.where(df > 0, 1.0 / df, 0.0).astype(np.float32)
        q = np.clip(np.rint(blk * inv[:, None]), -127, 127).astype(np.int8)
        # quant error report (f32 dequant vs f32 source)
        deq = df[:, None] * q.astype(np.float32)
        num = float(((deq - blk) ** 2).sum())
        den = float((blk ** 2).sum())
        rel = (num / max(den, 1e-30)) ** 0.5
        if rel > worst[0]:
            worst = (rel, name)
        qb = q.tobytes()
        sb = d.tobytes()
        e = {'name': name, 'shape': shape, 'dtype': 'q8_0',
             'offset': alloc(len(qb)), 'bytes': len(qb)}
        blobs.append(qb)
        e['scaleOffset'] = alloc(len(sb))
        e['scaleBytes'] = len(sb)
        blobs.append(sb)
        entries.append(e)
        if idx % 20 == 0:
            print(f'  [{idx + 1}/{len(names)}] {name} relL2 {rel:.2e}')
    src.close()

    hdr = json.dumps({'tensors': entries}).encode()
    base = 4 + 4 + 8 + len(hdr)
    base = (base + ALIGN - 1) // ALIGN * ALIGN  # align data section start
    with open(DST, 'wb') as f:
        f.write(b'ARTQ')
        f.write(struct.pack('<I', 1))
        f.write(struct.pack('<Q', base - 16))  # jsonLen incl. pad
        f.write(hdr.ljust(base - 16, b' '))
        cur = 0
        bi = 0
        for e in entries:
            for off, nb in ([(e['offset'], e['bytes'])]
                            + ([(e['scaleOffset'], e['scaleBytes'])]
                               if e['dtype'] == 'q8_0' else [])):
                f.write(b'\0' * (off - cur))
                f.write(blobs[bi])
                bi += 1
                cur = off + nb
    total = base + pos
    print(f'\nwrote {DST}: {total / 2**30:.2f} GiB '
          f'({len(entries)} tensors) in {time.time() - t0:.0f}s')
    print(f'worst per-tensor quant relL2: {worst[0]:.3e} ({worst[1]})')


if __name__ == '__main__':
    sys.exit(main())
