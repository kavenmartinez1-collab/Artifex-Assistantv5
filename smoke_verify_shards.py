"""Verify every tensor in every shard can be read by safetensors directly.
If this passes, the files are fine and the segfault is in transformers'
materialization/copy path, not on-disk corruption.

Run:  ./venv/Scripts/python.exe smoke_verify_shards.py
"""
import glob
import os
import faulthandler
import time

faulthandler.enable()

SHARD_DIR = "models/qwen3-vl-8b-instruct"

from safetensors import safe_open  # noqa: E402

shards = sorted(glob.glob(os.path.join(SHARD_DIR, "model-*-of-*.safetensors")))
print(f"Found {len(shards)} shards")

total_tensors = 0
total_bytes = 0
t0 = time.time()

for path in shards:
    print(f"\n{os.path.basename(path)} ({os.path.getsize(path)/1e9:.2f} GB)")
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"  {len(keys)} tensors")
        for k in keys:
            try:
                t = f.get_tensor(k)
                total_tensors += 1
                total_bytes += t.numel() * t.element_size()
            except Exception as e:
                print(f"  FAILED to read {k}: {e}")
                raise

print(f"\nAll {total_tensors} tensors readable. Total {total_bytes/1e9:.2f} GB in {time.time()-t0:.1f}s")
