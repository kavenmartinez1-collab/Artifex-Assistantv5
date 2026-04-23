"""Isolate Qwen3-VL load from the GUI. Run with:
    ./venv/Scripts/python.exe smoke_load_qwen3vl.py

Strategy: bypass transformers 5.2.0's meta-tensor materialize path, which
segfaults at torch.storage.__getitem__ on this model. We skip
`low_cpu_mem_usage=True` and `device_map="auto"` so the loader falls back
to the classic path: load to CPU first, then move to CUDA manually.
"""
import os
import faulthandler
import time

os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
faulthandler.enable()

MODEL_PATH = "models/qwen3-vl-8b-instruct"

print("=" * 60)
print("Qwen3-VL standalone load test (classic path)")
print("=" * 60)

print("\n[1/4] Importing torch + transformers...")
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

print(f"  torch:        {torch.__version__}")
print(f"  cuda avail:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  cuda device:  {torch.cuda.get_device_name(0)}")
    print(f"  cuda mem gb:  {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")

print("\n[2/4] Loading processor...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  done in {time.time() - t0:.1f}s")

print("\n[3/4] Loading model to CPU (classic path, no meta tensors)...")
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"  loaded to CPU in {time.time() - t0:.1f}s")

print("\n[4/4] Moving to CUDA...")
t0 = time.time()
if torch.cuda.is_available():
    model = model.to("cuda")
    print(f"  .to('cuda') in {time.time() - t0:.1f}s")
else:
    print("  no CUDA — staying on CPU")

print(f"\n  model type:   {type(model).__name__}")
print(f"  num params:   {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
print(f"  first param:  {next(model.parameters()).device} {next(model.parameters()).dtype}")

print("\nLoad succeeded. Safe to retry from GUI.")
