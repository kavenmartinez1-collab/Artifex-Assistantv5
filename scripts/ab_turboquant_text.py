"""Greedy A/B: TurboQuant+ KV cache OFF vs ON, same model, same prompt.

Loads the transformers engine once, generates greedily (temperature=0)
with the TurboQuant toggle off, then on, and compares:
  - common token prefix (must cover at least the pre-compression region:
    compression only touches tokens once seq_len > residual_length=128)
  - peak CUDA memory per run
  - tok/s per run

Run: ./venv/Scripts/python.exe scripts/ab_turboquant_text.py [model_dir] [max_tokens]
"""
import logging
import sys
import time

sys.path.insert(0, ".")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import torch
from core import config
from core.config import set_turboquant_kv
from core.engine_transformers import TransformersEngine

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "models/qwen3.5-9b"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 384

PROMPT = (
    "Explain, step by step, how a relational database executes a SQL query: "
    "parsing, planning, optimization, and execution. Be thorough."
)
MESSAGES = [{"role": "user", "content": PROMPT}]

config._active_model = MODEL_DIR
engine = TransformersEngine()
print(f"loading {MODEL_DIR} ...", flush=True)
engine.load()
print("loaded.", flush=True)


def run(label):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    text = engine.generate_streaming(
        MESSAGES, max_tokens=MAX_TOKENS, temperature=0, enable_thinking=False,
    )
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**2
    ids = engine.tokenizer.encode(text)
    print(f"\n[{label}] {len(ids)} tokens in {dt:.1f}s ({len(ids)/dt:.1f} tok/s), "
          f"peak CUDA alloc {peak:.0f} MB")
    return text, ids


text_off, ids_off = run("TurboQuant OFF")
set_turboquant_kv(True)
text_on, ids_on = run("TurboQuant ON ")

n = 0
while n < min(len(ids_off), len(ids_on)) and ids_off[n] == ids_on[n]:
    n += 1
print(f"\ncommon token prefix: {n} / off={len(ids_off)} on={len(ids_on)}")

with open("scripts/ab_turboquant_off.txt", "w", encoding="utf-8") as f:
    f.write(text_off)
with open("scripts/ab_turboquant_on.txt", "w", encoding="utf-8") as f:
    f.write(text_on)
print("outputs saved to scripts/ab_turboquant_{off,on}.txt")
