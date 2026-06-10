"""Smoke test: load Qwen3.5-9B Q4_K_M GGUF on Blackwell sm_120 CUDA build and generate."""
import os
import sys

# llama_cpp's internal loader uses winmode=RTLD_GLOBAL which DISABLES os.add_dll_directory
# lookups for the llama.dll load itself, but PATH is still searched. So prepend CUDA bin
# to PATH (and set CUDA_PATH so llama_cpp's loader also adds it via add_dll_directory for
# subsequent transitive loads).
CUDA_ROOT = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
CUDA_BIN = os.path.join(CUDA_ROOT, "bin")
if os.path.isdir(CUDA_BIN):
    os.environ["CUDA_PATH"] = CUDA_ROOT
    os.environ["PATH"] = CUDA_BIN + os.pathsep + os.environ.get("PATH", "")

from llama_cpp import Llama

MODEL = r"C:\Artifex-Assistant-V5\models\qwen3.5-9b-abliterated-gguf\Huihui-Qwen3.5-9B-abliterated.i1-Q4_K_M.gguf"

print(f"Loading: {MODEL}")
llm = Llama(
    model_path=MODEL,
    n_gpu_layers=-1,      # all layers on GPU
    n_ctx=2048,
    verbose=True,
)

print("\n=== Generating ===")
out = llm(
    "Q: What is the capital of France?\nA:",
    max_tokens=32,
    temperature=0.0,
    stop=["\n"],
)
print(out["choices"][0]["text"])
print("\n=== OK ===")
