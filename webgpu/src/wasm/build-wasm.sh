#!/usr/bin/env bash
# Build the freestanding wasm32 SIMD kernels with pinned LLVM 22.1.7.
# No emscripten, no libc — clang + wasm-ld only (supply-chain policy).
# Output goes to webgpu/public/wasm/ so vite serves it as a static asset.
set -euo pipefail

CLANG="/c/Program Files/LLVM/bin/clang.exe"
DIR="$(cd "$(dirname "$0")" && pwd)"
OUT="$DIR/../../public/wasm"
mkdir -p "$OUT"

"$CLANG" --target=wasm32 -O3 -msimd128 -mbulk-memory -nostdlib -ffreestanding \
  -Wl,--no-entry \
  -Wl,--export=__heap_base \
  -Wl,--initial-memory=16777216 \
  -Wl,--max-memory=1073741824 \
  -o "$OUT/q5k_gemv.wasm" \
  "$DIR/q5k_gemv.c"

echo "built: $OUT/q5k_gemv.wasm ($(stat -c%s "$OUT/q5k_gemv.wasm") bytes)"
