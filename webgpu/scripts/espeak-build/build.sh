#!/usr/bin/env bash
# Build espeak-ng (pinned commit 212928b) to WASM for the P5 G2P engine.
#
#   emsdk 3.1.74 · espeak-ng static lib · bridge = piper get_phonemes port
#
# Reuses piper's prebuilt espeak-ng-data (same commit) instead of running the
# native data-compile step (which can't run under an emscripten cross-build).
# Output: src/audio/espeak/espeak.{js,wasm,data}  (~1.2 MB, checked in)
#
# One-time bootstrap (from this dir; these trees are gitignored):
#   ../../../venv/Scripts/python.exe -m pip install cmake ninja
#   git clone https://github.com/emscripten-core/emsdk.git
#   ../../../venv/Scripts/python.exe emsdk/emsdk.py install 3.1.74
#   ../../../venv/Scripts/python.exe emsdk/emsdk.py activate 3.1.74
#   git clone https://github.com/espeak-ng/espeak-ng.git
#   git -C espeak-ng checkout 212928b394a96e8fd2096616bfd54e17845c48f6
#
# Then: bash build.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
ESPDK="$HERE/emsdk"
ESPEAK_SRC="$HERE/espeak-ng"
BUILD_DIR="$HERE/build-espeak"
DATA_SRC="$REPO_ROOT/venv/Lib/site-packages/piper/espeak-ng-data"
DATA_LOCAL="$HERE/espeak-ng-data"
OUT_DIR="$REPO_ROOT/webgpu/src/audio/espeak"

# --- emscripten env (emsdk_env.sh is flaky under git-bash; wire it by hand) ---
export EM_CONFIG="$ESPDK/.emscripten"
export PATH="$ESPDK/upstream/emscripten:$ESPDK/upstream/bin:$ESPDK/node/22.16.0_64bit/bin:$ESPDK/python/3.13.3_64bit:$REPO_ROOT/venv/Scripts:$PATH"
EMCMAKE="$ESPDK/upstream/emscripten/emcmake.bat"
EMCC="$ESPDK/upstream/emscripten/emcc.bat"

# --- stage the runtime data (from piper, same espeak-ng commit) ---
# en-us only: the shared phoneme binaries + en_dict + the lang voice defs. Drops
# every other language dict (ru_dict alone is 8 MB) to keep the bundle ~1 MB.
rm -rf "$DATA_LOCAL"
mkdir -p "$DATA_LOCAL"
cp "$DATA_SRC"/phondata "$DATA_SRC"/phonindex "$DATA_SRC"/phontab \
   "$DATA_SRC"/phondata-manifest "$DATA_SRC"/intonations "$DATA_SRC"/en_dict "$DATA_LOCAL/"
cp -r "$DATA_SRC"/lang "$DATA_LOCAL/"

# --- configure + build the static library only (skip CLI + data-compile) ---
"$EMCMAKE" cmake -S "$ESPEAK_SRC" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_ASYNC=OFF \
  -DUSE_MBROLA=OFF \
  -DUSE_LIBSONIC=OFF \
  -DUSE_LIBPCAUDIO=OFF \
  -DUSE_KLATT=OFF \
  -DUSE_SPEECHPLAYER=OFF \
  -DCMAKE_C_FLAGS="-D_FILE_OFFSET_BITS=64 -D__WINT_TYPE__=unsigned -Wno-builtin-macro-redefined -I$ESPEAK_SRC/src/ucd-tools/src/include"

cmake --build "$BUILD_DIR" --target espeak-ng

LIB_ESPEAK="$(find "$BUILD_DIR" -name 'libespeak-ng.a' | head -1)"
LIB_UCD="$(find "$BUILD_DIR" -name 'libucd.a' | head -1)"
echo "libespeak-ng: $LIB_ESPEAK"
echo "libucd:       $LIB_UCD"

# --- link the bridge + libs into a WASM ES module ---
mkdir -p "$OUT_DIR"
"$EMCC" -O3 espeak_bridge.c "$LIB_ESPEAK" "$LIB_UCD" \
  -I "$ESPEAK_SRC/src/include" \
  -o "$OUT_DIR/espeak.js" \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createEspeak \
  -sALLOW_MEMORY_GROWTH=1 \
  -sEXPORTED_FUNCTIONS=_bridge_init,_bridge_set_voice,_bridge_phonemize,_bridge_free,_malloc,_free \
  -sEXPORTED_RUNTIME_METHODS=ccall,cwrap,UTF8ToString,stringToUTF8,lengthBytesUTF8 \
  --preload-file "$DATA_LOCAL@/espeak-ng-data"

echo "== built =="
ls -la "$OUT_DIR"
