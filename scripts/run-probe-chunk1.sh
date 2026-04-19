#!/bin/bash
# run-probe-chunk1.sh — A/B variant of run-probe.sh that forces PREFILL_CHUNK=1.
# Used to isolate batched-GEMM precision vs chunked-prefill semantics.
#
# Outputs land in chunk1-prefixed files so they don't clobber the default-chunk run.
#
# Usage:
#   ./scripts/run-probe-chunk1.sh                    # "Hello Artifex?"
#   ./scripts/run-probe-chunk1.sh "What is Paris?"

set -e

PROMPT="${1:-Hello Artifex?}"
SERVER="http://127.0.0.1:5173"
MAX_TOKENS="${2:-60}"
HAILMARY_DIR="${HAILMARY_DIR:-models/qwen3.5-9b-HailMary}"
ENG_OUT="webgpu/debug-output.chunk1.json"
REF_OUT="reference_dump.json"  # reuse the same reference (token IDs may differ — see warning below)
PY="${PY:-./venv/Scripts/python.exe}"

DUMP_TAGS_JSON='{"prefill-end":0,"decode-1":1,"decode-10":10,"decode-50":50}'

echo "=== Probe (chunk=1): \"$PROMPT\" (maxTokens=$MAX_TOKENS) ==="

# Clear stale dump
curl -s -X POST "$SERVER/api/debug" -H "Content-Type: application/json" \
  -d '{"pending":true}' > /dev/null 2>&1 || true

# Queue the test with prefillChunk=1 to force single-row path through prefill.
curl -s -X POST "$SERVER/api/test" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"temperature\":0,\"maxTokens\":$MAX_TOKENS,\"dumpStats\":true,\"decodeDumpSteps\":[1,10,50],\"prefillChunk\":1}" \
  > /dev/null 2>&1

echo "[probe] test queued (prefillChunk=1, will be ~16x slower than default) — waiting ..."
for i in $(seq 1 240); do
    sleep 2
    RESULT=$(curl -s "$SERVER/api/debug" 2>/dev/null)
    if echo "$RESULT" | grep -q '"layerDump"'; then
        echo "[probe] engine dump received"
        break
    fi
    printf "."
done
echo ""

# Save the engine dump under a chunk1-specific name so it doesn't clobber the
# baseline at webgpu/debug-output.json.
cp webgpu/debug-output.json "$ENG_OUT"
echo "[probe] saved engine dump -> $ENG_OUT"

# Verify the reference's teacher tokens match the engine's tokenIds. If the
# engine produced different tokens with chunk=1 (it shouldn't at temp=0, but
# we check), we'd need to regenerate the reference.
ENG_TOKS=$("$PY" -c "
import json
print(json.dumps(json.load(open('$ENG_OUT')).get('tokenIds') or []))
")
REF_TOKS=$("$PY" -c "
import json
d = json.load(open('$REF_OUT'))
print(json.dumps(d.get('teacherTokens') or []))
" 2>/dev/null || echo "[]")
if [ "$ENG_TOKS" != "$REF_TOKS" ]; then
    echo "[probe] WARNING: chunk=1 engine produced different tokens than baseline:"
    echo "  engine: $ENG_TOKS"
    echo "  reference (cached): $REF_TOKS"
    echo "  -> regenerating reference for fair comparison ..."
    "$PY" scripts/divergence_probe.py \
        --prompt "$PROMPT" \
        --model "${REF_MODEL:-models/qwen3.5-9b}" \
        --out "reference_dump.chunk1.json" \
        --teacher-tokens "$ENG_TOKS" \
        --dump-tags "$DUMP_TAGS_JSON"
    REF_OUT="reference_dump.chunk1.json"
else
    echo "[probe] tokenIds match baseline — reusing $REF_OUT"
fi

echo ""
"$PY" scripts/compare_dumps.py \
    --engine "$ENG_OUT" \
    --reference "$REF_OUT" \
    --model-dir "$HAILMARY_DIR"
