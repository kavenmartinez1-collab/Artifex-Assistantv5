#!/bin/bash
# run-probe.sh — Divergence probe: dump engine per-layer state, compare vs PyTorch reference.
#
# Usage:
#   ./scripts/run-probe.sh                           # default "Hello Artifex?"
#   ./scripts/run-probe.sh "What is Paris known for?"
#
# Prereqs:
#   - Model loaded in browser at http://127.0.0.1:5173
#   - venv with transformers, torch — reference model at models/qwen3.5-9b
#
# Output:
#   - webgpu/debug-output.json  (engine dump, ride-along in POST body .layerDump)
#   - reference_dump.json       (PyTorch reference)
#   - stdout: per-layer cosine/rel-L2 + first-divergence flag + logits comparison

set -e

PROMPT="${1:-Hello Artifex?}"
SERVER="http://127.0.0.1:5173"
MAX_TOKENS="${2:-60}"  # need >= max(decodeDumpSteps) so all dump points trigger
REF_MODEL="${REF_MODEL:-models/qwen3.5-9b}"
REF_OUT="${REF_OUT:-reference_dump.json}"
ENG_OUT="${ENG_OUT:-webgpu/debug-output.json}"
HAILMARY_DIR="${HAILMARY_DIR:-models/qwen3.5-9b-HailMary}"
PY="${PY:-./venv/Scripts/python.exe}"

# Tag -> step map for the reference probe. step=0 = last prompt token (prefill end).
DUMP_TAGS_JSON='{"prefill-end":0,"decode-1":1,"decode-10":10,"decode-50":50}'

echo "=== Probe: \"$PROMPT\" (maxTokens=$MAX_TOKENS) ==="

# Clear stale dump
curl -s -X POST "$SERVER/api/debug" -H "Content-Type: application/json" \
  -d '{"pending":true}' > /dev/null 2>&1 || true

# Queue the test WITH dumpStats flag + decode-step schedule.
# decodeDumpSteps must be a subset of the steps named in DUMP_TAGS_JSON above.
curl -s -X POST "$SERVER/api/test" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"temperature\":0,\"maxTokens\":$MAX_TOKENS,\"dumpStats\":true,\"decodeDumpSteps\":[1,10,50]}" \
  > /dev/null 2>&1

echo "[probe] test queued — waiting for engine dump ..."
for i in $(seq 1 120); do
    sleep 2
    RESULT=$(curl -s "$SERVER/api/debug" 2>/dev/null)
    if echo "$RESULT" | grep -q '"layerDump"'; then
        echo "[probe] engine dump received"
        break
    fi
    printf "."
done
echo ""

if ! grep -q '"layerDump"' "$ENG_OUT" 2>/dev/null; then
    echo "ERROR: no layerDump in $ENG_OUT — engine may not have fired the dump"
    exit 1
fi

# Extract engine-generated tokenIds for teacher forcing the reference.
TEACHER_TOKENS=$("$PY" -c "
import json, sys
data = json.load(open('$ENG_OUT'))
tids = data.get('tokenIds') or []
print(json.dumps(tids))
" 2>/dev/null)
if [ -z "$TEACHER_TOKENS" ] || [ "$TEACHER_TOKENS" = "[]" ]; then
    echo "[probe] WARNING: no tokenIds in engine dump — reference will only have prefill-end"
    TEACHER_TOKENS="[]"
else
    NTOK=$(echo "$TEACHER_TOKENS" | "$PY" -c "import json,sys; print(len(json.load(sys.stdin)))")
    echo "[probe] teacher-forcing reference with $NTOK engine-generated tokens"
fi

# Produce the PyTorch reference (slow; --device cpu for 8 GB GPUs).
# Cache key includes prompt+token count, so a different prompt or token count
# regenerates the reference automatically.
if [ ! -f "$REF_OUT" ] || [ "$REGEN_REF" = "1" ]; then
    echo "[probe] running PyTorch reference (CPU, bf16 — will take minutes) ..."
    "$PY" scripts/divergence_probe.py \
        --prompt "$PROMPT" \
        --model "$REF_MODEL" \
        --out "$REF_OUT" \
        --teacher-tokens "$TEACHER_TOKENS" \
        --dump-tags "$DUMP_TAGS_JSON"
else
    echo "[probe] reusing existing $REF_OUT (set REGEN_REF=1 to regenerate)"
fi

# Compare (multi-tag, with HailMary's layer_types for SSM/FA annotation).
echo ""
"$PY" scripts/compare_dumps.py \
    --engine "$ENG_OUT" \
    --reference "$REF_OUT" \
    --model-dir "$HAILMARY_DIR"
