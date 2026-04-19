#!/bin/bash
# run-probe-audit.sh — multi-tag probe + per-tensor audit on suspect layers.
#
# Builds on run-probe.sh by adding `auditLayers` to the engine request and
# `--audit-layers` to the reference probe. The engine dumps every linear
# projection's last-row output for each audited layer; the reference does the
# same via auto-discovered nn.Linear hooks. compare_dumps.py aligns engine
# short-names (lin-qkv, q, gate, ...) to HF dotted paths via the per-family
# label map (scripts/compare_label_map.qwen35.json by default).
#
# Default audit set targets the persistent-weak layers identified in
# project_qwen35_divergence_2026-04-18.md: L21 (SSM, 0.69 cosine across all
# decode steps), L27 (FA, V-outliers), L28-L31 (deep distillation washout).
#
# To onboard a new model family (e.g., gemma4):
#   1. Drop scripts/compare_label_map.gemma4.json (same shape, remapped names).
#   2. Pass --label-map scripts/compare_label_map.gemma4.json to compare_dumps.py.
#   3. No engine or reference changes needed — labels are auto-discovered.
#
# Usage:
#   ./scripts/run-probe-audit.sh                          # default: "Hello Artifex?", L21/27/28-31
#   ./scripts/run-probe-audit.sh "What is Paris?"         # custom prompt
#   AUDIT_LAYERS='[21]' ./scripts/run-probe-audit.sh      # narrow audit

set -e

PROMPT="${1:-Hello Artifex?}"
SERVER="http://127.0.0.1:5173"
MAX_TOKENS="${2:-60}"
HAILMARY_DIR="${HAILMARY_DIR:-models/qwen3.5-9b-HailMary}"
REF_MODEL="${REF_MODEL:-models/qwen3.5-9b}"
ENG_OUT="webgpu/debug-output.audit.json"
REF_OUT="reference_dump.audit.json"
PY="${PY:-./venv/Scripts/python.exe}"
LABEL_MAP="${LABEL_MAP:-scripts/compare_label_map.qwen35.json}"
AUDIT_LAYERS="${AUDIT_LAYERS:-[21,27,28,29,30,31]}"

DUMP_TAGS_JSON='{"prefill-end":0,"decode-1":1,"decode-10":10,"decode-50":50}'

echo "=== Probe + audit: \"$PROMPT\" (maxTokens=$MAX_TOKENS, audit=$AUDIT_LAYERS) ==="

# Clear stale dump
curl -s -X POST "$SERVER/api/debug" -H "Content-Type: application/json" \
  -d '{"pending":true}' > /dev/null 2>&1 || true

# Queue the test with auditLayers set.
curl -s -X POST "$SERVER/api/test" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"temperature\":0,\"maxTokens\":$MAX_TOKENS,\"dumpStats\":true,\"decodeDumpSteps\":[1,10,50],\"auditLayers\":$AUDIT_LAYERS}" \
  > /dev/null 2>&1

echo "[probe] test queued — waiting for engine dump (audit adds GPU readbacks per projection per dump tag) ..."
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

cp webgpu/debug-output.json "$ENG_OUT"
echo "[probe] saved engine dump -> $ENG_OUT"

# Pull tokenIds for teacher forcing (so reference sees the same context).
ENG_TOKS=$("$PY" -c "
import json
print(json.dumps(json.load(open('$ENG_OUT')).get('tokenIds') or []))
")
echo "[probe] teacher tokens: $(echo "$ENG_TOKS" | "$PY" -c "import sys,json; print(len(json.load(sys.stdin)))") engine tokens"

# Run reference with same audit-layers list. Pass --audit-layers as a JSON array.
"$PY" scripts/divergence_probe.py \
    --prompt "$PROMPT" \
    --model "$REF_MODEL" \
    --out "$REF_OUT" \
    --teacher-tokens "$ENG_TOKS" \
    --dump-tags "$DUMP_TAGS_JSON" \
    --audit-layers "$AUDIT_LAYERS"

echo ""
"$PY" scripts/compare_dumps.py \
    --engine "$ENG_OUT" \
    --reference "$REF_OUT" \
    --model-dir "$HAILMARY_DIR" \
    --label-map "$LABEL_MAP"
