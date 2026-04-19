#!/bin/bash
# run-temp-test.sh — Generate at a specified temperature and dump the output text.
# Used to confirm whether the Chinese leak appears at temp>0 vs temp=0.
#
# Usage:
#   ./scripts/run-temp-test.sh "Hello Artifex?" 0.7 100

set -e

PROMPT="${1:-Hello Artifex?}"
TEMP="${2:-0.7}"
MAX_TOKENS="${3:-100}"
SERVER="http://127.0.0.1:5173"
PY="${PY:-./venv/Scripts/python.exe}"

echo "=== Temp test: \"$PROMPT\" temp=$TEMP maxTokens=$MAX_TOKENS ==="

# Clear stale dump
curl -s -X POST "$SERVER/api/debug" -H "Content-Type: application/json" \
  -d '{"pending":true}' > /dev/null 2>&1 || true

# Queue the test (no layer dump, just generation).
curl -s -X POST "$SERVER/api/test" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"temperature\":$TEMP,\"maxTokens\":$MAX_TOKENS}" \
  > /dev/null 2>&1

echo "[temp-test] queued — waiting ..."
for i in $(seq 1 60); do
    sleep 2
    RESULT=$(curl -s "$SERVER/api/debug" 2>/dev/null)
    if echo "$RESULT" | grep -q '"output"'; then
        echo "[temp-test] result received"
        break
    fi
    printf "."
done
echo ""

"$PY" scripts/inspect_output.py
