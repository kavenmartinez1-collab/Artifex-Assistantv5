#!/bin/bash
# run-test.sh — Send a test prompt to the running WebGPU engine and get results.
#
# Usage:
#   ./scripts/run-test.sh                          # default "Hello Artifex?" (matches PyTorch ref)
#   ./scripts/run-test.sh "What is the capital of France?"
#
# Prerequisites: model must be loaded in the browser at http://127.0.0.1:5173

PROMPT="${1:-Hello Artifex?}"
SERVER="http://127.0.0.1:5173"
MAX_TOKENS="${2:-50}"

echo "=== Sending test: \"$PROMPT\" (max_tokens=$MAX_TOKENS, temp=0) ==="

# Clear old results
curl -s -X POST "$SERVER/api/debug" \
  -H "Content-Type: application/json" \
  -d '{"pending":true}' > /dev/null 2>&1

# Queue the test
curl -s -X POST "$SERVER/api/test" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"$PROMPT\",\"temperature\":0,\"maxTokens\":$MAX_TOKENS}" > /dev/null 2>&1

echo "Queued. Waiting for inference..."

# Poll for result (up to 240s)
for i in $(seq 1 120); do
    sleep 2
    RESULT=$(curl -s "$SERVER/api/debug" 2>/dev/null)

    if echo "$RESULT" | grep -q '"output"'; then
        echo ""
        echo "============================================"
        echo "  GENERATION RESULT"
        echo "============================================"

        # Extract fields with python
        python3 -c "
import json, sys
d = json.loads('''$RESULT''')
print(f\"  Prompt:  {d['prompt']}\")
print(f\"  Output:  {d['output'][:200]}\")
print(f\"  Tokens:  {d['tokens']} | {d['tokPerSec']:.1f} tok/s | {d['stopReason']}\")
print(f\"  Prompt tokens: {d['promptTokens']}\")
print()

# Show debug logs (forward pass internals)
logs = d.get('consoleLogs', [])
debug_lines = [l for l in logs if 'DEBUG' in l or 'LAYER' in l or 'REF' in l or 'SSM' in l or 'L2' in l]
if debug_lines:
    print('============================================')
    print('  DEBUG VALUES (WebGPU)')
    print('============================================')
    for line in debug_lines:
        print(f'  {line}')
    print()

    # Compare with PyTorch reference
    print('============================================')
    print('  PYTORCH REFERENCE (original f32 model)')
    print('============================================')
    print('  embed[0:8]:    [-0.0071, 0.0082, -0.0001, -0.0008, 0.0007, -0.0074, -0.0026, 0.0029]')
    print('  L0 QKV Q[0:8]: [ 0.1381, 0.2031, 0.4639, 0.4961, 0.1077, -0.1190, -3.1806, 0.3653]')
    print('  L0 QKV K[0:8]: [ 1.9249, 2.0204, 0.3740, 0.8411, 0.6689, 2.5954, -0.7563, 0.7148]')
    print('  L0 QKV V[0:8]: [-0.3856, 4.9916, -3.6423, -3.2781, -5.9483, 1.9273, 2.0068, -0.5256]')
    print('  L0 output[0:8]:[-0.0384, 0.0628, 0.0014, -0.0143, 0.0133, 0.0258, 0.0468, -0.0352]')
    print()
else:
    print('  (No debug lines found — is isDebug enabled in forward-pass.ts?)')
    print('  All console logs:')
    for line in logs[-20:]:
        print(f'    {line}')
" 2>/dev/null || echo "$RESULT"

        exit 0
    fi
    printf "."
done

echo ""
echo "ERROR: Timed out (240s). Is the model loaded?"
exit 1
