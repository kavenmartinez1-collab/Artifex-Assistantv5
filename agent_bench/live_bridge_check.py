"""
Live end-to-end check of the WebGPU backend.

Run from the repo root:  ./venv/Scripts/python.exe -m agent_bench.live_bridge_check

Starts the bridge (port 8790), waits for the browser session, then:
  1. streams a direct generation through WebGpuEngine, and
  2. drives a one-goal AgentRunner in a temp git workspace over the
     same engine — the full agent stack on the browser GPU.
"""

import os
import subprocess
import sys
import tempfile
import time

from core.engine_webgpu import WebGpuEngine
from core.agent_loop import AgentRunner, AutonomyLevel, RunConfig, RunControl
from core.sandbox import RiskLevel
from core.sandbox.circuit_breaker import CircuitBreaker
from core.sandbox.human_gate import GateState

# The 27B WebGPU session runs at ctx=2048 (MAX_ATTN_SEQ_LEN kernel cap), so
# the full assistant prompt (~1300 tok) would starve the rounds. This leg
# verifies TRANSPORT + loop mechanics; a compact agent prompt is the point.
_COMPACT_AGENT_PROMPT = """You are Artifex, an AI agent with LIVE tools.
Writing a marker EXECUTES it:
- @read_file("path") — read a file
- ```python ... ``` — run python (also how you write files)
- @done("summary") — declare the GOAL complete
Work one step per turn toward the GOAL. Never announce without acting.
CWD: {cwd}"""


def main():
    engine = WebGpuEngine(handshake_timeout=900)
    print("[live] bridge up on port", engine._ensure_bridge().port, flush=True)
    print("[live] waiting for a browser session (load a model in the WebGPU GUI)...",
          flush=True)
    engine.load(status_callback=lambda s: print("[live]", s, flush=True))
    info = engine._ensure_bridge().state.session_info
    print("[live] ATTACHED:", info, flush=True)

    print("[live] --- direct generation ---", flush=True)
    t0 = time.monotonic()
    out = engine.generate_streaming(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "Count from 1 to 5, one number per line."}],
        max_tokens=768, temperature=0.6,
        on_token=lambda t: print(t, end="", flush=True))
    dt = time.monotonic() - t0
    print(f"\n[live] direct gen ok in {dt:.1f}s; stats={engine._last_gen_stats}",
          flush=True)
    # Transport assertions only — content quality belongs to the agent leg.
    # (A 256-token cap once truncated a chatty thinker mid-count and failed
    # a content assert here, killing the bridge mid-verification.)
    assert out.strip(), "empty response over the bridge"
    assert engine._last_gen_stats.get("completion_tokens"), "no usage stats"

    print("[live] --- agent loop over WebGPU ---", flush=True)
    ws = tempfile.mkdtemp(prefix="webgpu_agent_")
    subprocess.run(["git", "-C", ws, "init", "-q"], check=False)
    with open(os.path.join(ws, "config.ini"), "w", encoding="utf-8") as f:
        f.write("[server]\nport = 7373\n")

    events = []

    def emit(ev):
        if ev.kind not in ("assistant_chunk", "thinking_chunk"):
            events.append(ev.kind)
            print(f"[agent] {ev.kind} r{ev.round} "
                  f"{(ev.action.display if ev.action else ev.reason or ev.summary or '')[:70]}",
                  flush=True)

    def approval(action, decision, reason):
        from core.agent_loop import Decision
        if decision is not None and decision.risk_level == RiskLevel.CRITICAL:
            return Decision.DENY
        return Decision.APPROVE

    runner = AgentRunner(
        engine,
        build_system_prompt=lambda: _COMPACT_AGENT_PROMPT.format(cwd=ws),
        emit=emit, request_approval=approval,
        config=RunConfig(autonomy=AutonomyLevel.FULL_AUTO, max_rounds=6,
                         wall_clock_s=600, max_tokens=512,
                         sampler_preset="agent"),
        control=RunControl(),
        gate=GateState(interval=0, max_actions=0, risk_budget=0),
        breaker=CircuitBreaker(),
    )
    prev = os.getcwd()
    os.chdir(ws)
    try:
        result = runner.run(
            "Read config.ini, find the configured port, write just that "
            "number to answer.txt, then finish.", [])
    finally:
        os.chdir(prev)

    answer_path = os.path.join(ws, "answer.txt")
    answer = open(answer_path, encoding="utf-8").read().strip() \
        if os.path.isfile(answer_path) else None
    print(f"[live] agent status={result.status} rounds={result.rounds} "
          f"actions={result.actions_run} answer={answer!r}", flush=True)
    ok = result.status == "done" and answer == "7373"
    print("[live] E2E:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
