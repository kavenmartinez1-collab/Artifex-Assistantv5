"""
Agent bench — harness that drives the REAL Artifex agent stack
(core.agent_loop.AgentRunner + tools.agent_tools + core.sandbox) against a
live llama-server, records everything, and grades outcomes.

Design notes:
  * The engine is the production LlamaCppEngine; load() adopts an already-
    running llama-server on the configured port, so bench runs reuse the
    resident 35B instead of relaunching it.
  * Each scenario gets a fresh workspace that is git-init'd as its OWN repo.
    tools.agent_tools.git_commit_edit walks up to the nearest .git — without
    the per-workspace repo, agent auto-commits would land in the Artifex
    monorepo's history.
  * FULL_AUTO autonomy with an approval callback that denies CRITICAL-risk
    actions and auto-acknowledges breaker/gate pauses. Human gates are
    disabled (interval/action/risk = 0); the circuit breaker stays on so we
    can MEASURE runaway loops instead of silently suffering them.
  * sampling={} reproduces the pre-fix legacy behavior (only temperature in
    the request → llama-server compiled defaults apply). sampling=None uses
    the engine's new explicit DEFAULT_SAMPLING. A dict = full control.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime

from core.agent_loop import (
    AgentRunner, AutonomyLevel, Decision, RunConfig, RunControl,
)
from core.engine_llama_cpp import LlamaCppEngine
from core.config import get_llama_cpp_model_config
from core.prompts import build_assistant_prompt
from core.sampling import get_preset
from core.sandbox import RiskLevel
from core.sandbox.circuit_breaker import CircuitBreaker
from core.sandbox.human_gate import GateState
from tools.agent_tools import extract_agent_actions, detect_done, get_assistant_tools_prompt

BENCH_ROOT = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.join(BENCH_ROOT, "workspace")
RESULTS_ROOT = os.path.join(BENCH_ROOT, "results")

MODEL_NAME = os.environ.get("ARTIFEX_BENCH_MODEL", "qwen3.6-35b-a3b")
BENCH_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════
# Configs under test
# ═══════════════════════════════════════════════════════════════════════════
# Each config: sampling (None → engine DEFAULT_SAMPLING; {} → legacy
# server-default leak; dict → explicit), temperature (used only when the
# sampling dict doesn't carry one), enable_thinking, max_tokens.

CONFIGS = {
    # What every Artifex llama.cpp request did BEFORE this tuning session:
    # only temperature=0.7 sent; llama-server fills in top_k=20/top_p=0.95/
    # min_p=0.05 (build-dependent).
    "legacy": {
        "sampling": {},
        "temperature": 0.7,
        "enable_thinking": True,
    },
    # The new engine default: balanced-shaped explicit chain, caller temp.
    "neutral": {
        "sampling": None,
        "temperature": 0.7,
        "enable_thinking": True,
    },
    # Qwen3-series recommended thinking config == the initial "agent" preset.
    "agent": {
        "sampling": get_preset("agent", seed=BENCH_SEED),
        "enable_thinking": True,
    },
    # Colder variant — agent chain at temp 0.3.
    "agent-cold": {
        "sampling": get_preset("agent", temperature=0.3, seed=BENCH_SEED),
        "enable_thinking": True,
    },
    # Greedy decode.
    "greedy": {
        "sampling": get_preset("deterministic"),
        "enable_thinking": True,
    },
    # Best-guess non-thinking config (Qwen non-think rec: temp .7 top_p .8).
    "agent-nothink": {
        "sampling": get_preset("agent", temperature=0.7, top_p=0.8,
                               seed=BENCH_SEED),
        "enable_thinking": False,
    },
    # Repetition guard variant — presence penalty per Qwen repetition advice.
    "agent-presence": {
        "sampling": get_preset("agent", presence_penalty=1.0, seed=BENCH_SEED),
        "enable_thinking": True,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Engine proxy — metrics capture around the production engine
# ═══════════════════════════════════════════════════════════════════════════

class EngineProxy:
    """Delegates to LlamaCppEngine, recording per-call timing/tokens/chars."""

    def __init__(self, engine):
        self._engine = engine
        self.calls = []

    def generate_streaming(self, messages, max_tokens, temperature,
                           on_token=None, enable_thinking=True, sampling=None):
        think_chars = [0]
        content_chars = [0]
        state = {"in_think": False}

        def counting_on_token(t):
            if t == "<think>":
                state["in_think"] = True
            elif t == "</think>":
                state["in_think"] = False
            elif state["in_think"]:
                think_chars[0] += len(t)
            else:
                content_chars[0] += len(t)
            if on_token:
                on_token(t)

        t0 = time.monotonic()
        err = None
        try:
            out = self._engine.generate_streaming(
                messages, max_tokens=max_tokens, temperature=temperature,
                on_token=counting_on_token, enable_thinking=enable_thinking,
                sampling=sampling,
            )
        except Exception as e:  # noqa: BLE001 — record, then re-raise
            err = repr(e)
            raise
        finally:
            stats = dict(getattr(self._engine, "_last_gen_stats", {}) or {})
            self.calls.append({
                "wall_s": round(time.monotonic() - t0, 2),
                "prompt_tokens": stats.get("prompt_tokens"),
                "completion_tokens": stats.get("completion_tokens"),
                "finish_reason": stats.get("finish_reason"),
                "think_chars": think_chars[0],
                "content_chars": content_chars[0],
                "n_messages": len(messages),
                "error": err,
            })
        return out

    def get_context_size(self):
        return self._engine.get_context_size()

    def __getattr__(self, name):
        return getattr(self._engine, name)


def make_engine():
    engine = LlamaCppEngine(MODEL_NAME, get_llama_cpp_model_config(MODEL_NAME))
    engine.load()  # adopts the running server if healthy
    return engine


# ═══════════════════════════════════════════════════════════════════════════
# Workspace lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def _git(ws, *args):
    return subprocess.run(["git", "-C", ws, *args], capture_output=True,
                          text=True, timeout=15)


def make_workspace(run_id, name):
    ws = os.path.join(WORKSPACE_ROOT, run_id, name)
    os.makedirs(ws)
    _git(ws, "init", "-q")
    _git(ws, "config", "user.email", "bench@artifex.local")
    _git(ws, "config", "user.name", "Artifex Bench")
    return ws


def commit_setup(ws):
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "scenario setup")


# ═══════════════════════════════════════════════════════════════════════════
# System prompt (static per workspace → llama-server prefix cache stays warm)
# ═══════════════════════════════════════════════════════════════════════════

def make_system_prompt(ws):
    return build_assistant_prompt(
        get_assistant_tools_prompt(),
        cwd=ws,
        workspace_text=ws,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Micro probe runner (single-turn, no execution)
# ═══════════════════════════════════════════════════════════════════════════

def run_micro_suite(proxy, cfg, probes, ws, log=print):
    from core.prompts import build_autonomous_prompt
    base_prompt = make_system_prompt(ws)
    results = []
    for probe in probes:
        # Frame exactly like AgentRunner round 1: autonomous preamble + GOAL
        # in the system prompt, goal repeated as the user message. Probes are
        # meant to predict agent-run behavior, so they see the same context
        # (including the @done documentation, which lives in the preamble).
        messages = [
            {"role": "system",
             "content": build_autonomous_prompt(base_prompt, probe["prompt"])},
            {"role": "user", "content": probe["prompt"]},
        ]
        t0 = time.monotonic()
        try:
            resp = proxy.generate_streaming(
                messages,
                max_tokens=cfg.get("max_tokens", 3072),
                temperature=cfg.get("temperature", 0.7),
                enable_thinking=cfg["enable_thinking"],
                sampling=cfg["sampling"],
            )
            error = None
        except Exception as e:  # noqa: BLE001
            resp, error = "", repr(e)
        actions = extract_agent_actions(resp)
        done = detect_done(resp)
        score, note = probe["check"](actions, resp, done)
        call = proxy.calls[-1] if proxy.calls else {}
        row = {
            "probe": probe["name"],
            "score": score,
            "note": note,
            "actions": [a.type for a in actions],
            "wall_s": round(time.monotonic() - t0, 2),
            "completion_tokens": call.get("completion_tokens"),
            "think_chars": call.get("think_chars"),
            "content_chars": call.get("content_chars"),
            "finish_reason": call.get("finish_reason"),
            "error": error,
            "response": resp,
        }
        results.append(row)
        log(f"    [{probe['name']}] {score:.1f}  {note}  "
            f"({row['wall_s']}s, {row['completion_tokens']} tok)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Scenario runner (full agent loop)
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(proxy, cfg, scenario, run_id, log=print,
                 max_rounds=8, wall_clock_s=480.0):
    ws = make_workspace(run_id, scenario["name"])
    scenario["setup"](ws)
    commit_setup(ws)

    events = []
    action_records = []

    def emit(ev):
        rec = {
            "kind": ev.kind, "round": ev.round,
            "reason": ev.reason, "summary": ev.summary,
            "success": ev.success,
        }
        if ev.kind in ("assistant_message",):
            rec["text"] = ev.text
        if ev.action is not None:
            rec["action_type"] = ev.action.type
            rec["action_display"] = ev.action.display
        if ev.kind == "action_result":
            rec["output"] = (ev.output or "")[:2000]
            action_records.append({
                "type": ev.action.type, "ok": bool(ev.success),
                "display": ev.action.display,
            })
        if ev.kind not in ("assistant_chunk", "thinking_chunk"):
            events.append(rec)

    def approval(action, decision, reason):
        if decision is not None and decision.risk_level == RiskLevel.CRITICAL:
            events.append({"kind": "bench_denied_critical",
                           "action": getattr(action, "display", "")})
            return Decision.DENY
        events.append({"kind": "bench_auto_approved", "reason": reason})
        return Decision.APPROVE

    run_cfg = RunConfig(
        autonomy=AutonomyLevel.FULL_AUTO,
        max_rounds=max_rounds,
        wall_clock_s=wall_clock_s,
        max_tokens=cfg.get("max_tokens", 3072),
        temperature=cfg.get("temperature", 0.7),
        enable_thinking=cfg["enable_thinking"],
        sampling=cfg["sampling"],
        framing=True,
    )

    runner = AgentRunner(
        proxy,
        build_system_prompt=lambda: make_system_prompt(ws),
        emit=emit,
        request_approval=approval,
        config=run_cfg,
        control=RunControl(),
        gate=GateState(interval=0, max_actions=0, risk_budget=0),
        breaker=CircuitBreaker(),
    )

    history = []
    calls_before = len(proxy.calls)
    prev_cwd = os.getcwd()
    t0 = time.monotonic()
    try:
        os.chdir(ws)
        result = runner.run(scenario["goal"], history)
    finally:
        os.chdir(prev_cwd)
    wall = round(time.monotonic() - t0, 2)

    calls = proxy.calls[calls_before:]
    score, note = scenario["grade"](ws, result, action_records)

    row = {
        "scenario": scenario["name"],
        "score": score,
        "note": note,
        "status": result.status,
        "rounds": result.rounds,
        "actions_run": result.actions_run,
        "failed_actions": sum(1 for r in action_records if not r["ok"]),
        "action_types": [r["type"] for r in action_records],
        "done_marker": result.status == "done" and bool(result.summary),
        "wall_s": wall,
        "gen_wall_s": round(sum(c["wall_s"] for c in calls), 2),
        "completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
        "final_prompt_tokens": calls[-1]["prompt_tokens"] if calls else None,
        "think_chars": sum(c["think_chars"] for c in calls),
        "content_chars": sum(c["content_chars"] for c in calls),
        "workspace": ws,
        "events": events,
        "calls": calls,
        "history": history,
    }
    log(f"  [{scenario['name']}] score={score:.2f} status={result.status} "
        f"rounds={result.rounds} actions={result.actions_run} "
        f"fails={row['failed_actions']} {wall}s "
        f"({row['completion_tokens']} tok)  — {note}")
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Suite orchestration + persistence
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Round-trip through JSON with a fallback for stray types."""
    return json.loads(json.dumps(obj, default=repr))


def run_config(config_name, suite="both", scenario_names=None,
               probe_names=None, max_rounds=8, wall_clock_s=480.0,
               tag="", log=print):
    from agent_bench.scenarios import MICRO_PROBES, SCENARIOS

    cfg = dict(CONFIGS[config_name])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f"{stamp}_{config_name}" + (f"_{tag}" if tag else "")
    out_dir = os.path.join(RESULTS_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)

    log(f"=== config {config_name} (run {run_id}) ===")
    log(f"    sampling={cfg['sampling']}  thinking={cfg['enable_thinking']}")

    proxy = EngineProxy(make_engine())

    summary = {
        "run_id": run_id, "config": config_name, "tag": tag,
        "model": MODEL_NAME,
        "cfg": _sanitize(cfg),
        "max_rounds": max_rounds, "wall_clock_s": wall_clock_s,
        "started": stamp,
    }

    if suite in ("micro", "both"):
        probes = [p for p in MICRO_PROBES
                  if not probe_names or p["name"] in probe_names]
        log(f"-- micro suite ({len(probes)} probes)")
        micro_ws = make_workspace(run_id, "_micro")
        micro = run_micro_suite(proxy, cfg, probes, micro_ws, log=log)
        with open(os.path.join(out_dir, "micro.json"), "w",
                  encoding="utf-8") as f:
            json.dump(_sanitize(micro), f, indent=1)
        summary["micro_avg"] = round(
            sum(r["score"] for r in micro) / max(len(micro), 1), 3)
        summary["micro"] = [{k: r[k] for k in
                             ("probe", "score", "note", "wall_s",
                              "completion_tokens")} for r in micro]
        log(f"-- micro avg: {summary['micro_avg']:.3f}")

    if suite in ("full", "both"):
        scens = [s for s in SCENARIOS
                 if not scenario_names or s["name"] in scenario_names]
        log(f"-- full suite ({len(scens)} scenarios)")
        rows = []
        for scenario in scens:
            try:
                row = run_scenario(proxy, cfg, scenario, run_id, log=log,
                                   max_rounds=max_rounds,
                                   wall_clock_s=wall_clock_s)
            except Exception:  # noqa: BLE001 — one scenario must not kill the run
                log(f"  [{scenario['name']}] CRASHED:\n{traceback.format_exc()}")
                row = {"scenario": scenario["name"], "score": 0.0,
                       "note": "harness exception",
                       "crash": traceback.format_exc()}
            rows.append(row)
            with open(os.path.join(out_dir, f"{scenario['name']}.json"),
                      "w", encoding="utf-8") as f:
                json.dump(_sanitize(row), f, indent=1)
        summary["task_avg"] = round(
            sum(r["score"] for r in rows) / max(len(rows), 1), 3)
        summary["scenarios"] = [{k: r.get(k) for k in
                                 ("scenario", "score", "note", "status",
                                  "rounds", "actions_run", "failed_actions",
                                  "wall_s", "completion_tokens")}
                                for r in rows]
        clean_stops = sum(1 for r in rows if r.get("status") == "done")
        summary["clean_stop_rate"] = round(clean_stops / max(len(rows), 1), 3)
        log(f"-- task avg: {summary['task_avg']:.3f}  "
            f"clean stops: {clean_stops}/{len(rows)}")

    with open(os.path.join(out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(_sanitize(summary), f, indent=1)
    with open(os.path.join(RESULTS_ROOT, "index.jsonl"), "a",
              encoding="utf-8") as f:
        light = {k: v for k, v in summary.items()
                 if k not in ("micro", "scenarios")}
        f.write(json.dumps(light) + "\n")
    log(f"=== {config_name} done -> {out_dir}")
    return summary
