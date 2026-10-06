"""
Artifex Assistant V5 — plan first, then one fresh context per step.

A long goal done in one loop piles every file read, test run and dead end
into one context; by the third sub-task a small model is reasoning over 40k
tokens of mostly stale output, and it reasons worse (and slower) there.
Here the model first writes a short step plan. Each step then runs as its
own AgentRunner loop over a FRESH history that holds only the goal, the plan
and what the earlier steps reported, so every step starts in the model's
sharp, fast range.

The run's verifier (core/verify.py), pinned notes and repo map are shared
across steps; only the conversation is reset. A plan of one step is a normal
run.
"""

import re
import time
from typing import List

from core.agent_loop import AgentEvent, GenerationAborted, RunResult

MAX_STEPS = 6
PLAN_MAX_TOKENS = 6144

_PLAN_SYSTEM = (
    "You plan work for an autonomous coding agent. You do not do the work. "
    "Reply with ONLY the plan in the requested format.")

_PLAN_ASK = """Split the GOAL below into the FEWEST steps that can each be done and checked on
their own (at most {max_steps}). A step is one coherent change: a separate requested
feature, a separate bug to find and fix. Do not split one change into "read", "edit"
and "test" steps; every step already includes reading, editing and checking.
If the GOAL is one coherent change, give exactly ONE step.

Reply with exactly this format and nothing else:
PLAN:
1. <the step, specific enough to do without re-reading the GOAL>
2. ...

{repo_map}GOAL:
{goal}"""

_STEP_GOAL = """{goal}

---
This GOAL is being done in {n} steps, each in a fresh context. The plan:
{plan}

{progress}YOUR STEP NOW: step {k} of {n}: {step}
Do ONLY this step (later steps are done separately), check that it works, then emit
@done("what you changed, in which files"). Earlier steps' changes are already on disk."""


def parse_plan(text: str) -> List[str]:
    body = text.split("PLAN:", 1)[1] if "PLAN:" in text else text
    steps = []
    for line in body.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s+(.+?)\s*$", line)
        if m:
            steps.append(m.group(2))
    return steps[:MAX_STEPS]


def make_plan(runner, goal: str) -> List[str]:
    """One bounded generation; [] on any failure (= run without a plan)."""
    from core.inference import strip_think_blocks
    rm = (runner._repo_map + "\n\n") if runner._repo_map else ""
    messages = [
        {"role": "system", "content": _PLAN_SYSTEM},
        {"role": "user", "content": _PLAN_ASK.format(max_steps=MAX_STEPS, repo_map=rm,
                                                      goal=goal.strip())},
    ]
    parts = []

    def on_tok(t):
        if runner.control.stop_requested:
            raise GenerationAborted()
        parts.append(t)

    try:
        resp = runner.engine.generate_streaming(
            messages, max_tokens=PLAN_MAX_TOKENS, temperature=runner.config.temperature,
            on_token=on_tok, **runner._engine_gen_kwargs())
    except GenerationAborted:
        return []
    except Exception:
        return []
    text = resp if isinstance(resp, str) and resp else "".join(parts)
    return parse_plan(strip_think_blocks(text))


def run_planned(runner, goal: str, history: list) -> RunResult:
    cfg = runner.config
    runner._t0 = time.monotonic()
    runner._actions_run = 0
    runner._verifier = None
    runner._notes = []
    runner._repo_map = runner._build_repo_map() if cfg.repo_map else ""

    steps = make_plan(runner, goal)
    if len(steps) < 2:
        runner.emit(AgentEvent("plan", text="single step: running without a plan"))
        return runner._run(goal, history, keep_state=True)

    plan_text = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
    runner.emit(AgentEvent("plan", text=plan_text))
    history.append({"role": "user", "content": goal})
    history.append({"role": "assistant", "content": "PLAN:\n" + plan_text})

    # A step's own done/stopped is not the run's: hosts print "done" as the
    # final summary, so steps report as step_done / step_stopped instead.
    host_emit = runner.emit

    def step_emit(ev):
        if ev.kind in ("done", "stopped"):
            ev.kind = "step_" + ev.kind
        host_emit(ev)

    runner.emit = step_emit
    try:
        return _run_steps(runner, goal, history, steps, plan_text, host_emit)
    finally:
        runner.emit = host_emit


def _run_steps(runner, goal, history, steps, plan_text, host_emit) -> RunResult:
    cfg = runner.config
    n = len(steps)
    budget = cfg.max_rounds * 2      # whole run; one step may use at most max_rounds
    used = 0
    reports: List[str] = []
    statuses: List[str] = []
    for k, step in enumerate(steps, 1):
        if runner.control.stop_requested:
            statuses.append("stopped:user")
            break
        if runner._over_wall_clock():
            statuses.append("stopped:timeout")
            break
        left = budget - used
        if left <= 0:
            statuses.append("stopped:max_rounds")
            break
        before = list(runner._verifier.edited) if runner._verifier else []
        progress = ("Done so far:\n" + "\n".join(reports) + "\n\n") if reports else ""
        sub_goal = _STEP_GOAL.format(goal=goal.strip(), n=n, plan=plan_text,
                                     progress=progress, k=k, step=step)
        runner.emit(AgentEvent("step", text=f"step {k}/{n}: {step}"))
        res = runner._run(sub_goal, [], keep_state=True, max_rounds=min(cfg.max_rounds, left))
        used += res.rounds
        edited = [f for f in (runner._verifier.edited if runner._verifier else [])
                  if f not in before]
        files = f" (edited: {', '.join(edited)})" if edited else ""
        summary = (res.summary or res.status).strip().replace("\n", " ")[:500]
        reports.append(f"- step {k} [{res.status}]{files}: {summary}")
        statuses.append(res.status)
        history.append({"role": "assistant",
                        "content": f"[step {k}/{n}] {step}\n-> {res.status}{files}: {summary}"})
        if res.status == "stopped:user":
            break

    final = "done" if statuses and all(s == "done" for s in statuses) and len(statuses) == n \
        else next((s for s in statuses if s != "done"), "stopped:incomplete")
    summary = "\n".join(reports)
    host_emit(AgentEvent("done" if final == "done" else "stopped",
                         summary=summary, reason="" if final == "done" else final))
    return RunResult(status=final, rounds=used, actions_run=runner._actions_run,
                     summary=summary, history=history)
