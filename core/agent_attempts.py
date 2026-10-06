"""
Artifex Assistant V5 — best-of-N: several independent attempts, keep one.

Sampling is noisy: on the same task a small model is right on one run and
wrong on the next. With RunConfig.attempts = N the goal is attempted N times
from the same starting commit, each in a fresh context. Every attempt's
result is kept as a git ref, the workspace is reset between attempts, and
the winner is chosen by:

  1. the automatic checks (core/verify.py): an attempt that leaves
     regressions or broken imports loses to one that does not;
  2. a judge: the model, in a fresh context, reads the goal and every
     surviving attempt's diff and summary and names the best one.

Safety: this needs a git repository whose worktree is clean when the run
starts, because it resets the worktree between attempts. Otherwise it falls
back to a single normal run. Losing attempts stay reachable under
refs/artifex/attempts/<n>.
"""

import re
import subprocess
import time
from typing import List, Optional

from core.agent_loop import AgentEvent, GenerationAborted, RunResult

MAX_DIFF_CHARS = 12000
JUDGE_MAX_TOKENS = 6144


def _git(root: str, *args, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


def _repo_root() -> Optional[str]:
    import os
    from tools.agent_tools import _find_git_root
    return _find_git_root(os.getcwd())


def _clean(root: str) -> bool:
    r = _git(root, "status", "--porcelain")
    return r.returncode == 0 and not r.stdout.strip()


class _Attempt:
    def __init__(self, n: int):
        self.n = n
        self.result: Optional[RunResult] = None
        self.sha = ""
        self.diff = ""
        self.clean_checks = True
        self.check_text = ""


def run_attempts(runner, goal: str, history: list) -> RunResult:
    cfg = runner.config
    n = max(1, int(cfg.attempts))
    root = _repo_root()
    if n == 1 or not root or not _clean(root):
        if n > 1:
            runner.emit(AgentEvent("attempts", text="best-of-N needs a clean git "
                                   "worktree; running once"))
        return runner._run_once(goal, history)

    start = _git(root, "rev-parse", "HEAD").stdout.strip()
    t0 = time.monotonic()
    attempts: List[_Attempt] = []
    host_emit = runner.emit

    def attempt_emit(ev):
        if ev.kind in ("done", "stopped"):
            ev.kind = "attempt_" + ev.kind
        host_emit(ev)

    runner.emit = attempt_emit
    try:
        for i in range(1, n + 1):
            if runner.control.stop_requested:
                break
            if cfg.wall_clock_s and time.monotonic() - t0 >= cfg.wall_clock_s:
                break
            host_emit(AgentEvent("attempt", text=f"attempt {i} of {n}"))
            a = _Attempt(i)
            res = runner._run_once(goal, [])
            a.result = res
            # Anything the attempt wrote without an edit block (python/shell
            # writes) is committed too, so the attempt is one exact snapshot.
            if not _clean(root):
                _git(root, "add", "-A")
                _git(root, "commit", "-q", "-m", f"[agent] attempt {i}: uncommitted writes")
            a.sha = _git(root, "rev-parse", "HEAD").stdout.strip()
            _git(root, "update-ref", f"refs/artifex/attempts/{i}", a.sha)
            a.diff = _git(root, "diff", start, a.sha, timeout=60).stdout
            v = runner._verifier
            if v is not None and v.edited:
                rep = v.final_check()
                a.clean_checks = rep.clean
                a.check_text = rep.render()
            attempts.append(a)
            _git(root, "reset", "-q", "--hard", start)
            if runner.control.stop_requested:
                break
    finally:
        runner.emit = host_emit

    winner = _choose(runner, goal, attempts)
    if winner is None:
        host_emit(AgentEvent("stopped", reason="no attempt finished"))
        return RunResult(status="stopped:no_attempt", rounds=0,
                         actions_run=runner._actions_run, history=history)
    _git(root, "reset", "-q", "--hard", winner.sha)
    res = winner.result
    summary = (f"[attempt {winner.n} of {len(attempts)} kept] " + (res.summary or "")).strip()
    history.append({"role": "user", "content": goal})
    history.append({"role": "assistant", "content": summary})
    host_emit(AgentEvent("done" if res.status == "done" else "stopped",
                         summary=summary, reason="" if res.status == "done" else res.status))
    return RunResult(status=res.status, rounds=sum(a.result.rounds for a in attempts),
                     actions_run=runner._actions_run, summary=summary, history=history)


def _choose(runner, goal: str, attempts: List[_Attempt]) -> Optional[_Attempt]:
    done = [a for a in attempts if a.result and a.result.status == "done" and a.diff.strip()]
    pool = [a for a in done if a.clean_checks] or done
    if not pool:
        pool = [a for a in attempts if a.diff.strip()] or attempts
    if len(pool) <= 1:
        return pool[0] if pool else None
    if len({a.diff for a in pool}) == 1:
        return pool[0]
    pick = _judge(runner, goal, pool)
    return pick or pool[0]


_JUDGE_ASK = """Several independent attempts were made at the GOAL below. Each attempt's
summary and its full diff follow. Decide which attempt best accomplishes the GOAL:
correct, complete, fixes the actual cause rather than a symptom, and changes nothing
it should not. Read the diffs, not just the summaries.

GOAL:
{goal}

{attempts}

Reply with your reasoning, then a last line of exactly: BEST: <attempt number>"""


def _judge(runner, goal: str, pool: List[_Attempt]) -> Optional[_Attempt]:
    from core.inference import strip_think_blocks
    per = max(2000, MAX_DIFF_CHARS // len(pool))
    blocks = []
    for a in pool:
        diff = a.diff if len(a.diff) <= per else a.diff[:per] + "\n[... diff truncated ...]"
        checks = f"\nAutomatic checks: {a.check_text}" if a.check_text else ""
        blocks.append(f"=== ATTEMPT {a.n} ===\nSummary: {(a.result.summary or '')[:600]}"
                      f"{checks}\nDiff:\n{diff}")
    messages = [
        {"role": "system", "content": "You review code changes. Be strict and concrete."},
        {"role": "user", "content": _JUDGE_ASK.format(goal=goal.strip(),
                                                      attempts="\n\n".join(blocks))},
    ]
    parts = []

    def on_tok(t):
        if runner.control.stop_requested:
            raise GenerationAborted()
        parts.append(t)

    try:
        resp = runner.engine.generate_streaming(
            messages, max_tokens=JUDGE_MAX_TOKENS, temperature=0.3, on_token=on_tok,
            **runner._engine_gen_kwargs())
    except Exception:
        return None
    text = strip_think_blocks(resp if isinstance(resp, str) and resp else "".join(parts))
    m = re.findall(r"BEST:\s*(?:attempt\s*)?(\d+)", text, re.I)
    if not m:
        return None
    n = int(m[-1])
    pick = next((a for a in pool if a.n == n), None)
    runner.emit(AgentEvent("judge", text=f"judge picked attempt {n}" if pick
                           else f"judge named unknown attempt {n}"))
    return pick
