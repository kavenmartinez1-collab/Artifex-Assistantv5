"""
Agent bench — scenario and probe definitions for tuning Qwen3.6-35B-A3B
(and any future model) as an autonomous Artifex agent.

Two suites:

MICRO_PROBES — single-turn format-discipline checks. One generation each, no
tool execution; graded purely on what tools.agent_tools.extract_agent_actions
parses out of the response. Fast enough to sweep sampler configs.

SCENARIOS — multi-round tasks driven through the real core.agent_loop
AgentRunner with tools executing in a throwaway git-init'd workspace. Graded
programmatically on workspace end-state (the Sonnet-agent bar: did the task
actually get done).

Every grader returns (score 0.0-1.0, note). Scores are designed so partial
credit reflects genuinely useful partial behavior, not politeness.
"""

import json
import os
import re
import subprocess
import sys


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

_PYTHON = sys.executable  # harness runs under ./venv/Scripts/python.exe


def _write(ws, rel, content):
    path = os.path.join(ws, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _read(ws, rel):
    path = os.path.join(ws, rel)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _run_py(ws, *args, timeout=20):
    """Run the venv python in the workspace. Returns (rc, output)."""
    try:
        r = subprocess.run(
            [_PYTHON, *args], cwd=ws, capture_output=True, text=True,
            timeout=timeout, encoding="utf-8", errors="replace",
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:  # noqa: BLE001 — grader must never crash the bench
        return -1, f"grader error: {e}"


def _actions_of(actions, *types):
    return [a for a in actions if a.type in types]


def _shell_text(actions):
    return "\n".join(a.content for a in _actions_of(actions, "shell")).lower()


def _python_text(actions):
    return "\n".join(a.content for a in _actions_of(actions, "python"))


# ═══════════════════════════════════════════════════════════════════════════
# MICRO PROBES — single-turn format discipline
#   check(actions, text, done) -> (score, note)
# ═══════════════════════════════════════════════════════════════════════════

def _probe_read_check(actions, text, done):
    reads = [a for a in _actions_of(actions, "read_file", "grep")
             if "config.ini" in a.content]
    locates = [a for a in _actions_of(actions, "glob")
               if "config.ini" in a.content]
    banned_shell = any(k in _shell_text(actions)
                       for k in ("cat ", "type ", "get-content", "gc "))
    banned_py = "open(" in _python_text(actions)
    if reads and not banned_shell and not banned_py:
        return 1.0, "used read_file/grep"
    if locates and not banned_shell and not banned_py:
        return 0.8, "located via @glob first (reasonable round 1)"
    if reads or locates:
        return 0.5, "right tool plus banned shell/python read"
    if banned_shell or banned_py:
        return 0.2, "read via shell/python instead of tool"
    return 0.0, "no read action found"


def _probe_glob_check(actions, text, done):
    globs = [a for a in _actions_of(actions, "glob") if ".py" in a.content]
    if globs:
        return 1.0, "used @glob"
    sh = _shell_text(actions)
    if any(k in sh for k in ("ls", "dir", "get-childitem", "find ")):
        return 0.3, "listed via shell instead of @glob"
    return 0.0, "no file-listing action"


def _probe_symbol_check(actions, text, done):
    if any("parse_header" in a.content
           for a in _actions_of(actions, "find_symbol")):
        return 1.0, "used @find_symbol"
    if any("parse_header" in a.content for a in _actions_of(actions, "grep")):
        return 0.7, "used @grep (acceptable)"
    if "parse_header" in _shell_text(actions):
        return 0.2, "shell grep instead of tool"
    return 0.0, "no symbol search action"


def _probe_write_check(actions, text, done):
    py = _python_text(actions)
    if "hello.py" in py and ("open(" in py or "write_text" in py or "with open" in py):
        return 1.0, "python block writes hello.py"
    if "hello.py" in _shell_text(actions):
        return 0.3, "wrote via shell redirect (prompt says python blocks)"
    return 0.0, "no file-writing action"


def _probe_edit_check(actions, text, done):
    edits = _actions_of(actions, "edit_file")
    for a in edits:
        parts = a.content.split("\x00")
        if len(parts) != 3:
            continue
        path, old, new = parts
        if "mathx.py" in path and "a - b" in old and "a + b" in new:
            return 1.0, "well-formed edit block with correct fix"
    if edits:
        return 0.5, "edit block parsed but wrong path/old/new"
    if "a + b" in _python_text(actions) or "a + b" in _shell_text(actions):
        return 0.2, "fixed via rewrite instead of edit block"
    return 0.0, "no parseable edit block"


def _probe_chat_check(actions, text, done):
    if not actions:
        return 1.0, "no spurious tool calls"
    return 0.0, f"spurious actions: {[a.type for a in actions]}"


def _probe_list_tools_check(actions, text, done):
    # Prompt rule: "Tool markers are LIVE ... when listing tools, use prose
    # or backticks." Any parsed action means the model tripped its own tools.
    if not actions:
        return 1.0, "listed tools without triggering them"
    return 0.0, f"live markers fired while listing: {[a.type for a in actions]}"


def _probe_run_check(actions, text, done):
    if "pytest" in _shell_text(actions) or "pytest" in _python_text(actions):
        return 1.0, "ran pytest"
    if _actions_of(actions, "architecture", "glob"):
        return 0.5, "explored first instead of running (defensible round 1)"
    if _actions_of(actions, "shell", "python"):
        return 0.3, "ran something, but not pytest"
    return 0.0, "no run action"


def _probe_done_check(actions, text, done):
    if done is not None and not actions:
        return 1.0, "clean @done"
    if done is not None:
        return 0.5, "@done but with stray actions"
    return 0.0, "no @done marker"


def _probe_restraint_check(actions, text, done):
    reads = _actions_of(actions, "read_file")
    others = [a for a in actions if a.type != "read_file"]
    if 1 <= len(reads) <= 2 and not others:
        return 1.0, f"{len(reads)} read_file action(s), nothing else"
    if reads:
        return 0.4, f"reads plus extra actions: {[a.type for a in others]}"
    return 0.0, "no read actions"


MICRO_PROBES = [
    {
        "name": "read_file_discipline",
        "prompt": "What port is configured in config.ini?",
        "check": _probe_read_check,
    },
    {
        "name": "glob_discipline",
        "prompt": "List all Python files under src/.",
        "check": _probe_glob_check,
    },
    {
        "name": "symbol_search",
        "prompt": "Find where the function parse_header is defined in this project.",
        "check": _probe_symbol_check,
    },
    {
        "name": "write_new_file",
        "prompt": "Create a file hello.py that prints exactly: hello artifex",
        "check": _probe_write_check,
    },
    {
        "name": "edit_block_format",
        "prompt": (
            "The file mathx.py contains exactly:\n\n"
            "def add(a, b):\n"
            "    return a - b\n\n"
            "It has a bug — add() should add, not subtract. "
            "Fix it using an edit block."
        ),
        "check": _probe_edit_check,
    },
    {
        "name": "no_tool_chat",
        "prompt": "Explain in one sentence what a mutex is. Do not use any tools.",
        "check": _probe_chat_check,
    },
    {
        "name": "list_tools_trap",
        "prompt": "What tools do you have available? Just list them for me.",
        "check": _probe_list_tools_check,
    },
    {
        "name": "run_tests",
        "prompt": "Run the test suite with pytest and report the result.",
        "check": _probe_run_check,
    },
    {
        "name": "done_marker",
        "prompt": ("Everything is already finished — there is nothing to do. "
                   "End the run right now using your done marker."),
        "check": _probe_done_check,
    },
    {
        "name": "action_restraint",
        "prompt": "Read config.ini and sales.csv so we can discuss them.",
        "check": _probe_restraint_check,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# SCENARIOS — multi-round agent tasks
#   setup(ws)                      — create input files
#   goal                           — text handed to AgentRunner.run()
#   grade(ws, result, records) ->  (score, note)
#     result:  core.agent_loop.RunResult
#     records: harness per-action records [{type, ok, display}, ...]
# ═══════════════════════════════════════════════════════════════════════════

# ── s1: read a config value, write the answer ──────────────────────────────

def _s1_setup(ws):
    _write(ws, "config.ini",
           "[server]\nhost = 127.0.0.1\nport = 8443\n\n"
           "[auth]\ntoken = not-a-real-token\n")
    _write(ws, "README.txt",
           "Demo service. The default port is 9000 unless overridden in "
           "config.ini (the override is what actually applies).\n")


def _s1_grade(ws, result, records):
    ans = _read(ws, "answer.txt")
    if ans is None:
        return 0.0, "answer.txt not created"
    body = ans.strip()
    if body == "8443":
        return 1.0, "exact answer"
    if "8443" in body and "9000" not in body:
        return 0.8, f"correct but not clean: {body[:40]!r}"
    if "9000" in body:
        return 0.0, f"fell for the decoy: {body[:40]!r}"
    return 0.1, f"wrong content: {body[:40]!r}"


# ── s2: CSV aggregation to JSON ────────────────────────────────────────────

_S2_EXPECTED = {"north": 72.50, "south": 39.97, "west": 56.50}


def _s2_setup(ws):
    _write(ws, "sales.csv",
           "region,units,unit_price\n"
           "North,10,2.50\n"
           "North,4,10.00\n"
           "South,3,9.99\n"
           "West,7,3.00\n"
           "North,6,1.25\n"
           "South,20,0.50\n"
           "West,2,12.75\n"
           "West,5,2.00\n")


def _s2_grade(ws, result, records):
    raw = _read(ws, "summary.json")
    if raw is None:
        return 0.0, "summary.json not created"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0.1, "summary.json is not valid JSON"
    if not isinstance(data, dict):
        return 0.1, "summary.json is not an object"
    got = {str(k).lower(): v for k, v in data.items()}
    correct = 0
    for region, expected in _S2_EXPECTED.items():
        try:
            if abs(float(got.get(region)) - expected) < 0.02:
                correct += 1
        except (TypeError, ValueError):
            pass
    if correct == 3:
        return 1.0, "all three regions correct"
    return round(correct / 3, 2), f"{correct}/3 regions correct: {got}"


# ── s3: run failing test, fix the bug, verify ──────────────────────────────

def _s3_setup(ws):
    _write(ws, "stats.py",
           "def median(values):\n"
           "    s = sorted(values)\n"
           "    n = len(s)\n"
           "    return s[n // 2]\n")
    _write(ws, "test_stats.py",
           "from stats import median\n\n"
           "assert median([1, 3, 2]) == 2\n"
           "assert median([5]) == 5\n"
           "assert median([1, 2, 3, 4]) == 2.5, "
           "f'median([1,2,3,4]) = {median([1, 2, 3, 4])}'\n"
           "print('ALL TESTS PASSED')\n")


def _s3_grade(ws, result, records):
    rc, out = _run_py(ws, "test_stats.py")
    passed = rc == 0 and "ALL TESTS PASSED" in out
    ran_test = any(r["type"] in ("shell", "python")
                   and "test_stats" in r["display"] for r in records)
    score = 0.0
    notes = []
    if passed:
        score += 0.8
        notes.append("test passes")
    else:
        notes.append(f"test still fails: {out.strip()[:80]}")
    if ran_test:
        score += 0.2
        notes.append("agent ran the test itself")
    else:
        notes.append("agent never ran the test")
    return round(score, 2), "; ".join(notes)


# ── s4: code navigation report ─────────────────────────────────────────────

def _s4_setup(ws):
    _write(ws, "app/parser.py",
           "def parse_header(line):\n"
           "    key, _, value = line.partition(':')\n"
           "    return key.strip(), value.strip()\n\n\n"
           "def parse_body(text):\n"
           "    return text.splitlines()\n")
    _write(ws, "app/consumer.py",
           "from parser import parse_header\n\n\n"
           "def read_headers(lines):\n"
           "    return dict(parse_header(l) for l in lines if ':' in l)\n")
    _write(ws, "app/util.py",
           "from parser import parse_header\n\n\n"
           "def header_key(line):\n"
           "    return parse_header(line)[0]\n")
    _write(ws, "app/unrelated.py",
           "def helper():\n    return 42\n")


def _s4_grade(ws, result, records):
    report = _read(ws, "report.md")
    if report is None:
        return 0.0, "report.md not created"
    r = report.lower()
    score = 0.0
    notes = []
    if "parser.py" in r:
        score += 0.4
        notes.append("definition file named")
    if "consumer.py" in r:
        score += 0.3
        notes.append("consumer.py found")
    if "util.py" in r:
        score += 0.3
        notes.append("util.py found")
    if not notes:
        notes.append("report has none of the expected files")
    return round(score, 2), "; ".join(notes)


# ── s5: build a small CLI and use it ───────────────────────────────────────

def _s5_setup(ws):
    words = " ".join(f"alpha{i:02d}" for i in range(1, 58))  # exactly 57 words
    _write(ws, "sample.txt", words.replace("alpha20 ", "alpha20\n") + "\n")


def _s5_grade(ws, result, records):
    score = 0.0
    notes = []
    res = _read(ws, "result.txt")
    if res is not None and re.search(r"\b57\b", res.strip()):
        score += 0.5
        notes.append("result.txt has 57")
    else:
        notes.append(f"result.txt wrong/missing: {None if res is None else res.strip()[:30]!r}")
    if os.path.isfile(os.path.join(ws, "wordcount.py")):
        _write(ws, "_fresh_check.txt", "one two three four five six seven "
                                       "eight nine ten eleven twelve thirteen\n")
        rc, out = _run_py(ws, "wordcount.py", "_fresh_check.txt")
        if rc == 0 and re.search(r"\b13\b", out):
            score += 0.5
            notes.append("wordcount.py works on unseen file")
        else:
            notes.append(f"wordcount.py wrong on unseen file: {out.strip()[:60]!r}")
    else:
        notes.append("wordcount.py missing")
    return round(score, 2), "; ".join(notes)


# ── s6: surgical edit via the ```edit``` protocol ──────────────────────────

def _s6_setup(ws):
    _write(ws, "shapes.py",
           "import math\n\n\n"
           "def circle_area(r):\n"
           "    return 3.14 * r * r\n\n\n"
           "def square_area(side):\n"
           "    return side * side\n")
    _write(ws, "test_shapes.py",
           "import math\n\n"
           "from shapes import circle_area, square_area\n\n"
           "assert square_area(3) == 9\n"
           "assert abs(circle_area(2) - math.pi * 4) < 1e-9, "
           "f'circle_area(2) = {circle_area(2)}'\n"
           "print('ALL TESTS PASSED')\n")


def _s6_grade(ws, result, records):
    score = 0.0
    notes = []
    rc, out = _run_py(ws, "test_shapes.py")
    if rc == 0 and "ALL TESTS PASSED" in out:
        score += 0.5
        notes.append("test passes")
    else:
        notes.append(f"test fails: {out.strip()[:80]}")
    src = _read(ws, "shapes.py") or ""
    if "math.pi" in src and "3.14 *" not in src:
        score += 0.3
        notes.append("source uses math.pi")
    if any(r["type"] == "edit_file" and r["ok"] for r in records):
        score += 0.2
        notes.append("used the edit protocol")
    else:
        notes.append("edit protocol not used (rewrite?)")
    return round(score, 2), "; ".join(notes)


SCENARIOS = [
    {
        "name": "s1_read_answer",
        "setup": _s1_setup,
        "goal": ("Find the server port configured in config.ini and write just "
                 "that number to a new file answer.txt in the current "
                 "directory. Then finish."),
        "grade": _s1_grade,
    },
    {
        "name": "s2_csv_aggregate",
        "setup": _s2_setup,
        "goal": ("sales.csv has columns region,units,unit_price. Compute the "
                 "total revenue (units * unit_price) per region and write "
                 "summary.json in the current directory mapping each region "
                 "name to its total. Then finish."),
        "grade": _s2_grade,
    },
    {
        "name": "s3_bugfix",
        "setup": _s3_setup,
        "goal": ("test_stats.py currently fails. Run it, find the bug in "
                 "stats.py, fix the bug, and re-run the test to verify it "
                 "passes. Then finish."),
        "grade": _s3_grade,
    },
    {
        "name": "s4_code_nav",
        "setup": _s4_setup,
        "goal": ("Find where the function parse_header is defined in this "
                 "project and every file that uses it. Write report.md in the "
                 "current directory listing the definition file and each "
                 "usage file. Then finish."),
        "grade": _s4_grade,
    },
    {
        "name": "s5_build_cli",
        "setup": _s5_setup,
        "goal": ("Create wordcount.py: a script that prints the number of "
                 "whitespace-separated words in the file given as its first "
                 "command-line argument. Run it on sample.txt and write the "
                 "resulting number to result.txt. Then finish."),
        "grade": _s5_grade,
    },
    {
        "name": "s6_edit_refactor",
        "setup": _s6_setup,
        "goal": ("test_shapes.py fails because circle_area in shapes.py uses "
                 "3.14 instead of math.pi. Fix shapes.py using an edit block "
                 "(not a full rewrite), then run test_shapes.py to verify it "
                 "passes. Then finish."),
        "grade": _s6_grade,
    },
]
