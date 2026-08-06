"""
Re-score saved micro.json responses with the CURRENT graders + parser.

Model responses are stored verbatim in results/<run>/micro.json, so grader
recalibration and extract_agent_actions fixes can be re-applied offline
without re-running the model. Framing changes DO require a re-run (the
prompt differs); regrade only makes parser/grader changes comparable.

    ./venv/Scripts/python.exe -m agent_bench.regrade [run_dir ...]

With no args, regrades every run under agent_bench/results/.
"""

import json
import os
import sys

from agent_bench.harness import RESULTS_ROOT
from agent_bench.scenarios import MICRO_PROBES
from tools.agent_tools import extract_agent_actions, detect_done


def regrade_run(run_dir):
    micro_path = os.path.join(run_dir, "micro.json")
    if not os.path.isfile(micro_path):
        return None
    with open(micro_path, encoding="utf-8") as f:
        rows = json.load(f)
    probes = {p["name"]: p for p in MICRO_PROBES}
    out = []
    for row in rows:
        probe = probes.get(row["probe"])
        if probe is None:
            continue
        resp = row.get("response", "")
        actions = extract_agent_actions(resp)
        done = detect_done(resp)
        score, note = probe["check"](actions, resp, done)
        out.append({"probe": row["probe"], "old": row["score"],
                    "new": score, "note": note})
    avg_old = sum(r["old"] for r in out) / max(len(out), 1)
    avg_new = sum(r["new"] for r in out) / max(len(out), 1)
    return {"run": os.path.basename(run_dir), "rows": out,
            "avg_old": round(avg_old, 3), "avg_new": round(avg_new, 3)}


def main():
    targets = sys.argv[1:]
    if not targets:
        targets = sorted(
            os.path.join(RESULTS_ROOT, d) for d in os.listdir(RESULTS_ROOT)
            if os.path.isdir(os.path.join(RESULTS_ROOT, d)))
    for t in targets:
        res = regrade_run(t)
        if res is None:
            continue
        print(f"{res['run']}: micro {res['avg_old']:.3f} -> {res['avg_new']:.3f}")
        for r in res["rows"]:
            if abs(r["old"] - r["new"]) > 1e-9:
                print(f"    {r['probe']:<24} {r['old']:.1f} -> {r['new']:.1f}  {r['note'][:60]}")


if __name__ == "__main__":
    main()
