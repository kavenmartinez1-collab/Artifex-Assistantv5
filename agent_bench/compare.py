"""
Print a comparison table from results/index.jsonl, filtered by tag.

    ./venv/Scripts/python.exe -m agent_bench.compare sweep2
    ./venv/Scripts/python.exe -m agent_bench.compare sweep2 --probes
"""

import json
import os
import sys

from agent_bench.harness import RESULTS_ROOT


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    show_probes = "--probes" in sys.argv
    tag = args[0] if args else ""

    rows = []
    with open(os.path.join(RESULTS_ROOT, "index.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if not tag or d.get("tag") == tag:
                rows.append(d)

    print(f"{'config':<16} {'micro':>6} {'task':>6} {'stops':>6}   run")
    for d in rows:
        micro = d.get("micro_avg")
        task = d.get("task_avg")
        stops = d.get("clean_stop_rate")
        fmt = lambda v: f"{v:>6.3f}" if isinstance(v, (int, float)) else f"{'-':>6}"
        print(f"{d['config']:<16} {fmt(micro)} {fmt(task)} {fmt(stops)}   {d['run_id']}")

    if show_probes:
        print()
        names = None
        table = {}
        for d in rows:
            run_dir = os.path.join(RESULTS_ROOT, d["run_id"])
            micro_path = os.path.join(run_dir, "micro.json")
            if not os.path.isfile(micro_path):
                continue
            with open(micro_path, encoding="utf-8") as f:
                probes = json.load(f)
            table[d["config"]] = {p["probe"]: p["score"] for p in probes}
            if names is None:
                names = [p["probe"] for p in probes]
        if names:
            width = max(len(n) for n in names) + 1
            configs = list(table)
            print(" " * width + "  ".join(f"{c[:10]:>10}" for c in configs))
            for n in names:
                cells = "  ".join(f"{table[c].get(n, float('nan')):>10.1f}"
                                  for c in configs)
                print(f"{n:<{width}}{cells}")


if __name__ == "__main__":
    main()
