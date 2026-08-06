"""
Agent bench CLI.

Run from the repo root with the venv python:

    ./venv/Scripts/python.exe -m agent_bench.run_bench --config agent
    ./venv/Scripts/python.exe -m agent_bench.run_bench --config legacy,agent --suite micro
    ./venv/Scripts/python.exe -m agent_bench.run_bench --list

Requires a llama-server for the target model (default qwen3.6-35b-a3b) —
either already running on its configured port (adopted) or startable via
llama_cpp_config.json.
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description="Artifex agent bench")
    ap.add_argument("--config", default="agent",
                    help="comma-separated config names, or 'all'")
    ap.add_argument("--suite", default="both",
                    choices=("micro", "full", "both"))
    ap.add_argument("--scenarios", default="",
                    help="comma-separated scenario subset (full suite)")
    ap.add_argument("--probes", default="",
                    help="comma-separated probe subset (micro suite)")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--wall", type=float, default=480.0,
                    help="wall-clock cap per scenario, seconds")
    ap.add_argument("--tag", default="", help="run-id suffix")
    ap.add_argument("--list", action="store_true",
                    help="list configs/probes/scenarios and exit")
    args = ap.parse_args()

    from agent_bench.harness import CONFIGS, run_config
    from agent_bench.scenarios import MICRO_PROBES, SCENARIOS

    if args.list:
        print("configs:   " + ", ".join(CONFIGS))
        print("probes:    " + ", ".join(p["name"] for p in MICRO_PROBES))
        print("scenarios: " + ", ".join(s["name"] for s in SCENARIOS))
        return 0

    names = list(CONFIGS) if args.config == "all" else [
        c.strip() for c in args.config.split(",") if c.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        print(f"unknown config(s): {unknown}; use --list")
        return 2

    scenario_names = {s.strip() for s in args.scenarios.split(",") if s.strip()}
    probe_names = {p.strip() for p in args.probes.split(",") if p.strip()}

    summaries = []
    for name in names:
        summaries.append(run_config(
            name, suite=args.suite,
            scenario_names=scenario_names or None,
            probe_names=probe_names or None,
            max_rounds=args.max_rounds, wall_clock_s=args.wall,
            tag=args.tag,
        ))

    print("\n=== comparison ===")
    print(f"{'config':<16} {'micro':>6} {'task':>6} {'stops':>6}")
    for s in summaries:
        print(f"{s['config']:<16} "
              f"{s.get('micro_avg', float('nan')):>6.3f} "
              f"{s.get('task_avg', float('nan')):>6.3f} "
              f"{s.get('clean_stop_rate', float('nan')):>6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
