# agent_bench — feedback-test harness for Artifex agent tuning

Measures how well a local model drives the **real** Artifex agent stack
(`core/agent_loop.AgentRunner` + `tools/agent_tools` + `core/sandbox`) so the
framework can be tuned until the model operates as a genuine autonomous
assistant/agent. Built for and first run against **Qwen3.6-35B-A3B**
(UD-Q5_K_S, llama.cpp backend, `--cpu-moe` on the 5060 Ti 8 GB box).

## What it measures

Two suites, both graded programmatically (no LLM judging):

* **micro** — 10 single-turn format-discipline probes. One generation each,
  nothing executes; the grade is what `extract_agent_actions()` parses from
  the response. Catches: wrong tool for the job (shell `cat` instead of
  `@read_file`), live-marker accidents while *listing* tools, spurious tool
  calls in plain chat, malformed ```edit``` blocks, missing `@done`.
  Fast (~4-6 min/config) → used to sweep sampler configs.
* **full** — 6 multi-round scenarios through the real `AgentRunner` in
  FULL_AUTO, tools executing in a throwaway per-scenario workspace:
  read-and-answer, CSV→JSON aggregation, run-test/fix-bug/verify, code
  navigation report, build-a-CLI-and-use-it, surgical ```edit``` refactor.
  Graded on workspace end-state (files correct, tests actually pass, the
  fresh-input generalization check) plus stop discipline.

Per run it records: task score, rounds, per-action success, clean-stop rate,
wall time, prompt/completion tokens, think-vs-content chars, llama-server
decode rates — everything needed to compare configs honestly.

## Running

A llama-server for the target model must be running (it is adopted) or
startable from `llama_cpp_config.json`.

```bash
./venv/Scripts/python.exe -m agent_bench.run_bench --list
./venv/Scripts/python.exe -m agent_bench.run_bench --config all --suite micro
./venv/Scripts/python.exe -m agent_bench.run_bench --config agent,greedy --suite full
./venv/Scripts/python.exe -m agent_bench.run_bench --config agent --scenarios s3_bugfix
```

Target model: `ARTIFEX_BENCH_MODEL` env var (default `qwen3.6-35b-a3b`).
Results land in `agent_bench/results/<timestamp>_<config>/` (gitignored);
`results/index.jsonl` accumulates one summary row per run.

## Configs

Defined in `harness.py::CONFIGS`. Three special sampling values:

* `sampling={}` — **legacy**: reproduces pre-tuning behavior where only
  `temperature` was sent and llama-server's compiled-in defaults
  (min_p=0.05, top_k=20/40 depending on build) applied invisibly.
* `sampling=None` — the engine's new explicit `DEFAULT_SAMPLING`
  (balanced-shaped, `core/sampling.py`).
* a dict — full explicit control, usually `get_preset(name, seed=42)`.

Seed 42 is pinned for sampled configs so config comparisons aren't noise.

## Workspace safety

Each scenario workspace is `git init`-ed as its **own** repo before the agent
runs. This is load-bearing: `tools.agent_tools.git_commit_edit` auto-commits
successful ```edit``` actions to the *nearest* enclosing repo, which would be
the Artifex monorepo without the per-workspace repo. Keep it that way.

## Interpreting

`summary.json` per run; compare runs with the table `run_bench` prints or by
reading `results/index.jsonl`. Primary metrics: `task_avg` (did the work
get done), `micro_avg` (format discipline), `clean_stop_rate` (finished via
`done` rather than round/failure caps). Ties break on completion_tokens and
wall time — cheaper is better.

Findings and the tuned defaults that came out of this harness are written up
in `agent_bench/TUNING_REPORT.md`.
