# Qwen3.6-35B-A3B agent tuning — feedback-test report (2026-08-05)

Goal: tune the Artifex framework + GUI so the local Qwen3.6-35B-A3B
(UD-Q5_K_S, llama.cpp `--cpu-moe`, RTX 5060 Ti 8 GB) can operate as an
autonomous agent — the kind of task you'd hand a cloud Sonnet agent.
Method: build `agent_bench/`, run feedback tests against the live model,
fix what they expose, re-measure.

Model serving context: llama-server @ 131072 ctx, `-fa on --swa-full
--cpu-moe --jinja --reasoning-format deepseek`, ~39.5 tok/s decode,
~50-300 tok/s prompt processing (cache-warm rounds are mostly incremental).

## Framework bugs the feedback tests exposed (all fixed + regression-tested)

1. **Invisible sampler defaults (the Qwen3.5-collapse class).** The llama.cpp
   engine sent only `temperature`; llama-server silently applied its
   compiled-in request defaults — measured live via `/props`: top_k=20,
   top_p=0.95, **min_p=0.05**. Exactly the invisible-min_p failure mode that
   caused the 2026-04 Qwen3.5 coherence collapse. Fix: `core/sampling.py`
   preset contract (mirrors the WebGPU dropdown values exactly) + the engine
   now sends the FULL explicit chain on every request. `sampling={}`
   reproduces the legacy leak for A/B purposes.
2. **Multi-line edit replacements silently truncated.** `extract_agent_actions`
   parsed ```edit``` NEW blocks with a lazy `(.*?)$` under `re.MULTILINE` —
   every multi-line NEW lost everything after its first line. Real agent
   edits were breaking at the parser, not the model.
3. **Tool listings fired all 11 tools.** The system prompt promises
   backticked markers are inert; the extractor matched markers inside inline
   code spans (and `**bold**` wrappers) anyway. A model politely answering
   "what tools do you have?" executed its entire toolbox with placeholder
   args. Now: inline-code and bold-wrapped markers are documentation.
4. **Native `<tool_call>` forms unparsed.** Qwen3.6 under `--jinja`
   occasionally emits its trained syntax — hybrid `<tool_call>:glob("x")`
   and JSON `<tool_call>{"name":...,"arguments":{...}}</tool_call>`. Both now
   normalize onto the existing marker executors (unknown tool names are
   deliberately ignored).
5. **Streaming usage/timings never captured.** llama-server delivers
   usage in a final SSE chunk whose `choices` is empty; the engine's
   `if not choices: continue` skipped it before the read. Also now requests
   `stream_options.include_usage` and records llama-server `timings`
   (prompt/decode tok/s) into `_last_gen_stats`.
6. **Thinking/response misrouting.** llama.cpp + Ollama engines emit explicit
   `<think>` tags, but the agent loop's ThinkFilter assumed streams start
   inside a think block — with thinking off, whole responses displayed as
   "thinking". `stream_starts_in_think` is now per-engine and honored.
7. **Engine failures reported as clean "done".** An exception in
   `_generate` returned an empty response, which the loop treated as a
   normal completion. Now finishes as `stopped:error:<ExceptionName>`.
8. **RunConfig.enable_thinking was dead** — never passed to engines. Wired,
   plus `sampler_preset`/`sampling` fields, GUI dropdown + thinking toggle
   in the Agent tab, CLI `/run` defaults to the agent preset.
9. **Teaching-quality tool feedback.** `find_symbol("config.ini")` (a
   recurring Qwen3.6 drift: filename → symbol search) now redirects to
   `@glob`/`@read_file` instead of returning an empty non-answer; empty-OLD
   edits explain themselves instead of "appears N times".

## Micro suite (10 single-turn discipline probes, seed 42)

Scores under the CURRENT parser (regraded offline from saved responses).

Chat framing (plain assistant prompt — sweep1):

| config | micro |
|---|---|
| legacy (server-default leak, T=0.7) | 0.83 |
| neutral (explicit balanced chain, T=0.7) | 0.73 |
| agent (T=0.6 topk20 topp0.95 minp0) | 0.73 |
| agent-nothink (thinking off) | 0.73 |
| agent-cold (T=0.3) | 0.63 |
| greedy (T=0) | 0.63 |
| agent-presence (presence=1.0) | 0.63 |

Agent framing (autonomous preamble + GOAL — sweep2, the condition that
matches real runs):

| config | micro |
|---|---|
| legacy | 0.95 |
| agent-presence | 0.95 |
| greedy | 0.93 |
| agent | 0.85 |
| neutral | 0.83 |
| agent-cold | 0.83 |
| **agent-nothink** | **0.70** |

Findings:
* **The autonomous framing itself is worth ~+0.15** across configs — the
  preamble (`@done` docs, "tool markers are LIVE", one-step-per-turn) does
  real work. Micro probes must (and now do) use it.
* **Thinking ON is the single clearest win**: 0.70 vs 0.85+ under identical
  sampling. Non-thinking responses skip tool syntax more, botch edit
  blocks more, and answer prose where an action was required.
* Sampler differences among thinking configs are 1-2 probes at n=10 —
  within noise. The multi-round full suite is the decider.
* A consistent behavioral trait at every config: the model explores
  (`@architecture()`) before running tests when asked to "run the test
  suite" — defensible in round 1, scored 0.5.

## Full suite (6 multi-round scenarios through the real AgentRunner)

### Round 1 (full2) — pre-fix framework, fair four-way comparison

| config | task avg | clean stops | losses (every one a FRAMEWORK bug) |
|---|---|---|---|
| **agent-presence** | **0.833** | 5/6 | s5: heredoc body line-split into PowerShell |
| legacy | 0.750 | 5/6 | s4: nested-fence truncation; s5: UTF-16 redirect |
| greedy | 0.750 | 5/6 | s4: nested-fence truncation; s5: UTF-16 redirect |
| agent | 0.617 | 5/6 | s3: shell python≠venv (no pytest); s4: `<tool_call>` repetition collapse; s5: UTF-16 redirect |

The headline finding of round 1: **the model never failed a task for
capability reasons.** All 5.5 dropped points across 24 scenario runs trace
to five framework defects, each since fixed and regression-tested:

1. Nested markdown fences inside python string literals truncated the code
   at the interior ``` (s4, two configs). → AST-validated fence close.
2. PowerShell `>` redirects wrote UTF-16 files unreadable downstream (s5,
   three configs). → forced UTF-8 Out-File/console encoding.
3. Shell `python`/`pip`/`pytest` resolved outside the venv while python
   blocks ran inside it (s3, agent config). → venv Scripts on child PATH +
   VIRTUAL_ENV + python3/pip3 shims.
4. Qwen3.6 `<tool_call>` repetition collapse read as a prose "done" (s4,
   agent config). → format-retry: corrective feedback + another round
   (cap 2), plus native `<tool_call>` parsing for well-formed emissions.
5. Bash heredocs split into per-line "commands", executing the document
   body through PowerShell (s5, agent-presence). → heredocs stay whole
   and route to Git Bash.

Multi-round behavior otherwise: correct tool selection, real
run→diagnose→fix→verify loops (s3/s6 passed everywhere incl. recovery
from failed actions), clean `@done` stops in 20/24 runs, zero policy
blocks, zero circuit-breaker trips, ~30-80 s and 150-1400 completion
tokens per scenario at ~39 tok/s.

### Round 2 (full3) — post-fix validation

Same four configs, same scenarios, framework fixes 1-4 live (the heredoc
and create-file-edit fixes landed mid-run and apply from full4 on):

| config | task avg | clean stops | remaining loss |
|---|---|---|---|
| **agent** | **1.000** | 6/6 | — |
| **greedy** | **1.000** | 6/6 | — |
| agent-presence | 0.967 | 6/6 | s3: fixed the bug correctly but SKIPPED re-running the test |
| legacy | 0.833 | 6/6 | s5: create-file edit form (parsed to nothing; supported post-run) |

Timing (agent config): 29-62 s and 164-810 completion tokens per scenario,
6/6 ended with an explicit `@done`, zero failed actions.

### Round 3 (final2) — shipping configuration, all fixes live

One more failure mode surfaced and fixed between rounds: **announce-stall**
(the model narrates "Step 1: find the definition." and ends the turn — a
chat reflex the loop misread as a prose final answer; now detected and
nudged via the shared retry budget, with the rule added to the autonomous
preamble; s4 went 3/3 after the fix). The create-file ```edit``` form
(empty OLD, new path) also proved out live in s5.

Final numbers for the shipping config (agent preset, thinking ON, seed 42):

| suite | score |
|---|---|
| micro discipline | **0.930** |
| full 6-scenario task avg | **1.000** |
| clean `@done` stops | **6/6** |

### Preset decision

`agent` preset ships as the Qwen3-recommended thinking configuration —
**temp 0.6, top_k 20, top_p 0.95, min_p 0, penalties off** — because it
scored 1.000 WITH the run-the-test-before-done discipline intact:

* greedy also hit 1.000, but temp 0 regenerates failed attempts verbatim
  on retry rounds — diversity is what lets a retry escape a dead end.
* presence_penalty=1.0 had the best micro discipline (0.95) and
  mechanistically counters the one true model-level pathology observed
  (`<tool_call>` repetition collapse), but its one full-suite loss was
  skipping test verification — the more precious agent trait. Collapse is
  now handled reactively by the loop's format-retry; presence=1.0 remains
  the documented fallback if collapse recurs at scale.
* Thinking stays ON for agent runs (micro: 0.85+ vs 0.70 off; the think
  block is where tool choice and stop decisions get made).

## Tuned defaults (final)

* **llama.cpp engine** (`core/engine_llama_cpp.py`): every request carries a
  fully-explicit sampler chain. `sampling=None` → `DEFAULT_SAMPLING`
  (balanced-shaped, min_p 0, penalties off, caller's temperature).
  llama-server compiled defaults can no longer leak into generation.
* **Autonomous runs** (GUI Agent tab, CLI `/run`, `AgentRunner`): preset
  `agent` (temp 0.6 / top_k 20 / top_p 0.95 / min_p 0), thinking ON,
  max_tokens 2048+ (thinking headroom), format-retry cap 2.
* **Qt Agent tab**: Sampling preset dropdown (default Agent) + Thinking
  checkbox, wired into `RunConfig`.
* **Executor environment**: venv-activated shell PATH, UTF-8 PowerShell
  redirects, Git Bash for heredocs/bash syntax — shell and python blocks
  now live in the SAME world.
* **Protocol tolerance**: backtick/bold-wrapped markers inert; native
  `<tool_call>` hybrid + JSON forms parsed; create-file edit blocks
  (empty OLD, new path) supported; malformed attempts get one corrective
  retry instead of silently ending the run.

## Verdict

With the framework fixed and the preset pinned, Qwen3.6-35B-A3B on a
5060 Ti completes the full agent suite — read/locate, aggregate,
run-test→fix→verify, cross-file navigation reports, build-and-use a CLI,
surgical edits — at 1.000 with clean `@done` stops and ~40 tok/s. The
model was never the bottleneck; the harness was. It is now reasonable to
hand Artifex agent goals of this shape (single-workspace, file+shell+
python, ≤10 rounds) and expect Sonnet-agent-style completion.

## Reproduction

```bash
./venv/Scripts/python.exe -m agent_bench.run_bench --config all --suite micro
./venv/Scripts/python.exe -m agent_bench.run_bench --config legacy,agent,agent-presence,greedy --suite full
./venv/Scripts/python.exe -m agent_bench.compare sweep2 --probes
./venv/Scripts/python.exe -m agent_bench.regrade      # re-score saved runs offline
```
