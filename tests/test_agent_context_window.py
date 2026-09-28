"""Tests for the agent actually reaching the engine's full context window.

The failure these guard against: an agent run on the 73728-ctx split could
never use more than about 40k of it. Three independent throttles stacked up
— a 16384 request default, a flat 0.70-of-ctx input cap, and compaction at
0.60 of raw ctx — and the halving ladder in build_active_messages threw away
whatever the cap did grant. All four are exercised below.
"""

import pytest

from core.inference import (
    build_active_messages, context_input_budget, _count_tokens,
)
from core.agent_loop import AgentRunner, RunConfig, AutonomyLevel
from api.agent_api import _resolve_context_window, _CTX_FALLBACK


CTX = 73728          # the measured split on this box
MAX_TOKENS = 4096


def _history(pairs=120, chars=1600):
    """A session far too big for any single window."""
    hist = [{"role": "system", "content": "SYS"},
            {"role": "user", "content": "GOAL: refactor the thing"}]
    for i in range(pairs):
        hist.append({"role": "assistant", "content": f"a{i} " + "y" * chars})
        hist.append({"role": "user", "content": f"[TOOL OUTPUT] {i} " + "x" * chars})
    return hist


def _runner(max_tokens=MAX_TOKENS, **kw):
    r = AgentRunner.__new__(AgentRunner)
    r.config = RunConfig(autonomy=AutonomyLevel.FULL_AUTO,
                         max_tokens=max_tokens, **kw)
    return r


class TestBuildActiveMessagesBudget:
    def test_max_tokens_raises_the_input_cap(self):
        """The cap becomes ctx-minus-completion instead of a flat 70%."""
        hist = _history()
        _, old = build_active_messages(hist, 16384, engine_ctx=CTX)
        _, new = build_active_messages(hist, 16384, engine_ctx=CTX,
                                       max_tokens=MAX_TOKENS)
        assert _count_tokens(new) > _count_tokens(old)

    def test_fills_the_budget_rather_than_halving_into_it(self):
        """Regression: raising the cap bought ZERO history on its own.

        The shrink ladder stepped 200 -> 100 -> 50 messages, and both the
        old 51.6k cap and the new 67.3k one cleared the same 100-message
        rung, so the extra 15.7k went unused.
        """
        hist = _history()
        budget = context_input_budget(CTX, MAX_TOKENS)
        _, active = build_active_messages(hist, 16384, engine_ctx=CTX,
                                          max_tokens=MAX_TOKENS)
        kept = _count_tokens(active)
        assert kept <= budget, "must not exceed the input budget"
        # Uses most of what it was granted, not an arbitrary power-of-two cut.
        assert kept > budget * 0.8, f"only used {kept} of {budget}"

    def test_without_max_tokens_behaviour_is_unchanged(self):
        """The Qt GUI, the CLI and the WebGPU golden fixtures depend on this."""
        hist = _history()
        h1, a1 = build_active_messages(hist, 15, engine_ctx=CTX)
        h2, a2 = build_active_messages(hist, 15, engine_ctx=CTX,
                                       max_tokens=None)
        assert a1 == a2 and h1 == h2
        # And still capped by the historical 0.70 fraction.
        assert _count_tokens(a1) <= int(CTX * 0.70)

    def test_system_prompt_and_pinned_goal_survive(self):
        hist = _history()
        _, active = build_active_messages(hist, 16384, engine_ctx=CTX,
                                          max_tokens=MAX_TOKENS)
        assert active[0]["role"] == "system"
        assert "GOAL" in active[1]["content"]

    def test_always_keeps_a_usable_tail(self):
        """One enormous message must not reduce the active set to nothing."""
        hist = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "GOAL"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "z" * 4_000_000}]
        _, active = build_active_messages(hist, 16384, engine_ctx=CTX,
                                          max_tokens=MAX_TOKENS)
        assert len(active) >= 2


class TestCompactTrigger:
    def test_measured_against_budget_not_raw_ctx(self):
        r = _runner()
        trigger = r._compact_trigger(CTX)
        budget = context_input_budget(CTX, MAX_TOKENS)
        assert trigger == int(budget * 0.85)
        # Strictly later than the old flat 0.60-of-raw-ctx rule.
        assert trigger > int(CTX * 0.60)

    def test_folds_before_the_budget_is_blown(self):
        r = _runner()
        assert r._compact_trigger(CTX) <= context_input_budget(CTX, MAX_TOKENS)

    def test_threshold_is_configurable(self):
        assert (_runner(compact_threshold=0.5)._compact_trigger(CTX)
                < _runner(compact_threshold=0.95)._compact_trigger(CTX))

    @pytest.mark.parametrize("bad", [0.0, -1.0, 2.0])
    def test_nonsense_threshold_falls_back(self, bad):
        r = _runner(compact_threshold=bad)
        assert r._compact_trigger(CTX) == int(
            context_input_budget(CTX, MAX_TOKENS) * 0.85)

    def test_small_tiers_still_compact(self):
        r = _runner(max_tokens=2048)
        assert 0 < r._compact_trigger(8192) < 8192


class TestResolveContextWindow:
    class _Engine:
        def __init__(self, n):
            self.n = n

        def get_context_size(self):
            return self.n

    class _Dead:
        def get_context_size(self):
            raise RuntimeError("engine not loaded")

    def test_omitted_means_the_whole_loaded_window(self):
        assert _resolve_context_window(self._Engine(CTX), None) == CTX

    def test_explicit_smaller_is_honoured(self):
        assert _resolve_context_window(self._Engine(CTX), 16384) == 16384

    def test_explicit_bigger_is_clamped_to_what_is_loaded(self):
        """Asking for 72k on a 32k tier just means trimming every round."""
        assert _resolve_context_window(self._Engine(32768), CTX) == 32768

    def test_unknown_engine_size_falls_back(self):
        assert _resolve_context_window(self._Engine(0), None) == _CTX_FALLBACK

    def test_engine_that_raises_falls_back(self):
        assert _resolve_context_window(self._Dead(), None) == _CTX_FALLBACK

    def test_engine_that_raises_still_honours_an_explicit_request(self):
        assert _resolve_context_window(self._Dead(), 49152) == 49152


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
