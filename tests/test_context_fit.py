"""Tests for the pre-flight context assessment (core.inference).

The failure these guard against: at 75-85% of context, a big tool read or
long paste overflowed the engine at prefill because the input cap left no
room for the completion budget and chars/4 under-counted dense text.
"""

import pytest

from core.inference import (
    context_input_budget, ensure_context_fit, _count_tokens,
)


class TestContextInputBudget:
    def test_reserves_completion_and_margin(self):
        b = context_input_budget(32000, 12288)
        # input + completion must fit inside ctx
        assert b + 12288 <= 32000
        assert b > 0

    def test_floor_on_degenerate_budget(self):
        # Completion budget nearly the whole tier: floor keeps 25% for input
        b = context_input_budget(16384, 12288)
        assert b == 16384 // 4

    def test_zero_ctx(self):
        assert context_input_budget(0, 4096) == 0

    def test_no_completion_reservation(self):
        b = context_input_budget(32000, 0)
        margin = max(512, 32000 // 32)
        assert b == 32000 - margin


class TestEnsureContextFit:
    def _msgs(self, big_chars):
        return [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "the goal"},
            {"role": "assistant", "content": "working on it"},
            {"role": "user", "content": "X" * big_chars},
        ]

    def test_small_history_untouched(self):
        msgs = self._msgs(1000)
        out, info = ensure_context_fit(msgs, 32000, 12288)
        assert out == msgs
        assert not info["compacted"] and not info["trimmed"]

    def test_oversized_single_message_trimmed_to_budget(self):
        out, info = ensure_context_fit(self._msgs(400_000), 32000, 12288)
        assert info["trimmed"]
        assert info["est"] <= info["budget"]
        # generation room survives: input + completion fits ctx
        assert info["est"] + 12288 <= 32000

    def test_many_messages_compact_first(self):
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "goal"}]
        for i in range(40):
            msgs.append({"role": "assistant", "content": f"round {i}"})
            msgs.append({"role": "user", "content": f"[TOOL OUTPUT] {i} " + "z" * 4000})
        out, info = ensure_context_fit(msgs, 32000, 12288)
        assert info["est"] <= info["budget"]
        assert len(out) < len(msgs)

    def test_exact_counter_catches_underestimate(self):
        # Fake tokenizer reporting 30% more than chars/4 — the heuristic
        # alone would pass a prompt that really overflows.
        dense = lambda text: int(len(text) / 4 * 1.3)
        out, info = ensure_context_fit(
            self._msgs(280_000), 32000, 12288, count_fn=dense)
        assert info["trimmed"]
        # the REAL count of what survived fits the budget
        joined = "\n".join(m["content"] for m in out)
        assert dense(joined) <= info["budget"] * 1.02  # tiny slack for join newlines

    def test_broken_counter_is_nonfatal(self):
        def boom(text):
            raise RuntimeError("tokenize down")
        out, info = ensure_context_fit(
            self._msgs(1000), 32000, 12288, count_fn=boom)
        assert out  # no exception, messages returned

    def test_unknown_ctx_is_noop(self):
        msgs = self._msgs(400_000)
        out, info = ensure_context_fit(msgs, 0, 12288)
        assert out == msgs
        assert info["budget"] == 0


class TestAgentFeedbackCap:
    def _runner(self, ctx=32000):
        from core.agent_loop import AgentRunner, RunConfig, AutonomyLevel

        class FakeEngine:
            def get_context_size(self):
                return ctx

            def count_tokens(self, text):
                return len(text) // 4

        runner = AgentRunner.__new__(AgentRunner)
        runner.engine = FakeEngine()
        runner.config = RunConfig(autonomy=AutonomyLevel.FULL_AUTO,
                                  max_tokens=12288, context_window=15)
        runner.emit = lambda ev: None
        runner._round = 1
        return runner

    def test_small_feedback_untouched(self):
        r = self._runner()
        assert r._fit_feedback("short output") == "short output"

    def test_huge_feedback_truncated_with_marker(self):
        r = self._runner()
        fb = r._fit_feedback("Y" * 300_000)
        assert len(fb) < 300_000
        assert "TOOL OUTPUT TRUNCATED" in fb
        # head and tail both survive
        assert fb.startswith("Y") and fb.endswith("Y")
        # capped at half the input budget (plus the marker text)
        budget = context_input_budget(32000, 12288)
        assert len(fb) <= (budget // 2) * 4 + 400

    def test_no_ctx_no_cap(self):
        r = self._runner(ctx=0)
        big = "Y" * 300_000
        assert r._fit_feedback(big) == big


class TestAgentModelCompaction:
    def _runner(self, summary="- goal: x\n- state: y\n- next: z\n- done so far: lots"):
        from core.agent_loop import AgentRunner, RunConfig, AutonomyLevel

        class FakeEngine:
            def get_context_size(self):
                return 32000

            def count_tokens(self, text):
                return len(text) // 4

            def generate_streaming(self, msgs, max_tokens=0, temperature=0.7,
                                   on_token=None, **kw):
                if on_token:
                    on_token(summary)
                return summary

        class Ctl:
            stop_requested = False

        runner = AgentRunner.__new__(AgentRunner)
        runner.engine = FakeEngine()
        runner.config = RunConfig(autonomy=AutonomyLevel.FULL_AUTO,
                                  max_tokens=12288, context_window=15)
        runner.events = []
        runner.emit = lambda ev: runner.events.append(ev)
        runner._round = 2
        runner.control = Ctl()
        return runner

    def _big_history(self):
        hist = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "GOAL: refactor foo"}]
        for i in range(10):
            hist.append({"role": "assistant", "content": f"round {i}"})
            hist.append({"role": "user", "content": f"[TOOL OUTPUT] {i} " + "z" * 9000})
        return hist

    def test_model_synopsis_replaces_old_half(self):
        r = self._runner()
        new_hist, did = r._compact_if_needed(self._big_history(), 32000, 15)
        assert did
        assert any("COMPACTED SUMMARY" in m.get("content", "") for m in new_hist)
        assert new_hist[1]["content"].startswith("GOAL")     # pinned
        assert _count_tokens(new_hist) < 32000 * 0.60
        assert [e.kind for e in r.events] == ["compacted"]

    def test_below_threshold_untouched(self):
        r = self._runner()
        hist = [{"role": "system", "content": "s"},
                {"role": "user", "content": "g"},
                {"role": "assistant", "content": "a"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "done"}]
        new_hist, did = r._compact_if_needed(hist, 32000, 15)
        assert not did and new_hist == hist

    def test_degenerate_summary_falls_back_to_key_points(self):
        r = self._runner(summary="ok")   # < 40 chars — rejected
        new_hist, did = r._compact_if_needed(self._big_history(), 32000, 15)
        assert did
        # key-point compactor's tag, not the model synopsis tag
        assert any("key points" in e.reason for e in r.events)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
