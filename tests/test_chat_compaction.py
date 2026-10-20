"""Hysteresis chat compaction: fold to ~10% at 80% full, then leave the
prefix alone (core.inference.plan_compaction / compact_with_engine)."""

from core.inference import (COMPACT_REQUEST, COMPACT_TAG, _count_tokens,
                            apply_compaction, compact_with_engine,
                            context_input_budget, plan_compaction)

CTX = 32768
MAX_TOK = 4096


def _convo(n, chars=2000):
    msgs = [{"role": "system", "content": "You are helpful."}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i} " + "x" * chars})
    return msgs


def test_not_due_below_high_water():
    msgs = _convo(10)
    assert _count_tokens(msgs) < context_input_budget(CTX, MAX_TOK) * 0.8
    assert plan_compaction(msgs, CTX, MAX_TOK) == 0


def test_folds_down_to_target():
    msgs = _convo(60)
    budget = context_input_budget(CTX, MAX_TOK)
    assert _count_tokens(msgs) > budget * 0.8
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    assert cut > 0
    tail = msgs[1 + cut:]
    assert len(tail) >= 2
    assert _count_tokens(tail) <= budget * 0.10 or len(tail) == 2


def test_compacted_history_is_not_due_again_next_turn():
    """The point of the hysteresis: one compaction buys many sends."""
    msgs = _convo(60)
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    new = apply_compaction(msgs, cut, "short synopsis")
    new.append({"role": "assistant", "content": "y" * 2000})
    new.append({"role": "user", "content": "z" * 2000})
    assert plan_compaction(new, CTX, MAX_TOK) == 0


def test_message_count_backstop():
    msgs = _convo(170, chars=10)
    assert _count_tokens(msgs) < context_input_budget(CTX, MAX_TOK) * 0.8
    assert plan_compaction(msgs, CTX, MAX_TOK) > 0


def test_always_keeps_last_two_even_if_huge():
    msgs = _convo(8, chars=30000)
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    assert cut == len(msgs) - 1 - 2


def test_old_synopsis_is_folded_not_kept():
    msgs = apply_compaction(_convo(6), 4, "old synopsis")
    msgs += _convo(60)[1:]
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    assert cut > 1
    assert not any(m["content"].startswith("[EARLIER") for m in msgs[1 + cut:])


class _Engine:
    def __init__(self, reply="synopsis text", fail=False):
        self.reply, self.fail, self.sent = reply, fail, None

    def generate_streaming(self, messages, max_tokens, temperature, **kw):
        self.sent = messages
        if self.fail:
            raise RuntimeError("boom")
        return "<think>hmm</think>" + self.reply


def test_engine_summary_reuses_the_conversation_prefix():
    msgs = _convo(61)                      # ends on a new user turn
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    eng = _Engine()
    new = compact_with_engine(msgs, cut, eng)
    # Same system prompt and history as the last send, minus the new turn,
    # plus the instruction: the server's prompt cache covers all but that.
    assert eng.sent[:-1] == msgs[:-1]
    assert eng.sent[-1]["content"] == COMPACT_REQUEST
    assert new[1]["content"] == f"{COMPACT_TAG}\nsynopsis text"
    assert new[-1] == msgs[-1]


def test_engine_failure_falls_back_to_key_points():
    msgs = _convo(61)
    cut = plan_compaction(msgs, CTX, MAX_TOK)
    new = compact_with_engine(msgs, cut, _Engine(fail=True))
    assert new[1]["content"].startswith(COMPACT_TAG)
    assert "Q: m0" in new[1]["content"]
