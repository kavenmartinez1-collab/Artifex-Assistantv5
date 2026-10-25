"""Tests for the per-request web tool guards (api/web_tools.py, 2026-10-25).

Regression for chat chatcmpl-7b4b64da: one turn emitted 44 tool calls - the same
five @web_read refs about ten times each plus @search(...) with a literal "..."
query. All of them ran (the sites answered HTTP 429), 304K chars of results went
back and the next prompt reached ~80K tokens. The guards mirror Anthropic's
server tools: max_uses per tool per request (error result, request continues),
a per-request URL cache, bounded calls per turn, and capped content per turn.
"""

import pytest

import api.web_tools as web_tools

RESULTS = [{"title": f"Result {i}", "url": f"https://site{i}.example/page", "snippet": ""}
           for i in range(1, 17)]


@pytest.fixture
def gw(monkeypatch):
    calls = {"search": [], "fetch": []}

    def post(endpoint, payload):
        if endpoint == "/search":
            calls["search"].append(payload["query"])
            return True, {"results": RESULTS[:8] if len(calls["search"]) == 1 else RESULTS[8:]}
        calls["fetch"].append(payload["url"])
        return True, {"text": "x" * 20000, "title": payload["url"], "url": payload["url"]}

    monkeypatch.setattr(web_tools, "gateway_post", post)
    monkeypatch.delenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", raising=False)
    return calls


def _reads(*refs):
    return [{"type": "web_read", "ref": str(r)} for r in refs]


def _searches(*queries):
    return [{"type": "search", "query": q} for q in queries]


def _primed(gw):
    """A request state with two searches done (16 cached results)."""
    cache, state = [], web_tools.new_request_state()
    web_tools.execute_web_tools(_searches("cves this week", "cisa kev october"), cache, state)
    return cache, state


class TestIncidentReplay:
    def test_44_call_loop_runs_each_distinct_call_once(self, gw):
        cache, state = _primed(gw)
        loop = (_searches("...", "...", "...", "...")
                + _reads(9, 15, 1, 3, 5) * 8)              # 4 + 40 = 44 calls
        out = web_tools.execute_web_tools(loop, cache, state)
        distinct = {RESULTS[i - 1]["url"] for i in (9, 15, 1, 3, 5)}
        assert len(gw["fetch"]) == len(set(gw["fetch"]))      # no page twice (was ~10x each)
        assert set(gw["fetch"]) <= distinct and len(gw["fetch"]) >= 4
        assert gw["search"] == ["cves this week", "cisa kev october"]   # "..." never searched
        assert "has no real argument" in out
        assert "repeated tool call(s) in this turn were skipped" in out
        assert len(out) <= web_tools.WEB_ROUND_MAX_CHARS + 1500   # was 304K
        if len(gw["fetch"]) < 5:                               # the rest waits for next turn
            assert "over this turn's limits" in out

    def test_next_turn_repeats_are_not_refetched(self, gw):
        cache, state = _primed(gw)
        web_tools.execute_web_tools(_reads(1, 3), cache, state)
        out = web_tools.execute_web_tools(_reads(1, 3), cache, state)
        assert len(gw["fetch"]) == 2
        assert out.count("was already read in turn") == 2


class TestLimits:
    def test_max_uses_for_reads_returns_error_and_goes_on(self, gw, monkeypatch):
        monkeypatch.setattr(web_tools, "WEB_READ_MAX_USES", 3)
        monkeypatch.setattr(web_tools, "WEB_TOOLS_MAX_PER_ROUND", 99)
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_reads(1, 2, 3, 4, 5), cache, state)
        assert len(gw["fetch"]) == 3
        assert out.count("max_uses_exceeded") == 2

    def test_max_uses_for_searches(self, gw, monkeypatch):
        monkeypatch.setattr(web_tools, "WEB_SEARCH_MAX_USES", 2)
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_searches("a third query"), cache, state)
        assert gw["search"] == ["cves this week", "cisa kev october"]
        assert "max_uses_exceeded" in out

    def test_same_query_is_not_rerun(self, gw):
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_searches("CVEs  this week"), cache, state)
        assert len(gw["search"]) == 2
        assert "was already run in turn 1" in out

    def test_calls_per_turn_are_bounded(self, gw, monkeypatch):
        monkeypatch.setattr(web_tools, "WEB_TOOLS_MAX_PER_ROUND", 4)
        monkeypatch.setattr(web_tools, "WEB_READ_MAX_USES", 99)
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_reads(*range(1, 11)), cache, state)
        assert len(gw["fetch"]) == 4
        assert "over this turn's limits (4 tool calls" in out

    def test_turn_content_is_capped(self, gw, monkeypatch):
        monkeypatch.setattr(web_tools, "WEB_ROUND_MAX_CHARS", 30000)
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_reads(1, 2, 3, 4, 5), cache, state)
        assert len(out) < 30000 + 5 * 2500
        assert "[TRUNCATED" in out

    def test_placeholder_reads_are_refused(self, gw):
        cache, state = _primed(gw)
        out = web_tools.execute_web_tools(_reads("N", "<url>"), cache, state)
        assert gw["fetch"] == []
        assert out.count("has no real argument") == 2


class TestWithoutState:
    def test_legacy_path_runs_everything(self, gw):
        cache = []
        web_tools.execute_web_tools(_searches("q"), cache)
        web_tools.execute_web_tools(_reads(1, 1, 1), cache)
        assert len(gw["fetch"]) == 3


class TestCollapse:
    def test_repeated_call_lines_are_dropped(self):
        text = "Let me read these.\n" + "@web_read(9)\n@web_read(15)\n" * 6 + "@search(\"x\")"
        out = web_tools.collapse_repeated_tool_calls(text)
        assert out.count("@web_read(9)") == 1 and out.count("@web_read(15)") == 1
        assert "@search(\"x\")" in out
        assert "10 repeated tool-call line(s) removed" in out

    def test_text_without_repeats_is_unchanged(self):
        text = "Answer.\n@web_read(1)\n@web_read(2)"
        assert web_tools.collapse_repeated_tool_calls(text) == text


class TestClearOldToolResults:
    def _msgs(self, sizes):
        msgs = [{"role": "user", "content": "question"}]
        for n in sizes:
            msgs.append({"role": "assistant", "content": "@web_read(1)"})
            msgs.append({"role": "user", "content": web_tools.TOOL_RESULTS_PREFIX + "\n\n" + "y" * n})
        return msgs

    def test_clears_all_but_newest_two_over_trigger(self):
        msgs = self._msgs([30000, 30000, 30000, 30000])
        cleared = web_tools.clear_old_tool_results(msgs, keep=2, trigger_chars=60000)
        assert cleared > 59000
        tool_msgs = [m["content"] for m in msgs if m["content"].startswith(web_tools.TOOL_RESULTS_PREFIX)]
        assert "cleared to save context" in tool_msgs[0] and "cleared to save context" in tool_msgs[1]
        assert len(tool_msgs[2]) > 30000 and len(tool_msgs[3]) > 30000

    def test_under_trigger_nothing_changes(self):
        msgs = self._msgs([10000, 10000])
        before = [m["content"] for m in msgs]
        assert web_tools.clear_old_tool_results(msgs) == 0
        assert [m["content"] for m in msgs] == before
