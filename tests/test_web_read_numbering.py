"""Tests for @web_read(N) numbering across several @search() calls.

Regression for agent run b23bad6e17c1 (2026-09-30): two searches in one round,
the model asked for result 4 of the FIRST list and got result 4 of the second
(drugs.com) because each search overwrote the cache.
"""

import pytest

import api.web_tools as web_tools
import tools.agent_tools as agent_tools

BENCH = [
    {"title": "Reddit FP8 post", "url": "https://reddit.example/fp8", "snippet": ""},
    {"title": "Taj Mahal", "url": "https://taj.example/", "snippet": ""},
    {"title": "abliterlitics", "url": "https://abliterlitics.example/", "snippet": ""},
]
EYES = [
    {"title": "WebMD eye drops", "url": "https://webmd.example/", "snippet": ""},
    {"title": "Drugs.com dry eye", "url": "https://drugs.example/", "snippet": ""},
]


def _fake_gateway(fetched):
    def post(endpoint, payload):
        if endpoint == "/search":
            return True, {"results": BENCH if "bench" in payload["query"] else EYES}
        fetched.append(payload["url"])
        return True, {"text": "page", "title": payload["url"], "url": payload["url"]}
    return post


# ── Agent path (tools/agent_tools.py) ────────────────────────────────────────

@pytest.fixture
def agent_gw(monkeypatch):
    fetched = []
    monkeypatch.setattr(agent_tools, "_check_gateway", lambda: True)
    monkeypatch.setattr(agent_tools, "_gateway_post", _fake_gateway(fetched))
    monkeypatch.setattr(agent_tools, "_last_search_results", [])
    monkeypatch.setattr(agent_tools, "_search_result_base", 0)
    monkeypatch.delenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", raising=False)
    return fetched


class TestAgentNumbering:
    def test_second_search_continues_numbering(self, agent_gw):
        _, first = agent_tools.run_web_search("bench")
        _, second = agent_tools.run_web_search("eyes")
        assert "[1] Reddit FP8 post" in first and "[3] abliterlitics" in first
        assert "[4] WebMD eye drops" in second and "[5] Drugs.com dry eye" in second
        assert "[1] " not in second

    def test_read_resolves_against_the_search_that_printed_n(self, agent_gw):
        agent_tools.run_web_search("bench")
        agent_tools.run_web_search("eyes")
        ok, _ = agent_tools.run_web_read("3")
        assert ok and agent_gw == ["https://abliterlitics.example/"]
        ok, _ = agent_tools.run_web_read("5")
        assert ok and agent_gw[-1] == "https://drugs.example/"

    def test_out_of_range_names_valid_span(self, agent_gw):
        agent_tools.run_web_search("bench")
        ok, out = agent_tools.run_web_read("9")
        assert not ok and "1-3" in out and agent_gw == []

    def test_no_search_yet(self, agent_gw):
        ok, out = agent_tools.run_web_read("1")
        assert not ok and "No search results cached" in out

    def test_old_numbers_expire_past_keep_window(self, agent_gw, monkeypatch):
        monkeypatch.setattr(agent_tools, "_SEARCH_RESULTS_KEEP", 4)
        agent_tools.run_web_search("bench")   # 1-3
        agent_tools.run_web_search("eyes")    # 4-5 -> 1 drops out
        ok, out = agent_tools.run_web_read("1")
        assert not ok and "expired" in out
        ok, _ = agent_tools.run_web_read("2")
        assert ok and agent_gw == ["https://taj.example/"]

    def test_legacy_flag_restores_per_search_numbering(self, agent_gw, monkeypatch):
        monkeypatch.setenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", "1")
        agent_tools.run_web_search("bench")
        _, second = agent_tools.run_web_search("eyes")
        assert "[1] WebMD eye drops" in second
        agent_tools.run_web_read("2")
        assert agent_gw == ["https://drugs.example/"]


# ── Chat path (api/web_tools.py) ─────────────────────────────────────────────

@pytest.fixture
def chat_gw(monkeypatch):
    fetched = []
    monkeypatch.setattr(web_tools, "gateway_post", _fake_gateway(fetched))
    monkeypatch.delenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", raising=False)
    return fetched


class TestChatNumbering:
    def test_two_searches_then_read_by_first_list_number(self, chat_gw):
        cache = []
        out = web_tools.execute_web_tools(
            [{"type": "search", "query": "bench"}, {"type": "search", "query": "eyes"}], cache)
        assert "[3] abliterlitics" in out and "[5] Drugs.com dry eye" in out
        web_tools.execute_web_tools([{"type": "web_read", "ref": "3"}], cache)
        assert chat_gw == ["https://abliterlitics.example/"]

    def test_search_ordered_before_read_in_same_round(self, chat_gw):
        # extract_web_tools() emits searches before reads, so a read the model
        # wrote against the PREVIOUS round's list runs after a fresh search.
        cache = []
        web_tools.execute_web_tools([{"type": "search", "query": "bench"}], cache)
        web_tools.execute_web_tools(
            [{"type": "search", "query": "eyes"}, {"type": "web_read", "ref": "1"}], cache)
        assert chat_gw == ["https://reddit.example/fp8"]

    def test_legacy_flag_restores_per_search_numbering(self, chat_gw, monkeypatch):
        monkeypatch.setenv("ARTIFEX_WEB_READ_PER_SEARCH_NUMBERING", "1")
        cache = []
        web_tools.execute_web_tools(
            [{"type": "search", "query": "bench"}, {"type": "search", "query": "eyes"},
             {"type": "web_read", "ref": "2"}], cache)
        assert chat_gw == ["https://drugs.example/"]
