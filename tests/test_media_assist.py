"""Tests for the Create tab's AI-assisted prompt (core.services.media_tools).

The chat model is replaced by a fake async function, so these cover how its
reply is parsed and what lands in the pipeline kwargs: an edit that comes
back with a negative turns on true CFG, an image ignores it, and a failed
or useless reply keeps the user's own prompt.
"""

import asyncio

import pytest

from core.services import media_tools


@pytest.fixture
def fake_llm():
    calls = []

    def install(reply):
        async def llm(messages):
            calls.append(messages)
            if isinstance(reply, Exception):
                raise reply
            return reply
        media_tools.set_prompt_llm(llm)
        return calls

    yield install
    media_tools.set_prompt_llm(None)


def _job(tool):
    return {"id": "t1", "tool": tool, "started": None}


class TestParseReply:
    def test_json(self):
        assert media_tools.parse_assist_reply(
            '{"prompt": "a detailed one", "negative": "photograph"}', "short"
        ) == ("a detailed one", "photograph")

    def test_json_inside_prose_and_fences(self):
        reply = 'Sure!\n```json\n{"prompt": "detailed", "negative": ""}\n```'
        assert media_tools.parse_assist_reply(reply, "short") == ("detailed", "")

    def test_plain_prose_becomes_the_prompt(self):
        assert media_tools.parse_assist_reply("A cat on a red sofa.", "cat") == (
            "A cat on a red sofa.", "")

    def test_empty_or_broken_keeps_user_prompt(self):
        assert media_tools.parse_assist_reply("", "cat") == ("cat", "")
        assert media_tools.parse_assist_reply('{"prompt": ', "cat")[0] == "cat"
        assert media_tools.parse_assist_reply('{"prompt": ""}', "cat") == ("cat", "")

    def test_lengths_capped(self):
        p, n = media_tools.parse_assist_reply(
            '{"prompt": "%s", "negative": "%s"}' % ("x" * 3000, "y" * 900), "s")
        assert len(p) == 2000 and len(n) == 500


class TestAssist:
    def test_edit_negative_turns_on_true_cfg(self, fake_llm):
        calls = fake_llm('{"prompt": "Redraw the whole photo as South Park", '
                         '"negative": "photograph, photorealistic"}')
        job, kwargs = _job("edit"), {"prompt": "south park him", "image_path": "x.jpg"}
        asyncio.run(media_tools._assist(job, kwargs))
        assert kwargs["prompt"] == job["assisted_prompt"] == "Redraw the whole photo as South Park"
        assert kwargs["negative_prompt"] == job["negative_prompt"] == "photograph, photorealistic"
        assert kwargs["true_cfg_scale"] == media_tools._ASSIST_CFG
        assert calls[0][0]["content"] == media_tools._ASSIST_SYSTEM["edit"]
        assert calls[0][1]["content"] == "south park him"
        assert job["stage"] == "Waiting for the GPU" and job["started"]

    def test_edit_without_negative_stays_cfg_free(self, fake_llm):
        fake_llm('{"prompt": "Give the dog a red scarf; keep the rest", "negative": ""}')
        kwargs = {"prompt": "red scarf on dog"}
        asyncio.run(media_tools._assist(_job("edit"), kwargs))
        assert "true_cfg_scale" not in kwargs and "negative_prompt" not in kwargs

    def test_image_ignores_negative(self, fake_llm):
        fake_llm('{"prompt": "A detailed cat", "negative": "dog"}')
        kwargs = {"prompt": "cat", "width": 1024}
        asyncio.run(media_tools._assist(_job("image"), kwargs))
        assert kwargs == {"prompt": "A detailed cat", "width": 1024}

    def test_failure_keeps_prompt_and_says_so(self, fake_llm):
        fake_llm(RuntimeError("no model loaded"))
        job, kwargs = _job("edit"), {"prompt": "south park him"}
        asyncio.run(media_tools._assist(job, kwargs))
        assert kwargs == {"prompt": "south park him"}
        assert "no model loaded" in job["assist_error"]
        assert "assisted_prompt" not in job


def test_catalog_offers_assist_only_with_a_chat_model(fake_llm, monkeypatch):
    monkeypatch.setattr(media_tools, "list_models", lambda tool: [])
    from core.services import voice_io
    monkeypatch.setattr(voice_io, "list_voices", lambda: [])
    tools = {t["id"]: t["assist"] for t in media_tools.catalog()["tools"]}
    assert not any(tools.values())
    fake_llm("")
    tools = {t["id"]: t["assist"] for t in media_tools.catalog()["tools"]}
    assert {k for k, v in tools.items() if v} == {"image", "edit"}
