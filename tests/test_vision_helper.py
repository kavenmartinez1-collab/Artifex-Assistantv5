"""The CPU vision helper (core.vision_helper) for images the GPU can't serve.

The Create tab's prompt assist used to rewrite an edit request blind: for a
cat photo and "make it winter" it wrote a snowy forest with "no animals" in
the negative prompt. With a vision_helper configured, the assist sees the
source photo; the chat model stays as the fallback. The agent's image tools
fall back to the helper when the run's own model has no projector.
"""

import asyncio

import pytest
from PIL import Image

from core import vision_helper
from core.services import media_tools
from tools import image_tools

BLOCK = {"path": "x.gguf", "port": 8095, "assist_tools": ["edit", "image"]}


@pytest.fixture
def helper(monkeypatch):
    """Configured helper whose chat() records the messages it was given."""
    sent = {}

    def fake_chat(messages, max_tokens=600, **kw):
        sent["messages"] = messages
        if sent.get("fail"):
            raise ConnectionError("helper down")
        return '{"prompt": "An orange cat in a snowy garden", "negative": "no cat"}'

    monkeypatch.setattr(vision_helper, "config", lambda: dict(BLOCK))
    monkeypatch.setattr(vision_helper, "chat", fake_chat)
    yield sent
    media_tools.set_prompt_llm(None)


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "cat.png"
    Image.new("RGB", (64, 48), "orange").save(p)
    return str(p)


def test_edit_assist_sees_the_photo(helper, photo):
    job, kwargs = {"id": "j", "tool": "edit", "started": None}, {
        "prompt": "make it winter", "image_path": photo}
    asyncio.run(media_tools._assist(job, kwargs))
    user = helper["messages"][1]["content"]
    assert user[0]["type"] == "image_url"
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert user[1] == {"type": "text", "text": "make it winter"}
    assert helper["messages"][0]["content"] == media_tools._ASSIST_SYSTEM["edit"]
    assert kwargs["prompt"] == "An orange cat in a snowy garden"
    assert kwargs["negative_prompt"] == "no cat"


def test_image_assist_is_text_only(helper):
    kwargs = {"prompt": "a cat"}
    asyncio.run(media_tools._assist({"id": "j", "tool": "image", "started": None}, kwargs))
    assert helper["messages"][1]["content"] == "a cat"
    assert kwargs == {"prompt": "An orange cat in a snowy garden"}


def test_helper_failure_falls_back_to_the_chat_model(helper, photo):
    helper["fail"] = True
    async def llm(messages):
        return '{"prompt": "chat model prompt", "negative": ""}'
    media_tools.set_prompt_llm(llm)
    kwargs = {"prompt": "make it winter", "image_path": photo}
    asyncio.run(media_tools._assist({"id": "j", "tool": "edit", "started": None}, kwargs))
    assert kwargs["prompt"] == "chat model prompt"


def test_helper_failure_without_a_chat_model_keeps_the_prompt(helper):
    helper["fail"] = True
    job, kwargs = {"id": "j", "tool": "edit", "started": None}, {"prompt": "x"}
    asyncio.run(media_tools._assist(job, kwargs))
    assert kwargs == {"prompt": "x"} and "helper down" in job["assist_error"]


def test_assist_is_offered_with_only_the_helper(helper):
    media_tools.set_prompt_llm(None)
    assert media_tools.assist_available("edit")
    assert media_tools.assist_available("image")
    assert not media_tools.assist_available("music")


def test_agent_image_tools_fall_back_to_the_helper(monkeypatch, photo):
    monkeypatch.delenv("ARTIFEX_VISION_URL", raising=False)
    monkeypatch.setattr(image_tools, "check_vision_server",
                        lambda url=None: (False, "no projector"))
    monkeypatch.setattr(vision_helper, "config", lambda: dict(BLOCK))
    monkeypatch.setattr(vision_helper, "ensure_url", lambda: "http://127.0.0.1:8095")
    seen = {}

    def fake_ask(path, prompt, max_tokens=200, base_url=None):
        seen["url"] = base_url
        return "an orange square"
    monkeypatch.setattr(image_tools, "ask_about_image", fake_ask)
    ok, out = image_tools.run_view_image(photo)
    assert ok and "an orange square" in out
    assert seen["url"] == "http://127.0.0.1:8095"


def test_agent_image_tools_prefer_a_seeing_main_model(monkeypatch):
    monkeypatch.delenv("ARTIFEX_VISION_URL", raising=False)
    monkeypatch.setattr(image_tools, "check_vision_server", lambda url=None: (True, ""))
    monkeypatch.setattr(vision_helper, "ensure_url",
                        lambda: pytest.fail("helper must not start"))
    assert image_tools.resolve_vision_url() == ("http://127.0.0.1:8081", "")


def test_no_helper_keeps_the_old_message(monkeypatch):
    monkeypatch.delenv("ARTIFEX_VISION_URL", raising=False)
    monkeypatch.setattr(image_tools, "check_vision_server",
                        lambda url=None: (False, "launched without vision"))
    monkeypatch.setattr(vision_helper, "config", lambda: None)
    assert image_tools.resolve_vision_url() == (None, "launched without vision")
