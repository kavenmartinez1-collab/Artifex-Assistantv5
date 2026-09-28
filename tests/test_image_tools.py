"""Tests for the agent's image tools: @view_image and @describe_images.

The agent's turns are text-only, so these tools post each picture to the
loaded llama-server themselves. A fake server stands in for it here: /props
advertises (or withholds) vision, and /v1/chat/completions answers with a
description naming the image it was sent, so a test can tell which picture
landed on which catalog line.
"""

import http.server
import json
import threading

import pytest
from PIL import Image

from core.sandbox.fs_sandbox import extract_paths_from_content
from core.sandbox.policy import ACTION_RISK, RiskLevel
from tools.agent_tools import extract_agent_actions, rebase_action_paths
from tools import image_tools
from tools.image_tools import (
    catalogued_names, list_images, run_describe_images, run_view_image,
)


class _FakeLlama(http.server.BaseHTTPRequestHandler):
    vision = True
    fail_after = None     # 500 on every completion after this many
    calls = []

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/props":
            self._json(200, {"model_path": "C:/m/fake.gguf",
                             "modalities": {"vision": type(self).vision}})
        else:
            self._json(404, {})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        if cls.fail_after is not None and len(cls.calls) >= cls.fail_after:
            self._json(500, {"error": "boom"})
            return
        cls.calls.append(req)
        parts = req["messages"][0]["content"]
        url = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
        n = len(cls.calls)
        self._json(200, {"choices": [{"message": {
            # the empty think block Qwen3.x emits even with thinking off
            "content": f"<think>\n\n</think>\n\npicture {n}\nsecond line "
                       f"({len(url)} b64 chars)"}}]})


@pytest.fixture
def server(monkeypatch):
    _FakeLlama.vision = True
    _FakeLlama.fail_after = None
    _FakeLlama.calls = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _FakeLlama)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    monkeypatch.setenv("ARTIFEX_VISION_URL", f"http://127.0.0.1:{httpd.server_address[1]}")
    yield _FakeLlama
    httpd.shutdown()


@pytest.fixture
def photos(tmp_path):
    folder = tmp_path / "photos"
    (folder / "sub").mkdir(parents=True)
    (folder / ".git").mkdir()
    for name, color in (("b.png", "red"), ("A.jpg", "blue"), ("sub/c.webp", "green"),
                        (".git/x.png", "black")):
        Image.new("RGB", (2000, 1000), color).save(folder / name)
    (folder / "notes.txt").write_text("not an image")
    return folder


class TestParsing:
    def test_view_image_marker(self):
        acts = extract_agent_actions('@view_image("C:/p/a.jpg")\n'
                                     '@view_image("b.png", "what breed is it?")')
        assert [(a.type, a.content) for a in acts] == [
            ("view_image", "C:/p/a.jpg"), ("view_image", "b.png|what breed is it?")]

    def test_describe_images_marker_forms(self):
        acts = extract_agent_actions(
            '@describe_images("D:/pics")\n'
            '@describe_images("D:/pics", "list.md")\n'
            '@describe_images("D:/pics", "list.md", "name the dog breed")')
        assert [a.content for a in acts] == [
            "D:/pics", "D:/pics|list.md", "D:/pics|list.md|name the dog breed"]
        assert {a.type for a in acts} == {"describe_images"}

    def test_backticked_marker_is_inert(self):
        assert extract_agent_actions('use `@describe_images("D:/pics")` for that') == []

    def test_native_xml_and_json_calls(self):
        xml = ("<tool_call>\n<function=describe_images>\n"
               "<parameter=folder>D:/pics</parameter>\n"
               "<parameter=output>cat.md</parameter>\n</function>\n</tool_call>")
        js = '<tool_call>{"name": "view_image", "arguments": {"path": "a.jpg"}}</tool_call>'
        bare = "<tool_call>\n<function=view_image>\nz.png\n</function>\n</tool_call>"
        got = [(a.type, a.content) for a in
               extract_agent_actions(xml) + extract_agent_actions(js)
               + extract_agent_actions(bare)]
        assert got == [("describe_images", "D:/pics|cat.md"),
                       ("view_image", "a.jpg"), ("view_image", "z.png")]


class TestSandboxWiring:
    def test_risk_levels_registered(self):
        # ACTION_RISK doubles as the capability list — absent means refused.
        assert ACTION_RISK["view_image"] == RiskLevel.SAFE
        assert ACTION_RISK["describe_images"] == RiskLevel.MEDIUM

    def test_paths_extracted_for_sandbox(self):
        assert extract_paths_from_content("view_image", "C:/a.jpg|q") == ["C:/a.jpg"]
        assert extract_paths_from_content(
            "describe_images", "C:/pics|C:/out.md|a | b") == ["C:/pics", "C:/out.md"]
        assert extract_paths_from_content("describe_images", "C:/pics") == ["C:/pics"]

    def test_rebase(self, tmp_path):
        cwd = str(tmp_path)
        assert rebase_action_paths("view_image", "a.jpg|what?", cwd) == \
            f"{tmp_path / 'a.jpg'}|what?"
        # empty catalog stays empty (means "inside the folder")
        assert rebase_action_paths("describe_images", "pics||q", cwd) == \
            f"{tmp_path / 'pics'}||q"
        assert rebase_action_paths("describe_images", "pics|o.md", cwd) == \
            f"{tmp_path / 'pics'}|{tmp_path / 'o.md'}"


class TestDescribeImages:
    def test_lists_images_recursively_skipping_hidden(self, photos):
        assert list_images(str(photos)) == ["A.jpg", "b.png", "sub/c.webp"]

    def test_writes_catalog_then_resumes(self, server, photos):
        ok, out = run_describe_images(str(photos))
        catalog = photos / "image_catalog.md"
        assert ok, out
        assert "0 remaining" in out
        lines = catalog.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# Images in ")
        assert lines[2:] == [
            "- A.jpg — picture 1 second line (" + lines[2].split("(")[1],
            "- b.png — picture 2 second line (" + lines[3].split("(")[1],
            "- sub/c.webp — picture 3 second line (" + lines[4].split("(")[1],
        ]
        # images were shrunk to <= 1024 px before sending
        assert all(len(c["messages"][0]["content"]) == 2 for c in server.calls)

        (photos / "new.png").write_bytes((photos / "b.png").read_bytes())
        ok, out = run_describe_images(str(photos))
        assert ok and "Described 1 image" in out and "3 were already listed" in out
        assert catalogued_names(catalog) == {"A.jpg", "b.png", "sub/c.webp", "new.png"}

        ok, out = run_describe_images(str(photos))
        assert "Nothing left to do" in out
        assert len(server.calls) == 4

    def test_custom_catalog_and_prompt(self, server, photos, tmp_path):
        out_file = tmp_path / "out" / "list.md"
        ok, _ = run_describe_images(f"{photos}|{out_file}|is there a dog?")
        assert ok and out_file.exists()
        assert server.calls[0]["messages"][0]["content"][0]["text"] == "is there a dog?"
        assert server.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}

    def test_time_budget_stops_and_reports_remaining(self, server, photos):
        ok, out = run_describe_images(str(photos), time_budget=0)
        # one image is always done, so repeated calls can't stall at zero
        assert ok and "Described 1 image" in out
        assert "2 remaining (stopped: time budget)" in out
        assert "call @describe_images again" in out.lower()

    def test_server_failure_does_not_mark_images_done(self, server, photos):
        server.fail_after = 1
        ok, out = run_describe_images(str(photos))
        assert "2 remaining (stopped: vision server error" in out
        assert catalogued_names(photos / "image_catalog.md") == {"A.jpg"}
        server.fail_after = None
        ok, out = run_describe_images(str(photos))
        assert ok and "0 remaining" in out

    def test_unreadable_image_is_recorded_not_retried_forever(self, server, photos):
        (photos / "broken.jpg").write_bytes(b"not really a jpeg")
        ok, out = run_describe_images(str(photos))
        assert ok and "1 unreadable" in out and "0 remaining" in out
        text = (photos / "image_catalog.md").read_text(encoding="utf-8")
        assert "- broken.jpg — [could not read:" in text

    def test_text_only_model_is_refused_clearly(self, server, photos):
        server.vision = False
        ok, out = run_describe_images(str(photos))
        assert not ok and "without vision" in out and "fake.gguf" in out
        assert not (photos / "image_catalog.md").exists()
        assert server.calls == []

    def test_missing_folder(self, server, tmp_path):
        ok, out = run_describe_images(str(tmp_path / "nope"))
        assert not ok and "Folder not found" in out

    def test_no_server(self, monkeypatch, photos):
        monkeypatch.setenv("ARTIFEX_VISION_URL", "http://127.0.0.1:9")
        ok, out = run_describe_images(str(photos))
        assert not ok and "No model server" in out


class TestViewImage:
    def test_view_image_returns_description(self, server, photos):
        ok, out = run_view_image(f"{photos / 'A.jpg'}|what colour is it?")
        assert ok and out.startswith("[A.jpg]") and "picture 1" in out
        assert server.calls[0]["messages"][0]["content"][0]["text"] == "what colour is it?"

    def test_view_image_missing_file(self, server, tmp_path):
        ok, out = run_view_image(str(tmp_path / "gone.png"))
        assert not ok and "not found" in out
