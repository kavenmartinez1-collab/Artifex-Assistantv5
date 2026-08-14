"""Tests for llama.cpp engine — P1 features: grammar, cache_prompt, tokenize,
enable_thinking, configurable health timeout.
"""

import json
import os
import unittest
from unittest.mock import patch, MagicMock


class TestHealthTimeoutConfig(unittest.TestCase):
    """P1-T10: HEALTH_TIMEOUT is configurable via env var and per-model config."""

    def test_default_timeout_is_120(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_HEALTH_TIMEOUT", None)
            import importlib
            import core.engine_llama_cpp as mod
            importlib.reload(mod)
            self.assertEqual(mod.DEFAULT_HEALTH_TIMEOUT, 120)

    def test_env_var_overrides_default(self):
        with patch.dict(os.environ, {"ARTIFEX_HEALTH_TIMEOUT": "300"}):
            import importlib
            import core.engine_llama_cpp as mod
            importlib.reload(mod)
            self.assertEqual(mod.HEALTH_TIMEOUT, 300)

    def test_per_model_config_overrides_module(self):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {
            "path": "/fake/model.gguf",
            "health_timeout": 240,
        })
        self.assertEqual(engine._health_timeout, 240)

    def test_per_model_falls_back_to_module(self):
        from core.engine_llama_cpp import LlamaCppEngine, HEALTH_TIMEOUT
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        self.assertEqual(engine._health_timeout, HEALTH_TIMEOUT)


class TestGrammarPayload(unittest.TestCase):
    """P1-T6: Grammar and response_format pass through to llama-server payload."""

    def _make_engine(self):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        engine._loaded = True
        engine._base_url = "http://localhost:8081"
        engine._is_server_healthy = lambda: True
        return engine

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_grammar_in_payload(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "test"}],
            max_tokens=100, temperature=0.7,
            grammar='root ::= "yes" | "no"',
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["grammar"], 'root ::= "yes" | "no"')
        self.assertTrue(body["cache_prompt"])

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_response_format_in_payload(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        fmt = {"type": "json_object"}
        engine.generate_streaming(
            [{"role": "user", "content": "test"}],
            max_tokens=100, temperature=0.7,
            response_format=fmt,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["response_format"], {"type": "json_object"})

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_no_grammar_when_none(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "test"}],
            max_tokens=100, temperature=0.7,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertNotIn("grammar", body)
        self.assertNotIn("response_format", body)


class TestEnableThinking(unittest.TestCase):
    """enable_thinking=False is passed to llama-server via chat_template_kwargs."""

    def _make_engine(self):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        engine._loaded = True
        engine._base_url = "http://localhost:8081"
        engine._is_server_healthy = lambda: True
        return engine

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_thinking_disabled_sets_chat_template_kwargs(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=False,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        # enable_thinking goes through chat_template_kwargs, not a /no_think
        # text injection — the system message must be left untouched.
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["reasoning_format"], "none")
        self.assertEqual(body["messages"][0]["content"], "Be helpful.")

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_thinking_disabled_does_not_inject_system_message(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=False,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        # No synthetic system message — the user message stays first.
        self.assertEqual(body["messages"][0]["role"], "user")
        self.assertEqual(body["messages"][0]["content"], "hi")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_thinking_enabled_no_injection(self, mock_urlopen):
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "system", "content": "Be helpful."}, {"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=True,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        # Thinking enabled: no chat_template_kwargs, no reasoning_format override.
        self.assertEqual(body["messages"][0]["content"], "Be helpful.")
        self.assertNotIn("chat_template_kwargs", body)
        self.assertNotIn("reasoning_format", body)

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_reasoning_effort_in_chat_template_kwargs(self, mock_urlopen):
        """A valid reasoning_effort rides along in chat_template_kwargs."""
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=True,
            reasoning_effort="medium",
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["chat_template_kwargs"],
                         {"reasoning_effort": "medium"})
        # Thinking is still on — no reasoning_format override.
        self.assertNotIn("reasoning_format", body)

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_reasoning_effort_omitted_by_default(self, mock_urlopen):
        """No reasoning_effort argument leaves the payload untouched."""
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=True,
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertNotIn("chat_template_kwargs", body)

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_reasoning_effort_does_not_clobber_disabled_thinking(self, mock_urlopen):
        """With thinking off, the enable_thinking kwarg wins — there is no
        think block left to budget."""
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        engine.generate_streaming(
            [{"role": "user", "content": "hi"}],
            max_tokens=100, temperature=0.7,
            enable_thinking=False,
            reasoning_effort="low",
        )

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        body = json.loads(req.data.decode())
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["reasoning_format"], "none")

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_reasoning_content_wrapped_in_think_tags(self, mock_urlopen):
        """When llama-server emits reasoning_content, on_token receives <think> tags."""
        engine = self._make_engine()
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.__iter__ = MagicMock(return_value=iter([
            b'data: {"choices":[{"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}\n',
            b'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n',
            b'data: [DONE]\n',
        ]))
        mock_urlopen.return_value = mock_resp

        tokens = []
        result = engine.generate_streaming(
            [{"role": "user", "content": "test"}],
            max_tokens=100, temperature=0.7,
            on_token=lambda t: tokens.append(t),
        )
        raw = "".join(tokens)
        self.assertIn("<think>", raw)
        self.assertIn("hmm", raw)
        self.assertIn("</think>", raw)
        self.assertIn("answer", raw)
        self.assertIn("answer", result)
        self.assertNotIn("<think>", result)


class TestCountTokens(unittest.TestCase):
    """P1-T8: count_tokens uses llama-server /tokenize when available."""

    def test_fallback_heuristic(self):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        self.assertEqual(engine.count_tokens("hello world test"), len("hello world test") // 4)

    @patch("core.engine_llama_cpp.urllib.request.urlopen")
    def test_uses_server_tokenize(self, mock_urlopen):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        engine._loaded = True

        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps({"tokens": [1, 2, 3, 4, 5]}).encode()
        mock_urlopen.return_value = mock_resp

        count = engine.count_tokens("hello world test")
        self.assertEqual(count, 5)


class TestNeedsReloadDocumented(unittest.TestCase):
    """P1-T11: needs_reload() has per-engine docstrings."""

    def test_llama_cpp_docstring(self):
        from core.engine_llama_cpp import LlamaCppEngine
        self.assertIn("False", LlamaCppEngine.needs_reload.__doc__)

    def test_ollama_docstring(self):
        from core.engine_ollama import OllamaEngine
        self.assertIn("False", OllamaEngine.needs_reload.__doc__)

    def test_transformers_docstring(self):
        from core.engine_transformers import TransformersEngine
        self.assertIn("True", TransformersEngine.needs_reload.__doc__)


class TestAdoptedServerUnload(unittest.TestCase):
    """Regression: unload() must terminate the underlying server even when
    load() adopted a process we didn't spawn. Without this, a model/ctx-tier
    switch can be a no-op at the OS level even though the queue thinks it
    succeeded — the next adoption re-grabs the same orphan."""

    def _make_engine(self):
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/model.gguf"})
        engine._loaded = True
        return engine

    def test_kill_process_calls_listener_kill_when_no_subprocess(self):
        """When adopted (self._process is None) and a healthy server
        exists, _kill_process must fall through to _kill_listener_on_port."""
        engine = self._make_engine()
        engine._process = None
        engine._is_server_healthy = lambda: True
        called = {"n": 0}
        engine._kill_listener_on_port = lambda: called.update(n=called["n"] + 1)
        engine._kill_process()
        self.assertEqual(called["n"], 1)
        self.assertIsNone(engine._process)

    def test_kill_process_skips_listener_kill_when_no_server(self):
        """If no server is healthy, no PID lookup is attempted (avoids
        killing unrelated processes that may be on the port)."""
        engine = self._make_engine()
        engine._process = None
        engine._is_server_healthy = lambda: False
        called = {"n": 0}
        engine._kill_listener_on_port = lambda: called.update(n=called["n"] + 1)
        engine._kill_process()
        self.assertEqual(called["n"], 0)

    def test_kill_process_uses_subprocess_when_we_own_it(self):
        """When we have a Popen handle, _kill_listener_on_port is NOT
        called — terminate the subprocess directly."""
        engine = self._make_engine()
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        engine._process = proc
        engine._is_server_healthy = lambda: True  # would trip listener path if reached
        called = {"n": 0}
        engine._kill_listener_on_port = lambda: called.update(n=called["n"] + 1)
        engine._kill_process()
        self.assertEqual(called["n"], 0)
        proc.terminate.assert_called_once()
        self.assertIsNone(engine._process)

    def test_kill_listener_on_port_only_kills_llama_server(self):
        """The port-based killer must scope to processes named
        'llama-server' so an unrelated service that happens to bind
        the port is not taken down."""
        from core.engine_llama_cpp import LlamaCppEngine
        engine = LlamaCppEngine("test", {"path": "/fake/m.gguf", "port": 8081})

        unrelated = MagicMock()
        unrelated.info = {"pid": 100, "name": "node.exe", "exe": "C:/.../node.exe"}
        # Even if it claims the port, executable name must rule it out.
        unrelated.net_connections.return_value = []

        target = MagicMock()
        target.info = {"pid": 200, "name": "llama-server.exe",
                       "exe": "C:/.../llama-server.exe"}
        target.pid = 200
        conn = MagicMock()
        conn.laddr = MagicMock(port=8081)
        import psutil as _psutil
        conn.status = _psutil.CONN_LISTEN
        target.net_connections.return_value = [conn]

        with patch("psutil.process_iter", return_value=[unrelated, target]):
            engine._kill_listener_on_port()

        unrelated.terminate.assert_not_called()
        target.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
