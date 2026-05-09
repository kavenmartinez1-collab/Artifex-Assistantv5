"""Tests for the claude_cli engine — subprocess invocation hygiene,
auth probe behavior, OpenAI-shape integration, and message conversion.
"""

import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch


class TestMessageConversion(unittest.TestCase):
    """OpenAI-format messages → (system_prompt, user_prompt) for claude --print."""

    def test_system_and_user_split(self):
        from core.engine_claude_cli import _messages_to_cli_inputs
        msgs = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi."},
        ]
        sys_p, user_p = _messages_to_cli_inputs(msgs)
        self.assertEqual(sys_p, "Be helpful.")
        self.assertEqual(user_p, "Hi.")

    def test_assistant_history_folded_into_user(self):
        """claude --print is single-turn; prior assistant turns get
        labeled and concatenated into the user prompt."""
        from core.engine_claude_cli import _messages_to_cli_inputs
        msgs = [
            {"role": "user", "content": "First Q"},
            {"role": "assistant", "content": "First A"},
            {"role": "user", "content": "Follow-up"},
        ]
        sys_p, user_p = _messages_to_cli_inputs(msgs)
        self.assertEqual(sys_p, "")
        self.assertIn("First Q", user_p)
        self.assertIn("[Previous assistant response]", user_p)
        self.assertIn("First A", user_p)
        self.assertIn("Follow-up", user_p)

    def test_multimodal_content_drops_images_keeps_text(self):
        """Image parts in OpenAI multimodal arrays are dropped — vision
        routes through other backends.  Text parts survive."""
        from core.engine_claude_cli import _messages_to_cli_inputs
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ],
        }]
        _sys, user_p = _messages_to_cli_inputs(msgs)
        self.assertEqual(user_p, "Describe this:")


class TestProbeAuthState(unittest.TestCase):
    """probe_auth_state must check both binary presence + ~/.claude/
    directory; either missing → ok=False with a useful reason."""

    def test_binary_missing_returns_false(self):
        from core.engine_claude_cli import probe_auth_state
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, reason = probe_auth_state("missing-claude")
        self.assertFalse(ok)
        self.assertIn("not on PATH", reason)

    def test_version_nonzero_exit_returns_false(self):
        from core.engine_claude_cli import probe_auth_state
        fake_proc = MagicMock(returncode=1, stderr=b"some error")
        with patch("subprocess.run", return_value=fake_proc):
            ok, reason = probe_auth_state()
        self.assertFalse(ok)
        self.assertIn("exit 1", reason)

    def test_oauth_dir_missing_returns_false(self):
        from core.engine_claude_cli import probe_auth_state
        fake_proc = MagicMock(returncode=0, stdout=b"1.0.0", stderr=b"")
        with patch("subprocess.run", return_value=fake_proc), \
             patch("os.path.isdir", return_value=False):
            ok, reason = probe_auth_state()
        self.assertFalse(ok)
        self.assertIn("claude /login", reason)

    def test_both_present_returns_true(self):
        from core.engine_claude_cli import probe_auth_state
        fake_proc = MagicMock(returncode=0, stdout=b"1.0.0", stderr=b"")
        with patch("subprocess.run", return_value=fake_proc), \
             patch("os.path.isdir", return_value=True):
            ok, reason = probe_auth_state()
        self.assertTrue(ok)
        self.assertEqual(reason, "")


class TestEngineLifecycle(unittest.TestCase):
    """ClaudeCliEngine.load() / is_loaded() / unload() behavior."""

    def test_load_raises_when_unauthed(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-sonnet-4-6")
        with patch("core.engine_claude_cli.probe_auth_state",
                   return_value=(False, "not installed")):
            with self.assertRaises(RuntimeError) as cm:
                engine.load()
        self.assertIn("claude CLI not available", str(cm.exception))
        self.assertFalse(engine.is_loaded())

    def test_load_succeeds_when_authed(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-sonnet-4-6")
        with patch("core.engine_claude_cli.probe_auth_state",
                   return_value=(True, "")):
            engine.load()
        self.assertTrue(engine.is_loaded())

    def test_unload_releases(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-sonnet-4-6")
        with patch("core.engine_claude_cli.probe_auth_state",
                   return_value=(True, "")):
            engine.load()
        engine.unload()
        # After unload, engine still reports authed via the cache, but
        # the loaded flag is false until next load().
        self.assertFalse(engine._loaded)

    def test_needs_reload_is_always_false(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-sonnet-4-6")
        self.assertFalse(engine.needs_reload())

    def test_get_context_size_from_config(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine(
            "claude-sonnet-4-6", {"context_length": 200000},
        )
        self.assertEqual(engine.get_context_size(), 200000)


class TestEngineGeneration(unittest.TestCase):
    """Subprocess invocation: list args, no shell, ANTHROPIC_API_KEY stripped."""

    def _make_engine(self):
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-sonnet-4-6")
        # Force the engine to think it's loaded without running probes.
        engine._loaded = True
        engine._auth_ok = True
        engine._auth_checked_at = 1e18  # far future, skips re-probe
        return engine

    def test_generate_uses_subprocess_exec_not_shell(self):
        """Regression: the downstream-service version had subprocess_shell with
        shell-pipe interpolation in its long-prompt path.  The new
        engine must use list args + stdin redirection only."""
        engine = self._make_engine()
        fake_proc = MagicMock(
            returncode=0,
            stdout=b"hello world",
            stderr=b"",
        )
        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.7,
            )
        call = mock_run.call_args
        # First positional arg must be a list (exec form), not a string.
        self.assertIsInstance(call.args[0], list)
        self.assertEqual(call.args[0][0], engine._cli_path)
        # No shell=True kwarg (subprocess.run defaults to False, but
        # confirm it's not flipped by mistake).
        self.assertNotEqual(call.kwargs.get("shell"), True)
        # User prompt goes via stdin, not in args.
        self.assertEqual(call.kwargs.get("input"), b"hi")

    def test_anthropic_api_key_stripped_from_env(self):
        """Force OAuth subscription path — the API key env var would
        otherwise switch the CLI into billed mode."""
        engine = self._make_engine()
        fake_proc = MagicMock(returncode=0, stdout=b"x", stderr=b"")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}), \
             patch("subprocess.run", return_value=fake_proc) as mock_run:
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.7,
            )
        env_passed = mock_run.call_args.kwargs["env"]
        self.assertNotIn("ANTHROPIC_API_KEY", env_passed)

    def test_model_arg_passed(self):
        engine = self._make_engine()
        fake_proc = MagicMock(returncode=0, stdout=b"x", stderr=b"")
        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.7,
            )
        cmd = mock_run.call_args.args[0]
        # --model claude-sonnet-4-6 should be in the args list.
        self.assertIn("--model", cmd)
        self.assertIn("claude-sonnet-4-6", cmd)

    def test_claude_code_sentinel_skips_model_arg(self):
        """The claude-code sentinel runs the CLI without --model so it
        uses its current default."""
        from core.engine_claude_cli import ClaudeCliEngine
        engine = ClaudeCliEngine("claude-code")
        engine._loaded = True
        engine._auth_ok = True
        engine._auth_checked_at = 1e18
        fake_proc = MagicMock(returncode=0, stdout=b"x", stderr=b"")
        with patch("subprocess.run", return_value=fake_proc) as mock_run:
            engine.generate_streaming(
                [{"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.7,
            )
        cmd = mock_run.call_args.args[0]
        self.assertNotIn("--model", cmd)

    def test_nonzero_exit_raises_with_stderr(self):
        engine = self._make_engine()
        fake_proc = MagicMock(
            returncode=2, stdout=b"", stderr=b"auth required",
        )
        with patch("subprocess.run", return_value=fake_proc):
            with self.assertRaises(RuntimeError) as cm:
                engine.generate_streaming(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=100, temperature=0.7,
                )
        self.assertIn("auth required", str(cm.exception))
        self.assertIn("with code 2", str(cm.exception))

    def test_timeout_raises(self):
        engine = self._make_engine()
        with patch("subprocess.run",
                   side_effect=subprocess.TimeoutExpired("claude", 5)):
            with self.assertRaises(TimeoutError):
                engine.generate_streaming(
                    [{"role": "user", "content": "hi"}],
                    max_tokens=100, temperature=0.7,
                )

    def test_on_token_emits_full_response_as_single_chunk(self):
        """Non-streaming CLI invocation; emit the whole response as one
        token chunk so streaming-interface callers still receive it."""
        engine = self._make_engine()
        fake_proc = MagicMock(returncode=0, stdout=b"the answer", stderr=b"")
        emitted = []
        with patch("subprocess.run", return_value=fake_proc):
            result = engine.generate_streaming(
                [{"role": "user", "content": "hi"}],
                max_tokens=100, temperature=0.7,
                on_token=emitted.append,
            )
        self.assertEqual(result, "the answer")
        self.assertEqual(emitted, ["the answer"])


class TestDiscoveryAndConfig(unittest.TestCase):
    """Wiring: model_discovery picks up claude_cli config; engine_factory
    builds the right class; set_active_model handles the new backend."""

    def test_discover_claude_cli_reads_config(self):
        from core.model_discovery import _discover_claude_cli
        fake_models = {
            "claude-sonnet-4-6": {
                "context_length": 200000,
                "recommended_max_completion": 8192,
                "capabilities": ["text"],
            },
        }
        with patch("core.config.get_claude_cli_models", return_value=fake_models), \
             patch("core.engine_claude_cli.probe_auth_state",
                   return_value=(True, "")):
            models = _discover_claude_cli()
        self.assertEqual(len(models), 1)
        m = models[0]
        self.assertEqual(m["id"], "claude-sonnet-4-6")
        self.assertEqual(m["backend"], "claude_cli")
        self.assertEqual(m["context_length"], 200000)
        self.assertEqual(m["recommended_max_completion"], 8192)
        self.assertTrue(m["claude_cli_authed"])

    def test_discover_claude_cli_when_unauthed_marks_false(self):
        from core.model_discovery import _discover_claude_cli
        fake_models = {"claude-sonnet-4-6": {"context_length": 200000}}
        with patch("core.config.get_claude_cli_models", return_value=fake_models), \
             patch("core.engine_claude_cli.probe_auth_state",
                   return_value=(False, "not installed")):
            models = _discover_claude_cli()
        self.assertFalse(models[0]["claude_cli_authed"])

    def test_engine_factory_creates_claude_cli_engine(self):
        # The factory imports these names at module load time, so patching
        # core.config wouldn't reach the factory's bound references.  Patch
        # the names where they're actually used.
        from core.engine_factory import create_engine
        from core.engine_claude_cli import ClaudeCliEngine
        with patch("core.engine_factory.get_active_backend", return_value="claude_cli"), \
             patch("core.engine_factory.get_active_claude_cli_model",
                   return_value="claude-sonnet-4-6"), \
             patch("core.engine_factory.get_claude_cli_model_config",
                   return_value={"context_length": 200000}):
            engine = create_engine()
        self.assertIsInstance(engine, ClaudeCliEngine)
        self.assertEqual(engine.model_name, "claude-sonnet-4-6")

    def test_engine_factory_raises_when_no_models(self):
        from core.engine_factory import create_engine
        with patch("core.engine_factory.get_active_backend", return_value="claude_cli"), \
             patch("core.engine_factory.get_active_claude_cli_model",
                   return_value=None):
            with self.assertRaises(RuntimeError) as cm:
                create_engine()
        self.assertIn("claude_cli_config.json", str(cm.exception))

    def test_set_active_model_logs_unknown_claude_cli(self):
        import core.config as cfg
        original_backend = cfg._active_backend
        try:
            cfg._active_backend = "claude_cli"
            with patch("core.config.get_claude_cli_models",
                       return_value={"claude-sonnet-4-6": {}}), \
                 self.assertLogs("core.config", level="WARNING") as logs:
                ok = cfg.set_active_model("nonexistent-claude")
            self.assertFalse(ok)
            self.assertTrue(any("nonexistent-claude" in m for m in logs.output))
        finally:
            cfg._active_backend = original_backend


if __name__ == "__main__":
    unittest.main()
