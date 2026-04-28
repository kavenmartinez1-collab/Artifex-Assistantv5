"""Tests for core.sandbox.ratchet — P6-T31 self-modification ratchet."""

import os
import unittest
from unittest.mock import patch

from core.sandbox.ratchet import (
    is_protected,
    _extract_write_paths,
    _ratchet_hook,
    install,
)
from core.sandbox.policy import RiskLevel, clear_policy_hooks, check_policy


class TestProtectedPaths(unittest.TestCase):
    """Protected path detection."""

    def test_sandbox_module_protected(self):
        self.assertIsNotNone(is_protected("core/sandbox/policy.py"))
        self.assertIsNotNone(is_protected("core\\sandbox\\audit.py"))

    def test_core_config_protected(self):
        self.assertIsNotNone(is_protected("core/config.py"))

    def test_ci_workflows_protected(self):
        self.assertIsNotNone(is_protected(".github/workflows/tests.yml"))

    def test_env_file_protected(self):
        self.assertIsNotNone(is_protected(".env"))

    def test_pyproject_protected(self):
        self.assertIsNotNone(is_protected("pyproject.toml"))

    def test_normal_files_not_protected(self):
        self.assertIsNone(is_protected("core/engine_ollama.py"))
        self.assertIsNone(is_protected("ui/cli_assistant.py"))
        self.assertIsNone(is_protected("tools/agent_tools.py"))

    def test_extra_patterns_from_env(self):
        with patch.dict(os.environ, {"ARTIFEX_RATCHET_EXTRA": "secrets.yaml"}):
            self.assertIsNotNone(is_protected("deploy/secrets.yaml"))


class TestWritePathExtraction(unittest.TestCase):
    """Extract write targets from action content."""

    def test_edit_file(self):
        paths = _extract_write_paths("edit_file", "core/sandbox/policy.py\x00old\x00new")
        self.assertEqual(paths, ["core/sandbox/policy.py"])

    def test_shell_redirect(self):
        paths = _extract_write_paths("shell", "echo hack > .env")
        self.assertIn(".env", paths)

    def test_shell_git_add(self):
        paths = _extract_write_paths("shell", "git add core/sandbox/policy.py")
        self.assertIn("core/sandbox/policy.py", paths)

    def test_no_write_in_read(self):
        paths = _extract_write_paths("read_file", "core/sandbox/policy.py")
        self.assertEqual(paths, [])


class TestRatchetHook(unittest.TestCase):
    """Policy hook blocks writes to protected paths."""

    def test_blocks_sandbox_edit(self):
        d = _ratchet_hook("edit_file", "core/sandbox/policy.py\x00old\x00new", RiskLevel.MEDIUM)
        self.assertIsNotNone(d)
        self.assertFalse(d.allowed)
        self.assertEqual(d.matched_rule, "self_mod_ratchet")

    def test_blocks_env_redirect(self):
        d = _ratchet_hook("shell", "echo SECRET > .env", RiskLevel.HIGH)
        self.assertIsNotNone(d)
        self.assertFalse(d.allowed)

    def test_allows_normal_edit(self):
        d = _ratchet_hook("edit_file", "ui/cli_assistant.py\x00old\x00new", RiskLevel.MEDIUM)
        self.assertIsNone(d)

    def test_ignores_reads(self):
        d = _ratchet_hook("read_file", "core/sandbox/policy.py|1", RiskLevel.SAFE)
        self.assertIsNone(d)

    def test_ignores_glob(self):
        d = _ratchet_hook("glob", "core/sandbox/*.py", RiskLevel.SAFE)
        self.assertIsNone(d)


class TestRatchetInstall(unittest.TestCase):
    """install() registers the hook."""

    def setUp(self):
        clear_policy_hooks()

    def tearDown(self):
        clear_policy_hooks()

    def test_install_blocks_sandbox_edit(self):
        install()
        with patch.dict(os.environ, {"ARTIFEX_POLICY": "auto", "ARTIFEX_AGENT_KEY": "k"}):
            d = check_policy("edit_file", "core/sandbox/policy.py\x00old\x00new")
            self.assertFalse(d.allowed)
            self.assertEqual(d.matched_rule, "self_mod_ratchet")


if __name__ == "__main__":
    unittest.main()
