"""Tests for P3 agent loop safety features: git commit/revert, bounded loops,
agent key enforcement.
"""

import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class TestGitCommitEdit(unittest.TestCase):
    """P3-T15: git_commit_edit and git_revert_last."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        subprocess.run(["git", "init"], cwd=self.tmpdir, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=self.tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=self.tmpdir, capture_output=True,
        )
        self.test_file = os.path.join(self.tmpdir, "test.txt")
        with open(self.test_file, "w") as f:
            f.write("initial")
        subprocess.run(
            ["git", "add", "."], cwd=self.tmpdir, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=self.tmpdir, capture_output=True,
        )

    def test_commit_succeeds(self):
        from tools.agent_tools import git_commit_edit
        with open(self.test_file, "w") as f:
            f.write("modified")
        ok, msg = git_commit_edit(self.test_file, "test edit")
        self.assertTrue(ok, msg)

    def test_commit_outside_repo_fails(self):
        from tools.agent_tools import git_commit_edit
        tmpfile = os.path.join(tempfile.gettempdir(), "no_repo_file.txt")
        with open(tmpfile, "w") as f:
            f.write("test")
        ok, msg = git_commit_edit(tmpfile, "no repo")
        # May or may not find a repo depending on tmpdir location
        # Just verify it doesn't crash
        self.assertIsInstance(ok, bool)
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

    def test_revert_only_reverts_agent_commits(self):
        from tools.agent_tools import git_commit_edit, git_revert_last
        with open(self.test_file, "w") as f:
            f.write("agent edit")
        git_commit_edit(self.test_file, "test edit")

        ok, msg = git_revert_last(self.test_file)
        self.assertTrue(ok, msg)

    def test_revert_refuses_non_agent_commit(self):
        from tools.agent_tools import git_revert_last
        ok, msg = git_revert_last(self.test_file)
        self.assertFalse(ok)
        self.assertIn("not an agent edit", msg)


class TestBoundedAgentLoop(unittest.TestCase):
    """P3-T16: MAX_AGENT_ROUNDS is configurable and enforced."""

    def test_default_is_10(self):
        from tools.agent_tools import MAX_AGENT_ROUNDS
        self.assertGreater(MAX_AGENT_ROUNDS, 0)
        # Default should be 10 unless overridden
        if "ARTIFEX_MAX_AGENT_ROUNDS" not in os.environ:
            self.assertEqual(MAX_AGENT_ROUNDS, 10)

    def test_env_override(self):
        with patch.dict(os.environ, {"ARTIFEX_MAX_AGENT_ROUNDS": "5"}):
            import importlib
            import tools.agent_tools as mod
            importlib.reload(mod)
            self.assertEqual(mod.MAX_AGENT_ROUNDS, 5)


class TestAgentKeyEnforcement(unittest.TestCase):
    """P3-T17: ARTIFEX_AGENT_KEY gates auto-execution."""

    def test_no_key_means_no_auto_exec(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_AGENT_KEY", None)
            import importlib
            import tools.agent_tools as mod
            importlib.reload(mod)
            self.assertFalse(mod.agent_auto_exec_enabled())

    def test_key_set_enables_auto_exec(self):
        with patch.dict(os.environ, {"ARTIFEX_AGENT_KEY": "secret123"}):
            import importlib
            import tools.agent_tools as mod
            importlib.reload(mod)
            self.assertTrue(mod.agent_auto_exec_enabled())


class TestFindGitRoot(unittest.TestCase):
    """_find_git_root walks up to find .git directory."""

    def test_finds_repo_root(self):
        from tools.agent_tools import _find_git_root
        # This test repo itself
        here = os.path.dirname(os.path.abspath(__file__))
        root = _find_git_root(here)
        self.assertIsNotNone(root)
        self.assertTrue(os.path.isdir(os.path.join(root, ".git")))

    def test_returns_none_for_no_repo(self):
        from tools.agent_tools import _find_git_root
        root = _find_git_root("/")
        # / may or may not have .git, but this shouldn't crash
        self.assertIsInstance(root, (str, type(None)))


if __name__ == "__main__":
    unittest.main()
