"""Tests for core.sandbox.fs_sandbox — P6-T23 filesystem sandbox."""

import os
import tempfile
import unittest
from unittest.mock import patch

from core.sandbox.fs_sandbox import (
    is_path_denied,
    is_path_within_sandbox,
    check_path,
    extract_paths_from_content,
    _fs_sandbox_hook,
    install,
)
from core.sandbox.policy import RiskLevel, clear_policy_hooks, check_policy


class TestDenyPatterns(unittest.TestCase):
    """Sensitive paths are always denied."""

    def test_ssh_denied(self):
        self.assertIsNotNone(is_path_denied("/home/user/.ssh/id_rsa"))
        self.assertIsNotNone(is_path_denied("C:\\Users\\me\\.ssh\\config"))

    def test_aws_denied(self):
        self.assertIsNotNone(is_path_denied("/home/user/.aws/credentials"))

    def test_env_file_denied(self):
        self.assertIsNotNone(is_path_denied("/app/.env"))
        self.assertIsNotNone(is_path_denied("C:\\project\\.env.local"))

    def test_credentials_json_denied(self):
        self.assertIsNotNone(is_path_denied("/app/credentials.json"))

    def test_etc_shadow_denied(self):
        self.assertIsNotNone(is_path_denied("/etc/shadow"))

    def test_normal_file_allowed(self):
        self.assertIsNone(is_path_denied("/home/user/project/main.py"))
        self.assertIsNone(is_path_denied("C:\\Users\\me\\code\\app.py"))

    def test_git_config_denied(self):
        self.assertIsNotNone(is_path_denied("/project/.git/config"))

    def test_npmrc_denied(self):
        self.assertIsNotNone(is_path_denied("/home/user/.npmrc"))


class TestSandboxRoots(unittest.TestCase):
    """Paths must be within allowed root directories."""

    def test_project_dir_allowed(self):
        from core.config import BASE_DIR
        self.assertTrue(is_path_within_sandbox(
            os.path.join(BASE_DIR, "core", "config.py")))

    def test_temp_dir_allowed(self):
        tmp = os.path.join(tempfile.gettempdir(), "test_file.txt")
        self.assertTrue(is_path_within_sandbox(tmp))

    def test_random_system_dir_denied(self):
        if os.name == "nt":
            self.assertFalse(is_path_within_sandbox("C:\\Windows\\System32\\cmd.exe"))
        else:
            self.assertFalse(is_path_within_sandbox("/usr/bin/python3"))

    def test_extra_roots_from_env(self):
        extra = os.path.join(tempfile.gettempdir(), "extra_sandbox_root")
        with patch.dict(os.environ, {"ARTIFEX_SANDBOX_ROOTS": extra}):
            path = os.path.join(extra, "subdir", "file.txt")
            self.assertTrue(is_path_within_sandbox(path))


class TestCheckPath(unittest.TestCase):
    """Combined deny + sandbox check."""

    def test_denied_path_rejected(self):
        from core.config import BASE_DIR
        path = os.path.join(BASE_DIR, ".env")
        result = check_path(path)
        self.assertIsNotNone(result)
        self.assertIn("sensitive", result)

    def test_outside_sandbox_rejected(self):
        if os.name == "nt":
            result = check_path("D:\\random\\place\\file.txt")
        else:
            result = check_path("/opt/random/file.txt")
        self.assertIsNotNone(result)
        self.assertIn("outside sandbox", result)

    def test_valid_project_file_allowed(self):
        from core.config import BASE_DIR
        path = os.path.join(BASE_DIR, "core", "config.py")
        self.assertIsNone(check_path(path))


class TestPathExtraction(unittest.TestCase):
    """Extract paths from different action content formats."""

    def test_read_file(self):
        paths = extract_paths_from_content("read_file", "/home/user/file.py|1")
        self.assertEqual(paths, ["/home/user/file.py"])

    def test_read_file_url_ignored(self):
        paths = extract_paths_from_content("read_file", "https://example.com|1")
        self.assertEqual(paths, [])

    def test_edit_file(self):
        paths = extract_paths_from_content("edit_file", "/app/main.py\x00old\x00new")
        self.assertEqual(paths, ["/app/main.py"])

    def test_glob(self):
        paths = extract_paths_from_content("glob", "*.py|/app/src")
        self.assertEqual(paths, ["/app/src"])

    def test_grep(self):
        paths = extract_paths_from_content("grep", "pattern|/app/src|i")
        self.assertEqual(paths, ["/app/src"])

    def test_shell_extracts_paths(self):
        paths = extract_paths_from_content("shell", "cat /etc/hosts")
        self.assertIn("/etc/hosts", paths)

    def test_read_function(self):
        paths = extract_paths_from_content("read_function", "core/main.py|my_func")
        self.assertEqual(paths, ["core/main.py"])


class TestFsSandboxHook(unittest.TestCase):
    """The policy hook blocks denied paths."""

    def test_denies_sensitive_path(self):
        decision = _fs_sandbox_hook("read_file", "/home/user/.ssh/id_rsa|1", RiskLevel.SAFE)
        self.assertIsNotNone(decision)
        self.assertFalse(decision.allowed)
        self.assertIn("filesystem sandbox", decision.reason)

    def test_allows_normal_path(self):
        from core.config import BASE_DIR
        path = os.path.join(BASE_DIR, "core", "config.py")
        decision = _fs_sandbox_hook("read_file", f"{path}|1", RiskLevel.SAFE)
        self.assertIsNone(decision)

    def test_no_paths_returns_none(self):
        decision = _fs_sandbox_hook("search", "python tutorials", RiskLevel.LOW)
        self.assertIsNone(decision)


class TestFsSandboxInstall(unittest.TestCase):
    """Integration: install() registers the hook."""

    def setUp(self):
        clear_policy_hooks()

    def tearDown(self):
        clear_policy_hooks()

    def test_install_blocks_sensitive_reads(self):
        install()
        with patch.dict(os.environ, {"ARTIFEX_POLICY": "auto", "ARTIFEX_AGENT_KEY": "k"}):
            d = check_policy("read_file", "/home/user/.ssh/id_rsa|1")
            self.assertFalse(d.allowed)
            self.assertEqual(d.matched_rule, "fs_sandbox")


if __name__ == "__main__":
    unittest.main()
