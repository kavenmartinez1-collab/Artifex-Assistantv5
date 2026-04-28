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


class TestPrefixCollision(unittest.TestCase):
    """Sandbox roots must use exact-or-child match, not raw startswith.

    Without a separator-aware check, /x/projEVIL passes a /x/proj root.
    Tests patch _get_allowed_roots so the project dir + tempdir don't
    accidentally cover both sides of the collision pair.
    """

    def test_prefix_collision_outside_sandbox(self):
        if os.name == "nt":
            real_root = "D:\\workspaces\\proj"
            evil_file = "D:\\workspaces\\projEVIL\\secret.txt"
        else:
            real_root = "/workspaces/proj"
            evil_file = "/workspaces/projEVIL/secret.txt"

        with patch("core.sandbox.fs_sandbox._get_allowed_roots", return_value=[real_root]):
            self.assertFalse(
                is_path_within_sandbox(evil_file),
                "Path /x/projEVIL/... must not match /x/proj sandbox root",
            )

    def test_exact_root_allowed(self):
        if os.name == "nt":
            real_root = "D:\\workspaces\\proj"
        else:
            real_root = "/workspaces/proj"
        with patch("core.sandbox.fs_sandbox._get_allowed_roots", return_value=[real_root]):
            self.assertTrue(is_path_within_sandbox(real_root))

    def test_child_of_root_allowed(self):
        if os.name == "nt":
            real_root = "D:\\workspaces\\proj"
            child = "D:\\workspaces\\proj\\src\\main.py"
        else:
            real_root = "/workspaces/proj"
            child = "/workspaces/proj/src/main.py"
        with patch("core.sandbox.fs_sandbox._get_allowed_roots", return_value=[real_root]):
            self.assertTrue(is_path_within_sandbox(child))

    def test_helper_separator_check(self):
        """Direct unit test of the separator-aware comparison helper."""
        from core.sandbox.fs_sandbox import _is_under_root
        if os.name == "nt":
            self.assertTrue(_is_under_root("D:\\proj", "D:\\proj"))
            self.assertTrue(_is_under_root("D:\\proj\\src", "D:\\proj"))
            self.assertFalse(_is_under_root("D:\\projEVIL", "D:\\proj"))
            self.assertFalse(_is_under_root("D:\\projEVIL\\x", "D:\\proj"))
        else:
            self.assertTrue(_is_under_root("/proj", "/proj"))
            self.assertTrue(_is_under_root("/proj/src", "/proj"))
            self.assertFalse(_is_under_root("/projEVIL", "/proj"))
            self.assertFalse(_is_under_root("/projEVIL/x", "/proj"))


@unittest.skipUnless(
    os.name != "nt" or hasattr(os, "symlink"),
    "Symlink creation requires admin or Developer Mode on Windows; "
    "skip if unavailable",
)
class TestSymlinkEscape(unittest.TestCase):
    """Symlinks inside the sandbox pointing outside must not bypass the check.

    If is_path_within_sandbox uses os.path.abspath instead of os.path.realpath,
    a symlink at sandbox/sneaky → /etc/passwd reads as inside the sandbox even
    though following the link reads /etc/passwd.
    """

    def _can_symlink(self, parent):
        """Best-effort check; on Windows without privilege, os.symlink raises."""
        try:
            target = os.path.join(parent, "_t")
            link = os.path.join(parent, "_l")
            with open(target, "w") as f:
                f.write("x")
            os.symlink(target, link)
            os.remove(link)
            os.remove(target)
            return True
        except (OSError, NotImplementedError):
            return False

    def test_symlink_to_outside_path_blocked(self):
        with tempfile.TemporaryDirectory() as parent:
            sandbox_root = os.path.join(parent, "sbx")
            outside_dir = os.path.join(parent, "outside")
            os.makedirs(sandbox_root)
            os.makedirs(outside_dir)

            if not self._can_symlink(parent):
                self.skipTest("symlink creation not available on this platform/config")

            secret = os.path.join(outside_dir, "secret.txt")
            with open(secret, "w") as f:
                f.write("nope")

            sneaky_link = os.path.join(sandbox_root, "sneaky")
            os.symlink(secret, sneaky_link)

            # Patch the allowed roots so only sandbox_root counts —
            # without this, tempfile.gettempdir() (which is `parent`'s ancestor)
            # would whitelist both directories.
            with patch(
                "core.sandbox.fs_sandbox._get_allowed_roots",
                return_value=[sandbox_root],
            ):
                self.assertFalse(
                    is_path_within_sandbox(sneaky_link),
                    "symlink target outside sandbox must be rejected after realpath resolution",
                )

    def test_symlink_to_sensitive_path_caught_via_resolve(self):
        """A sandbox symlink pointing at a deny-listed file must be caught."""
        with tempfile.TemporaryDirectory() as parent:
            sandbox_root = os.path.join(parent, "sbx")
            os.makedirs(sandbox_root)

            if not self._can_symlink(parent):
                self.skipTest("symlink creation not available on this platform/config")

            # Create a "fake credentials" file that matches a deny pattern.
            fake_creds = os.path.join(parent, "credentials.json")
            with open(fake_creds, "w") as f:
                f.write("{}")

            link = os.path.join(sandbox_root, "harmless.txt")
            os.symlink(fake_creds, link)

            with patch.dict(os.environ, {"ARTIFEX_SANDBOX_ROOTS": sandbox_root}):
                reason = check_path(link)
                self.assertIsNotNone(reason)
                self.assertIn("symlink", reason.lower())


if __name__ == "__main__":
    unittest.main()
