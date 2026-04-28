"""Tests for core.sandbox.egress — P6-T30 network egress policy."""

import os
import unittest
from unittest.mock import patch

from core.sandbox.egress import (
    EgressMode,
    get_egress_mode,
    extract_urls,
    check_url_egress,
    _egress_hook,
    install,
)
from core.sandbox.policy import RiskLevel, clear_policy_hooks, check_policy


class TestEgressMode(unittest.TestCase):
    """ARTIFEX_EGRESS_MODE env var handling."""

    def test_default_is_open(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_EGRESS_MODE", None)
            self.assertEqual(get_egress_mode(), EgressMode.OPEN)

    def test_allowlist(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "allowlist"}):
            self.assertEqual(get_egress_mode(), EgressMode.ALLOWLIST)

    def test_denylist(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "denylist"}):
            self.assertEqual(get_egress_mode(), EgressMode.DENYLIST)

    def test_invalid_falls_back_to_open(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "yolo"}):
            self.assertEqual(get_egress_mode(), EgressMode.OPEN)


class TestUrlExtraction(unittest.TestCase):
    """URL extraction from action content."""

    def test_web_read_url(self):
        urls = extract_urls("web_read", "https://example.com/page")
        self.assertEqual(urls, ["https://example.com/page"])

    def test_web_read_number_ignored(self):
        urls = extract_urls("web_read", "3")
        self.assertEqual(urls, [])

    def test_web_read_bare_domain(self):
        urls = extract_urls("web_read", "example.com")
        self.assertEqual(urls, ["https://example.com"])

    def test_download_url(self):
        urls = extract_urls("download", "https://cdn.example.com/file.zip|myfile.zip")
        self.assertEqual(urls, ["https://cdn.example.com/file.zip"])

    def test_shell_urls(self):
        urls = extract_urls("shell", "curl https://api.github.com/repos/foo/bar")
        self.assertIn("https://api.github.com/repos/foo/bar", urls)

    def test_no_urls_in_read_file(self):
        urls = extract_urls("read_file", "/home/user/file.py|1")
        self.assertEqual(urls, [])


class TestCheckUrlEgress(unittest.TestCase):
    """URL egress checks."""

    def test_open_allows_all(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "open"}):
            self.assertIsNone(check_url_egress("https://evil.com"))

    def test_allowlist_allows_listed(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "allowlist",
            "ARTIFEX_EGRESS_ALLOW": "github.com,pypi.org",
        }):
            self.assertIsNone(check_url_egress("https://github.com/foo"))
            self.assertIsNone(check_url_egress("https://api.github.com/repos"))

    def test_allowlist_denies_unlisted(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "allowlist",
            "ARTIFEX_EGRESS_ALLOW": "github.com",
        }):
            result = check_url_egress("https://evil.com/payload")
            self.assertIsNotNone(result)
            self.assertIn("not in allowlist", result)

    def test_allowlist_empty_denies_all(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "allowlist",
            "ARTIFEX_EGRESS_ALLOW": "",
        }):
            result = check_url_egress("https://github.com")
            self.assertIsNotNone(result)

    def test_denylist_blocks_listed(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "denylist",
            "ARTIFEX_EGRESS_DENY": "evil.com,malware.net",
        }):
            result = check_url_egress("https://evil.com/payload")
            self.assertIsNotNone(result)
            self.assertIn("denylist", result)

    def test_denylist_allows_unlisted(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "denylist",
            "ARTIFEX_EGRESS_DENY": "evil.com",
        }):
            self.assertIsNone(check_url_egress("https://github.com"))

    def test_subdomain_matching(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "denylist",
            "ARTIFEX_EGRESS_DENY": "evil.com",
        }):
            result = check_url_egress("https://sub.evil.com/page")
            self.assertIsNotNone(result)


class TestEgressHook(unittest.TestCase):
    """Policy hook integration."""

    def test_blocks_denied_url(self):
        with patch.dict(os.environ, {
            "ARTIFEX_EGRESS_MODE": "denylist",
            "ARTIFEX_EGRESS_DENY": "evil.com",
        }):
            d = _egress_hook("web_read", "https://evil.com", RiskLevel.LOW)
            self.assertIsNotNone(d)
            self.assertFalse(d.allowed)

    def test_allows_open_mode(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "open"}):
            d = _egress_hook("web_read", "https://anywhere.com", RiskLevel.LOW)
            self.assertIsNone(d)

    def test_no_urls_passes(self):
        with patch.dict(os.environ, {"ARTIFEX_EGRESS_MODE": "allowlist", "ARTIFEX_EGRESS_ALLOW": ""}):
            d = _egress_hook("glob", "*.py", RiskLevel.SAFE)
            self.assertIsNone(d)


class TestEgressInstall(unittest.TestCase):
    """install() registers the hook."""

    def setUp(self):
        clear_policy_hooks()

    def tearDown(self):
        clear_policy_hooks()

    def test_install_blocks_denied_domain(self):
        install()
        with patch.dict(os.environ, {
            "ARTIFEX_POLICY": "auto",
            "ARTIFEX_AGENT_KEY": "k",
            "ARTIFEX_EGRESS_MODE": "denylist",
            "ARTIFEX_EGRESS_DENY": "evil.com",
        }):
            d = check_policy("download", "https://evil.com/malware.exe")
            self.assertFalse(d.allowed)
            self.assertEqual(d.matched_rule, "egress_policy")


if __name__ == "__main__":
    unittest.main()
