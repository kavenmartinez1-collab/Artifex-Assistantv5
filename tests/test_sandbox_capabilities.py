"""Tests for core.sandbox.capabilities — P6-T25 capability-based permissions."""

import os
import unittest
from unittest.mock import patch

from core.sandbox.capabilities import (
    ALL_CAPABILITIES,
    get_capabilities,
    set_capabilities,
    reset_capabilities,
    has_capability,
    grant,
    revoke,
    _capabilities_hook,
    install,
)
from core.sandbox.policy import RiskLevel, clear_policy_hooks, check_policy


class TestDefaultCapabilities(unittest.TestCase):
    """Default: all capabilities granted."""

    def setUp(self):
        reset_capabilities()

    def tearDown(self):
        reset_capabilities()

    def test_all_granted_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_CAPABILITIES", None)
            reset_capabilities()
            caps = get_capabilities()
            self.assertEqual(caps, ALL_CAPABILITIES)

    def test_has_all_action_types(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_CAPABILITIES", None)
            reset_capabilities()
            self.assertTrue(has_capability("shell"))
            self.assertTrue(has_capability("python"))
            self.assertTrue(has_capability("read_file"))


class TestEnvCapabilities(unittest.TestCase):
    """ARTIFEX_CAPABILITIES env var restricts capabilities."""

    def setUp(self):
        reset_capabilities()

    def tearDown(self):
        reset_capabilities()

    def test_restrict_to_read_only(self):
        with patch.dict(os.environ, {"ARTIFEX_CAPABILITIES": "read_file,glob,grep"}):
            reset_capabilities()
            caps = get_capabilities()
            self.assertEqual(caps, {"read_file", "glob", "grep"})
            self.assertTrue(has_capability("read_file"))
            self.assertFalse(has_capability("shell"))

    def test_unknown_capabilities_ignored(self):
        with patch.dict(os.environ, {"ARTIFEX_CAPABILITIES": "read_file,fly_rocket"}):
            reset_capabilities()
            caps = get_capabilities()
            self.assertEqual(caps, {"read_file"})

    def test_empty_means_all(self):
        with patch.dict(os.environ, {"ARTIFEX_CAPABILITIES": ""}):
            reset_capabilities()
            caps = get_capabilities()
            self.assertEqual(caps, ALL_CAPABILITIES)


class TestCapabilityMutation(unittest.TestCase):
    """Grant/revoke individual capabilities."""

    def setUp(self):
        reset_capabilities()

    def tearDown(self):
        reset_capabilities()

    def test_revoke(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_CAPABILITIES", None)
            reset_capabilities()
            self.assertTrue(has_capability("shell"))
            revoke("shell")
            self.assertFalse(has_capability("shell"))
            self.assertTrue(has_capability("read_file"))

    def test_grant(self):
        set_capabilities({"read_file"})
        self.assertFalse(has_capability("shell"))
        grant("shell")
        self.assertTrue(has_capability("shell"))

    def test_grant_unknown_ignored(self):
        set_capabilities({"read_file"})
        grant("fly_rocket")
        self.assertNotIn("fly_rocket", get_capabilities())

    def test_set_capabilities(self):
        set_capabilities({"glob", "grep"})
        caps = get_capabilities()
        self.assertEqual(caps, {"glob", "grep"})


class TestCapabilitiesHook(unittest.TestCase):
    """Policy hook denies actions outside capability set."""

    def setUp(self):
        reset_capabilities()

    def tearDown(self):
        reset_capabilities()

    def test_allowed_capability(self):
        set_capabilities({"read_file", "glob"})
        d = _capabilities_hook("read_file", "file.py|1", RiskLevel.SAFE)
        self.assertIsNone(d)

    def test_denied_capability(self):
        set_capabilities({"read_file", "glob"})
        d = _capabilities_hook("shell", "ls", RiskLevel.HIGH)
        self.assertIsNotNone(d)
        self.assertFalse(d.allowed)
        self.assertIn("capability", d.reason)


class TestCapabilitiesInstall(unittest.TestCase):
    """Integration: install() registers the hook."""

    def setUp(self):
        clear_policy_hooks()
        reset_capabilities()

    def tearDown(self):
        clear_policy_hooks()
        reset_capabilities()

    def test_install_denies_revoked_action(self):
        install()
        set_capabilities({"read_file", "glob"})
        with patch.dict(os.environ, {"ARTIFEX_POLICY": "auto", "ARTIFEX_AGENT_KEY": "k"}):
            d = check_policy("shell", "ls")
            self.assertFalse(d.allowed)
            self.assertEqual(d.matched_rule, "capability_denied")

    def test_install_allows_granted_action(self):
        install()
        set_capabilities({"read_file", "glob", "shell"})
        with patch.dict(os.environ, {"ARTIFEX_POLICY": "auto", "ARTIFEX_AGENT_KEY": "k"}):
            d = check_policy("shell", "ls")
            self.assertTrue(d.allowed)


if __name__ == "__main__":
    unittest.main()
