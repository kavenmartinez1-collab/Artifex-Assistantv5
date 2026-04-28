"""Tests for core.sandbox.dry_run — P6-T27 dry-run mode."""

import os
import unittest
from unittest.mock import patch

from core.sandbox.dry_run import (
    is_enabled,
    enable,
    disable,
    reset,
    dry_run_result,
)


class TestDryRunEnvVar(unittest.TestCase):
    """ARTIFEX_DRY_RUN env var handling."""

    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_default_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_DRY_RUN", None)
            reset()
            self.assertFalse(is_enabled())

    def test_enabled_with_1(self):
        with patch.dict(os.environ, {"ARTIFEX_DRY_RUN": "1"}):
            reset()
            self.assertTrue(is_enabled())

    def test_enabled_with_true(self):
        with patch.dict(os.environ, {"ARTIFEX_DRY_RUN": "true"}):
            reset()
            self.assertTrue(is_enabled())

    def test_enabled_with_yes(self):
        with patch.dict(os.environ, {"ARTIFEX_DRY_RUN": "yes"}):
            reset()
            self.assertTrue(is_enabled())

    def test_disabled_with_0(self):
        with patch.dict(os.environ, {"ARTIFEX_DRY_RUN": "0"}):
            reset()
            self.assertFalse(is_enabled())


class TestDryRunProgrammatic(unittest.TestCase):
    """Programmatic enable/disable."""

    def setUp(self):
        reset()

    def tearDown(self):
        reset()

    def test_enable(self):
        self.assertFalse(is_enabled())
        enable()
        self.assertTrue(is_enabled())

    def test_disable(self):
        enable()
        self.assertTrue(is_enabled())
        disable()
        self.assertFalse(is_enabled())

    def test_reset(self):
        enable()
        reset()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTIFEX_DRY_RUN", None)
            reset()
            self.assertFalse(is_enabled())


class TestDryRunResult(unittest.TestCase):
    """Dry-run result message generation."""

    def test_basic_message(self):
        msg = dry_run_result("shell", "ls -la")
        self.assertIn("[DRY RUN]", msg)
        self.assertIn("shell", msg)
        self.assertIn("ls -la", msg)

    def test_truncates_long_content(self):
        msg = dry_run_result("python", "x" * 300)
        self.assertLessEqual(len(msg), 300)

    def test_empty_content(self):
        msg = dry_run_result("glob", "")
        self.assertIn("(empty)", msg)


if __name__ == "__main__":
    unittest.main()
