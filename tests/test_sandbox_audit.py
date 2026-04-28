"""Tests for core.sandbox.audit — P6-T26 audit log + replay."""

import json
import os
import tempfile
import unittest

from core.sandbox.audit import (
    AuditEntry,
    AuditLog,
    load_audit_log,
    replay_summary,
)
from core.sandbox.policy import PolicyDecision, RiskLevel


def _make_decision(allowed=True, risk=RiskLevel.SAFE, rule="test"):
    return PolicyDecision(
        allowed=allowed,
        requires_confirmation=False,
        risk_level=risk,
        reason="test",
        matched_rule=rule,
    )


class TestAuditEntry(unittest.TestCase):
    """AuditEntry serialization."""

    def test_round_trip_json(self):
        entry = AuditEntry(
            timestamp=1000.5,
            session_id="sess-1",
            round_num=3,
            action_type="shell",
            content_preview="ls -la",
            risk_level="SAFE",
            policy_decision="allowed",
            matched_rule="auto_moderate",
            outcome="success",
            error="",
        )
        line = entry.to_json()
        restored = AuditEntry.from_json(line)
        self.assertEqual(restored.session_id, "sess-1")
        self.assertEqual(restored.round_num, 3)
        self.assertEqual(restored.action_type, "shell")
        self.assertEqual(restored.outcome, "success")

    def test_json_is_valid(self):
        entry = AuditEntry(
            timestamp=1.0, session_id="s", round_num=0,
            action_type="glob", content_preview="*.py",
            risk_level="SAFE", policy_decision="allowed",
            matched_rule="x", outcome="ok",
        )
        data = json.loads(entry.to_json())
        self.assertEqual(data["action_type"], "glob")


class TestAuditLog(unittest.TestCase):
    """In-memory + file audit logging."""

    def test_record_and_get_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("test-session", log_dir=tmpdir)
            d = _make_decision()
            log.record("glob", "*.py", d, outcome="success")
            log.record("shell", "ls", d, outcome="success")
            entries = log.get_entries()
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].action_type, "glob")
            self.assertEqual(entries[1].action_type, "shell")
            log.close()

    def test_writes_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("file-test", log_dir=tmpdir)
            d = _make_decision()
            log.record("read_file", "config.py|1", d, outcome="success")
            log.close()

            path = os.path.join(tmpdir, "file-test.jsonl")
            self.assertTrue(os.path.isfile(path))
            with open(path, "r") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertEqual(data["action_type"], "read_file")

    def test_round_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("round-test", log_dir=tmpdir)
            d = _make_decision()
            log.set_round(1)
            log.record("glob", "*.py", d)
            log.set_round(2)
            log.record("shell", "ls", d)
            entries = log.get_entries()
            self.assertEqual(entries[0].round_num, 1)
            self.assertEqual(entries[1].round_num, 2)
            log.close()

    def test_content_preview_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("trunc-test", log_dir=tmpdir)
            d = _make_decision()
            long_content = "x" * 500
            log.record("shell", long_content, d)
            entries = log.get_entries()
            self.assertEqual(len(entries[0].content_preview), 200)
            log.close()

    def test_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with AuditLog("ctx-test", log_dir=tmpdir) as log:
                log.record("glob", "*.py", _make_decision())
            path = os.path.join(tmpdir, "ctx-test.jsonl")
            self.assertTrue(os.path.isfile(path))

    def test_denied_action_recorded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("deny-test", log_dir=tmpdir)
            d = _make_decision(allowed=False, risk=RiskLevel.CRITICAL, rule="proc_blocklist")
            log.record("shell", "sudo rm -rf /", d, outcome="denied")
            entries = log.get_entries()
            self.assertEqual(entries[0].policy_decision, "denied")
            self.assertEqual(entries[0].risk_level, "CRITICAL")
            log.close()


class TestReplay(unittest.TestCase):
    """Load and summarize audit logs."""

    def test_load_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log = AuditLog("replay-test", log_dir=tmpdir)
            d_ok = _make_decision()
            d_deny = _make_decision(allowed=False, risk=RiskLevel.HIGH)
            log.record("glob", "*.py", d_ok, outcome="success")
            log.record("shell", "ls", d_ok, outcome="success")
            log.record("shell", "sudo rm /", d_deny, outcome="denied")
            log.record("python", "print('hi')", d_ok, outcome="success", error="timeout")
            log.close()

            path = os.path.join(tmpdir, "replay-test.jsonl")
            entries = load_audit_log(path)
            self.assertEqual(len(entries), 4)

            summary = replay_summary(entries)
            self.assertEqual(summary["total_actions"], 4)
            self.assertEqual(summary["denied"], 1)
            self.assertIn("shell", summary["by_action_type"])
            self.assertEqual(summary["by_action_type"]["shell"], 2)
            self.assertEqual(len(summary["errors"]), 1)
            self.assertEqual(summary["errors"][0]["error"], "timeout")


if __name__ == "__main__":
    unittest.main()
