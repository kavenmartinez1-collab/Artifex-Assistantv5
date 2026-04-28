"""
Artifex Assistant V5 — Audit log + replay (P6-T26).

Records every agent action with timestamps, risk levels, policy decisions,
and outcomes. Supports JSON-lines format for machine parsing and replay.

The audit log is append-only during a session. Replay reads the log and
re-evaluates actions against the current policy (dry-run, no execution).

Log location: SESSION_DIR/audit/<session_id>.jsonl
              or ARTIFEX_AUDIT_DIR env var.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import TextIO

from core.sandbox.policy import PolicyDecision, RiskLevel

_log = logging.getLogger(__name__)

# ── Audit entry ──────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    timestamp: float
    session_id: str
    round_num: int
    action_type: str
    content_preview: str
    risk_level: str
    policy_decision: str
    matched_rule: str
    outcome: str
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "AuditEntry":
        return cls(**json.loads(line))


# ── Audit logger ─────────────────────────────────────────────────────────────

class AuditLog:
    """Append-only audit logger for a single session."""

    def __init__(self, session_id: str, log_dir: str | None = None):
        self.session_id = session_id
        self._round = 0
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []

        if log_dir is None:
            log_dir = os.environ.get("ARTIFEX_AUDIT_DIR", "")
        if not log_dir:
            from core.config import SESSION_DIR
            log_dir = os.path.join(SESSION_DIR, "audit")

        self._log_dir = log_dir
        self._file: TextIO | None = None
        self._file_path: str | None = None

    def _ensure_file(self):
        if self._file is not None:
            return
        os.makedirs(self._log_dir, exist_ok=True)
        self._file_path = os.path.join(self._log_dir, f"{self.session_id}.jsonl")
        self._file = open(self._file_path, "a", encoding="utf-8")

    @property
    def file_path(self) -> str | None:
        return self._file_path

    def set_round(self, n: int):
        """Update the current agent round number."""
        self._round = n

    def record(
        self,
        action_type: str,
        content: str,
        decision: PolicyDecision,
        outcome: str = "pending",
        error: str = "",
    ) -> AuditEntry:
        """Record an action to the audit log."""
        preview = content[:200] if content else ""

        entry = AuditEntry(
            timestamp=time.time(),
            session_id=self.session_id,
            round_num=self._round,
            action_type=action_type,
            content_preview=preview,
            risk_level=decision.risk_level.name,
            policy_decision="allowed" if decision.allowed else "denied",
            matched_rule=decision.matched_rule,
            outcome=outcome,
            error=error,
        )

        with self._lock:
            self._entries.append(entry)
            try:
                self._ensure_file()
                self._file.write(entry.to_json() + "\n")
                self._file.flush()
            except OSError as e:
                _log.warning("Audit write failed: %s", e)

        return entry

    def get_entries(self) -> list[AuditEntry]:
        """Get all entries for this session (in-memory)."""
        with self._lock:
            return list(self._entries)

    def close(self):
        """Flush and close the audit log file."""
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Replay ───────────────────────────────────────────────────────────────────

def load_audit_log(path: str) -> list[AuditEntry]:
    """Load audit entries from a JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(AuditEntry.from_json(line))
    return entries


def replay_summary(entries: list[AuditEntry]) -> dict:
    """Generate a summary of an audit log for review."""
    total = len(entries)
    denied = sum(1 for e in entries if e.policy_decision == "denied")
    by_risk = {}
    by_type = {}
    errors = []

    for e in entries:
        by_risk[e.risk_level] = by_risk.get(e.risk_level, 0) + 1
        by_type[e.action_type] = by_type.get(e.action_type, 0) + 1
        if e.error:
            errors.append({"round": e.round_num, "action": e.action_type, "error": e.error})

    return {
        "total_actions": total,
        "denied": denied,
        "by_risk_level": by_risk,
        "by_action_type": by_type,
        "errors": errors[:20],
    }
