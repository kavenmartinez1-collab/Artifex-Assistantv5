"""
Artifex Assistant V5 — Human gates (P6-T28).

Configurable checkpoints that pause auto-execution for human review.
Gates trigger based on:
  1. Round interval — pause every N rounds (e.g., every 3 rounds).
  2. Risk threshold — pause when cumulative risk exceeds a threshold.
  3. Action count — pause after N total actions in a session.

Configured via:
  ARTIFEX_GATE_INTERVAL   — pause every N rounds (0 = disabled, default 5)
  ARTIFEX_GATE_MAX_ACTIONS — max actions before mandatory pause (default 25)
  ARTIFEX_GATE_RISK_BUDGET — cumulative risk points before pause (default 15)

Risk points: SAFE=0, LOW=1, MEDIUM=2, HIGH=4, CRITICAL=8.
"""

import logging
import os
import threading

from core.sandbox.policy import RiskLevel

_log = logging.getLogger(__name__)

# ── Risk point mapping ───────────────────────────────────────────────────────

RISK_POINTS: dict[RiskLevel, int] = {
    RiskLevel.SAFE: 0,
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 4,
    RiskLevel.CRITICAL: 8,
}

# ── Configuration ────────────────────────────────────────────────────────────

GATE_INTERVAL = int(os.environ.get("ARTIFEX_GATE_INTERVAL", "5"))
GATE_MAX_ACTIONS = int(os.environ.get("ARTIFEX_GATE_MAX_ACTIONS", "25"))
GATE_RISK_BUDGET = int(os.environ.get("ARTIFEX_GATE_RISK_BUDGET", "15"))


# ── Gate state (per session) ─────────────────────────────────────────────────

class GateState:
    """Tracks gate triggers for one agent session."""

    def __init__(
        self,
        interval: int = GATE_INTERVAL,
        max_actions: int = GATE_MAX_ACTIONS,
        risk_budget: int = GATE_RISK_BUDGET,
    ):
        self.interval = interval
        self.max_actions = max_actions
        self.risk_budget = risk_budget
        self._lock = threading.Lock()
        self._action_count = 0
        self._risk_accumulated = 0
        self._last_gate_round = 0

    def record_action(self, risk: RiskLevel):
        """Record an executed action. Call before checking the gate."""
        with self._lock:
            self._action_count += 1
            self._risk_accumulated += RISK_POINTS.get(risk, 0)

    def should_gate(self, current_round: int) -> str | None:
        """Check if a human gate should trigger.

        Returns the reason string if gate should trigger, None otherwise.
        """
        with self._lock:
            if self.max_actions > 0 and self._action_count >= self.max_actions:
                return f"action limit reached ({self._action_count}/{self.max_actions})"

            if self.risk_budget > 0 and self._risk_accumulated >= self.risk_budget:
                return f"risk budget exhausted ({self._risk_accumulated}/{self.risk_budget})"

            if (self.interval > 0
                    and current_round > 0
                    and current_round - self._last_gate_round >= self.interval):
                return f"round interval ({self.interval} rounds)"

        return None

    def acknowledge_gate(self, current_round: int):
        """Mark that the human acknowledged the gate. Resets interval counter."""
        with self._lock:
            self._last_gate_round = current_round
            self._risk_accumulated = 0

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self._action_count = 0
            self._risk_accumulated = 0
            self._last_gate_round = 0

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def risk_accumulated(self) -> int:
        return self._risk_accumulated
