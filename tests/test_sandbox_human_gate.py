"""Tests for core.sandbox.human_gate — P6-T28 human gates."""

import unittest

from core.sandbox.human_gate import GateState, RISK_POINTS
from core.sandbox.policy import RiskLevel


class TestRiskPoints(unittest.TestCase):
    """Risk point values."""

    def test_safe_is_zero(self):
        self.assertEqual(RISK_POINTS[RiskLevel.SAFE], 0)

    def test_critical_is_highest(self):
        self.assertGreater(
            RISK_POINTS[RiskLevel.CRITICAL],
            RISK_POINTS[RiskLevel.HIGH],
        )

    def test_all_levels_covered(self):
        for level in RiskLevel:
            self.assertIn(level, RISK_POINTS)


class TestGateStateInterval(unittest.TestCase):
    """Round interval gate."""

    def test_no_gate_before_interval(self):
        g = GateState(interval=3, max_actions=0, risk_budget=0)
        self.assertIsNone(g.should_gate(1))
        self.assertIsNone(g.should_gate(2))

    def test_gate_at_interval(self):
        g = GateState(interval=3, max_actions=0, risk_budget=0)
        reason = g.should_gate(3)
        self.assertIsNotNone(reason)
        self.assertIn("round interval", reason)

    def test_acknowledge_resets_interval(self):
        g = GateState(interval=3, max_actions=0, risk_budget=0)
        self.assertIsNotNone(g.should_gate(3))
        g.acknowledge_gate(3)
        self.assertIsNone(g.should_gate(4))
        self.assertIsNone(g.should_gate(5))
        self.assertIsNotNone(g.should_gate(6))

    def test_disabled_when_zero(self):
        g = GateState(interval=0, max_actions=0, risk_budget=0)
        self.assertIsNone(g.should_gate(100))


class TestGateStateActionLimit(unittest.TestCase):
    """Action count gate."""

    def test_gate_at_limit(self):
        g = GateState(interval=0, max_actions=3, risk_budget=0)
        g.record_action(RiskLevel.SAFE)
        g.record_action(RiskLevel.SAFE)
        self.assertIsNone(g.should_gate(1))
        g.record_action(RiskLevel.SAFE)
        reason = g.should_gate(1)
        self.assertIsNotNone(reason)
        self.assertIn("action limit", reason)

    def test_disabled_when_zero(self):
        g = GateState(interval=0, max_actions=0, risk_budget=0)
        for _ in range(100):
            g.record_action(RiskLevel.HIGH)
        self.assertIsNone(g.should_gate(50))


class TestGateStateRiskBudget(unittest.TestCase):
    """Cumulative risk budget gate."""

    def test_safe_actions_no_gate(self):
        g = GateState(interval=0, max_actions=0, risk_budget=10)
        for _ in range(100):
            g.record_action(RiskLevel.SAFE)
        self.assertIsNone(g.should_gate(1))

    def test_high_actions_trigger_gate(self):
        g = GateState(interval=0, max_actions=0, risk_budget=10)
        g.record_action(RiskLevel.HIGH)  # 4
        g.record_action(RiskLevel.HIGH)  # 8
        self.assertIsNone(g.should_gate(1))
        g.record_action(RiskLevel.HIGH)  # 12
        reason = g.should_gate(1)
        self.assertIsNotNone(reason)
        self.assertIn("risk budget", reason)

    def test_acknowledge_resets_budget(self):
        g = GateState(interval=0, max_actions=0, risk_budget=10)
        g.record_action(RiskLevel.CRITICAL)  # 8
        g.record_action(RiskLevel.MEDIUM)    # 10
        self.assertIsNotNone(g.should_gate(1))
        g.acknowledge_gate(1)
        self.assertIsNone(g.should_gate(2))

    def test_disabled_when_zero(self):
        g = GateState(interval=0, max_actions=0, risk_budget=0)
        g.record_action(RiskLevel.CRITICAL)
        g.record_action(RiskLevel.CRITICAL)
        self.assertIsNone(g.should_gate(1))


class TestGateStateReset(unittest.TestCase):
    """Reset clears all counters."""

    def test_reset(self):
        g = GateState(interval=3, max_actions=5, risk_budget=10)
        for _ in range(5):
            g.record_action(RiskLevel.HIGH)
        self.assertIsNotNone(g.should_gate(5))
        g.reset()
        self.assertEqual(g.action_count, 0)
        self.assertEqual(g.risk_accumulated, 0)
        self.assertIsNone(g.should_gate(1))


class TestGateStateMultipleTriggers(unittest.TestCase):
    """First matching gate wins."""

    def test_action_limit_before_interval(self):
        g = GateState(interval=10, max_actions=2, risk_budget=0)
        g.record_action(RiskLevel.SAFE)
        g.record_action(RiskLevel.SAFE)
        reason = g.should_gate(1)
        self.assertIsNotNone(reason)
        self.assertIn("action limit", reason)


if __name__ == "__main__":
    unittest.main()
