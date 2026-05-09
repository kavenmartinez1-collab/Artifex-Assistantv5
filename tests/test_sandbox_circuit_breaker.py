"""Tests for core.sandbox.circuit_breaker — P6-T32 circuit breakers."""

import unittest

from core.sandbox.circuit_breaker import CircuitBreaker, BreakerConfig


class TestErrorRateBreaker(unittest.TestCase):
    """Error rate trip."""

    def test_no_trip_below_threshold(self):
        cb = CircuitBreaker(BreakerConfig(error_window=10, error_threshold=0.5))
        for _ in range(8):
            cb.record_success()
        for _ in range(2):
            cb.record_failure()
        self.assertIsNone(cb.check())

    def test_trips_at_threshold(self):
        cb = CircuitBreaker(BreakerConfig(error_window=10, error_threshold=0.5))
        for _ in range(5):
            cb.record_success()
        for _ in range(5):
            cb.record_failure()
        reason = cb.check()
        self.assertIsNotNone(reason)
        self.assertIn("error rate", reason)

    def test_trips_above_threshold(self):
        cb = CircuitBreaker(BreakerConfig(error_window=4, error_threshold=0.5))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        reason = cb.check()
        self.assertIsNotNone(reason)

    def test_window_slides(self):
        cb = CircuitBreaker(BreakerConfig(error_window=4, error_threshold=0.5))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        self.assertIsNotNone(cb.check())
        cb.record_success()
        cb.record_success()
        cb.record_success()
        self.assertIsNone(cb.check())


class TestRepetitionBreaker(unittest.TestCase):
    """Same action repetition trip."""

    def test_no_trip_with_variety(self):
        cb = CircuitBreaker(BreakerConfig(repetition_limit=3, error_window=100))
        cb.record_success("ls")
        cb.record_success("pwd")
        cb.record_success("ls")
        self.assertIsNone(cb.check())

    def test_trips_on_repetition(self):
        cb = CircuitBreaker(BreakerConfig(repetition_limit=3, error_window=100))
        cb.record_success("ls -la")
        cb.record_success("ls -la")
        cb.record_success("ls -la")
        reason = cb.check()
        self.assertIsNotNone(reason)
        self.assertIn("repeated", reason)

    def test_different_content_ok(self):
        cb = CircuitBreaker(BreakerConfig(repetition_limit=3, error_window=100))
        cb.record_success("ls -la")
        cb.record_success("ls -la")
        cb.record_success("pwd")
        self.assertIsNone(cb.check())


class TestRateLimitBreaker(unittest.TestCase):
    """Actions-per-minute rate limit."""

    def test_no_trip_at_low_rate(self):
        cb = CircuitBreaker(BreakerConfig(rate_limit_per_minute=100, error_window=200))
        for _ in range(5):
            cb.record_success()
        self.assertIsNone(cb.check())

    def test_trips_at_high_rate(self):
        cb = CircuitBreaker(BreakerConfig(rate_limit_per_minute=5, error_window=100))
        for _ in range(6):
            cb.record_success()
        reason = cb.check()
        self.assertIsNotNone(reason)
        self.assertIn("rate limit", reason)


class TestTripAndReset(unittest.TestCase):
    """Trip/reset/acknowledge lifecycle."""

    def test_manual_trip(self):
        cb = CircuitBreaker()
        self.assertFalse(cb.is_tripped)
        cb.trip("test reason")
        self.assertTrue(cb.is_tripped)
        self.assertEqual(cb.trip_reason, "test reason")

    def test_reset_clears_trip(self):
        cb = CircuitBreaker()
        cb.trip("test")
        self.assertTrue(cb.is_tripped)
        cb.reset()
        self.assertFalse(cb.is_tripped)
        self.assertEqual(cb.trip_reason, "")

    def test_acknowledge_clears_trip_keeps_history(self):
        cb = CircuitBreaker(BreakerConfig(error_window=4))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        reason = cb.check_and_trip()
        self.assertIsNotNone(reason)
        self.assertTrue(cb.is_tripped)
        cb.acknowledge()
        self.assertFalse(cb.is_tripped)

    def test_check_and_trip(self):
        cb = CircuitBreaker(BreakerConfig(error_window=4, error_threshold=0.5))
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        reason = cb.check_and_trip()
        self.assertIsNotNone(reason)
        self.assertTrue(cb.is_tripped)

    def test_already_tripped_returns_reason(self):
        cb = CircuitBreaker(BreakerConfig(cooldown_seconds=9999))
        cb.trip("original")
        reason = cb.check()
        self.assertEqual(reason, "original")


class TestCooldown(unittest.TestCase):
    """Auto-reset after cooldown."""

    def test_auto_reset_after_cooldown(self):
        # 10ms cooldown / 20ms sleep was tighter than Windows'
        # time.sleep precision (~15ms scheduler quantum), so this test
        # tripped ~20% of runs in isolation.  100ms / 200ms gives
        # comfortable headroom on every supported platform.
        cb = CircuitBreaker(BreakerConfig(cooldown_seconds=0.1))
        cb.trip("test")
        self.assertTrue(cb.is_tripped)
        import time
        time.sleep(0.2)
        self.assertFalse(cb.is_tripped)


if __name__ == "__main__":
    unittest.main()
