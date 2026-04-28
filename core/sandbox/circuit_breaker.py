"""
Artifex Assistant V5 — Circuit breakers (P6-T32).

Rate-limits and error-trips for agent actions. Prevents runaway loops where
the agent repeatedly hits errors or spams the same failing action.

Breakers track:
  1. Error rate — if >50% of recent actions fail, trip the breaker.
  2. Repetition — if the same action content appears N times, trip.
  3. Rate — if actions-per-minute exceed a threshold, trip.

A tripped breaker pauses auto-execution and requires human acknowledgement
or a cooldown period before resuming.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# ── Breaker configuration ───────────────────────────────────────────────────

@dataclass
class BreakerConfig:
    error_window: int = 10
    error_threshold: float = 0.5
    repetition_limit: int = 3
    rate_limit_per_minute: int = 30
    cooldown_seconds: float = 60.0


# ── Circuit breaker state ───────────────────────────────────────────────────

class CircuitBreaker:
    """Per-session circuit breaker."""

    def __init__(self, config: BreakerConfig | None = None):
        self.config = config or BreakerConfig()
        self._lock = threading.Lock()
        self._results: deque[bool] = deque(maxlen=self.config.error_window)
        self._recent_content: deque[str] = deque(maxlen=self.config.repetition_limit + 1)
        self._action_times: deque[float] = deque(maxlen=self.config.rate_limit_per_minute + 1)
        self._tripped = False
        self._trip_reason = ""
        self._trip_time = 0.0

    @property
    def is_tripped(self) -> bool:
        with self._lock:
            if self._tripped and self.config.cooldown_seconds > 0:
                elapsed = time.monotonic() - self._trip_time
                if elapsed >= self.config.cooldown_seconds:
                    self._tripped = False
                    self._trip_reason = ""
                    _log.info("Circuit breaker auto-reset after cooldown")
            return self._tripped

    @property
    def trip_reason(self) -> str:
        return self._trip_reason

    def record_success(self, content: str = ""):
        """Record a successful action."""
        with self._lock:
            self._results.append(True)
            if content:
                self._recent_content.append(content[:100])
            self._action_times.append(time.monotonic())

    def record_failure(self, content: str = ""):
        """Record a failed action."""
        with self._lock:
            self._results.append(False)
            if content:
                self._recent_content.append(content[:100])
            self._action_times.append(time.monotonic())

    def check(self) -> str | None:
        """Check if the breaker should trip.

        Returns the trip reason if tripped, None if OK.
        Does NOT modify state — call trip() separately if this returns a reason.
        """
        with self._lock:
            if self._tripped:
                return self._trip_reason

            if len(self._results) >= self.config.error_window:
                failures = sum(1 for r in self._results if not r)
                rate = failures / len(self._results)
                if rate >= self.config.error_threshold:
                    return f"error rate {rate:.0%} exceeds {self.config.error_threshold:.0%}"

            if len(self._recent_content) >= self.config.repetition_limit:
                last_n = list(self._recent_content)[-self.config.repetition_limit:]
                if len(set(last_n)) == 1:
                    return f"same action repeated {self.config.repetition_limit} times"

            if len(self._action_times) >= self.config.rate_limit_per_minute:
                oldest = self._action_times[0]
                newest = self._action_times[-1]
                if newest - oldest < 60.0:
                    return f"rate limit: {len(self._action_times)} actions in <60s"

        return None

    def trip(self, reason: str):
        """Trip the breaker."""
        with self._lock:
            self._tripped = True
            self._trip_reason = reason
            self._trip_time = time.monotonic()
            _log.warning("Circuit breaker tripped: %s", reason)

    def check_and_trip(self) -> str | None:
        """Check and auto-trip if needed. Returns reason or None."""
        reason = self.check()
        if reason and not self._tripped:
            self.trip(reason)
        return reason

    def reset(self):
        """Manual reset (human acknowledgement)."""
        with self._lock:
            self._tripped = False
            self._trip_reason = ""
            self._results.clear()
            self._recent_content.clear()
            self._action_times.clear()
            _log.info("Circuit breaker manually reset")

    def acknowledge(self):
        """Acknowledge a trip without full reset. Clears trip but keeps history."""
        with self._lock:
            self._tripped = False
            self._trip_reason = ""
