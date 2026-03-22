"""Tests for core/resilience.py — error classification, retry logic."""

import pytest
from core.resilience import is_recoverable, is_oom, retry_with_backoff, engine_recovery


class TestErrorClassification:
    """Test exception classification."""

    def test_oom_is_recoverable(self):
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert is_recoverable(exc) is True
        assert is_oom(exc) is True

    def test_connection_error_is_recoverable(self):
        exc = ConnectionError("Connection refused")
        assert is_recoverable(exc) is True
        assert is_oom(exc) is False

    def test_timeout_is_recoverable(self):
        exc = TimeoutError("Request timed out")
        assert is_recoverable(exc) is True

    def test_file_not_found_not_recoverable(self):
        exc = FileNotFoundError("Model not found: /path/to/model")
        assert is_recoverable(exc) is False

    def test_value_error_not_recoverable(self):
        exc = ValueError("Invalid parameter")
        assert is_recoverable(exc) is False


class TestRetryWithBackoff:
    """Test the retry decorator."""

    def test_succeeds_on_first_try(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def always_works():
            call_count[0] += 1
            return "ok"

        assert always_works() == "ok"
        assert call_count[0] == 1

    def test_retries_on_recoverable_error(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def fails_then_works():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection refused")
            return "recovered"

        assert fails_then_works() == "recovered"
        assert call_count[0] == 3

    def test_raises_non_recoverable_immediately(self):
        call_count = [0]

        @retry_with_backoff(max_retries=3, initial_delay=0.01)
        def always_fails():
            call_count[0] += 1
            raise ValueError("Bad input")

        with pytest.raises(ValueError):
            always_fails()
        assert call_count[0] == 1  # No retry for non-recoverable

    def test_exhausts_retries(self):
        call_count = [0]

        @retry_with_backoff(max_retries=2, initial_delay=0.01)
        def always_timeout():
            call_count[0] += 1
            raise ConnectionError("timed out")

        with pytest.raises(ConnectionError):
            always_timeout()
        assert call_count[0] == 3  # 1 initial + 2 retries


class TestEngineRecovery:
    """Test the engine recovery context manager."""

    def test_no_error(self):
        from unittest.mock import MagicMock
        engine = MagicMock()
        with engine_recovery(engine) as ctx:
            ctx.response = "all good"
        assert ctx.should_retry is False
        assert ctx.response == "all good"

    def test_oom_triggers_retry(self):
        from unittest.mock import MagicMock
        engine = MagicMock()
        with engine_recovery(engine) as ctx:
            raise RuntimeError("CUDA out of memory")
        assert ctx.should_retry is True

    def test_non_recoverable_propagates(self):
        from unittest.mock import MagicMock
        engine = MagicMock()
        with pytest.raises(FileNotFoundError):
            with engine_recovery(engine) as ctx:
                raise FileNotFoundError("Model not found")
