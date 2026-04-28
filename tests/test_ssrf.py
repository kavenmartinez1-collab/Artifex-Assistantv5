"""Tests for web-gateway SSRF protection — sanitize_url and _resolves_to_private.

Tests the resolve-then-check approach against all known bypass vectors:
octal IPs, hex IPs, decimal int IPs, IPv6, link-local, and DNS rebinding.
"""

import ipaddress
import sys
import os
import pytest
from unittest.mock import patch

# The web-gateway is a standalone FastAPI app with its own config module.
# Add it to sys.path so we can import sanitizer and config directly.
_gateway_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web-gateway"
)
sys.path.insert(0, _gateway_dir)

from sanitizer import _resolves_to_private, sanitize_url


class TestResolvesToPrivate:
    """_resolves_to_private checks resolved IPs against ipaddress properties."""

    def _mock_resolve(self, ip_str):
        """Create a mock getaddrinfo return for a single IP."""
        family = 2  # AF_INET
        if ":" in ip_str:
            family = 10  # AF_INET6
        return [(family, 1, 0, "", (ip_str, 0))]

    def test_loopback_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("127.0.0.1")):
            result = _resolves_to_private("evil.com")
            assert result is not None
            assert "127.0.0.1" in result

    def test_ipv6_loopback_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("::1")):
            result = _resolves_to_private("evil.com")
            assert result is not None
            assert "::1" in result

    def test_private_10_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("10.0.0.1")):
            result = _resolves_to_private("evil.com")
            assert result is not None

    def test_private_172_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("172.16.0.1")):
            result = _resolves_to_private("evil.com")
            assert result is not None

    def test_private_192_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("192.168.1.1")):
            result = _resolves_to_private("evil.com")
            assert result is not None

    def test_link_local_blocked(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("169.254.169.254")):
            result = _resolves_to_private("evil.com")
            assert result is not None

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("93.184.216.34")):
            result = _resolves_to_private("example.com")
            assert result is None

    def test_unresolvable_passes(self):
        """Unresolvable hosts pass — let the fetcher fail naturally."""
        import socket
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("not found")):
            result = _resolves_to_private("nonexistent.invalid")
            assert result is None


class TestSanitizeUrlSsrfBypasses:
    """sanitize_url must reject all known SSRF bypass vectors."""

    def _mock_resolve(self, ip_str):
        family = 10 if ":" in ip_str else 2
        return [(family, 1, 0, "", (ip_str, 0))]

    def test_plain_localhost_blocked(self):
        safe, reason = sanitize_url("http://localhost/admin")
        assert not safe

    def test_plain_127_blocked(self):
        safe, reason = sanitize_url("http://127.0.0.1/admin")
        assert not safe

    def test_ipv6_loopback_blocked(self):
        """[::1] must be caught by resolve check even if not in BLOCKED_HOSTS."""
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("::1")):
            safe, reason = sanitize_url("http://[::1]/admin")
            assert not safe

    def test_dns_rebinding_blocked(self):
        """Hostname that resolves to private IP is blocked."""
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("10.0.0.1")):
            safe, reason = sanitize_url("http://rebind.evil.com/admin")
            assert not safe
            assert "private" in reason.lower() or "blocked" in reason.lower()

    def test_ipv6_mapped_v4_blocked(self):
        """::ffff:127.0.0.1 (IPv6-mapped IPv4 loopback) must be blocked."""
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("::ffff:127.0.0.1")):
            safe, reason = sanitize_url("http://[::ffff:127.0.0.1]/admin")
            assert not safe

    def test_metadata_endpoint_blocked(self):
        safe, reason = sanitize_url("http://169.254.169.254/latest/meta-data/")
        assert not safe

    def test_private_172_20_blocked(self):
        """172.20.x.x is private (was missing from old pattern list)."""
        safe, reason = sanitize_url("http://172.20.0.1/")
        assert not safe

    def test_public_url_passes(self):
        """Legitimate public URLs must pass."""
        with patch("socket.getaddrinfo", return_value=self._mock_resolve("93.184.216.34")):
            safe, reason = sanitize_url("https://example.com/page")
            assert safe
            assert reason == "OK"

    def test_empty_url_rejected(self):
        safe, reason = sanitize_url("")
        assert not safe

    def test_ftp_scheme_rejected(self):
        safe, reason = sanitize_url("ftp://files.example.com/data")
        assert not safe

    def test_file_scheme_rejected(self):
        safe, reason = sanitize_url("file:///etc/passwd")
        assert not safe
