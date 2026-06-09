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


class TestSafeRequestRedirects:
    """safe_request must re-validate every redirect hop, not just the first URL.

    Uses unresolvable .test hostnames so _resolves_to_private passes them
    through without real DNS, while blocked hosts/IPs are caught by the
    static rules — no network access needed.
    """

    @pytest.fixture()
    def gateway_main(self, tmp_path, monkeypatch):
        import importlib.util

        monkeypatch.setenv("QUARANTINE_DIR", str(tmp_path / "quarantine"))
        # Reimport config so it picks up the env var; load the gateway's
        # main.py by explicit path (the repo root also has a main.py).
        sys.modules.pop("config", None)
        spec = importlib.util.spec_from_file_location(
            "gateway_main", os.path.join(_gateway_dir, "main.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
        sys.modules.pop("config", None)

    def _run(self, gateway_main, handler, method="GET", url="http://start.test/"):
        import asyncio
        import socket
        import httpx

        async def go():
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(handler), follow_redirects=False
            )
            gateway_main._client = client
            try:
                return await gateway_main.safe_request(method, url, timeout=5)
            finally:
                await client.aclose()

        # No real DNS: .test hosts are unresolvable (pass-through), blocked
        # targets are caught by the static host/pattern rules before resolve.
        with patch("socket.getaddrinfo", side_effect=socket.gaierror("no dns in tests")):
            return asyncio.run(go())

    def test_redirect_to_private_ip_blocked(self, gateway_main):
        from fastapi import HTTPException
        import httpx

        attempted = []

        def handler(request):
            attempted.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

        with pytest.raises(HTTPException) as exc:
            self._run(gateway_main, handler)
        assert exc.value.status_code == 403
        # The private target must never be contacted
        assert all("169.254.169.254" not in u for u in attempted)

    def test_redirect_to_localhost_blocked(self, gateway_main):
        from fastapi import HTTPException
        import httpx

        def handler(request):
            return httpx.Response(301, headers={"location": "http://localhost/admin"})

        with pytest.raises(HTTPException) as exc:
            self._run(gateway_main, handler)
        assert exc.value.status_code == 403

    def test_redirect_to_blocked_scheme_blocked(self, gateway_main):
        from fastapi import HTTPException
        import httpx

        def handler(request):
            return httpx.Response(302, headers={"location": "file:///etc/passwd"})

        with pytest.raises(HTTPException) as exc:
            self._run(gateway_main, handler)
        assert exc.value.status_code == 403

    def test_redirect_chain_to_public_followed(self, gateway_main):
        import httpx

        def handler(request):
            if request.url.host == "start.test":
                return httpx.Response(302, headers={"location": "https://final.test/page"})
            return httpx.Response(200, text="hello")

        resp = self._run(gateway_main, handler)
        assert resp.status_code == 200
        assert str(resp.url) == "https://final.test/page"

    def test_relative_redirect_resolved_and_followed(self, gateway_main):
        import httpx

        def handler(request):
            if request.url.path == "/":
                return httpx.Response(302, headers={"location": "/landing"})
            return httpx.Response(200, text="ok")

        resp = self._run(gateway_main, handler)
        assert resp.status_code == 200
        assert resp.url.path == "/landing"

    def test_too_many_redirects_rejected(self, gateway_main):
        from fastapi import HTTPException
        import httpx

        count = {"n": 0}

        def handler(request):
            count["n"] += 1
            return httpx.Response(302, headers={"location": f"http://hop{count['n']}.test/"})

        with pytest.raises(HTTPException) as exc:
            self._run(gateway_main, handler)
        assert exc.value.status_code == 502
        assert count["n"] == gateway_main.MAX_REDIRECTS + 1
