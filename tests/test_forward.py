"""Tests for upstream header stripping on forward."""

from inferencache.proxy.forward import _STRIP_RESPONSE_HEADERS


def test_strip_response_headers_includes_hop_by_hop() -> None:
    for name in ("date", "server", "set-cookie", "cf-ray", "cf-cache-status"):
        assert name in _STRIP_RESPONSE_HEADERS
