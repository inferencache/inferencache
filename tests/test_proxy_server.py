"""Tests for proxy server site detection and create_app routing."""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from inferencache.proxy.intercept import InterceptResult
from inferencache.proxy.server import (
    _cache_headers,
    _with_cache_headers,
    create_app,
    site_is_built,
)
from fastapi.responses import Response


def test_site_is_built_false_for_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "site"
    empty.mkdir()
    assert site_is_built(empty) is False


def test_site_is_built_true_when_index_present(tmp_path: Path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<html></html>", encoding="utf-8")
    assert site_is_built(site) is True


def test_site_is_built_false_for_missing_dir(tmp_path: Path) -> None:
    assert site_is_built(tmp_path / "missing") is False


@pytest.mark.asyncio
async def test_create_app_serve_site_unbuilt_returns_503(tmp_path: Path, monkeypatch) -> None:
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr("inferencache.proxy.server._SITE_DIR", site)

    app = create_app(cache_dir=tmp_path / "cache", serve_site=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 503
    assert "Site not built" in resp.json()["error"]


@pytest.mark.asyncio
async def test_create_app_serve_site_built_serves_static(tmp_path: Path, monkeypatch) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<html><body>ok</body></html>", encoding="utf-8")
    monkeypatch.setattr("inferencache.proxy.server._SITE_DIR", site)

    app = create_app(cache_dir=tmp_path / "cache", serve_site=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
    assert resp.status_code == 200
    assert "ok" in resp.text


def test_create_app_serve_site_false_no_root_route(tmp_path: Path, monkeypatch) -> None:
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.setattr("inferencache.proxy.server._SITE_DIR", site)

    app = create_app(cache_dir=tmp_path / "cache", serve_site=False)
    routes = [getattr(r, "path", None) for r in app.routes]
    assert "/" not in routes


def test_cache_headers_on_miss_uses_best_similarity() -> None:
    result = InterceptResult(
        hit=False,
        hit_type="miss",
        cached_response=None,
        model="gpt-4o-mini",
        prompt="say hi",
        similarity=0.0,
        latency_ms=12.34,
        best_similarity=0.42,
    )
    headers = _cache_headers(result)
    assert headers["X-Cache"] == "miss"
    assert headers["X-Cache-Similarity"] == "0.42"
    assert headers["X-Cache-Latency-Ms"] == "12.34"


def test_cache_headers_on_hit_uses_similarity() -> None:
    result = InterceptResult(
        hit=True,
        hit_type="exact",
        cached_response={},
        model="gpt-4o-mini",
        prompt="say hi",
        similarity=1.0,
        latency_ms=0.5,
        best_similarity=0.9,
    )
    headers = _cache_headers(result)
    assert headers["X-Cache"] == "exact"
    assert headers["X-Cache-Similarity"] == "1.0"


def test_with_cache_headers_merges_into_forwarded_response() -> None:
    upstream = Response(
        content=b'{"ok": true}',
        status_code=200,
        headers={"content-type": "application/json", "date": "upstream"},
    )
    merged = _with_cache_headers(upstream, {"X-Cache": "miss", "X-Cache-Latency-Ms": "1.0"})
    assert merged.headers["X-Cache"] == "miss"
    assert merged.headers["X-Cache-Latency-Ms"] == "1.0"
    assert merged.headers["content-type"] == "application/json"
    assert merged.body == b'{"ok": true}'
