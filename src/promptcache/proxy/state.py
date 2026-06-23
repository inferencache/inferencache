"""
Shared runtime state for the proxy server and control API.

Both intercept.py and control/* import from here so engine instances,
cache directory, and SSE broadcast queues stay in sync.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..analytics import CacheAnalytics
from ..engine import CacheConfig, CacheEngine

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "promptcache"

_cache_dir: Path = _DEFAULT_CACHE_DIR
_engines: dict[str, CacheEngine] = {}
_analytics: CacheAnalytics | None = None
_client_queues: set[asyncio.Queue[str]] = set()
_batch_running: bool = False


def init_state(cache_dir: Path | None = None) -> None:
    """Called once at app startup."""
    global _cache_dir, _analytics
    if cache_dir is not None:
        _cache_dir = cache_dir
    _cache_dir.mkdir(parents=True, exist_ok=True)
    _analytics = None


def get_cache_dir() -> Path:
    return _cache_dir


def get_engine(
    model: str,
    threshold: float = 0.85,
    provider: str = "openai",
    *,
    default_endpoint: str = "proxy",
) -> CacheEngine:
    key = f"{model}:{provider}"
    if key not in _engines:
        _engines[key] = CacheEngine(
            CacheConfig(
                cache_dir=_cache_dir,
                model=model,
                provider=provider,
                threshold=threshold,
                embedder_preset="balanced",
                default_endpoint=default_endpoint,
            )
        )
    return _engines[key]


def get_engine_for_model(model: str, path: str = "") -> CacheEngine:
    """Return an engine for proxy intercept, inferring provider from model/path."""
    provider = "anthropic" if "/messages" in path or model.startswith("claude") else "openai"
    return get_engine(model, provider=provider, default_endpoint="proxy")


def get_analytics() -> CacheAnalytics:
    global _analytics
    if _analytics is None:
        _analytics = CacheAnalytics(_cache_dir)
    return _analytics


def all_engines() -> dict[str, CacheEngine]:
    return _engines


async def broadcast_sse(msg: str) -> None:
    for q in list(_client_queues):
        await q.put(msg)


def register_sse_client() -> asyncio.Queue[str]:
    q: asyncio.Queue[str] = asyncio.Queue()
    _client_queues.add(q)
    return q


def unregister_sse_client(q: asyncio.Queue[str]) -> None:
    _client_queues.discard(q)


def is_batch_running() -> bool:
    return _batch_running


def set_batch_running(running: bool) -> None:
    global _batch_running
    _batch_running = running


def query_index_db(sql: str, params: tuple = ()) -> list[dict]:
    import sqlite3

    db_path = _cache_dir / "index.db"
    if not db_path.exists():
        return []
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def write_index_db(sql: str, params: tuple = ()) -> None:
    import sqlite3

    from fastapi import HTTPException

    db_path = _cache_dir / "index.db"
    if not db_path.exists():
        raise HTTPException(404, "Cache not initialised — run a test suite first")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(sql, params)
        conn.commit()
