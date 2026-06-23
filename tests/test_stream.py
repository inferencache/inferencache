"""
tests/test_stream.py

Tests for the streaming helpers in CacheEngine:
  - stream_cached: sync iterator over a cached response string
  - astream_cached: async iterator variant
  - collect_stream / acollect_stream: consume a stream back to a string
"""

from __future__ import annotations

import asyncio
import pytest

from inferencache.engine import CacheConfig, CacheEngine


class _FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.0] * 384

    def model_id(self) -> str:
        return "fake"


@pytest.fixture
def engine(tmp_path):
    config = CacheConfig(
        cache_dir=tmp_path / "stream_test",
        model="test-model",
        embedder=_FakeEmbedder(),
        stream_chunk_size=8,
        stream_delay=0.0,
    )
    e = CacheEngine(config)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# sync stream_cached
# ---------------------------------------------------------------------------


def test_stream_cached_empty_string(engine):
    chunks = list(engine.stream_cached(""))
    assert chunks == [] or "".join(chunks) == ""


def test_stream_cached_full_content_preserved(engine):
    text = "Hello world! This is a test of the streaming cache reconstitution."
    assert "".join(engine.stream_cached(text)) == text


def test_stream_cached_respects_chunk_size(engine):
    engine._config.stream_chunk_size = 5
    text = "0123456789"  # 10 chars → 2 chunks of 5
    chunks = list(engine.stream_cached(text))
    assert len(chunks) == 2
    assert chunks == ["01234", "56789"]


def test_stream_cached_handles_odd_length(engine):
    engine._config.stream_chunk_size = 4
    text = "abcde"  # 5 chars → [0:4] + [4:5]
    chunks = list(engine.stream_cached(text))
    assert "".join(chunks) == "abcde"
    assert len(chunks) == 2


def test_stream_cached_large_text(engine):
    engine._config.stream_chunk_size = 100
    text = "x" * 1000
    chunks = list(engine.stream_cached(text))
    assert "".join(chunks) == text
    assert len(chunks) == 10


# ---------------------------------------------------------------------------
# collect_stream
# ---------------------------------------------------------------------------


def test_collect_stream_roundtrip(engine):
    text = "Collect this stream into a single string."
    stream = engine.stream_cached(text)
    result = CacheEngine.collect_stream(stream)
    assert result == text


def test_collect_stream_empty(engine):
    result = CacheEngine.collect_stream(iter([]))
    assert result == ""


# ---------------------------------------------------------------------------
# async stream_cached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_cached_full_content(engine):
    text = "Async streaming reconstitution test."
    chunks = []
    async for chunk in engine.astream_cached(text):
        chunks.append(chunk)
    assert "".join(chunks) == text


@pytest.mark.asyncio
async def test_astream_cached_chunk_size(engine):
    engine._config.stream_chunk_size = 6
    text = "abcdefghijkl"  # 12 chars → 2 chunks of 6
    chunks = []
    async for chunk in engine.astream_cached(text):
        chunks.append(chunk)
    assert chunks == ["abcdef", "ghijkl"]


@pytest.mark.asyncio
async def test_acollect_stream_roundtrip(engine):
    text = "Round-trip test for async stream collection."
    result = await CacheEngine.acollect_stream(engine.astream_cached(text))
    assert result == text


# ---------------------------------------------------------------------------
# Integration: stream from cache after store
# ---------------------------------------------------------------------------


def test_stream_cache_hit_via_decorator(tmp_path):
    """
    Verify that the @cache(streaming=True) decorator yields chunks from
    the cache on a second call rather than invoking the underlying function.
    """
    from inferencache import CacheConfig, cache

    call_count = 0

    config = CacheConfig(
        cache_dir=tmp_path / "decorator_stream_test",
        model="test-model",
        embedder=_FakeEmbedder(),
    )

    @cache(config=config, streaming=True)
    def stream_ask(prompt: str):
        nonlocal call_count
        call_count += 1
        yield "Hello "
        yield "world"

    # First call — real function executes
    result1 = "".join(stream_ask("test prompt"))
    assert result1 == "Hello world"
    assert call_count == 1

    # Second call — should come from cache, function NOT called again
    result2 = "".join(stream_ask("test prompt"))
    assert result2 == "Hello world"
    assert call_count == 1  # unchanged


@pytest.mark.asyncio
async def test_stream_async_decorator(tmp_path):
    """
    Verify the async @cache decorator works for non-streaming async functions.
    """
    from inferencache import CacheConfig, cache

    call_count = 0

    config = CacheConfig(
        cache_dir=tmp_path / "async_decorator_test",
        model="test-model",
        embedder=_FakeEmbedder(),
    )

    @cache(config=config)
    async def async_ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "async response"

    r1 = await async_ask("async prompt")
    assert r1 == "async response"
    assert call_count == 1

    r2 = await async_ask("async prompt")
    assert r2 == "async response"
    assert call_count == 1
