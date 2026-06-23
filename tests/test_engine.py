"""
tests/test_engine.py

Tests for CacheEngine: exact match, semantic match, miss, write-back,
threshold enforcement, and enabled/disabled toggle.

Uses a temporary directory for the cache so tests are isolated.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from promptcache.engine import CacheConfig, CacheEngine, CacheResult
from promptcache.embed import Embedder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """
    Deterministic embedder for testing.

    Returns a unit vector where only the first element is non-zero,
    mapped from the hash of the input string. This guarantees that:
      - identical strings → identical vectors (exact match logic works)
      - different strings → orthogonal vectors (cosine sim = 0)

    For semantic similarity tests we override embed() directly.
    """

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 384
        idx = hash(text) % 384
        vec[idx] = 1.0
        return vec

    def model_id(self) -> str:
        return "fake-embedder"


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "promptcache_test"


@pytest.fixture
def config(tmp_cache_dir):
    return CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
    )


@pytest.fixture
def engine(config):
    e = CacheEngine(config)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Basic hit/miss
# ---------------------------------------------------------------------------


def test_miss_on_empty_cache(engine):
    result = engine.lookup("What is the capital of France?")
    assert result.hit is False
    assert result.hit_type == "miss"
    assert result.response is None


def test_exact_hit_after_store(engine):
    prompt = "What is the capital of France?"
    response = "Paris"

    # First lookup should miss
    result = engine.lookup(prompt)
    assert not result.hit

    # Store the response
    engine.store(prompt, response)

    # Second lookup should be an exact hit
    result = engine.lookup(prompt)
    assert result.hit is True
    assert result.hit_type == "exact"
    assert result.response == response
    assert result.similarity == 1.0


def test_exact_hit_returns_correct_response(engine):
    engine.store("prompt A", "response A")
    engine.store("prompt B", "response B")

    r1 = engine.lookup("prompt A")
    r2 = engine.lookup("prompt B")

    assert r1.response == "response A"
    assert r2.response == "response B"


def test_miss_increments_on_each_miss(engine):
    engine.lookup("never stored")
    engine.lookup("also never stored")
    stats = engine.cache_store._conn.execute(
        "SELECT miss_count FROM stats WHERE id = 1"
    ).fetchone()
    assert stats["miss_count"] == 2


# ---------------------------------------------------------------------------
# Semantic match
# ---------------------------------------------------------------------------


def test_semantic_hit_with_similar_embeddings(tmp_cache_dir):
    """
    Force a semantic hit by using an embedder that returns near-identical
    vectors for two different prompt strings.
    """
    similar_vec = [0.1] * 384

    class SimilarEmbedder:
        call_count = 0

        def embed(self, text: str) -> list[float]:
            SimilarEmbedder.call_count += 1
            return similar_vec[:]  # same vector for every input

        def model_id(self) -> str:
            return "similar-embedder"

    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.80,
        embedder=SimilarEmbedder(),
    )
    engine = CacheEngine(config)

    try:
        original_prompt = "What is the weather today?"
        similar_prompt = "How is the weather right now?"

        engine.store(original_prompt, "Sunny and warm")

        result = engine.lookup(similar_prompt)
        # Both prompts produce identical vectors → cosine sim = 1.0 → hit
        assert result.hit is True
        assert result.hit_type == "semantic"
        assert result.response == "Sunny and warm"
        assert result.similarity > 0.80
    finally:
        engine.close()


def test_no_semantic_hit_below_threshold(tmp_cache_dir):
    """
    Verify that a semantic hit is NOT returned when similarity is below
    the configured threshold.
    """
    import random

    rng = random.Random(42)

    class OrthogonalEmbedder:
        def embed(self, text: str) -> list[float]:
            # Each call returns a different random unit vector.
            vec = [rng.gauss(0, 1) for _ in range(384)]
            norm = sum(x**2 for x in vec) ** 0.5
            return [x / norm for x in vec]

        def model_id(self) -> str:
            return "orthogonal-embedder"

    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.99,  # very high — random vectors will never pass
        embedder=OrthogonalEmbedder(),
    )
    engine = CacheEngine(config)

    try:
        engine.store("some prompt", "some response")
        result = engine.lookup("completely different prompt")
        assert result.hit is False
    finally:
        engine.close()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_cached_yields_full_content(engine):
    full_text = "This is a cached streaming response. " * 10
    chunks = list(engine.stream_cached(full_text))
    assert "".join(chunks) == full_text


def test_stream_cached_chunk_size(engine):
    engine._config.stream_chunk_size = 10
    text = "0123456789abcdefghij"  # 20 chars
    chunks = list(engine.stream_cached(text))
    assert len(chunks) == 2
    assert chunks[0] == "0123456789"
    assert chunks[1] == "abcdefghij"


# ---------------------------------------------------------------------------
# Enabled / disabled
# ---------------------------------------------------------------------------


def test_disabled_engine_always_misses(engine):
    engine.set_enabled(False)
    engine.store("test prompt", "test response")
    result = engine.lookup("test prompt")
    assert result.hit is False
    assert result.hit_type == "miss"


def test_re_enable_engine(engine):
    engine.set_enabled(False)
    engine.set_enabled(True)
    engine.store("re-enabled prompt", "response")
    result = engine.lookup("re-enabled prompt")
    assert result.hit is True


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_set_threshold_updates_config(engine):
    engine.set_threshold(0.92)
    assert engine.config.threshold == 0.92


def test_set_threshold_rejects_out_of_range(engine):
    with pytest.raises(ValueError):
        engine.set_threshold(1.5)
    with pytest.raises(ValueError):
        engine.set_threshold(-0.1)


# ---------------------------------------------------------------------------
# Write-back size limit
# ---------------------------------------------------------------------------


def test_oversized_response_not_cached(engine):
    engine._config.max_response_tokens = 10  # 10 tokens → ~40 chars
    huge_response = "x" * 1000  # way over limit
    engine.store("prompt for huge response", huge_response)
    result = engine.lookup("prompt for huge response")
    assert result.hit is False


def test_normal_response_is_cached(engine):
    engine._config.max_response_tokens = 1000
    response = "A perfectly normal response."
    engine.store("normal prompt", response)
    result = engine.lookup("normal prompt")
    assert result.hit is True
    assert result.response == response


# ---------------------------------------------------------------------------
# Latency tracking
# ---------------------------------------------------------------------------


def test_result_has_latency(engine):
    result = engine.lookup("any prompt")
    assert result.latency_ms >= 0.0
    assert isinstance(result.latency_ms, float)


def test_hit_result_has_latency(engine):
    engine.store("prompt", "response")
    result = engine.lookup("prompt")
    assert result.hit is True
    assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# Multi-tier (opt-in)
# ---------------------------------------------------------------------------


def test_legacy_path_uses_lookup_legacy_directly(engine):
    """Default config delegates to _lookup_legacy unchanged."""
    engine.store("legacy prompt", "legacy response")
    result = engine._lookup_legacy("legacy prompt")
    assert result.hit is True
    assert result.hit_type == "exact"
    assert result.tier1_hit is True
    assert result.tier1_tokens_saved > 0


def test_tier_auto_routes_code_threshold(tmp_cache_dir):
    """tier='auto' uses CODE threshold 0.92 for code prompts."""
    similar_vec = [0.1] * 384

    class SimilarEmbedder:
        def embed(self, text: str) -> list[float]:
            return similar_vec[:]

        def model_id(self) -> str:
            return "similar-embedder"

    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.99,  # global would block; router should use 0.92
        embedder=SimilarEmbedder(),
        tier="auto",
        provider="openai",
    )
    engine = CacheEngine(config)
    try:
        engine.store("def foo(): pass", "code response")
        result = engine.lookup("def bar(): pass")
        assert result.hit is True
        assert result.tier1_hit is True
    finally:
        engine.close()


def test_session_aware_prevents_cross_session_hit(tmp_cache_dir):
    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
        session_aware=True,
    )
    engine = CacheEngine(config)
    try:
        history_a = ["context A"]
        session_hash = engine._session_lookup._session_hash(history_a)
        engine.store(
            "fix the bug",
            "fixed",
            session_hash=session_hash,
        )
        result = engine.lookup(
            "fix the bug",
            session_history=["different context"],
        )
        assert result.hit is False
    finally:
        engine.close()


def test_prefix_warnings_on_miss_with_tier_auto(tmp_cache_dir):
    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
        tier="auto",
        provider="openai",
    )
    engine = CacheEngine(config)
    try:
        result = engine.lookup(
            "unknown",
            system_prompt="Help {user} today",
        )
        assert result.hit is False
        assert len(result.prefix_warnings) >= 1
    finally:
        engine.close()
