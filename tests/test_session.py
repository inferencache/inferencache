"""
tests/test_session.py — SessionAwareLookup three-check sequence.
"""

from __future__ import annotations

import time

import pytest

from promptcache.engine import CacheConfig, CacheEngine
from promptcache.session import SessionAwareLookup
from promptcache.store import CacheEntry


class FakeEmbedder:
    """Orthogonal unit vectors keyed by prompt hash."""

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 384
        vec[hash(text) % 384] = 1.0
        return vec

    def model_id(self) -> str:
        return "fake-embedder"


class UniformEmbedder:
    """All prompts map to the same vector (cosine sim = 1.0)."""

    def embed(self, text: str) -> list[float]:
        return [0.1] * 384

    def model_id(self) -> str:
        return "uniform-embedder"


@pytest.fixture
def engine(tmp_path):
    config = CacheConfig(
        cache_dir=tmp_path / "session_cache",
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
    )
    e = CacheEngine(config)
    yield e
    e.close()


@pytest.fixture
def lookup(engine):
    return SessionAwareLookup(engine.cache_store, FakeEmbedder())


def _store_with_session(engine, prompt, response, session_hash):
    entry = CacheEntry(
        prompt=prompt,
        model="test-model",
        response=response,
        created_at=time.time(),
        metadata={"session_hash": session_hash},
    )
    engine.cache_store.write(entry, embedding=FakeEmbedder().embed(prompt))


# ---------------------------------------------------------------------------
# _session_hash
# ---------------------------------------------------------------------------


def test_session_hash_deterministic(lookup):
    h1 = lookup._session_hash(["turn1", "turn2", "turn3"])
    h2 = lookup._session_hash(["turn1", "turn2", "turn3"])
    assert h1 == h2
    assert len(h1) == 16


def test_session_hash_window_of_three(lookup):
    long_history = ["a", "b", "c", "d", "e"]
    short = lookup._session_hash(["c", "d", "e"])
    from_long = lookup._session_hash(long_history)
    assert short == from_long


def test_session_hash_differs_by_history(lookup):
    h1 = lookup._session_hash(["hello", "world"])
    h2 = lookup._session_hash(["foo", "bar"])
    assert h1 != h2


# ---------------------------------------------------------------------------
# _is_stateless
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "What is Python?",
        "explain recursion",
        "Define polymorphism",
        "Write a sort function",
        "How does caching work?",
        "How do I install pip?",
    ],
)
def test_is_stateless_true(lookup, prompt):
    assert lookup._is_stateless(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Fix this function",
        "Continue from above",
        "Make it faster",
    ],
)
def test_is_stateless_false(lookup, prompt):
    assert lookup._is_stateless(prompt) is False


# ---------------------------------------------------------------------------
# Three-check lookup
# ---------------------------------------------------------------------------


def test_check1_exact_hit_same_session(engine, lookup):
    history = ["prev turn 1", "prev turn 2"]
    ctx = lookup._session_hash(history)
    _store_with_session(engine, "fix the bug", "fixed response", ctx)

    result = lookup.lookup("fix the bug", history, "test-model", 0.85)
    assert result.hit is True
    assert result.hit_type == "exact"
    assert result.source == "tier1"
    assert result.response == "fixed response"


def test_check1_misses_different_session(engine, lookup):
    history_a = ["session A turn"]
    history_b = ["session B turn"]
    ctx_a = lookup._session_hash(history_a)
    _store_with_session(engine, "fix the bug", "fixed A", ctx_a)

    result = lookup.lookup("fix the bug", history_b, "test-model", 0.85)
    assert result.hit is False


def test_check2_stateless_cross_session_hit(engine, lookup):
    _store_with_session(engine, "What is REST?", "REST is...", "old_session")

    result = lookup.lookup(
        "What is REST?",
        ["completely different context"],
        "test-model",
        0.85,
    )
    assert result.hit is True
    assert result.hit_type == "exact"
    assert result.source == "tier1_stateless"


def test_check3_semantic_session_filtered(engine, tmp_path):
    """Semantic hit within session; miss when session context differs."""
    from unittest.mock import patch

    config = CacheConfig(
        cache_dir=tmp_path / "semantic_session",
        model="test-model",
        threshold=0.5,
        embedder=UniformEmbedder(),
    )
    eng = CacheEngine(config)
    sa = SessionAwareLookup(eng.cache_store, UniformEmbedder())

    history_a = ["context A"]
    ctx_a = sa._session_hash(history_a)
    stored = CacheEntry(
        prompt="explain caching basics",
        model="test-model",
        response="Caching saves API calls.",
        created_at=time.time(),
        metadata={"session_hash": ctx_a},
    )

    with patch.object(
        eng.cache_store,
        "query_semantic",
        return_value=[(stored, 0.92)],
    ) as mock_qs:
        result = sa.lookup(
            "explain caching fundamentals",
            history_a,
            "test-model",
            0.5,
        )
        mock_qs.assert_called_once()
        call_kwargs = mock_qs.call_args.kwargs
        assert call_kwargs.get("session_hash") == ctx_a

    assert result.hit is True
    assert result.hit_type == "semantic"
    assert result.source == "tier1_session"

    with patch.object(eng.cache_store, "query_semantic", return_value=[]):
        result_other = sa.lookup(
            "explain caching fundamentals",
            ["different context entirely"],
            "test-model",
            0.5,
        )
    assert result_other.hit is False

    eng.close()


def test_miss_on_empty_cache(lookup):
    result = lookup.lookup("unknown prompt", [], "test-model", 0.85)
    assert result.hit is False
    assert result.hit_type == "miss"
