"""Tests for temporal validity in CacheEngine lookup and store."""

from __future__ import annotations

import time

import pytest

from inferencache.engine import CacheConfig, CacheEngine
from inferencache.ttl import TTLClass


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 384
        idx = hash(text) % 384
        vec[idx] = 1.0
        return vec

    def model_id(self) -> str:
        return "fake-embedder"


@pytest.fixture
def engine(tmp_path):
    config = CacheConfig(
        cache_dir=tmp_path / "ttl_test",
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
    )
    e = CacheEngine(config)
    yield e
    e.close()


def test_store_persists_ttl_fields(engine):
    engine.store("What is cosine similarity?", "A formula.", ttl_override=TTLClass.EPHEMERAL)
    entry = engine.cache_store.get_exact("What is cosine similarity?", "test-model")
    assert entry is not None
    assert entry.ttl_class == TTLClass.EPHEMERAL.value
    assert entry.expires_at is not None


def test_expired_entry_returns_stale_miss(engine):
    prompt = "What is cosine similarity?"
    engine.store(prompt, "Paris", ttl_override=TTLClass.EPHEMERAL)

    entry = engine.cache_store.get_exact(prompt, "test-model")
    assert entry is not None
    entry.expires_at = time.time() - 1
    engine.cache_store.write(entry)

    result = engine.lookup(prompt)
    assert result.hit is False
    assert result.hit_type == "stale_miss"


def test_session_mismatch_returns_stale_miss(engine):
    prompt = "Summarize this file"
    engine.store(
        prompt,
        "Summary",
        session_id="session-abc",
        ttl_override=TTLClass.SESSION,
    )

    result = engine.lookup(prompt, session_id="session-xyz")
    assert result.hit is False
    assert result.hit_type == "stale_miss"


def test_session_match_returns_hit(engine):
    prompt = "Summarize this file"
    engine.store(
        prompt,
        "Summary",
        session_id="session-abc",
        ttl_override=TTLClass.SESSION,
    )

    result = engine.lookup(prompt, session_id="session-abc")
    assert result.hit is True
    assert result.hit_type == "exact"
