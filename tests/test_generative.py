"""Integration tests for generative reuse lookup."""

from __future__ import annotations

import math

import pytest

from inferencache.adapt import AdaptationEngine
from inferencache.engine import CacheConfig, CacheEngine


class MockAdaptationClient:
    _model = "gpt-4o-mini"

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        return "def parse_tsv(path):\n    ...", 80, 40


class ControlledSimilarityEmbedder:
    """Returns vectors with a fixed cosine similarity for non-stored prompts."""

    def __init__(self, similarity: float = 0.85) -> None:
        self.similarity = similarity
        self._stored: set[str] = set()

    def mark_stored(self, prompt: str) -> None:
        self._stored.add(prompt)

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 384
        if text in self._stored:
            vec[0] = 1.0
            return vec
        sim = self.similarity
        vec[0] = sim
        vec[1] = math.sqrt(max(0.0, 1.0 - sim * sim))
        return vec

    def model_id(self) -> str:
        return "controlled-embedder"


@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "generative_test"


def _make_engine(tmp_cache_dir, *, similarity: float = 0.85, enabled: bool = True):
    embedder = ControlledSimilarityEmbedder(similarity=similarity)
    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.92,
        generative_reuse_enabled=enabled,
        generative_reuse_floor=0.78,
        adaptation_client=MockAdaptationClient() if enabled else None,
        embedder=embedder,
    )
    engine = CacheEngine(config)
    return engine, embedder


def test_lookup_generative_zone(tmp_cache_dir):
    engine, embedder = _make_engine(tmp_cache_dir)
    try:
        stored = "Parse a CSV file into dicts"
        embedder.mark_stored(stored)
        engine.store(stored, "def parse_csv(): ...")

        result = engine.lookup("Parse a TSV file into dicts")
        assert result.hit is True
        assert result.hit_type == "generative"
        assert result.adaptation_model == "gpt-4o-mini"
        assert "tsv" in (result.response or "").lower()
    finally:
        engine.close()


def test_lookup_below_floor_is_miss(tmp_cache_dir):
    engine, embedder = _make_engine(tmp_cache_dir, similarity=0.40)
    try:
        stored = "Parse a CSV file into dicts"
        embedder.mark_stored(stored)
        engine.store(stored, "def parse_csv(): ...")

        result = engine.lookup("Write a poem about autumn")
        assert result.hit is False
        assert result.hit_type == "miss"
    finally:
        engine.close()


def test_generative_disabled_falls_through_to_miss(tmp_cache_dir):
    engine, embedder = _make_engine(tmp_cache_dir, enabled=False)
    try:
        stored = "Parse a CSV file into dicts"
        embedder.mark_stored(stored)
        engine.store(stored, "def parse_csv(): ...")

        result = engine.lookup("Parse a TSV file into dicts")
        assert result.hit is False
        assert result.hit_type == "miss"
    finally:
        engine.close()


def test_adaptation_failure_falls_through_to_miss(tmp_cache_dir):
    class FailingClient:
        _model = "gpt-4o-mini"

        def complete(self, system: str, user: str) -> tuple[str, int, int]:
            return "ADAPTATION_FAILED", 0, 0

    embedder = ControlledSimilarityEmbedder(similarity=0.85)
    config = CacheConfig(
        cache_dir=tmp_cache_dir,
        model="test-model",
        threshold=0.92,
        generative_reuse_enabled=True,
        generative_reuse_floor=0.78,
        adaptation_client=FailingClient(),
        embedder=embedder,
    )
    engine = CacheEngine(config)
    try:
        stored = "Parse a CSV file into dicts"
        embedder.mark_stored(stored)
        engine.store(stored, "def parse_csv(): ...")

        result = engine.lookup("Parse a TSV file into dicts")
        assert result.hit is False
        assert result.hit_type == "miss"
    finally:
        engine.close()
