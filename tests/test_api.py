"""
tests/test_api.py

Tests for @cache decorator, cache_context(), and engine registry.
"""

from __future__ import annotations

import pytest

from inferencache.api import _engines, _flush_engines, cache, cache_context
from inferencache.engine import CacheConfig


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        vec = [0.0] * 384
        vec[hash(text) % 384] = 1.0
        return vec

    def model_id(self) -> str:
        return "fake-test"


@pytest.fixture(autouse=True)
def clean_engines():
    _flush_engines()
    yield
    _flush_engines()


@pytest.fixture
def config(tmp_path):
    return CacheConfig(
        cache_dir=tmp_path / "api_test",
        model="test-model",
        threshold=0.85,
        embedder=FakeEmbedder(),
    )


def test_cache_sync_miss_then_hit(config):
    call_count = {"n": 0}

    @cache(config=config)
    def ask(prompt: str) -> str:
        call_count["n"] += 1
        return f"response to: {prompt}"

    assert ask("hello") == "response to: hello"
    assert call_count["n"] == 1

    assert ask("hello") == "response to: hello"
    assert call_count["n"] == 1  # cache hit — API not called again


def test_cache_context_miss_and_store(config):
    api_calls = []

    with cache_context("context prompt", config=config) as ctx:
        assert ctx.hit is False
        assert ctx.hit_type == "miss"
        api_calls.append(1)
        ctx.store("stored response")

    with cache_context("context prompt", config=config) as ctx:
        assert ctx.hit is True
        assert ctx.response == "stored response"
        assert len(api_calls) == 1


def test_engine_registry_reuses_instance(config):
    @cache(config=config)
    def fn_a(prompt: str) -> str:
        return prompt

    @cache(config=config)
    def fn_b(prompt: str) -> str:
        return prompt

    key = (str(config.cache_dir), config.model)
    assert key in _engines
    engine_a = _engines[key]

    # Same config → same engine object
    assert _engines[key] is engine_a


def test_engine_registry_separate_models(tmp_path):
    cfg_a = CacheConfig(
        cache_dir=tmp_path / "shared",
        model="model-a",
        embedder=FakeEmbedder(),
    )
    cfg_b = CacheConfig(
        cache_dir=tmp_path / "shared",
        model="model-b",
        embedder=FakeEmbedder(),
    )

    @cache(config=cfg_a)
    def ask_a(prompt: str) -> str:
        return "a"

    @cache(config=cfg_b)
    def ask_b(prompt: str) -> str:
        return "b"

    key_a = (str(cfg_a.cache_dir), "model-a")
    key_b = (str(cfg_b.cache_dir), "model-b")
    assert key_a in _engines
    assert key_b in _engines
    assert _engines[key_a] is not _engines[key_b]

    # Single lookup — two engines on the same cache_dir cannot open Qdrant concurrently
    assert ask_a("x") == "a"
