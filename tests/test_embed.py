"""
tests/test_embed.py

Tests for embedder factory and protocol compliance.
"""

from __future__ import annotations

import pytest

from inferencache.embed import (
    Embedder,
    Qwen3Embedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


def test_get_embedder_fast():
    emb = get_embedder("fast")
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert "MiniLM" in emb._model_name or "mini" in emb._model_name.lower()


def test_get_embedder_balanced():
    emb = get_embedder("balanced")
    assert isinstance(emb, SentenceTransformerEmbedder)
    assert "bge-small" in emb._model_name


def test_get_embedder_accurate():
    emb = get_embedder("accurate")
    assert isinstance(emb, Qwen3Embedder)


def test_get_embedder_unknown_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_embedder("nonexistent")


def test_embedder_protocol():
    for preset in ("fast", "balanced", "accurate"):
        emb = get_embedder(preset)
        assert isinstance(emb, Embedder)
        mid = emb.model_id()
        assert isinstance(mid, str)
        assert len(mid) > 0
        assert mid == emb.model_id()  # stable


def test_model_ids_are_distinct():
    ids = {get_embedder(p).model_id() for p in ("fast", "balanced", "accurate")}
    assert len(ids) == 3
