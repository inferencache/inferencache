"""
tests/test_mcp.py

Tests for the MCP tool implementations in mcp/tools.py.

These tests call tool functions directly — no MCP server is started.
This is possible because tools.py is decoupled from server.py.
"""

from __future__ import annotations

import time
import pytest
from pathlib import Path

from promptcache.store import CacheEntry, CacheStore
from promptcache.mcp.tools import (
    tool_clear_cache,
    tool_get_cached_entry,
    tool_get_stats,
    tool_list_recent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "mcp_test_cache"


@pytest.fixture
def populated_store(cache_dir):
    """A CacheStore pre-populated with a few entries."""
    store = CacheStore(cache_dir=cache_dir, collection_name="test")
    for i in range(5):
        entry = CacheEntry(
            prompt=f"test prompt {i}",
            model="gpt-4o",
            response=f"response {i}",
            created_at=time.time(),
        )
        store.write(entry)
        if i < 3:
            store.increment_hit(entry.prompt_hash, "exact")
    store.record_miss()
    store.record_miss()
    store.close()
    return cache_dir


# ---------------------------------------------------------------------------
# tool_get_stats
# ---------------------------------------------------------------------------


def test_get_stats_no_cache(tmp_path):
    result = tool_get_stats(tmp_path / "nonexistent")
    assert result["total_entries"] == 0
    assert "message" in result


def test_get_stats_returns_correct_counts(populated_store):
    result = tool_get_stats(populated_store, model="gpt-4o")
    assert result["total_entries"] == 5
    assert result["total_hits"] == 3
    assert "hit_rate" in result
    assert "hit_rate_pct" in result
    assert "estimated_cost_saved_usd" in result
    assert isinstance(result["top_entries"], list)


def test_get_stats_hit_rate_calculation(populated_store):
    result = tool_get_stats(populated_store)
    # 3 hits / (3 hits + 2 misses) = 0.6
    assert abs(result["hit_rate"] - 0.6) < 0.01


def test_get_stats_cost_estimation(populated_store):
    result = tool_get_stats(populated_store, model="gpt-4o", avg_tokens=100)
    # 3 hits * 100 tokens * 0.000005 cost/token = 0.0015
    assert result["estimated_cost_saved_usd"] > 0


# ---------------------------------------------------------------------------
# tool_list_recent
# ---------------------------------------------------------------------------


def test_list_recent_no_cache(tmp_path):
    result = tool_list_recent(tmp_path / "nonexistent")
    assert result["entries"] == []


def test_list_recent_returns_entries(populated_store):
    result = tool_list_recent(populated_store, limit=3)
    assert result["count"] == 3
    assert len(result["entries"]) == 3


def test_list_recent_entry_structure(populated_store):
    result = tool_list_recent(populated_store, limit=1)
    entry = result["entries"][0]
    assert "prompt" in entry
    assert "model" in entry
    assert "hit_count" in entry
    assert "created_at" in entry
    assert "prompt_hash" in entry


def test_list_recent_respects_limit(populated_store):
    result = tool_list_recent(populated_store, limit=2)
    assert result["count"] == 2


# ---------------------------------------------------------------------------
# tool_get_cached_entry
# ---------------------------------------------------------------------------


def test_get_cached_entry_not_found(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    store = CacheStore(cache_dir=cache_dir, collection_name="test")
    store.close()

    result = tool_get_cached_entry(
        cache_dir=cache_dir,
        prompt="this was never cached",
        model="gpt-4o",
    )
    assert result["found"] is False


def test_get_cached_entry_exact_hit(cache_dir):
    store = CacheStore(cache_dir=cache_dir, collection_name="test")
    entry = CacheEntry(
        prompt="What is 2+2?",
        model="gpt-4o",
        response="4",
        created_at=time.time(),
    )
    store.write(entry)
    store.close()

    result = tool_get_cached_entry(
        cache_dir=cache_dir,
        prompt="What is 2+2?",
        model="gpt-4o",
    )
    assert result["found"] is True
    assert result["hit_type"] == "exact"
    assert result["similarity"] == 1.0
    assert "4" in result["response_preview"]


def test_get_cached_entry_response_preview_truncated(cache_dir):
    long_response = "x" * 1000
    store = CacheStore(cache_dir=cache_dir, collection_name="test")
    entry = CacheEntry(
        prompt="long response prompt",
        model="gpt-4o",
        response=long_response,
        created_at=time.time(),
    )
    store.write(entry)
    store.close()

    result = tool_get_cached_entry(
        cache_dir=cache_dir,
        prompt="long response prompt",
        model="gpt-4o",
    )
    assert result["found"] is True
    assert len(result["response_preview"]) <= 310  # 300 + "..."
    assert result["response_length"] == 1000


# ---------------------------------------------------------------------------
# tool_clear_cache
# ---------------------------------------------------------------------------


def test_clear_cache_no_confirm(populated_store):
    result = tool_clear_cache(populated_store, model=None)
    # confirm is not checked in tool_clear_cache itself (that's server.py's job)
    # but we can test the deletion still works
    assert result["success"] is True


def test_clear_cache_all(populated_store):
    result = tool_clear_cache(populated_store)
    assert result["success"] is True
    assert result["deleted"] == 5

    # Verify store is empty
    store = CacheStore(cache_dir=populated_store, collection_name="verify")
    stats = store.stats()
    store.close()
    assert stats.total_entries == 0


def test_clear_cache_by_model(populated_store):
    # Add an entry for a different model
    store = CacheStore(cache_dir=populated_store, collection_name="test")
    entry = CacheEntry(
        prompt="other model prompt",
        model="claude-sonnet",
        response="other response",
        created_at=time.time(),
    )
    store.write(entry)
    store.close()

    result = tool_clear_cache(populated_store, model="gpt-4o")
    assert result["success"] is True
    assert result["deleted"] == 5
    assert result["model_filter"] == "gpt-4o"

    # Claude entry should remain
    store2 = CacheStore(cache_dir=populated_store, collection_name="verify")
    stats = store2.stats()
    store2.close()
    assert stats.total_entries == 1


def test_clear_nonexistent_cache(tmp_path):
    result = tool_clear_cache(tmp_path / "does_not_exist")
    assert result["success"] is True
    assert result["deleted"] == 0
