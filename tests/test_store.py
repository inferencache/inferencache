"""
tests/test_store.py

Tests for CacheStore: SQLite read/write, exact lookup by hash,
stats aggregation, hit counting, and cache clear.

ChromaDB tests are skipped if qdrant-client is not installed.
"""

from __future__ import annotations

import time
import pytest
from pathlib import Path

from promptcache.store import CacheEntry, CacheStore, _hash_key


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    s = CacheStore(
        cache_dir=tmp_path / "test_store",
        collection_name="test-collection",
    )
    yield s
    s.close()


def make_entry(prompt: str = "test prompt", model: str = "test-model") -> CacheEntry:
    return CacheEntry(
        prompt=prompt,
        model=model,
        response="test response",
        created_at=time.time(),
    )


# ---------------------------------------------------------------------------
# Write + exact lookup
# ---------------------------------------------------------------------------


def test_write_and_exact_lookup(store):
    entry = make_entry()
    store.write(entry)

    found = store.get_exact(entry.prompt, entry.model)
    assert found is not None
    assert found.prompt == entry.prompt
    assert found.response == entry.response


def test_exact_lookup_miss(store):
    result = store.get_exact("nonexistent prompt", "test-model")
    assert result is None


def test_exact_lookup_model_isolation(store):
    entry_a = make_entry(prompt="same prompt", model="model-a")
    entry_b = make_entry(prompt="same prompt", model="model-b")
    entry_b.response = "different response"

    store.write(entry_a)
    store.write(entry_b)

    found_a = store.get_exact("same prompt", "model-a")
    found_b = store.get_exact("same prompt", "model-b")

    assert found_a is not None
    assert found_b is not None
    assert found_a.response != found_b.response


def test_write_overwrites_existing(store):
    entry = make_entry()
    store.write(entry)

    updated = make_entry()
    updated.response = "updated response"
    store.write(updated)

    found = store.get_exact(entry.prompt, entry.model)
    assert found.response == "updated response"


# ---------------------------------------------------------------------------
# Hit counting
# ---------------------------------------------------------------------------


def test_increment_hit_exact(store):
    entry = make_entry()
    store.write(entry)
    store.increment_hit(entry.prompt_hash, hit_type="exact")
    store.increment_hit(entry.prompt_hash, hit_type="exact")

    found = store.get_exact(entry.prompt, entry.model)
    assert found.hit_count == 2


def test_increment_hit_semantic(store):
    entry = make_entry()
    store.write(entry)
    store.increment_hit(entry.prompt_hash, hit_type="semantic")

    found = store.get_exact(entry.prompt, entry.model)
    assert found.hit_count == 1


# ---------------------------------------------------------------------------
# Miss recording
# ---------------------------------------------------------------------------


def test_record_miss(store):
    store.record_miss()
    store.record_miss()
    row = store._conn.execute(
        "SELECT miss_count FROM stats WHERE id = 1"
    ).fetchone()
    assert row["miss_count"] == 2


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats_empty_store(store):
    stats = store.stats()
    assert stats.total_entries == 0
    assert stats.total_hits == 0
    assert stats.hit_rate == 0.0


def test_stats_with_entries(store):
    for i in range(5):
        entry = make_entry(prompt=f"prompt {i}")
        store.write(entry)
        store.increment_hit(entry.prompt_hash, hit_type="exact")

    store.record_miss()
    store.record_miss()

    stats = store.stats()
    assert stats.total_entries == 5
    assert stats.total_hits == 5
    # 5 hits / (5 hits + 2 misses) = 0.714...
    assert abs(stats.hit_rate - 5 / 7) < 0.001


def test_stats_top_entries(store):
    for i in range(3):
        entry = make_entry(prompt=f"prompt {i}")
        store.write(entry)
        for _ in range(i + 1):  # prompt 0 → 1 hit, prompt 1 → 2 hits, etc.
            store.increment_hit(entry.prompt_hash)

    stats = store.stats(top_n=3)
    assert len(stats.top_entries) == 3
    # Top entry should have the most hits
    assert stats.top_entries[0]["hit_count"] >= stats.top_entries[1]["hit_count"]


# ---------------------------------------------------------------------------
# list_recent
# ---------------------------------------------------------------------------


def test_list_recent(store):
    for i in range(5):
        entry = make_entry(prompt=f"prompt {i}")
        store.write(entry)

    recent = store.list_recent(limit=3)
    assert len(recent) == 3


def test_list_recent_empty(store):
    assert store.list_recent() == []


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_all(store):
    for i in range(5):
        store.write(make_entry(prompt=f"prompt {i}"))

    deleted = store.clear()
    assert deleted == 5

    stats = store.stats()
    assert stats.total_entries == 0


def test_clear_by_model(store):
    store.write(make_entry(prompt="p1", model="model-a"))
    store.write(make_entry(prompt="p2", model="model-a"))
    store.write(make_entry(prompt="p3", model="model-b"))

    deleted = store.clear(model="model-a")
    assert deleted == 2

    stats = store.stats()
    assert stats.total_entries == 1


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_round_trip(store):
    entry = make_entry()
    entry.metadata = {"source": "test", "version": 1}
    store.write(entry)

    found = store.get_exact(entry.prompt, entry.model)
    assert found.metadata == {"source": "test", "version": 1}


# ---------------------------------------------------------------------------
# Schema migration (calls table + entries columns)
# ---------------------------------------------------------------------------


def test_sqlite_migration_adds_entries_columns(store):
    """endpoint and session_id must exist on entries after init."""
    cols = {
        row[1] for row in store._conn.execute("PRAGMA table_info(entries)").fetchall()
    }
    assert "endpoint" in cols
    assert "session_id" in cols


def test_sqlite_migration_creates_calls_table(store):
    """calls table must exist with all required columns and indexes."""
    cols = {
        row[1] for row in store._conn.execute("PRAGMA table_info(calls)").fetchall()
    }
    required = {
        "id", "prompt_hash", "model", "provider", "endpoint", "session_id",
        "session_hash",
        "hit_type", "similarity", "latency_ms", "tokens_input", "tokens_output",
        "cost_usd",
        "tier1_cached_input_tokens", "tier2_cached_input_tokens",
        "tier3_hit", "tier2_cost_saved", "tier3_cost_saved",
        "false_positive", "timestamp",
    }
    assert required.issubset(cols)


def test_calls_migration_adds_tier_columns(tmp_path):
    """Pre-migration calls table gains tier columns via ALTER."""
    import sqlite3

    cache_dir = tmp_path / "legacy_calls"
    cache_dir.mkdir()
    db_path = cache_dir / "index.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_hash TEXT NOT NULL,
            model TEXT NOT NULL,
            provider TEXT NOT NULL,
            endpoint TEXT,
            session_id TEXT,
            hit_type TEXT NOT NULL,
            similarity REAL,
            latency_ms REAL NOT NULL,
            tokens_input INTEGER,
            tokens_output INTEGER,
            cost_usd REAL,
            false_positive INTEGER NOT NULL DEFAULT 0,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    store = CacheStore(cache_dir=cache_dir, collection_name="legacy_col")
    cols = {
        row[1] for row in store._conn.execute("PRAGMA table_info(calls)").fetchall()
    }
    store.close()

    assert "session_hash" in cols
    assert "tier1_cached_input_tokens" in cols
    assert "tier2_cached_input_tokens" in cols
    assert "tier3_hit" in cols
    assert "tier2_cost_saved" in cols
    assert "tier3_cost_saved" in cols


def test_get_exact_filters_by_session_hash(store):
    """get_exact with session_hash only returns session-scoped entries."""
    import time as _time

    entry_a = CacheEntry(
        prompt="scoped prompt",
        model="test-model",
        response="response A",
        created_at=_time.time(),
        metadata={"session_hash": "session_a"},
    )
    store.write(entry_a)

    assert store.get_exact("scoped prompt", "test-model", session_hash="session_a") is not None
    assert store.get_exact("scoped prompt", "test-model", session_hash="session_b") is None
    assert store.get_exact("scoped prompt", "test-model") is not None


def test_write_call_event_persists_tier_fields(store):
    """Per-tier savings columns are persisted correctly."""
    row_id = store.write_call_event(
        prompt_hash="tier_test",
        model="gpt-4o",
        provider="openai",
        hit_type="miss",
        latency_ms=50.0,
        session_hash="abc123def456",
        tier1_cached_input_tokens=0,
        tier2_cached_input_tokens=512,
        tier3_hit=1,
        tier2_cost_saved=0.001,
        tier3_cost_saved=0.002,
        tokens_input=1000,
        tokens_output=200,
    )
    row = store._conn.execute(
        "SELECT * FROM calls WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["session_hash"] == "abc123def456"
    assert row["tier1_cached_input_tokens"] == 0
    assert row["tier2_cached_input_tokens"] == 512
    assert row["tier3_hit"] == 1
    assert row["tier2_cost_saved"] == pytest.approx(0.001)
    assert row["tier3_cost_saved"] == pytest.approx(0.002)


def test_sqlite_migration_creates_calls_indexes(store):
    """All four calls indexes must be present."""
    indexes = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='calls'"
        ).fetchall()
    }
    assert "idx_calls_timestamp" in indexes
    assert "idx_calls_endpoint" in indexes
    assert "idx_calls_session" in indexes
    assert "idx_calls_hit_type" in indexes


def test_write_call_event_returns_int_lastrowid(store):
    """write_call_event() must return the row ID as a plain int."""
    row_id = store.write_call_event(
        prompt_hash="aabbcc",
        model="test-model",
        provider="openai",
        hit_type="exact",
        latency_ms=5.0,
    )
    assert isinstance(row_id, int)
    assert row_id >= 1


def test_write_call_event_multiple_rows(store):
    """Each call produces a distinct row; IDs are strictly increasing."""
    id1 = store.write_call_event(
        prompt_hash="hash1", model="m", provider="openai",
        hit_type="exact", latency_ms=1.0,
    )
    id2 = store.write_call_event(
        prompt_hash="hash2", model="m", provider="openai",
        hit_type="miss", latency_ms=200.0,
    )
    assert id2 > id1


def test_write_call_event_stores_all_fields(store):
    """All optional fields are persisted correctly."""
    import time as _time
    before = _time.time()
    row_id = store.write_call_event(
        prompt_hash="abc123",
        model="gpt-4o",
        provider="openai",
        hit_type="semantic",
        latency_ms=12.5,
        endpoint="/api/chat",
        session_id="sess-1",
        similarity=0.91,
        tokens_input=50,
        tokens_output=100,
        cost_usd=0.0005,
    )
    row = store._conn.execute(
        "SELECT * FROM calls WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["hit_type"] == "semantic"
    assert row["endpoint"] == "/api/chat"
    assert row["session_id"] == "sess-1"
    assert abs(row["similarity"] - 0.91) < 0.001
    assert row["tokens_input"] == 50
    assert row["tokens_output"] == 100
    assert row["cost_usd"] == pytest.approx(0.0005)
    assert row["false_positive"] == 0
    assert row["timestamp"] >= before


def test_flag_false_positive_sets_and_clears(store):
    """flag_false_positive sets and clears the flag on the row."""
    row_id = store.write_call_event(
        prompt_hash="fp_test", model="m", provider="openai",
        hit_type="semantic", latency_ms=5.0,
    )
    store.flag_false_positive(row_id, flagged=True)
    row = store._conn.execute(
        "SELECT false_positive FROM calls WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["false_positive"] == 1

    store.flag_false_positive(row_id, flagged=False)
    row = store._conn.execute(
        "SELECT false_positive FROM calls WHERE id = ?", (row_id,)
    ).fetchone()
    assert row["false_positive"] == 0


def test_migration_is_idempotent(tmp_path):
    """Opening the same cache_dir twice does not raise OperationalError."""
    s1 = CacheStore(cache_dir=tmp_path / "idempotent", collection_name="col")
    s1.close()
    s2 = CacheStore(cache_dir=tmp_path / "idempotent", collection_name="col")
    s2.close()


# ---------------------------------------------------------------------------
# Hash key
# ---------------------------------------------------------------------------


def test_hash_key_is_deterministic():
    h1 = _hash_key("hello world", "gpt-4o")
    h2 = _hash_key("hello world", "gpt-4o")
    assert h1 == h2


def test_hash_key_differs_by_model():
    h1 = _hash_key("hello", "model-a")
    h2 = _hash_key("hello", "model-b")
    assert h1 != h2


def test_hash_key_differs_by_prompt():
    h1 = _hash_key("prompt A", "model")
    h2 = _hash_key("prompt B", "model")
    assert h1 != h2
