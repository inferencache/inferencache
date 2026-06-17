"""
tests/test_analytics.py

Tests for CacheAnalytics: all six DuckDB query methods over a seeded
calls event log.
"""

from __future__ import annotations

import time

import pytest

from promptcache.analytics import CacheAnalytics
from promptcache.store import CacheEntry, CacheStore

MODEL = "gpt-4o-mini"
PROVIDER = "openai"


@pytest.fixture
def seeded_store(tmp_path):
    """Populate index.db with entries and 35+ call events."""
    store = CacheStore(cache_dir=tmp_path / "analytics", collection_name="test-analytics")
    now = time.time()

    endpoints = ["dashboard/run-suite", "/api/chat", "/api/summarize"]
    hit_types = ["exact", "semantic", "miss"]

    for i in range(35):
        prompt = f"test prompt {i % 10}"
        phash = f"hash_{i:03d}"
        hit_type = hit_types[i % 3]
        similarity = 0.92 if hit_type == "semantic" else (1.0 if hit_type == "exact" else None)
        ts = now - (35 - i) * 120  # spread over ~70 minutes

        store.write_call_event(
            prompt_hash=phash,
            model=MODEL,
            provider=PROVIDER,
            hit_type=hit_type,
            latency_ms=float(5 + i),
            endpoint=endpoints[i % 3],
            session_id=f"sess-{i // 5}",
            similarity=similarity,
            tokens_input=50 if hit_type == "miss" else None,
            tokens_output=100 if hit_type == "miss" else None,
            cost_usd=0.0001 if hit_type == "miss" else None,
        )

        # Backdate timestamp (write_call_event uses time.time())
        store._conn.execute(
            "UPDATE calls SET timestamp = ? WHERE id = (SELECT MAX(id) FROM calls)",
            (ts,),
        )

    # Entry + flagged semantic call for false_positive_queue
    entry = CacheEntry(
        prompt="original cached prompt",
        model=MODEL,
        response='{"answer": "cached"}',
        created_at=now,
        prompt_hash="fp_hash",
    )
    store.write(entry)
    fp_id = store.write_call_event(
        prompt_hash="fp_hash",
        model=MODEL,
        provider=PROVIDER,
        hit_type="semantic",
        latency_ms=8.0,
        similarity=0.87,
    )
    store.flag_false_positive(fp_id, flagged=True)

    yield store
    store.close()


@pytest.fixture
def analytics(seeded_store, tmp_path):
    a = CacheAnalytics(tmp_path / "analytics")
    yield a
    a.close()


def test_hit_rate_over_time(analytics):
    rows = analytics.hit_rate_over_time(MODEL, window_hours=24, bucket_minutes=30)
    assert len(rows) >= 1
    row = rows[0]
    assert "time_bucket" in row
    assert "exact_hits" in row
    assert "semantic_hits" in row
    assert "misses" in row
    assert row["total_calls"] >= 1


def test_cost_saved_cumulative(analytics):
    rows = analytics.cost_saved_cumulative(MODEL, window_hours=24)
    assert len(rows) >= 1
    assert "timestamp" in rows[0]
    assert "cost_saved" in rows[0]
    assert "cumulative_saved" in rows[0]


def test_endpoint_breakdown(analytics):
    rows = analytics.endpoint_breakdown(MODEL, window_hours=24, limit=10)
    assert len(rows) >= 1
    row = rows[0]
    assert "endpoint" in row
    assert "total_calls" in row
    assert "hit_rate" in row
    assert "cost_saved_usd" in row


def test_similarity_distribution(analytics):
    rows = analytics.similarity_distribution(MODEL, window_hours=24, buckets=20)
    assert len(rows) >= 1
    assert "bucket_floor" in rows[0]
    assert "count" in rows[0]


def test_alert_check(analytics):
    state = analytics.alert_check(MODEL)
    assert "hit_rate" in state
    assert "quality" in state
    assert "cost_rate" in state
    assert state["hit_rate"]["status"] in ("ok", "warn", "alert")
    assert state["quality"]["status"] in ("ok", "warn", "alert")
    assert state["cost_rate"]["status"] in ("ok", "warn", "alert")


def test_alert_check_budget_param(analytics):
    """Lower budget should trigger cost_rate alert more easily."""
    strict = analytics.alert_check(MODEL, budget_usd_per_hour=0.00001)
    relaxed = analytics.alert_check(MODEL, budget_usd_per_hour=1000.0)
    assert strict["cost_rate"]["hourly_usd"] == relaxed["cost_rate"]["hourly_usd"]
    # With tiny budget, any spend should alert
    if strict["cost_rate"]["hourly_usd"] > 0.00001:
        assert strict["cost_rate"]["status"] == "alert"
    assert relaxed["cost_rate"]["status"] in ("ok", "warn")


def test_false_positive_queue(analytics):
    rows = analytics.false_positive_queue(MODEL, limit=10)
    assert len(rows) >= 1
    row = rows[0]
    assert row["id"] is not None
    assert row["original_prompt"] == "original cached prompt"
    assert row["cached_response"] == '{"answer": "cached"}'
    assert row["similarity"] == pytest.approx(0.87, abs=0.001)
