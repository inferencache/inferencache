"""
analytics.py — DuckDB analytics layer for inferencache.

Reads the SQLite index.db (written by CacheStore) via DuckDB's sqlite
extension. DuckDB is attached READ-ONLY so it never modifies the live
cache database.

All public methods return plain Python dicts/lists suitable for JSON
serialisation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

__all__ = ["CacheAnalytics"]

# Approximate USD per output token for common models.
# Used by the model_cost_per_token DuckDB UDF.
_MODEL_COSTS: dict[str, float] = {
    # GPT-5.x
    "gpt-5.5": 0.00003, "gpt-5.4": 0.000015, "gpt-5.4-mini": 0.0000045,
    "gpt-5.3-chat-latest": 0.000015, "gpt-5.2": 0.000014,
    "gpt-5.1": 0.000012, "gpt-5.1-mini": 0.000003,
    "gpt-5": 0.00001, "gpt-5-mini": 0.000003,
    # GPT-4.1
    "gpt-4.1": 0.000008, "gpt-4.1-mini": 0.0000016, "gpt-4.1-nano": 0.0000004,
    # Reasoning
    "o4-mini": 0.0000044, "o3": 0.00004, "o3-mini": 0.0000044,
    "o1": 0.00006, "o1-mini": 0.000012,
    # GPT-4o / legacy
    "gpt-4o": 0.000005, "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001, "gpt-4": 0.00003, "gpt-3.5-turbo": 0.0000005,
    # Anthropic
    "claude-opus-4-8": 0.000025, "claude-sonnet-4-6": 0.000015,
    "claude-haiku-4-5-20251001": 0.000004,
    "claude-3-5-sonnet-20241022": 0.000003,
    "claude-3-haiku-20240307": 0.00000025,
}
_DEFAULT_COST = 0.000005  # fallback: ~$5/M tokens


class CacheAnalytics:
    """
    OLAP analytics over the inferencache calls event log.

    Attaches the SQLite index.db read-only so DuckDB never writes to it.

    Args:
        cache_dir: Same directory passed to CacheStore — contains index.db.
        extra_costs: Optional additional model→cost_per_token mapping that
                     extends (or overrides) the built-in table.
    """

    def __init__(
        self,
        cache_dir: Path,
        extra_costs: dict[str, float] | None = None,
    ) -> None:
        self._db_path = Path(cache_dir) / "index.db"
        self._costs = {**_MODEL_COSTS, **(extra_costs or {})}
        self._conn = self._init_duckdb()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_duckdb(self):
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for CacheAnalytics. "
                "Install it with: pip install duckdb"
            ) from exc

        conn = duckdb.connect(database=":memory:")

        # Load sqlite extension (built-in since duckdb 0.10)
        try:
            conn.execute("INSTALL sqlite; LOAD sqlite;")
        except Exception:
            try:
                conn.execute("LOAD sqlite;")
            except Exception:
                pass

        if self._db_path.exists():
            conn.execute(
                f"ATTACH '{self._db_path}' AS cache (TYPE SQLITE, READ_ONLY);"
            )

        # Register cost UDF
        costs = self._costs

        def model_cost_per_token(model: str) -> float:
            return costs.get(model, _DEFAULT_COST)

        conn.create_function("model_cost_per_token", model_cost_per_token)

        return conn

    def _ensure_attached(self) -> None:
        """Re-attach index.db if it has appeared since init."""
        if not self._db_path.exists():
            return
        try:
            # Will error if already attached; that's fine.
            self._conn.execute(
                f"ATTACH '{self._db_path}' AS cache (TYPE SQLITE, READ_ONLY);"
            )
        except Exception:
            pass

    def _q(self, sql: str, params: list[Any] | None = None):
        """Execute a query and return list-of-dict rows."""
        self._ensure_attached()
        if not self._db_path.exists():
            return []
        try:
            result = self._conn.execute(sql, params or [])
            cols = [d[0] for d in result.description]
            return [dict(zip(cols, row)) for row in result.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def hit_rate_over_time(
        self,
        model: str,
        window_hours: int = 24,
        bucket_minutes: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Time-bucketed hit rate for the Analytics tab stacked area chart.

        Returns rows: {time_bucket, exact_hits, semantic_hits, misses, total_calls}
        time_bucket is a Unix timestamp (seconds) aligned to bucket_minutes.
        """
        now = int(time.time())
        cutoff = now - window_hours * 3600
        bucket_secs = bucket_minutes * 60

        return self._q(
            f"""
            SELECT
                FLOOR(timestamp / {bucket_secs}) * {bucket_secs}  AS time_bucket,
                COUNT(*) FILTER (WHERE hit_type = 'exact')          AS exact_hits,
                COUNT(*) FILTER (WHERE hit_type = 'semantic')       AS semantic_hits,
                COUNT(*) FILTER (WHERE hit_type = 'miss')           AS misses,
                COUNT(*)                                             AS total_calls
            FROM cache.calls
            WHERE timestamp > ?
              AND model = ?
            GROUP BY time_bucket
            ORDER BY time_bucket
            """,
            [cutoff, model],
        )

    def cost_saved_cumulative(
        self,
        model: str,
        window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Cumulative cost saved over the window.

        Returns rows: {timestamp, cost_saved, cumulative_saved}
        Cost saved per hit = COALESCE(tokens_output, 200) * model_cost_per_token(model).
        """
        now = int(time.time())
        cutoff = now - window_hours * 3600

        return self._q(
            """
            SELECT
                timestamp,
                CASE WHEN hit_type != 'miss'
                    THEN COALESCE(tokens_output, 200) * model_cost_per_token(model)
                    ELSE 0.0
                END AS cost_saved,
                SUM(
                    CASE WHEN hit_type != 'miss'
                        THEN COALESCE(tokens_output, 200) * model_cost_per_token(model)
                        ELSE 0.0
                    END
                ) OVER (ORDER BY timestamp) AS cumulative_saved
            FROM cache.calls
            WHERE timestamp > ?
              AND model = ?
            ORDER BY timestamp
            """,
            [cutoff, model],
        )

    def endpoint_breakdown(
        self,
        model: str,
        window_hours: int = 24,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Aggregated stats per endpoint for the Analytics tab drilldown table.

        Returns rows: {endpoint, total_calls, cache_hits, hit_rate,
                       avg_latency_ms, total_cost_usd, cost_saved_usd}
        """
        now = int(time.time())
        cutoff = now - window_hours * 3600

        return self._q(
            f"""
            SELECT
                COALESCE(endpoint, 'unknown')                       AS endpoint,
                COUNT(*)                                            AS total_calls,
                COUNT(*) FILTER (WHERE hit_type != 'miss')         AS cache_hits,
                ROUND(
                    COUNT(*) FILTER (WHERE hit_type != 'miss') * 1.0 / COUNT(*), 4
                )                                                   AS hit_rate,
                ROUND(AVG(latency_ms), 1)                          AS avg_latency_ms,
                ROUND(SUM(COALESCE(cost_usd, 0.0)), 8)             AS total_cost_usd,
                ROUND(SUM(
                    CASE WHEN hit_type != 'miss'
                        THEN COALESCE(tokens_output, 200) * model_cost_per_token(model)
                        ELSE 0.0
                    END
                ), 8)                                               AS cost_saved_usd
            FROM cache.calls
            WHERE timestamp > ?
              AND model = ?
            GROUP BY endpoint
            ORDER BY total_calls DESC
            LIMIT {int(limit)}
            """,
            [cutoff, model],
        )

    def similarity_distribution(
        self,
        model: str,
        window_hours: int = 24,
        buckets: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Histogram of similarity scores for semantic hits.

        Powers the threshold tuner chart in the Tuning tab.
        Returns rows: {bucket_floor, count}
        bucket_floor is the lower bound of the similarity bucket (0.0–1.0).
        """
        now = int(time.time())
        cutoff = now - window_hours * 3600

        return self._q(
            f"""
            SELECT
                FLOOR(similarity * {int(buckets)}) / {int(buckets)} AS bucket_floor,
                COUNT(*)                                             AS count
            FROM cache.calls
            WHERE hit_type = 'semantic'
              AND model = ?
              AND timestamp > ?
              AND similarity IS NOT NULL
            GROUP BY bucket_floor
            ORDER BY bucket_floor
            """,
            [model, cutoff],
        )

    def alert_check(self, model: str, budget_usd_per_hour: float = 1.0) -> dict[str, Any]:
        """
        Compute current alert state for the three alert types.

        Args:
            model: LLM model to filter calls.
            budget_usd_per_hour: Hourly spend threshold for cost_rate alerts.

        Returns:
            {
              hit_rate:  { status: 'ok'|'warn'|'alert', value, drop },
              quality:   { status: 'ok'|'warn'|'alert', risky_ratio },
              cost_rate: { status: 'ok'|'warn'|'alert', hourly_usd },
            }
        """
        now = int(time.time())

        # ── Hit rate alert: compare last 15min to prior 15min ────────
        recent_rows = self._q(
            """
            SELECT
                COUNT(*) FILTER (WHERE hit_type != 'miss') * 1.0
                    / NULLIF(COUNT(*), 0) AS rate
            FROM cache.calls
            WHERE timestamp > ? AND model = ?
            """,
            [now - 900, model],
        )
        prior_rows = self._q(
            """
            SELECT
                COUNT(*) FILTER (WHERE hit_type != 'miss') * 1.0
                    / NULLIF(COUNT(*), 0) AS rate
            FROM cache.calls
            WHERE timestamp BETWEEN ? AND ? AND model = ?
            """,
            [now - 1800, now - 900, model],
        )
        recent_rate = (recent_rows[0]["rate"] or 0.0) if recent_rows else 0.0
        prior_rate  = (prior_rows[0]["rate"] or 0.0) if prior_rows else 0.0
        drop = prior_rate - recent_rate
        if drop > 0.20:
            hr_status = "alert"
        elif drop > 0.10:
            hr_status = "warn"
        else:
            hr_status = "ok"

        # ── Semantic quality alert: risky band (threshold, threshold+0.05) ──
        # We approximate: treat scores 0.80–0.85 as the risky band.
        quality_rows = self._q(
            """
            SELECT
                COUNT(*) FILTER (WHERE similarity < 0.85 AND similarity >= 0.80)
                    * 1.0 / NULLIF(COUNT(*) FILTER (WHERE hit_type = 'semantic'), 0)
                    AS risky_ratio
            FROM cache.calls
            WHERE timestamp > ? AND model = ?
            """,
            [now - 3600, model],
        )
        risky_ratio = (quality_rows[0]["risky_ratio"] or 0.0) if quality_rows else 0.0
        if risky_ratio > 0.30:
            q_status = "alert"
        elif risky_ratio > 0.15:
            q_status = "warn"
        else:
            q_status = "ok"

        # ── Cost rate: extrapolated hourly spend ─────────────────────
        cost_rows = self._q(
            """
            SELECT SUM(COALESCE(cost_usd, 0.0)) AS window_cost
            FROM cache.calls
            WHERE timestamp > ? AND model = ?
            """,
            [now - 3600, model],
        )
        hourly_usd = (cost_rows[0]["window_cost"] or 0.0) if cost_rows else 0.0
        warn_threshold = budget_usd_per_hour * 0.25
        if hourly_usd > budget_usd_per_hour:
            c_status = "alert"
        elif hourly_usd > warn_threshold:
            c_status = "warn"
        else:
            c_status = "ok"

        return {
            "hit_rate":  {"status": hr_status,  "value": round(recent_rate, 4), "drop": round(drop, 4)},
            "quality":   {"status": q_status,   "risky_ratio": round(risky_ratio, 4)},
            "cost_rate": {"status": c_status,   "hourly_usd": round(hourly_usd, 6)},
        }

    def false_positive_queue(
        self,
        model: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Return flagged semantic hits for the Tuning tab review queue.

        Returns rows: {id, prompt_hash, similarity, timestamp,
                       original_prompt, cached_response}
        """
        return self._q(
            f"""
            SELECT
                c.id,
                c.prompt_hash,
                COALESCE(c.similarity, 0) AS similarity,
                c.timestamp,
                e.prompt    AS original_prompt,
                e.response  AS cached_response
            FROM cache.calls c
            JOIN cache.entries e ON c.prompt_hash = e.prompt_hash
            WHERE c.false_positive = 1
              AND c.model = ?
            ORDER BY c.timestamp DESC
            LIMIT {int(limit)}
            """,
            [model],
        )

    def tier_breakdown(
        self,
        model: str,
        window_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """
        Per-tier savings breakdown for the Analytics tab.

        Returns three rows: tier1_semantic, tier2_prefix, tier3_inference.

        Tier 2/3 aggregations filter on tokens_input IS NOT NULL to exclude
        lookup-miss rows (dashboard double-logging).
        """
        now = int(time.time())
        cutoff = now - window_hours * 3600

        rows = self._q(
            """
            SELECT
                COALESCE(SUM(tier1_cached_input_tokens)
                    FILTER (WHERE hit_type IN ('exact', 'semantic')), 0)
                    AS tier1_tokens,
                COALESCE(SUM(
                    CASE WHEN hit_type IN ('exact', 'semantic')
                        THEN COALESCE(tier1_cached_input_tokens, 0)
                            * model_cost_per_token(model)
                        ELSE 0.0
                    END
                ), 0.0) AS tier1_cost,
                COUNT(*) FILTER (WHERE hit_type IN ('exact', 'semantic'))
                    AS tier1_hits,
                COALESCE(SUM(tier2_cached_input_tokens)
                    FILTER (WHERE tokens_input IS NOT NULL
                        AND tier2_cached_input_tokens > 0), 0)
                    AS tier2_tokens,
                COALESCE(SUM(tier2_cost_saved)
                    FILTER (WHERE tokens_input IS NOT NULL), 0.0)
                    AS tier2_cost,
                COUNT(*) FILTER (WHERE tokens_input IS NOT NULL
                    AND tier2_cached_input_tokens > 0)
                    AS tier2_hits,
                COALESCE(SUM(tier3_cost_saved)
                    FILTER (WHERE tokens_input IS NOT NULL), 0.0)
                    AS tier3_cost,
                COALESCE(SUM(tier3_hit)
                    FILTER (WHERE tokens_input IS NOT NULL), 0)
                    AS tier3_hits
            FROM cache.calls
            WHERE timestamp > ?
              AND model = ?
            """,
            [cutoff, model],
        )

        if not rows:
            return self._empty_tier_breakdown()

        row = rows[0]
        return [
            {
                "tier": "tier1_semantic",
                "label": "Tier 1 — Semantic cache",
                "tokens_saved": int(row.get("tier1_tokens") or 0),
                "cost_saved": round(float(row.get("tier1_cost") or 0.0), 8),
                "hit_count": int(row.get("tier1_hits") or 0),
            },
            {
                "tier": "tier2_prefix",
                "label": "Tier 2 — Prefix cache",
                "tokens_saved": int(row.get("tier2_tokens") or 0),
                "cost_saved": round(float(row.get("tier2_cost") or 0.0), 8),
                "hit_count": int(row.get("tier2_hits") or 0),
            },
            {
                "tier": "tier3_inference",
                "label": "Tier 3 — Inference cache",
                "tokens_saved": 0,
                "cost_saved": round(float(row.get("tier3_cost") or 0.0), 8),
                "hit_count": int(row.get("tier3_hits") or 0),
            },
        ]

    @staticmethod
    def _empty_tier_breakdown() -> list[dict[str, Any]]:
        return [
            {
                "tier": "tier1_semantic",
                "label": "Tier 1 — Semantic cache",
                "tokens_saved": 0,
                "cost_saved": 0.0,
                "hit_count": 0,
            },
            {
                "tier": "tier2_prefix",
                "label": "Tier 2 — Prefix cache",
                "tokens_saved": 0,
                "cost_saved": 0.0,
                "hit_count": 0,
            },
            {
                "tier": "tier3_inference",
                "label": "Tier 3 — Inference cache",
                "tokens_saved": 0,
                "cost_saved": 0.0,
                "hit_count": 0,
            },
        ]

    def close(self) -> None:
        """Release the DuckDB in-memory connection."""
        try:
            self._conn.close()
        except Exception:
            pass
