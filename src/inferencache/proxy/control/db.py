"""
SQLite persistence for inferencache dashboard run history.

DB location: ~/.cache/inferencache/runs.db
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..state import get_cache_dir

_RUNS_EXTRA_COLS = [
    ("batch_id", "TEXT    DEFAULT ''"),
    ("cache_mode", "TEXT    DEFAULT 'warm'"),
    ("delay_between_ms", "INTEGER DEFAULT 200"),
    ("status", "TEXT    DEFAULT 'complete'"),
    ("tokens_saved", "INTEGER DEFAULT 0"),
    ("cost_saved", "REAL    DEFAULT 0"),
    ("api_errors", "INTEGER DEFAULT 0"),
]
_CALLS_EXTRA_COLS = [
    ("best_similarity", "REAL DEFAULT 0"),
    ("prompt_hash", "TEXT DEFAULT ''"),
    ("lookup_ms", "REAL DEFAULT 0"),
    ("group_id", "TEXT DEFAULT ''"),
    ("endpoint", "TEXT DEFAULT ''"),
    ("session_id", "TEXT DEFAULT ''"),
    ("model", "TEXT DEFAULT ''"),
    ("call_id", "INTEGER"),
    ("tokens_input", "INTEGER"),
    ("tokens_output", "INTEGER"),
]


def _db_path() -> Path:
    return get_cache_dir() / "runs.db"


def _tuning_path() -> Path:
    return get_cache_dir() / "cache_tuning.json"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate_columns(conn: sqlite3.Connection, table: str, cols: list[tuple[str, str]]) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, typedef in cols:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typedef}")


def ensure_schema() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id            TEXT PRIMARY KEY,
                created_at    TEXT NOT NULL,
                suite_name    TEXT NOT NULL,
                model         TEXT NOT NULL,
                provider      TEXT NOT NULL,
                threshold     REAL NOT NULL,
                repeat_factor INTEGER NOT NULL,
                total_calls   INTEGER NOT NULL DEFAULT 0,
                cache_hits    INTEGER NOT NULL DEFAULT 0,
                exact_hits    INTEGER NOT NULL DEFAULT 0,
                semantic_hits INTEGER NOT NULL DEFAULT 0,
                total_tokens  INTEGER NOT NULL DEFAULT 0,
                total_cost_usd REAL NOT NULL DEFAULT 0,
                total_time_ms  REAL NOT NULL DEFAULT 0,
                hit_rate       REAL NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_calls (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                prompt_index     INTEGER NOT NULL,
                prompt_preview   TEXT,
                hit_type         TEXT    NOT NULL,
                latency_ms       REAL    NOT NULL DEFAULT 0,
                tokens_used      INTEGER NOT NULL DEFAULT 0,
                cost_usd         REAL    NOT NULL DEFAULT 0,
                similarity       REAL    NOT NULL DEFAULT 0,
                response_preview TEXT
            )
        """)
        _migrate_columns(conn, "runs", _RUNS_EXTRA_COLS)
        _migrate_columns(conn, "run_calls", _CALLS_EXTRA_COLS)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS run_errors (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         TEXT    NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                prompt_index   INTEGER,
                prompt_preview TEXT,
                model          TEXT,
                message        TEXT    NOT NULL
            )
        """)
        conn.execute("""
            UPDATE runs SET hit_rate = CAST(cache_hits AS REAL) / total_calls
            WHERE total_calls > 0 AND hit_rate = 0 AND cache_hits > 0
        """)


def save_run(
    run_id: str,
    config: dict[str, Any],
    summary: dict[str, Any],
    calls: list[dict[str, Any]],
    errors: list[dict[str, Any]] | None = None,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    errors = errors or []
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (id, created_at, suite_name, model, provider, threshold, repeat_factor,
                 total_calls, cache_hits, exact_hits, semantic_hits,
                 total_tokens, total_cost_usd, total_time_ms, hit_rate,
                 batch_id, cache_mode, delay_between_ms, status,
                 tokens_saved, cost_saved, api_errors)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                created_at,
                config.get("suite_name", ""),
                config.get("model", ""),
                config.get("provider", "openai"),
                config.get("threshold", 0.85),
                config.get("repeat_factor", 1),
                summary.get("total_calls", 0),
                summary.get("cache_hits", 0),
                summary.get("exact_hits", 0),
                summary.get("semantic_hits", 0),
                summary.get("total_tokens", 0),
                summary.get("total_cost_usd", 0.0),
                summary.get("total_time_ms", 0.0),
                summary.get("hit_rate", 0.0),
                config.get("batch_id", ""),
                config.get("cache_mode", "warm"),
                config.get("delay_between_ms", 200),
                config.get("status", "complete"),
                summary.get("tokens_saved", 0),
                summary.get("cost_saved", 0.0),
                summary.get("api_errors", 0),
            ),
        )
        conn.execute("DELETE FROM run_calls WHERE run_id = ?", (run_id,))
        conn.executemany(
            """
            INSERT INTO run_calls
                (run_id, prompt_index, prompt_preview, hit_type,
                 latency_ms, tokens_used, cost_usd, similarity, response_preview,
                 best_similarity, prompt_hash, lookup_ms, group_id,
                 endpoint, session_id, model, call_id, tokens_input, tokens_output)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    run_id,
                    c.get("prompt_index", i),
                    c.get("prompt_preview", ""),
                    c.get("hit_type", "miss"),
                    c.get("latency_ms", 0.0),
                    c.get("tokens_used", 0),
                    c.get("cost_usd", 0.0),
                    c.get("similarity", 0.0),
                    c.get("response_preview", ""),
                    c.get("best_similarity", 0.0),
                    c.get("prompt_hash", ""),
                    c.get("lookup_ms", 0.0),
                    c.get("group_id", ""),
                    c.get("endpoint", ""),
                    c.get("session_id", ""),
                    c.get("model", ""),
                    c.get("call_id"),
                    c.get("tokens_input"),
                    c.get("tokens_output"),
                )
                for i, c in enumerate(calls)
            ],
        )
        conn.execute("DELETE FROM run_errors WHERE run_id = ?", (run_id,))
        if errors:
            conn.executemany(
                """
                INSERT INTO run_errors
                    (run_id, prompt_index, prompt_preview, model, message)
                VALUES (?,?,?,?,?)
                """,
                [
                    (
                        run_id,
                        e.get("prompt_index"),
                        e.get("prompt_preview", ""),
                        e.get("model", ""),
                        e.get("message", "Unknown error"),
                    )
                    for e in errors
                ],
            )


def count_runs() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
        return row[0] if row else 0


def list_runs(limit: int = 50, offset: int = 0, batch_id: str | None = None) -> list[dict]:
    with _connect() as conn:
        if batch_id:
            rows = conn.execute(
                "SELECT * FROM runs WHERE batch_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (batch_id, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        calls = conn.execute(
            """
            SELECT prompt_index, prompt_preview, hit_type,
                   latency_ms, tokens_used, cost_usd, similarity, response_preview,
                   best_similarity, prompt_hash, lookup_ms, group_id,
                   endpoint, session_id, model, call_id, tokens_input, tokens_output
            FROM run_calls WHERE run_id = ? ORDER BY prompt_index
            """,
            (run_id,),
        ).fetchall()
        err_rows = conn.execute(
            """
            SELECT prompt_index, prompt_preview, model, message
            FROM run_errors WHERE run_id = ? ORDER BY prompt_index IS NULL, prompt_index, id
            """,
            (run_id,),
        ).fetchall()
        return {
            **dict(row),
            "calls": [dict(c) for c in calls],
            "errors": [dict(e) for e in err_rows],
        }


def find_batch_cell(
    batch_id: str,
    suite_name: str,
    model: str,
    threshold: float,
    cache_mode: str,
) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id FROM runs
            WHERE batch_id = ? AND suite_name = ? AND model = ?
              AND threshold = ? AND cache_mode = ? AND status = 'complete'
            LIMIT 1
            """,
            (batch_id, suite_name, model, threshold, cache_mode),
        ).fetchone()
        return row[0] if row else None


def delete_run(run_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        return cur.rowcount > 0


def get_all_runs_for_analysis(batch_id: str | None = None) -> list[dict]:
    with _connect() as conn:
        if batch_id:
            rows = conn.execute(
                "SELECT * FROM runs WHERE batch_id = ? ORDER BY created_at",
                (batch_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM runs WHERE status = 'complete' ORDER BY created_at"
            ).fetchall()
        result = []
        for row in rows:
            run = dict(row)
            calls = conn.execute(
                "SELECT * FROM run_calls WHERE run_id = ? ORDER BY prompt_index",
                (run["id"],),
            ).fetchall()
            run["calls"] = [dict(c) for c in calls]
            result.append(run)
        return result


def save_tuning(recommendations: list[dict]) -> None:
    path = _tuning_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "recommendations": recommendations,
            },
            indent=2,
        )
    )


def load_tuning() -> dict:
    path = _tuning_path()
    if not path.exists():
        return {"recommendations": []}
    return json.loads(path.read_text())
