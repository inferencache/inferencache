"""
store.py — Two-tier persistence layer.

Tier 1 — SQLite (exact match index + event log)
    Keyed by SHA-256(prompt + model). Sub-millisecond lookup.
    Also holds the `calls` table: one row per LLM call for analytics.

Tier 2 — Qdrant (vector store, embedded mode)
    Stores embeddings for semantic search.
    Queried only when the exact-match check in Tier 1 misses.

Both tiers write to the same cache_dir on disk. SQLite uses
`index.db`; Qdrant uses a `qdrant/` subdirectory.

Thread safety: SQLite writes use WAL mode; Qdrant embedded client
handles its own concurrency. Safe for multi-threaded use within a
single process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["CacheStore", "CacheEntry", "StoreStats"]

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-level Qdrant singleton registry
#
# Qdrant embedded mode holds an exclusive file lock on its storage directory.
# Two problems arise without this registry:
#   1. Multiple CacheStore instances in the same process (one per model/provider
#      combo) each try to open a QdrantClient on the same path → lock conflict.
#   2. Hot-reload: the new worker process can start before the old one fully
#      releases the lock → spurious startup failure.
#
# Solution: one shared QdrantClient per (path) per process, protected by a
# threading.Lock, plus a retry loop to survive the brief reload race window.
# ---------------------------------------------------------------------------
_qdrant_clients: dict[str, Any] = {}
_qdrant_init_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached prompt/response pair."""

    prompt: str
    model: str
    response: str
    """The full response as a JSON-serialisable string."""

    created_at: float
    """Unix timestamp when the entry was first stored."""

    hit_count: int = 0
    """How many times this entry has been returned as a cache hit."""

    prompt_hash: str = ""
    """SHA-256(prompt + model). Computed automatically if empty."""

    embedding_id: str = ""
    """Stable ID used by the vector store. Derived from prompt_hash."""

    metadata: dict[str, Any] | None = None
    """Arbitrary metadata attached at write time."""

    ttl_class: str = "permanent"
    """TTLClass value stored at write time."""

    expires_at: float | None = None
    """Unix timestamp when entry expires; None = no time-based expiry."""

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            self.prompt_hash = _hash_key(self.prompt, self.model)
        if not self.embedding_id:
            self.embedding_id = self.prompt_hash


@dataclass
class StoreStats:
    """Aggregate statistics read by the CLI and MCP server."""

    total_entries: int
    exact_hits: int
    semantic_hits: int
    total_hits: int
    hit_rate: float
    """total_hits / (total_hits + misses) — 0.0 to 1.0"""
    top_entries: list[dict[str, Any]]
    """Top N entries by hit_count, each as a plain dict."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_key(prompt: str, model: str) -> str:
    """Return a hex SHA-256 digest for the (prompt, model) pair."""
    raw = f"{model}:{prompt}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _metadata_session_hash(metadata: dict[str, Any] | None) -> str | None:
    """Extract session_hash from entry metadata, if present."""
    if not metadata:
        return None
    value = metadata.get("session_hash")
    return str(value) if value is not None else None


def _qdrant_point_id(prompt_hash: str) -> int:
    """Derive a stable int64 Qdrant point ID from a SHA-256 hex digest."""
    return int(prompt_hash[:16], 16) % (2**63)


# ---------------------------------------------------------------------------
# CacheStore
# ---------------------------------------------------------------------------


class CacheStore:
    """
    Manages all reads and writes for both storage tiers.

    Args:
        cache_dir: Directory where index.db and the Qdrant data are
                   stored. Created if it does not exist.
        collection_name: Qdrant collection name. Must include the
                         embedder model_id to prevent dimension mismatches
                         between embedder presets.
        embedding_dim: Expected vector dimension. Validated against any
                       existing Qdrant collection on first access — raises
                       ValueError if there is a mismatch.
    """

    def __init__(
        self,
        cache_dir: Path,
        collection_name: str,
        embedding_dim: int = 384,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._collection_name = collection_name
        self._embedding_dim = embedding_dim

        # Initialise SQLite (Tier 1)
        self._db_path = self._cache_dir / "index.db"
        self._conn = self._init_sqlite()

        # Qdrant client is lazy — created on first vector operation
        self._qdrant_client = None

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def write(
        self,
        entry: CacheEntry,
        embedding: list[float] | None = None,
    ) -> None:
        """
        Persist a new entry to both tiers.

        Args:
            entry: The CacheEntry to store.
            embedding: Pre-computed embedding vector. If provided, the
                       entry is also written to Qdrant for semantic
                       search. If None, only the SQLite tier is updated.
        """
        self._sqlite_write(entry)
        if embedding is not None:
            self._qdrant_write(entry, embedding)

    def increment_hit(self, prompt_hash: str, hit_type: str = "exact") -> None:
        """Increment the hit_count for an entry and record the hit type."""
        with self._conn:
            self._conn.execute(
                """
                UPDATE entries
                SET hit_count = hit_count + 1,
                    last_hit_at = ?,
                    last_hit_type = ?
                WHERE prompt_hash = ?
                """,
                (time.time(), hit_type, prompt_hash),
            )

    def write_call_event(
        self,
        prompt_hash: str,
        model: str,
        provider: str,
        hit_type: str,
        latency_ms: float,
        endpoint: str | None = None,
        session_id: str | None = None,
        similarity: float | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
        session_hash: str | None = None,
        tier1_cached_input_tokens: int | None = None,
        tier2_cached_input_tokens: int | None = None,
        tier3_hit: int | None = None,
        tier2_cost_saved: float | None = None,
        tier3_cost_saved: float | None = None,
        adaptation_model: str | None = None,
        adaptation_tokens_in: int | None = None,
        adaptation_tokens_out: int | None = None,
        adaptation_cost_usd: float | None = None,
    ) -> int:
        """
        Insert one row into the calls event log.

        Called after every lookup (hit or miss) and after every store().
        Returns the auto-increment row ID (lastrowid) as int.
        """
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT INTO calls (
                    prompt_hash, model, provider, endpoint, session_id,
                    session_hash,
                    hit_type, similarity, latency_ms,
                    tokens_input, tokens_output, cost_usd,
                    tier1_cached_input_tokens, tier2_cached_input_tokens,
                    tier3_hit, tier2_cost_saved, tier3_cost_saved,
                    adaptation_model, adaptation_tokens_in,
                    adaptation_tokens_out, adaptation_cost_usd,
                    timestamp
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    prompt_hash, model, provider, endpoint, session_id,
                    session_hash,
                    hit_type, similarity, latency_ms,
                    tokens_input, tokens_output, cost_usd,
                    tier1_cached_input_tokens or 0,
                    tier2_cached_input_tokens or 0,
                    tier3_hit or 0,
                    tier2_cost_saved or 0.0,
                    tier3_cost_saved or 0.0,
                    adaptation_model,
                    adaptation_tokens_in,
                    adaptation_tokens_out,
                    adaptation_cost_usd,
                    time.time(),
                ),
            )
        return int(cur.lastrowid)

    def flag_false_positive(self, call_id: int, flagged: bool = True) -> None:
        """
        Mark or clear the false_positive flag on a calls row.

        Only rows where hit_type = 'semantic' should be flagged; the
        caller is responsible for enforcing this constraint.
        """
        with self._conn:
            self._conn.execute(
                "UPDATE calls SET false_positive = ? WHERE id = ?",
                (1 if flagged else 0, call_id),
            )

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_exact(
        self,
        prompt: str,
        model: str,
        session_hash: str | None = None,
    ) -> CacheEntry | None:
        """
        Look up an entry by exact (prompt, model) match.

        When session_hash is provided, only returns the entry if its
        metadata.session_hash matches (session-scoped exact match).

        Returns None if no match exists.
        """
        key = _hash_key(prompt, model)
        row = self._conn.execute(
            "SELECT * FROM entries WHERE prompt_hash = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        if session_hash is not None:
            stored = _metadata_session_hash(entry.metadata)
            if stored != session_hash:
                return None
        return entry

    @staticmethod
    def _matches_session_hash(entry: CacheEntry, session_hash: str | None) -> bool:
        if session_hash is None:
            return True
        return _metadata_session_hash(entry.metadata) == session_hash

    def query_semantic(
        self,
        embedding: list[float],
        model: str,
        threshold: float,
        top_k: int = 5,
        session_hash: str | None = None,
    ) -> list[tuple[CacheEntry, float]]:
        """
        Query Qdrant for semantically similar entries.

        Qdrant uses native cosine similarity on normalised vectors —
        scores are 0.0–1.0 directly. No L2→cosine conversion needed.

        Args:
            embedding: Query vector.
            model: Only return entries cached for this model.
            threshold: Minimum cosine similarity to include.
            top_k: Maximum candidates to retrieve from Qdrant.

        Returns:
            List of (CacheEntry, similarity_score) tuples, sorted by
            score descending. Empty list if Qdrant has no entries or
            no matches clear the threshold.
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
        except ImportError:
            return []

        client = self._get_qdrant_client()
        if client is None:
            return []

        try:
            response = client.query_points(
                collection_name=self._collection_name,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="model", match=MatchValue(value=model))]
                ),
                limit=top_k,
                score_threshold=threshold,
                with_payload=True,
            )
            results = response.points
        except Exception:
            return []

        hits: list[tuple[CacheEntry, float]] = []
        for r in results:
            entry = self.get_exact_by_hash(r.payload["prompt_hash"])
            if entry and self._matches_session_hash(entry, session_hash):
                hits.append((entry, round(r.score, 4)))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def get_exact_by_hash(self, prompt_hash: str) -> CacheEntry | None:
        """Look up an entry directly by its pre-computed hash."""
        row = self._conn.execute(
            "SELECT * FROM entries WHERE prompt_hash = ?", (prompt_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def list_recent(self, limit: int = 20) -> list[CacheEntry]:
        """Return the most recently created entries."""
        rows = self._conn.execute(
            "SELECT * FROM entries ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def clear(self, model: str | None = None) -> int:
        """
        Delete cache entries (and matching Qdrant points).

        Args:
            model: If given, only delete entries for this model.
                   If None, clear everything.

        Returns:
            Number of entries deleted.
        """
        if model:
            result = self._conn.execute(
                "DELETE FROM entries WHERE model = ?", (model,)
            )
            self._conn.commit()
            client = self._get_qdrant_client()
            if client is not None:
                try:
                    from qdrant_client.models import (
                        Filter, FieldCondition, MatchValue, FilterSelector,
                    )
                    client.delete(
                        collection_name=self._collection_name,
                        points_selector=FilterSelector(
                            filter=Filter(
                                must=[FieldCondition(key="model", match=MatchValue(value=model))]
                            )
                        ),
                    )
                except Exception:
                    pass
        else:
            result = self._conn.execute("DELETE FROM entries")
            self._conn.commit()
            self._reset_stats()
            # Drop and recreate the Qdrant collection for a clean slate
            client = self._get_qdrant_client()
            if client is not None:
                try:
                    from qdrant_client.models import Distance, VectorParams
                    client.delete_collection(self._collection_name)
                    client.create_collection(
                        collection_name=self._collection_name,
                        vectors_config=VectorParams(
                            size=self._embedding_dim,
                            distance=Distance.COSINE,
                        ),
                    )
                except Exception:
                    qdrant_path = str(self._cache_dir / "qdrant")
                    _qdrant_clients.pop(qdrant_path, None)
                    self._qdrant_client = None  # force re-init next access

        return result.rowcount

    def prune_expired(self) -> int:
        """
        Delete entries past their expires_at. Returns count deleted.
        Also removes corresponding Qdrant vectors.
        """
        now = time.time()
        expired_rows = self._conn.execute(
            "SELECT prompt_hash FROM entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            (now,),
        ).fetchall()

        if not expired_rows:
            return 0

        hashes = [row[0] for row in expired_rows]
        placeholders = ",".join("?" * len(hashes))
        with self._conn:
            self._conn.execute(
                f"DELETE FROM entries WHERE prompt_hash IN ({placeholders})",
                hashes,
            )

        client = self._get_qdrant_client()
        if client is not None:
            try:
                point_ids = [_qdrant_point_id(h) for h in hashes]
                for col in client.get_collections().collections:
                    try:
                        client.delete(
                            collection_name=col.name,
                            points_selector=point_ids,
                        )
                    except Exception:
                        pass
            except Exception:
                pass

        return len(hashes)

    def stats(self, top_n: int = 10) -> StoreStats:
        """Return aggregate statistics across the entire store."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*) as total_entries,
                COALESCE(SUM(hit_count), 0) as total_hits,
                COALESCE(SUM(CASE WHEN last_hit_type = 'exact' THEN hit_count ELSE 0 END), 0) as exact_hits,
                COALESCE(SUM(CASE WHEN last_hit_type = 'semantic' THEN hit_count ELSE 0 END), 0) as semantic_hits
            FROM entries
            """
        ).fetchone()

        total_entries = row["total_entries"]
        total_hits = row["total_hits"]
        exact_hits = row["exact_hits"]
        semantic_hits = row["semantic_hits"]

        misses_row = self._conn.execute(
            "SELECT COALESCE(SUM(miss_count), 0) as total_misses FROM stats"
        ).fetchone()
        total_misses = misses_row["total_misses"] if misses_row else 0

        denominator = total_hits + total_misses
        hit_rate = total_hits / denominator if denominator > 0 else 0.0

        top_rows = self._conn.execute(
            """
            SELECT prompt, model, hit_count, created_at
            FROM entries
            ORDER BY hit_count DESC
            LIMIT ?
            """,
            (top_n,),
        ).fetchall()

        top_entries = [
            {
                "prompt": r["prompt"][:120],
                "model": r["model"],
                "hit_count": r["hit_count"],
                "created_at": r["created_at"],
            }
            for r in top_rows
        ]

        return StoreStats(
            total_entries=total_entries,
            exact_hits=exact_hits,
            semantic_hits=semantic_hits,
            total_hits=total_hits,
            hit_rate=round(hit_rate, 4),
            top_entries=top_entries,
        )

    def list_entries(self, limit: int = 500) -> list[dict]:
        """Return cache entries for the map view, ordered by hit count."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT prompt_hash, prompt, model, created_at,
                   hit_count, last_hit_at
            FROM entries
            ORDER BY hit_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def record_miss(self) -> None:
        """Record that a lookup resulted in a real API call."""
        with self._conn:
            self._conn.execute(
                "UPDATE stats SET miss_count = miss_count + 1 WHERE id = 1"
            )

    # ------------------------------------------------------------------
    # SQLite internals
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entries (
                prompt_hash     TEXT PRIMARY KEY,
                prompt          TEXT NOT NULL,
                model           TEXT NOT NULL,
                response        TEXT NOT NULL,
                created_at      REAL NOT NULL,
                hit_count       INTEGER NOT NULL DEFAULT 0,
                last_hit_at     REAL,
                last_hit_type   TEXT,
                metadata        TEXT,
                endpoint        TEXT,
                session_id      TEXT,
                ttl_class       TEXT NOT NULL DEFAULT 'permanent',
                expires_at      REAL
            );

            CREATE INDEX IF NOT EXISTS idx_entries_model
                ON entries (model);

            CREATE INDEX IF NOT EXISTS idx_entries_hit_count
                ON entries (hit_count DESC);

            CREATE TABLE IF NOT EXISTS stats (
                id          INTEGER PRIMARY KEY CHECK (id = 1),
                miss_count  INTEGER NOT NULL DEFAULT 0
            );

            INSERT OR IGNORE INTO stats (id, miss_count) VALUES (1, 0);

            CREATE TABLE IF NOT EXISTS calls (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_hash                 TEXT NOT NULL,
                model                       TEXT NOT NULL,
                provider                    TEXT NOT NULL,
                endpoint                    TEXT,
                session_id                  TEXT,
                session_hash                TEXT,
                hit_type                    TEXT NOT NULL,
                similarity                  REAL,
                latency_ms                  REAL NOT NULL,
                tokens_input                INTEGER,
                tokens_output               INTEGER,
                cost_usd                    REAL,
                tier1_cached_input_tokens   INTEGER DEFAULT 0,
                tier2_cached_input_tokens   INTEGER DEFAULT 0,
                tier3_hit                   INTEGER NOT NULL DEFAULT 0,
                tier2_cost_saved            REAL DEFAULT 0,
                tier3_cost_saved            REAL DEFAULT 0,
                false_positive              INTEGER NOT NULL DEFAULT 0,
                timestamp                   REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_calls_timestamp
                ON calls (timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_calls_endpoint
                ON calls (endpoint, timestamp DESC);

            CREATE INDEX IF NOT EXISTS idx_calls_session
                ON calls (session_id);

            CREATE INDEX IF NOT EXISTS idx_calls_hit_type
                ON calls (hit_type);
            """
        )
        self._migrate_sqlite_schema(conn)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_entries_expires_at
                ON entries (expires_at)
                WHERE expires_at IS NOT NULL
            """
        )
        conn.commit()
        return conn

    _SCHEMA_MIGRATIONS: dict[str, list[tuple[str, str]]] = {
        "entries": [
            ("endpoint", "TEXT"),
            ("session_id", "TEXT"),
            ("ttl_class", "TEXT NOT NULL DEFAULT 'permanent'"),
            ("expires_at", "REAL"),
        ],
        "calls": [
            ("session_hash", "TEXT"),
            ("tier1_cached_input_tokens", "INTEGER DEFAULT 0"),
            ("tier2_cached_input_tokens", "INTEGER DEFAULT 0"),
            ("tier3_hit", "INTEGER NOT NULL DEFAULT 0"),
            ("tier2_cost_saved", "REAL DEFAULT 0"),
            ("tier3_cost_saved", "REAL DEFAULT 0"),
            ("adaptation_model", "TEXT"),
            ("adaptation_tokens_in", "INTEGER"),
            ("adaptation_tokens_out", "INTEGER"),
            ("adaptation_cost_usd", "REAL"),
        ],
    }

    @staticmethod
    def _migrate_sqlite_schema(conn: sqlite3.Connection) -> None:
        """
        Apply incremental schema changes for existing index.db files.

        SQLite does not support IF NOT EXISTS on ALTER TABLE, so we
        guard each ALTER with a PRAGMA table_info check and catch the
        OperationalError that fires if the column already exists.
        """
        for table, columns in CacheStore._SCHEMA_MIGRATIONS.items():
            existing_cols = {
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col, typedef in columns:
                if col not in existing_cols:
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
                    except sqlite3.OperationalError:
                        pass

    def _sqlite_write(self, entry: CacheEntry) -> None:
        metadata_json = json.dumps(entry.metadata) if entry.metadata else None
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO entries
                    (prompt_hash, prompt, model, response, created_at, hit_count,
                     metadata, ttl_class, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.prompt_hash,
                    entry.prompt,
                    entry.model,
                    entry.response,
                    entry.created_at,
                    entry.hit_count,
                    metadata_json,
                    entry.ttl_class,
                    entry.expires_at,
                ),
            )

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        metadata = None
        if row["metadata"]:
            try:
                metadata = json.loads(row["metadata"])
            except (json.JSONDecodeError, TypeError):
                metadata = None

        keys = row.keys()
        ttl_class = row["ttl_class"] if "ttl_class" in keys else "permanent"
        expires_at = row["expires_at"] if "expires_at" in keys else None

        return CacheEntry(
            prompt=row["prompt"],
            model=row["model"],
            response=row["response"],
            created_at=row["created_at"],
            hit_count=row["hit_count"],
            prompt_hash=row["prompt_hash"],
            embedding_id=row["prompt_hash"],
            metadata=metadata,
            ttl_class=ttl_class,
            expires_at=expires_at,
        )

    def _reset_stats(self) -> None:
        with self._conn:
            self._conn.execute("UPDATE stats SET miss_count = 0 WHERE id = 1")

    # ------------------------------------------------------------------
    # Qdrant internals
    # ------------------------------------------------------------------

    def _get_qdrant_client(self):
        """Return the Qdrant client, initialising it if needed.

        Uses a process-level singleton so that all CacheStore instances
        sharing the same cache_dir reuse a single QdrantClient (Qdrant
        embedded mode allows only one client per path per process).

        Retries up to 5 times with exponential back-off to survive the
        brief window during a hot-reload where the previous worker process
        has not yet released its file lock.

        Raises ValueError if an existing collection's vector dimension does
        not match self._embedding_dim.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            return None

        qdrant_path = str(self._cache_dir / "qdrant")

        with _qdrant_init_lock:
            # Another CacheStore instance in this process may have already
            # opened this path — reuse its client.
            if qdrant_path in _qdrant_clients:
                client = _qdrant_clients[qdrant_path]
                self._qdrant_client = client
                self._ensure_qdrant_collection(client)
                return client

            # Retry to survive hot-reload lock races (old process dying).
            last_err: Exception | None = None
            for attempt in range(5):
                try:
                    client = QdrantClient(path=qdrant_path)
                    last_err = None
                    break
                except RuntimeError as exc:
                    if "already accessed" not in str(exc):
                        raise
                    last_err = exc
                    if attempt < 4:
                        delay = 0.4 * (2 ** attempt)  # 0.4 → 0.8 → 1.6 → 3.2 s
                        _log.warning(
                            "Qdrant lock held by another process (attempt %d/5); "
                            "retrying in %.1fs…", attempt + 1, delay,
                        )
                        time.sleep(delay)

            if last_err is not None:
                _log.warning(
                    "Qdrant: could not acquire lock after 5 attempts — "
                    "semantic search disabled for this process. (%s)", last_err,
                )
                return None

            self._ensure_qdrant_collection(client)
            _qdrant_clients[qdrant_path] = client
            self._qdrant_client = client
            return client

    def _ensure_qdrant_collection(self, client: Any) -> None:
        """Create the Qdrant collection if it does not exist, or validate
        the vector dimension of an existing one."""
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in client.get_collections().collections}

        if self._collection_name in existing:
            info = client.get_collection(self._collection_name)
            stored_dim = info.config.params.vectors.size
            if stored_dim != self._embedding_dim:
                raise ValueError(
                    f"Qdrant collection '{self._collection_name}' stores "
                    f"{stored_dim}-dimensional vectors, but the configured embedder "
                    f"produces {self._embedding_dim}-dimensional vectors. "
                    f"Clear the cache or switch to a matching embedder preset."
                )
        else:
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def _qdrant_write(self, entry: CacheEntry, embedding: list[float]) -> None:
        try:
            from qdrant_client.models import PointStruct
        except ImportError:
            return

        client = self._get_qdrant_client()
        if client is None:
            return

        client.upsert(
            collection_name=self._collection_name,
            points=[
                PointStruct(
                    id=_qdrant_point_id(entry.prompt_hash),
                    vector=embedding,
                    payload={
                        "prompt_hash": entry.prompt_hash,
                        "model": entry.model,
                        "created_at": entry.created_at,
                        "prompt": entry.prompt[:500],
                    },
                )
            ],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection. Qdrant manages its own lifecycle."""
        if self._conn:
            self._conn.close()

    def __repr__(self) -> str:
        return (
            f"CacheStore(cache_dir={str(self._cache_dir)!r}, "
            f"collection={self._collection_name!r})"
        )
