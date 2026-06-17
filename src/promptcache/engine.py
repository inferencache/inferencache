"""
engine.py — Core cache engine.

The CacheEngine is the single coordinator between the public API,
the embedder, and the store. It owns:

  1. The two-check lookup sequence (exact → semantic)
  2. Write-back logic (when a real API call returns, store the result)
  3. Stream reconstitution (yield chunks from a cached string so
     callers that expect a generator see no difference)
  4. Configuration validation

CacheEngine is the only place that knows about both embed.py and
store.py. Neither of those modules imports the other.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

from .embed import Embedder, get_default_embedder
from .store import CacheEntry, CacheStore, _hash_key

__all__ = ["CacheEngine", "CacheConfig", "CacheResult"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "promptcache"
_DEFAULT_THRESHOLD = 0.85
_DEFAULT_STREAM_CHUNK_SIZE = 32
_DEFAULT_STREAM_DELAY = 0.0


@dataclass
class CacheConfig:
    """
    All tunable parameters for a CacheEngine instance.

    Sensible defaults work out of the box. Override only what you need.

    Args:
        cache_dir: Where to store index.db and Qdrant data.
        threshold: Minimum cosine similarity [0.0–1.0] for a semantic
                   hit. Higher = stricter matching.
                   Recommended range: 0.80–0.92.
        model: LLM model string used as part of the cache key.
        embedder: Custom Embedder implementation. When None, the preset
                  selected by embedder_preset is used.
        embedder_preset: 'fast' | 'balanced' | 'accurate'. Ignored when
                         embedder is provided explicitly.
        provider: LLM provider ('openai' | 'anthropic'). Recorded in the
                  calls event log for cost analytics.
        default_endpoint: Fallback endpoint label for call events when
                          lookup()/store() are called without endpoint.
        max_response_tokens: Responses longer than this (in rough char
                             count / 4) are not cached. Set 0 to disable.
        stream_chunk_size: Characters per chunk when reconstituting a
                           cached response as a stream.
        stream_delay: Seconds to sleep between chunks (0 = instant).
        enabled: Master switch. When False, always passes through to the
                 real API without touching the cache.
    """

    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    threshold: float = _DEFAULT_THRESHOLD
    model: str = "unknown"
    embedder: Embedder | None = None
    embedder_preset: str = "balanced"
    provider: str = "unknown"
    default_endpoint: str | None = None
    max_response_tokens: int = 8192
    stream_chunk_size: int = _DEFAULT_STREAM_CHUNK_SIZE
    stream_delay: float = _DEFAULT_STREAM_DELAY
    enabled: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {self.threshold}")
        if self.embedder is None:
            from .embed import get_embedder
            self.embedder = get_embedder(self.embedder_preset)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class CacheResult:
    """
    Returned by CacheEngine.lookup() to communicate what happened.

    Attributes:
        hit: True if a cached response was found (exact or semantic).
        hit_type: 'exact', 'semantic', or 'miss'.
        response: The cached response string if hit=True, else None.
        similarity: Cosine similarity score for semantic hits; 1.0 for
                    exact hits; 0.0 for misses.
        best_similarity: Highest semantic candidate score on miss (even if
                         below threshold). Useful for threshold tuning.
        entry: The full CacheEntry for hits; None for misses.
        latency_ms: Time spent in the lookup, in milliseconds.
        call_id: Auto-increment row ID from the calls event log.
                 None when caching is disabled (enabled=False).
    """

    hit: bool
    hit_type: str  # 'exact' | 'semantic' | 'miss'
    response: str | None = None
    similarity: float = 0.0
    best_similarity: float = 0.0
    entry: CacheEntry | None = None
    latency_ms: float = 0.0
    call_id: int | None = None


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CacheEngine:
    """
    Coordinates exact-match lookup, semantic search, and write-back.

    Typical usage — see api.py for the higher-level decorator/context:

        engine = CacheEngine(CacheConfig(model="gpt-4o", threshold=0.88))

        result = engine.lookup(prompt)
        if result.hit:
            return result.response

        response = call_real_api(prompt)
        engine.store(prompt, response)
        return response
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()
        self._store = self._build_store()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def lookup(
        self,
        prompt: str,
        endpoint: str | None = None,
        session_id: str | None = None,
    ) -> CacheResult:
        """
        Run the two-check lookup sequence for the given prompt.

        Check order:
          1. Exact match  (SHA-256 key in SQLite — sub-ms)
          2. Semantic match (embedding + Qdrant query)

        Writes one row to the calls event log regardless of outcome.

        Returns a CacheResult regardless of outcome. The caller
        inspects result.hit to decide whether to use result.response
        or proceed with the real API call.

        Args:
            prompt: The prompt to look up.
            endpoint: Optional label for the calling function/route.
            session_id: Optional session grouping identifier.
        """
        if not self._config.enabled:
            return CacheResult(hit=False, hit_type="miss")

        t0 = time.perf_counter()
        effective_endpoint = endpoint or self._config.default_endpoint

        # ── Check 1: exact match ──────────────────────────────────────
        entry = self._store.get_exact(prompt, self._config.model)
        if entry is not None:
            self._store.increment_hit(entry.prompt_hash, hit_type="exact")
            latency = round((time.perf_counter() - t0) * 1000, 2)
            call_id = self._store.write_call_event(
                prompt_hash=entry.prompt_hash,
                model=self._config.model,
                provider=self._config.provider,
                hit_type="exact",
                latency_ms=latency,
                endpoint=effective_endpoint,
                session_id=session_id,
                similarity=1.0,
            )
            return CacheResult(
                hit=True,
                hit_type="exact",
                response=entry.response,
                similarity=1.0,
                best_similarity=1.0,
                entry=entry,
                latency_ms=latency,
                call_id=call_id,
            )

        # ── Check 2: semantic match ───────────────────────────────────
        embedding = self._config.embedder.embed(prompt)
        # Query all candidates (threshold=0) so we capture best_similarity on miss
        all_hits = self._store.query_semantic(
            embedding=embedding,
            model=self._config.model,
            threshold=0.0,
        )
        best_similarity = all_hits[0][1] if all_hits else 0.0
        passing = [(e, s) for e, s in all_hits if s >= self._config.threshold]
        if passing:
            best_entry, score = passing[0]
            self._store.increment_hit(best_entry.prompt_hash, hit_type="semantic")
            latency = round((time.perf_counter() - t0) * 1000, 2)
            call_id = self._store.write_call_event(
                prompt_hash=best_entry.prompt_hash,
                model=self._config.model,
                provider=self._config.provider,
                hit_type="semantic",
                latency_ms=latency,
                endpoint=effective_endpoint,
                session_id=session_id,
                similarity=score,
            )
            return CacheResult(
                hit=True,
                hit_type="semantic",
                response=best_entry.response,
                similarity=score,
                best_similarity=score,
                entry=best_entry,
                latency_ms=latency,
                call_id=call_id,
            )

        # ── Miss ──────────────────────────────────────────────────────
        self._store.record_miss()
        latency = round((time.perf_counter() - t0) * 1000, 2)
        call_id = self._store.write_call_event(
            prompt_hash=_hash_key(prompt, self._config.model),
            model=self._config.model,
            provider=self._config.provider,
            hit_type="miss",
            latency_ms=latency,
            endpoint=effective_endpoint,
            session_id=session_id,
        )
        return CacheResult(
            hit=False,
            hit_type="miss",
            similarity=0.0,
            best_similarity=best_similarity,
            latency_ms=latency,
            call_id=call_id,
        )

    def store(
        self,
        prompt: str,
        response: str,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
        endpoint: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Persist a (prompt, response) pair after a real API call.

        Writes to SQLite and Qdrant. Also writes one row to the calls
        event log with the real API token counts and cost.

        Args:
            prompt: The exact prompt that was sent to the LLM.
            response: The full response string to cache.
            tokens_input: Actual input token count from the API response.
            tokens_output: Actual output token count from the API response.
            cost_usd: Actual cost in USD for this API call.
            endpoint: Optional label for the calling function/route.
            session_id: Optional session grouping identifier.
            metadata: Optional arbitrary metadata attached to the entry.
        """
        if not self._config.enabled:
            return

        if self._config.max_response_tokens > 0:
            if len(response) > self._config.max_response_tokens * 4:
                # Very rough char-to-token ratio. Skips caching of huge
                # responses without importing a tokenizer.
                return

        entry = CacheEntry(
            prompt=prompt,
            model=self._config.model,
            response=response,
            created_at=time.time(),
            metadata=metadata,
        )

        embedding = self._config.embedder.embed(prompt)
        self._store.write(entry, embedding=embedding)

        effective_endpoint = endpoint or self._config.default_endpoint
        self._store.write_call_event(
            prompt_hash=entry.prompt_hash,
            model=self._config.model,
            provider=self._config.provider,
            hit_type="miss",
            latency_ms=0.0,
            endpoint=effective_endpoint,
            session_id=session_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost_usd=cost_usd,
        )

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------

    def stream_cached(self, response: str) -> Iterator[str]:
        """
        Yield chunks from a cached response string.

        Callers that use `for chunk in stream:` see identical behaviour
        whether the response came from the cache or the real API.
        """
        chunk_size = self._config.stream_chunk_size
        delay = self._config.stream_delay
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield chunk
            if delay > 0:
                time.sleep(delay)

    async def astream_cached(self, response: str) -> AsyncIterator[str]:
        """Async variant of stream_cached for async callers."""
        chunk_size = self._config.stream_chunk_size
        delay = self._config.stream_delay
        for i in range(0, len(response), chunk_size):
            chunk = response[i : i + chunk_size]
            yield chunk
            if delay > 0:
                await asyncio.sleep(delay)

    @staticmethod
    def collect_stream(stream: Iterator[str]) -> str:
        """Consume a streaming response and return the full string."""
        return "".join(stream)

    @staticmethod
    async def acollect_stream(stream: AsyncIterator[str]) -> str:
        """Async variant of collect_stream."""
        parts = []
        async for chunk in stream:
            parts.append(chunk)
        return "".join(parts)

    # ------------------------------------------------------------------
    # Config + store access
    # ------------------------------------------------------------------

    @property
    def config(self) -> CacheConfig:
        return self._config

    @property
    def cache_store(self) -> CacheStore:
        """Direct access to the underlying persistence layer."""
        return self._store

    def set_threshold(self, threshold: float) -> None:
        """Update the similarity threshold at runtime."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
        self._config.threshold = threshold

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable caching at runtime."""
        self._config.enabled = enabled

    def close(self) -> None:
        """Release resources held by the underlying store."""
        self._store.close()

    def __repr__(self) -> str:
        return (
            f"CacheEngine(model={self._config.model!r}, "
            f"threshold={self._config.threshold}, "
            f"enabled={self._config.enabled})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_store(self) -> CacheStore:
        embedder_id = self._config.embedder.model_id()
        model_slug = (
            self._config.model.replace("/", "-")
            .replace(":", "-")
            .replace(".", "-")
            .lower()
        )
        # Collection name includes embedder ID to prevent dimension collisions
        collection_name = f"pc-{model_slug}-{embedder_id}"[:63]

        embedding_dim = self._get_embedding_dim()

        return CacheStore(
            cache_dir=self._config.cache_dir,
            collection_name=collection_name,
            embedding_dim=embedding_dim,
        )

    def _get_embedding_dim(self) -> int:
        """Return the embedding dimension for the configured embedder."""
        embedder = self._config.embedder
        if hasattr(embedder, "dimension"):
            return int(embedder.dimension)
        # Fall back to embedding a minimal string to measure the dimension.
        # This triggers lazy model loading for unknown embedder types.
        return len(embedder.embed("x"))
