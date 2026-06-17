"""
session.py — Session-aware Tier 1 lookup.

Extends exact + semantic matching with session context awareness to
prevent false hits where identical prompts appear in different
conversation contexts.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .embed import Embedder
from .store import CacheEntry, CacheStore

__all__ = ["SessionAwareLookup", "SessionLookupResult"]

_STATELESS_PATTERNS = [
    re.compile(r"^what is ", re.IGNORECASE),
    re.compile(r"^explain ", re.IGNORECASE),
    re.compile(r"^define ", re.IGNORECASE),
    re.compile(r"^write a .* function", re.IGNORECASE),
    re.compile(r"^how do(es)? ", re.IGNORECASE),
]


@dataclass
class SessionLookupResult:
    """Result from SessionAwareLookup.lookup()."""

    hit: bool
    hit_type: str  # 'exact' | 'semantic' | 'miss'
    response: str | None = None
    similarity: float = 0.0
    best_similarity: float = 0.0
    entry: CacheEntry | None = None
    source: str = ""  # tier1 | tier1_stateless | tier1_session


class SessionAwareLookup:
    """
    Three-check lookup sequence with session context filtering.

    Check order:
      1. Exact match scoped to session_hash
      2. Exact match ignoring session (stateless prompts only)
      3. Semantic match filtered by session_hash
    """

    def __init__(self, store: CacheStore, embedder: Embedder) -> None:
        self._store = store
        self._embedder = embedder

    def lookup(
        self,
        prompt: str,
        session_history: list[str],
        model: str,
        threshold: float,
        session_hash: str | None = None,
    ) -> SessionLookupResult:
        ctx_hash = session_hash or self._session_hash(session_history)

        # Check 1: exact match with session context
        entry = self._store.get_exact(prompt, model, session_hash=ctx_hash)
        if entry is not None:
            return SessionLookupResult(
                hit=True,
                hit_type="exact",
                response=entry.response,
                similarity=1.0,
                best_similarity=1.0,
                entry=entry,
                source="tier1",
            )

        # Check 2: exact match ignoring session (stateless prompts)
        if self._is_stateless(prompt):
            entry = self._store.get_exact(prompt, model)
            if entry is not None:
                return SessionLookupResult(
                    hit=True,
                    hit_type="exact",
                    response=entry.response,
                    similarity=1.0,
                    best_similarity=1.0,
                    entry=entry,
                    source="tier1_stateless",
                )

        # Check 3: semantic match with session context filtering
        embedding = self._embedder.embed(prompt)
        all_hits = self._store.query_semantic(
            embedding=embedding,
            model=model,
            threshold=0.0,
            session_hash=ctx_hash,
        )
        best_similarity = all_hits[0][1] if all_hits else 0.0
        passing = [(e, s) for e, s in all_hits if s >= threshold]
        if passing:
            best_entry, score = passing[0]
            return SessionLookupResult(
                hit=True,
                hit_type="semantic",
                response=best_entry.response,
                similarity=score,
                best_similarity=score,
                entry=best_entry,
                source="tier1_session",
            )

        return SessionLookupResult(
            hit=False,
            hit_type="miss",
            similarity=0.0,
            best_similarity=best_similarity,
        )

    def _session_hash(self, history: list[str], window: int = 3) -> str:
        """Hash the last N messages to create a session context fingerprint."""
        recent = history[-window:] if len(history) >= window else history
        combined = "|||".join(recent)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _is_stateless(self, prompt: str) -> bool:
        """Stateless prompts are safe to match across sessions."""
        return any(p.match(prompt) for p in _STATELESS_PATTERNS)
