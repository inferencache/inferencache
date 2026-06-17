"""
mcp/tools.py — Tool implementations for the promptcache MCP server.

All functions here are pure: they take a cache_dir and args, open a
CacheStore, do their work, and return a plain dict. This makes them
directly testable without starting the MCP server process.

server.py imports these and wraps them in the MCP protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_MODEL_COSTS: dict[str, float] = {
    "gpt-4o": 0.000005,
    "gpt-4o-mini": 0.0000006,
    "gpt-4-turbo": 0.00001,
    "gpt-3.5-turbo": 0.0000005,
    "claude-3-5-sonnet-20241022": 0.000003,
    "claude-3-opus-20240229": 0.000015,
    "claude-3-haiku-20240307": 0.00000025,
    "claude-sonnet-4-20250514": 0.000003,
    "gemini-1.5-pro": 0.0000035,
    "gemini-1.5-flash": 0.00000035,
    "unknown": 0.000003,
}


def _cost_per_token(model: str) -> float:
    if model in _MODEL_COSTS:
        return _MODEL_COSTS[model]
    for key in _MODEL_COSTS:
        if model.startswith(key):
            return _MODEL_COSTS[key]
    return _MODEL_COSTS["unknown"]


def _open_store(cache_dir: Path):
    """Open a read-only CacheStore for CLI/MCP tool use."""
    from ..store import CacheStore

    return CacheStore(cache_dir=cache_dir, collection_name="__mcp__")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_get_stats(
    cache_dir: Path,
    model: str = "unknown",
    top_n: int = 10,
    avg_tokens: int = 200,
) -> dict[str, Any]:
    """
    Return cache statistics as a plain dict.

    Includes hit rate breakdown, estimated tokens saved, estimated cost
    saved, and the top entries by hit count.
    """
    if not cache_dir.exists():
        return {
            "total_entries": 0,
            "total_hits": 0,
            "hit_rate": 0.0,
            "message": f"No cache found at {cache_dir}. Run your app with promptcache enabled first.",
        }

    store = _open_store(cache_dir)
    try:
        stats = store.stats(top_n=top_n)
    finally:
        store.close()

    cpt = _cost_per_token(model)
    tokens_saved = stats.total_hits * avg_tokens
    cost_saved = tokens_saved * cpt

    return {
        "total_entries": stats.total_entries,
        "total_hits": stats.total_hits,
        "exact_hits": stats.exact_hits,
        "semantic_hits": stats.semantic_hits,
        "hit_rate": stats.hit_rate,
        "hit_rate_pct": f"{stats.hit_rate * 100:.1f}%",
        "estimated_tokens_saved": tokens_saved,
        "estimated_cost_saved_usd": round(cost_saved, 6),
        "estimated_cost_saved_formatted": _fmt_cost(cost_saved),
        "model_used_for_estimate": model,
        "cost_per_token_usd": cpt,
        "top_entries": stats.top_entries,
    }


def tool_list_recent(
    cache_dir: Path,
    limit: int = 20,
) -> dict[str, Any]:
    """Return the most recently created cache entries."""
    if not cache_dir.exists():
        return {"entries": [], "message": f"No cache found at {cache_dir}"}

    store = _open_store(cache_dir)
    try:
        entries = store.list_recent(limit=limit)
    finally:
        store.close()

    return {
        "count": len(entries),
        "entries": [
            {
                "prompt": e.prompt[:120] + ("..." if len(e.prompt) > 120 else ""),
                "model": e.model,
                "hit_count": e.hit_count,
                "created_at": e.created_at,
                "prompt_hash": e.prompt_hash[:12] + "...",
            }
            for e in entries
        ],
    }


def tool_get_cached_entry(
    cache_dir: Path,
    prompt: str,
    model: str = "unknown",
    threshold: float = 0.85,
) -> dict[str, Any]:
    """
    Look up a prompt in the cache (exact match first, then semantic).

    Returns the cached response and metadata if found, or a not-found
    message if the prompt isn't in the cache.
    """
    if not cache_dir.exists():
        return {"found": False, "message": f"No cache found at {cache_dir}"}

    store = _open_store(cache_dir)
    try:
        # Try exact match first
        entry = store.get_exact(prompt, model)
        if entry is not None:
            return {
                "found": True,
                "hit_type": "exact",
                "similarity": 1.0,
                "model": entry.model,
                "hit_count": entry.hit_count,
                "created_at": entry.created_at,
                "response_preview": entry.response[:300] + (
                    "..." if len(entry.response) > 300 else ""
                ),
                "response_length": len(entry.response),
            }

        # Try semantic match
        try:
            from ..embed import get_default_embedder

            embedder = get_default_embedder()
            embedding = embedder.embed(prompt)
            hits = store.query_semantic(
                embedding=embedding,
                model=model,
                threshold=threshold,
                top_k=3,
            )
            if hits:
                best_entry, score = hits[0]
                return {
                    "found": True,
                    "hit_type": "semantic",
                    "similarity": score,
                    "model": best_entry.model,
                    "hit_count": best_entry.hit_count,
                    "created_at": best_entry.created_at,
                    "matched_prompt_preview": best_entry.prompt[:120],
                    "response_preview": best_entry.response[:300] + (
                        "..." if len(best_entry.response) > 300 else ""
                    ),
                    "response_length": len(best_entry.response),
                }
        except Exception:
            pass  # Embedding unavailable — fall through to not-found

        return {
            "found": False,
            "message": "No matching cache entry found for this prompt.",
            "threshold_used": threshold,
        }
    finally:
        store.close()


def tool_set_threshold(threshold: float) -> dict[str, Any]:
    """
    Validate and return a new similarity threshold value.

    The MCP server stores the active threshold in process memory; this
    function performs validation only.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold}")
    return {"success": True, "threshold": threshold}


def tool_clear_cache(
    cache_dir: Path,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Delete cache entries. Returns count of deleted entries.
    """
    if not cache_dir.exists():
        return {"success": True, "deleted": 0, "message": "Cache directory not found; nothing to clear."}

    store = _open_store(cache_dir)
    try:
        deleted = store.clear(model=model)
    finally:
        store.close()

    return {
        "success": True,
        "deleted": deleted,
        "model_filter": model or "all",
        "message": f"Deleted {deleted} entr{'y' if deleted == 1 else 'ies'}.",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_cost(dollars: float) -> str:
    if dollars >= 1.0:
        return f"${dollars:.2f}"
    elif dollars >= 0.01:
        return f"${dollars:.4f}"
    else:
        return f"${dollars:.6f}"
