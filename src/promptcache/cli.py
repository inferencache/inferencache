"""
cli.py — Command-line interface for promptcache.

Commands:
    promptcache stats   — Print hit rate, cost saved, top cached queries
    promptcache clear   — Flush the cache (optionally by model)
    promptcache config  — Show active configuration

Entry point registered in pyproject.toml as:
    [project.scripts]
    promptcache = "promptcache.cli:main"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Rough per-token cost estimates (USD) for common models.
# Users can override via --cost-per-token.
# Source: public pricing pages as of early 2026.
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
    "unknown": 0.000003,  # conservative default
}

_CHARS_PER_TOKEN = 4  # rough approximation


def _get_cost_per_token(model: str, override: float | None) -> float:
    if override is not None:
        return override
    # Try exact match, then prefix match
    if model in _MODEL_COSTS:
        return _MODEL_COSTS[model]
    for key in _MODEL_COSTS:
        if model.startswith(key):
            return _MODEL_COSTS[key]
    return _MODEL_COSTS["unknown"]


def _format_cost(dollars: float) -> str:
    if dollars >= 1.0:
        return f"${dollars:.2f}"
    elif dollars >= 0.01:
        return f"${dollars:.4f}"
    else:
        return f"${dollars:.6f}"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "promptcache"


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"No cache found at {cache_dir}")
        print("Run your app with promptcache enabled first.")
        return 1

    try:
        from .store import CacheStore

        # We need a collection name to open the store. Use a placeholder
        # since stats() only queries SQLite, not ChromaDB.
        store = CacheStore(
            cache_dir=cache_dir,
            collection_name="__cli__",
        )
        stats = store.stats(top_n=args.top)
    except Exception as exc:
        print(f"Error reading cache: {exc}", file=sys.stderr)
        return 1

    cost_per_token = _get_cost_per_token(args.model, args.cost_per_token)

    # Estimate tokens saved: total_hits * average_response_length
    # We use a rough heuristic of 200 tokens per cached response.
    avg_tokens = args.avg_tokens
    tokens_saved = stats.total_hits * avg_tokens
    cost_saved = tokens_saved * cost_per_token

    if args.json:
        output = {
            "total_entries": stats.total_entries,
            "total_hits": stats.total_hits,
            "exact_hits": stats.exact_hits,
            "semantic_hits": stats.semantic_hits,
            "hit_rate": stats.hit_rate,
            "estimated_tokens_saved": tokens_saved,
            "estimated_cost_saved_usd": cost_saved,
            "top_entries": stats.top_entries,
        }
        print(json.dumps(output, indent=2))
        return 0

    # Human-readable output
    hit_pct = f"{stats.hit_rate * 100:.1f}%"
    exact_pct = (
        f"{stats.exact_hits / stats.total_hits * 100:.0f}%"
        if stats.total_hits > 0
        else "—"
    )
    semantic_pct = (
        f"{stats.semantic_hits / stats.total_hits * 100:.0f}%"
        if stats.total_hits > 0
        else "—"
    )

    print()
    print("  promptcache stats")
    print("  " + "─" * 42)
    print(f"  Cache dir        {cache_dir}")
    print(f"  Entries stored   {stats.total_entries:,}")
    print()
    print(f"  Hit rate         {hit_pct}")
    print(f"    exact          {stats.exact_hits:,}  ({exact_pct})")
    print(f"    semantic       {stats.semantic_hits:,}  ({semantic_pct})")
    print()
    print(f"  Est. tokens saved   {tokens_saved:,}")
    print(f"  Est. cost saved     {_format_cost(cost_saved)}")
    print(f"  (model: {args.model}, {cost_per_token:.7f} USD/token)")

    if stats.top_entries:
        print()
        print(f"  Top {len(stats.top_entries)} cached prompts by hit count:")
        print("  " + "─" * 42)
        for i, entry in enumerate(stats.top_entries, 1):
            prompt_preview = entry["prompt"]
            if len(prompt_preview) > 60:
                prompt_preview = prompt_preview[:57] + "..."
            print(f"  {i:>2}. [{entry['hit_count']:>4}×]  {prompt_preview!r}")

    print()
    return 0


# ---------------------------------------------------------------------------
# clear command
# ---------------------------------------------------------------------------


def cmd_clear(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"No cache found at {cache_dir}")
        return 0

    model_filter = getattr(args, "model", None)

    if not args.yes:
        target = f"entries for model '{model_filter}'" if model_filter else "entire cache"
        confirm = input(f"Delete {target} at {cache_dir}? [y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborted.")
            return 0

    try:
        from .store import CacheStore

        store = CacheStore(cache_dir=cache_dir, collection_name="__cli__")
        deleted = store.clear(model=model_filter)
        store.close()
    except Exception as exc:
        print(f"Error clearing cache: {exc}", file=sys.stderr)
        return 1

    noun = "entries" if deleted != 1 else "entry"
    print(f"Deleted {deleted:,} {noun}.")
    return 0


# ---------------------------------------------------------------------------
# config command
# ---------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)

    print()
    print("  promptcache config")
    print("  " + "─" * 42)
    print(f"  cache_dir    {cache_dir}")
    print(f"  threshold    0.85  (default; override via CacheConfig)")
    print(f"  embedder     SentenceTransformerEmbedder(BAAI/bge-small-en-v1.5, preset=balanced)")
    print()
    print("  Override in code:")
    print("    from promptcache import CacheConfig")
    print("    config = CacheConfig(")
    print(f'        cache_dir="{cache_dir}",')
    print("        threshold=0.88,")
    print('        model="gpt-4o",')
    print("    )")
    print()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptcache",
        description="Semantic LLM response caching — inspect and manage your cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_default_cache_dir()),
        help="Path to cache directory (default: ~/.cache/promptcache)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── stats ──
    stats_p = sub.add_parser("stats", help="Show cache statistics")
    stats_p.add_argument(
        "--model",
        default="unknown",
        help="Model name for cost estimation (e.g. gpt-4o)",
    )
    stats_p.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of top entries to show (default: 10)",
    )
    stats_p.add_argument(
        "--avg-tokens",
        type=int,
        default=200,
        metavar="N",
        help="Assumed avg response length in tokens for cost estimation (default: 200)",
    )
    stats_p.add_argument(
        "--cost-per-token",
        type=float,
        default=None,
        metavar="USD",
        help="Override cost per output token in USD",
    )
    stats_p.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    stats_p.set_defaults(func=cmd_stats)

    # ── clear ──
    clear_p = sub.add_parser("clear", help="Delete cached entries")
    clear_p.add_argument(
        "--model",
        default=None,
        help="Only delete entries for this model (default: all models)",
    )
    clear_p.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )
    clear_p.set_defaults(func=cmd_clear)

    # ── config ──
    config_p = sub.add_parser("config", help="Show active configuration")
    config_p.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
