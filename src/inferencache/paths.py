"""Shared cache directory resolution with legacy promptcache fallback."""

from __future__ import annotations

from pathlib import Path

_CACHE_NAME = "inferencache"
_LEGACY_CACHE_NAME = "promptcache"


def default_cache_dir() -> Path:
    """Return ~/.cache/inferencache, falling back to legacy ~/.cache/promptcache if needed."""
    home = Path.home()
    new_dir = home / ".cache" / _CACHE_NAME
    legacy_dir = home / ".cache" / _LEGACY_CACHE_NAME

    if new_dir.exists():
        return new_dir

    if legacy_dir.exists() and any(legacy_dir.iterdir()):
        return legacy_dir

    return new_dir
