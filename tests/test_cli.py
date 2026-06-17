"""
tests/test_cli.py

Tests for promptcache CLI commands: stats, clear, config.
"""

from __future__ import annotations

import json
import time

import pytest

from promptcache.cli import main
from promptcache.store import CacheEntry, CacheStore


@pytest.fixture
def populated_cache(tmp_path):
    cache_dir = tmp_path / "cli_cache"
    store = CacheStore(cache_dir=cache_dir, collection_name="__cli__")
    entry = CacheEntry(
        prompt="What is Python?",
        model="gpt-4o",
        response="A programming language.",
        created_at=time.time(),
    )
    store.write(entry)
    store.increment_hit(entry.prompt_hash, hit_type="exact")
    store.close()
    return cache_dir


def test_stats_json(populated_cache, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--cache-dir", str(populated_cache), "stats", "--json", "--model", "gpt-4o"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["total_entries"] == 1
    assert data["total_hits"] == 1
    assert "hit_rate" in data


def test_stats_human_readable(populated_cache, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--cache-dir", str(populated_cache), "stats"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "promptcache stats" in out
    assert "Hit rate" in out


def test_clear_with_yes(populated_cache):
    with pytest.raises(SystemExit) as exc:
        main(["--cache-dir", str(populated_cache), "clear", "-y"])
    assert exc.value.code == 0

    store = CacheStore(cache_dir=populated_cache, collection_name="__cli__")
    stats = store.stats()
    store.close()
    assert stats.total_entries == 0


def test_config(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--cache-dir", "/tmp/test-cache", "config"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "promptcache config" in out
    assert "/tmp/test-cache" in out
    assert "CacheConfig" in out


def test_stats_missing_cache(tmp_path, capsys):
    missing = tmp_path / "nonexistent"
    with pytest.raises(SystemExit) as exc:
        main(["--cache-dir", str(missing), "stats"])
    assert exc.value.code == 1
