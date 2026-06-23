"""
tests/test_prefix.py — PrefixOptimizer per provider.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from promptcache.prefix import (
    DYNAMIC_SYSTEM_PROMPT_INDICATORS,
    PrefixConfig,
    PrefixOptimizer,
)


@pytest.fixture
def optimizer():
    return PrefixOptimizer()


SAMPLE_CONTEXT = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there"},
    {"role": "user", "content": "What is caching?"},
]

SYSTEM = "You are a helpful assistant."


def test_anthropic_system_has_cache_control(optimizer):
    config = PrefixConfig(provider="anthropic", ttl="5min")
    result = optimizer.optimize("", SYSTEM, SAMPLE_CONTEXT, config)

    assert result.modified is True
    assert result.system is not None
    assert result.system[0]["cache_control"] == {"type": "ephemeral"}
    assert result.system[0]["text"] == SYSTEM
    assert result.expected_savings_tier == "prefix"


def test_anthropic_penultimate_history_marked(optimizer):
    config = PrefixConfig(provider="anthropic")
    result = optimizer.optimize("", SYSTEM, SAMPLE_CONTEXT, config)

    assert "cache_control" not in result.messages[0]
    assert result.messages[1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in result.messages[2]


def test_anthropic_1hr_ttl(optimizer):
    config = PrefixConfig(provider="anthropic", ttl="1hr")
    result = optimizer.optimize("", SYSTEM, SAMPLE_CONTEXT, config)

    assert result.system[0]["cache_control"]["ttl"] == 3600


def test_openai_stable_no_warnings(optimizer):
    config = PrefixConfig(provider="openai")
    result = optimizer.optimize("", SYSTEM, SAMPLE_CONTEXT, config)

    assert result.modified is False
    assert result.warnings == []
    assert result.messages == SAMPLE_CONTEXT


def test_openai_dynamic_system_warnings(optimizer):
    dynamic_system = f"You are helping {{user}} today in {datetime.now().year}"
    config = PrefixConfig(provider="openai")
    result = optimizer.optimize("", dynamic_system, SAMPLE_CONTEXT, config)

    assert result.modified is False
    assert len(result.warnings) == 1
    assert "dynamic content" in result.warnings[0].lower()


def test_unknown_provider_passthrough(optimizer):
    config = PrefixConfig(provider="cohere")
    result = optimizer.optimize("", SYSTEM, SAMPLE_CONTEXT, config)

    assert result.modified is False
    assert result.messages == SAMPLE_CONTEXT


def test_dynamic_indicators_constant_matches_openai_check(optimizer):
    """SYNC constant covers all indicators checked by _optimize_openai."""
    for indicator in DYNAMIC_SYSTEM_PROMPT_INDICATORS:
        config = PrefixConfig(provider="openai")
        result = optimizer.optimize("", f"System {indicator} here", [], config)
        assert result.warnings


def test_analyze_system_prompt_stable(optimizer):
    analysis = optimizer.analyze_system_prompt("You are a helpful coding assistant.")
    assert analysis["stability_score"] == 1.0
    assert analysis["warnings"] == []


def test_analyze_system_prompt_dynamic(optimizer):
    analysis = optimizer.analyze_system_prompt("Help {user} with {date} tasks")
    assert analysis["stability_score"] < 1.0
    assert len(analysis["warnings"]) >= 2
