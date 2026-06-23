"""
tests/test_router.py — TierRouter classification and thresholds.
"""

from __future__ import annotations

import pytest

from inferencache.router import (
    THRESHOLDS,
    CallContext,
    PromptType,
    TierRouter,
)


@pytest.fixture
def router():
    return TierRouter()


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_classify_code_def(router):
    ctx = CallContext()
    decision = router.route("def calculate_sum(a, b):\n    return a + b", ctx)
    assert decision.prompt_type == PromptType.CODE
    assert decision.threshold == 0.92


def test_classify_code_fenced(router):
    ctx = CallContext()
    decision = router.route("```python\nprint('hi')\n```", ctx)
    assert decision.prompt_type == PromptType.CODE


def test_classify_deterministic(router):
    ctx = CallContext()
    decision = router.route("What is the capital of France?", ctx)
    assert decision.prompt_type == PromptType.DETERMINISTIC
    assert decision.threshold == 0.95


def test_classify_rag_long_context(router):
    long_context = "x" * 2500
    ctx = CallContext(system_prompt="")
    decision = router.route(long_context, ctx)
    assert decision.prompt_type == PromptType.RAG
    assert decision.threshold == 0.88


def test_classify_conversational_default(router):
    ctx = CallContext()
    decision = router.route("Can you help me think through this idea?", ctx)
    assert decision.prompt_type == PromptType.CONVERSATIONAL
    assert decision.threshold == 0.82


# ---------------------------------------------------------------------------
# Threshold mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt_type,expected",
    [
        (PromptType.CODE, 0.92),
        (PromptType.DETERMINISTIC, 0.95),
        (PromptType.RAG, 0.88),
        (PromptType.CONVERSATIONAL, 0.82),
    ],
)
def test_threshold_values(prompt_type, expected):
    assert THRESHOLDS[prompt_type] == expected


# ---------------------------------------------------------------------------
# Prefix + session flags
# ---------------------------------------------------------------------------


def test_prefix_enabled_above_1024_tokens(router):
    long_prompt = "word " * 600  # ~3000 chars → ~750 tokens... need more
    long_prompt = "word " * 900  # ~4500 chars → ~1125 tokens
    ctx = CallContext()
    decision = router.route(long_prompt, ctx)
    assert decision.prefix_enabled is True


def test_prefix_disabled_below_1024_tokens(router):
    ctx = CallContext()
    decision = router.route("Short question?", ctx)
    assert decision.prefix_enabled is False


def test_session_aware_multi_turn(router):
    ctx = CallContext(turn_count=3)
    decision = router.route("Continue please", ctx)
    assert decision.session_aware is True


def test_session_aware_single_turn(router):
    ctx = CallContext(turn_count=1)
    decision = router.route("Hello", ctx)
    assert decision.session_aware is False


def test_prefix_ttl_high_frequency(router):
    ctx = CallContext(session_frequency="high")
    decision = router.route("Hello", ctx)
    assert decision.prefix_ttl == "1hr"


def test_prefix_ttl_low_frequency(router):
    ctx = CallContext(session_frequency="low")
    decision = router.route("Hello", ctx)
    assert decision.prefix_ttl == "5min"
