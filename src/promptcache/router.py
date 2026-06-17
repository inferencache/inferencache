"""
router.py — Tier routing and prompt classification.

Classifies incoming prompts and decides which caching tiers to attempt
and in what configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PromptType",
    "THRESHOLDS",
    "CallContext",
    "TierDecision",
    "TierRouter",
]


class PromptType(Enum):
    CODE = "code"
    DETERMINISTIC = "deterministic"
    RAG = "rag"
    CONVERSATIONAL = "conversational"


THRESHOLDS: dict[PromptType, float] = {
    PromptType.CODE: 0.92,
    PromptType.DETERMINISTIC: 0.95,
    PromptType.RAG: 0.88,
    PromptType.CONVERSATIONAL: 0.82,
}

_CODE_PATTERNS = [
    re.compile(r"```", re.IGNORECASE),
    re.compile(r"\bdef\s+\w+\s*\(", re.IGNORECASE),
    re.compile(r"\bfunction\s+\w+", re.IGNORECASE),
    re.compile(r"\bclass\s+\w+", re.IGNORECASE),
    re.compile(r"\b(import|from)\s+\w+", re.IGNORECASE),
    re.compile(r"\bfix\b.*\b(bug|error|function|code)\b", re.IGNORECASE),
    re.compile(r"\brefactor\b", re.IGNORECASE),
]

_DETERMINISTIC_PATTERNS = [
    re.compile(r"^what is ", re.IGNORECASE),
    re.compile(r"^define ", re.IGNORECASE),
    re.compile(r"^list (all |the )?", re.IGNORECASE),
    re.compile(r"^how many ", re.IGNORECASE),
    re.compile(r"^when (was|did) ", re.IGNORECASE),
]

_RAG_MIN_CHARS = 2000


@dataclass
class CallContext:
    """Context for a single LLM call."""

    provider: str = "unknown"
    system_prompt: str = ""
    turn_count: int = 1
    session_frequency: str = "low"  # "low" | "high"


@dataclass
class TierDecision:
    """Routing decision for a single prompt."""

    prompt_type: PromptType
    threshold: float
    prefix_enabled: bool
    session_aware: bool
    prefix_ttl: str = "5min"


class TierRouter:
    """Classifies prompts and decides tier configuration."""

    def route(self, prompt: str, context: CallContext) -> TierDecision:
        prompt_type = self._classify(prompt, context)
        prompt_tokens = self._estimate_tokens(prompt, context)
        threshold = THRESHOLDS[prompt_type]

        return TierDecision(
            prompt_type=prompt_type,
            threshold=threshold,
            prefix_enabled=prompt_tokens > 1024,
            session_aware=context.turn_count > 1,
            prefix_ttl="1hr" if context.session_frequency == "high" else "5min",
        )

    def _classify(self, prompt: str, context: CallContext) -> PromptType:
        if self._is_code_query(prompt):
            return PromptType.CODE
        if self._is_deterministic(prompt):
            return PromptType.DETERMINISTIC
        if self._has_long_context(prompt, context):
            return PromptType.RAG
        return PromptType.CONVERSATIONAL

    def _is_code_query(self, prompt: str) -> bool:
        return any(p.search(prompt) for p in _CODE_PATTERNS)

    def _is_deterministic(self, prompt: str) -> bool:
        return any(p.match(prompt.strip()) for p in _DETERMINISTIC_PATTERNS)

    def _has_long_context(self, prompt: str, context: CallContext) -> bool:
        combined_len = len(prompt) + len(context.system_prompt)
        return combined_len >= _RAG_MIN_CHARS

    def _estimate_tokens(self, prompt: str, context: CallContext) -> int:
        """Rough token estimate: ~4 chars per token."""
        total_chars = len(prompt) + len(context.system_prompt)
        return max(1, total_chars // 4)
