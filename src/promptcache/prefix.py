"""
prefix.py — Tier 2 prefix cache optimizer.

Restructures prompts to maximize provider-side prefix cache hit rate.
Stable content first, dynamic content last.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "PrefixConfig",
    "OptimizedRequest",
    "PrefixOptimizer",
    "DYNAMIC_SYSTEM_PROMPT_INDICATORS",
]

# SYNC: keep in sync with frontend-next/src/lib/prefixPatterns.ts
DYNAMIC_SYSTEM_PROMPT_INDICATORS = [
    "{user}",
    "{date}",
    "{session_id}",
]


@dataclass
class PrefixConfig:
    """Configuration for prefix optimization."""

    provider: str
    ttl: str = "5min"  # "5min" | "1hr"


@dataclass
class OptimizedRequest:
    """Result of prefix optimization."""

    messages: list[dict[str, Any]]
    system: list[dict[str, Any]] | None = None
    modified: bool = False
    warnings: list[str] = field(default_factory=list)
    expected_savings_tier: str | None = None


class PrefixOptimizer:
    """
    Restructures prompts to maximize provider-side prefix cache hit rate.

    Anthropic: injects cache_control: ephemeral markers.
    OpenAI: automatic caching — warns on dynamic system prompt content.
    """

    def optimize(
        self,
        prompt: str,
        system_prompt: str,
        context: list[dict[str, Any]],
        config: PrefixConfig,
    ) -> OptimizedRequest:
        if config.provider == "anthropic":
            return self._optimize_anthropic(system_prompt, context, config)
        if config.provider == "openai":
            return self._optimize_openai(system_prompt, context)
        return OptimizedRequest(messages=list(context), modified=False)

    def _optimize_anthropic(
        self,
        system_prompt: str,
        context: list[dict[str, Any]],
        config: PrefixConfig,
    ) -> OptimizedRequest:
        cache_control: dict[str, Any] = {"type": "ephemeral"}
        if config.ttl == "1hr":
            cache_control["ttl"] = 3600

        system_with_cache = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": cache_control,
            }
        ]

        optimized_messages: list[dict[str, Any]] = []
        if not context:
            context = [{"role": "user", "content": ""}]

        for i, msg in enumerate(context[:-1]):
            if i == len(context) - 2:
                optimized_messages.append({**msg, "cache_control": {"type": "ephemeral"}})
            else:
                optimized_messages.append(dict(msg))

        optimized_messages.append(dict(context[-1]))

        return OptimizedRequest(
            system=system_with_cache,
            messages=optimized_messages,
            modified=True,
            expected_savings_tier="prefix",
        )

    def _optimize_openai(
        self,
        system_prompt: str,
        context: list[dict[str, Any]],
    ) -> OptimizedRequest:
        dynamic_indicators = [
            *DYNAMIC_SYSTEM_PROMPT_INDICATORS,
            str(datetime.now().year),
        ]
        has_dynamic = any(d in system_prompt for d in dynamic_indicators)
        warnings: list[str] = []
        if has_dynamic:
            warnings.append(
                "System prompt contains dynamic content — will reduce prefix cache hits"
            )

        return OptimizedRequest(
            messages=list(context),
            modified=False,
            warnings=warnings,
        )

    def analyze_system_prompt(self, system_prompt: str) -> dict[str, Any]:
        """
        Analyze system prompt stability for dashboard tuning UI.

        Returns stability_score [0.0–1.0] and any dynamic-content warnings.
        """
        dynamic_indicators = [
            *DYNAMIC_SYSTEM_PROMPT_INDICATORS,
            str(datetime.now().year),
        ]
        warnings: list[str] = []
        for indicator in dynamic_indicators:
            if indicator in system_prompt:
                warnings.append(f"Dynamic content detected: {indicator!r}")

        if not system_prompt.strip():
            return {"stability_score": 0.0, "warnings": ["System prompt is empty"]}

        # Penalise each dynamic indicator; stable prompts score 1.0
        penalty = min(len(warnings) * 0.25, 1.0)
        stability_score = round(1.0 - penalty, 2)
        if warnings and not any("reduce prefix" in w for w in warnings):
            warnings.append(
                "System prompt contains dynamic content — will reduce prefix cache hits"
            )

        return {"stability_score": stability_score, "warnings": warnings}
