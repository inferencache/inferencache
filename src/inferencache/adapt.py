"""
adapt.py — Generative reuse adaptation engine.

Given a cached prompt, cached response, and new prompt, calls a cheap
fast model and returns the adapted response. No knowledge of the cache
or thresholds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "AdaptationClient",
    "AdaptationResult",
    "AdaptationEngine",
    "ADAPTATION_SYSTEM_PROMPT",
    "build_adaptation_prompt",
]


class AdaptationClient(Protocol):
  """Minimal interface for the adaptation model client."""

  def complete(self, system: str, user: str) -> tuple[str, int, int]:
    """Returns (response_text, tokens_in, tokens_out)."""
    ...


@dataclass
class AdaptationResult:
    response: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model: str


ADAPTATION_SYSTEM_PROMPT = """You are a response adaptation assistant.
You will be given:
1. An original request
2. A cached response to that request
3. A new request that is structurally similar but differs in specifics

Your job is to adapt the cached response to fit the new request.

Rules:
- Change ONLY what is necessary to address the differences in the new request
- Preserve the structure, depth, format, and style of the cached response
- If the cached response is code, change variable names / types / logic
  only where the new request differs — do not rewrite working sections
- If you cannot adapt confidently (the requests are too different in intent),
  respond with exactly: ADAPTATION_FAILED
- Do not explain what you changed. Return only the adapted response."""


def build_adaptation_prompt(
    cached_prompt: str,
    cached_response: str,
    new_prompt: str,
) -> str:
    return f"""ORIGINAL REQUEST:
{cached_prompt}

CACHED RESPONSE:
{cached_response}

NEW REQUEST:
{new_prompt}

Adapted response:"""


class AdaptationEngine:
    """Calls a cheap fast model to adapt a cached response to a new prompt."""

    FAILURE_SENTINEL = "ADAPTATION_FAILED"

    def __init__(
        self,
        client: AdaptationClient,
        model: str,
        max_cached_response_chars: int = 4000,
    ) -> None:
        self._client = client
        self._model = model
        self._max_chars = max_cached_response_chars

    def adapt(
        self,
        cached_prompt: str,
        cached_response: str,
        new_prompt: str,
    ) -> AdaptationResult | None:
        """
        Returns AdaptationResult on success, None on ADAPTATION_FAILED
        or if the adaptation model signals it cannot adapt confidently.
        """
        truncated_response = cached_response[: self._max_chars]
        if len(cached_response) > self._max_chars:
            truncated_response += "\n[... response truncated for adaptation ...]"

        user_prompt = build_adaptation_prompt(
            cached_prompt, truncated_response, new_prompt
        )

        t0 = time.perf_counter()
        try:
            text, tok_in, tok_out = self._client.complete(
                system=ADAPTATION_SYSTEM_PROMPT,
                user=user_prompt,
            )
        except Exception:
            return None
        latency_ms = (time.perf_counter() - t0) * 1000

        if text.strip() == self.FAILURE_SENTINEL:
            return None

        return AdaptationResult(
            response=text,
            tokens_input=tok_in,
            tokens_output=tok_out,
            latency_ms=latency_ms,
            model=self._model,
        )
