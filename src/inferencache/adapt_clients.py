"""
adapt_clients.py — Provider-specific AdaptationClient implementations.

Kept separate from adapt.py to avoid importing provider SDKs in the
core adaptation logic.
"""

from __future__ import annotations

__all__ = ["OpenAIAdaptationClient", "AnthropicAdaptationClient"]


class OpenAIAdaptationClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        msg = resp.choices[0].message.content or ""
        usage = resp.usage
        return (
            msg,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )


class AnthropicAdaptationClient:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        return text, resp.usage.input_tokens, resp.usage.output_tokens
