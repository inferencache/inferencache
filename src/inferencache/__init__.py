"""
inferencache — Semantic LLM response caching.

Drop-in caching for any LLM API call. Two checks before the request
ever leaves your machine: exact match (SHA-256, sub-ms) then semantic
match (embedding + cosine similarity). Zero external services required.

Quickstart::

    from inferencache import cache, CacheConfig

    @cache(config=CacheConfig(model="gpt-4o"))
    def ask(prompt: str) -> str:
        return openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

    # First call -> real API. Subsequent identical/similar calls -> cache.
    response = ask("What is the capital of France?")

Context manager usage::

    from inferencache import cache_context, CacheConfig

    with cache_context(prompt, config=CacheConfig(model="gpt-4o")) as ctx:
        if ctx.hit:
            return ctx.response
        response = call_real_api(prompt)
        ctx.store(response)
        return response
"""

from .api import CacheContext, CacheResult, cache, cache_context
from .engine import CacheConfig, CacheEngine

__all__ = [
    "cache",
    "cache_context",
    "CacheConfig",
    "CacheEngine",
    "CacheContext",
    "CacheResult",
]

__version__ = "0.1.0"
