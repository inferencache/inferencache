"""
api.py — Public API: @cache decorator and cache_context() context manager.

This is the only module users need to import. Everything else is internal.

    from promptcache import cache, CacheConfig

    config = CacheConfig(model="gpt-4o", threshold=0.88)

    @cache(config=config)
    def ask(prompt: str) -> str:
        return openai_client.chat.completions.create(...).choices[0].message.content

Streaming usage:

    @cache(config=config, streaming=True)
    def ask_stream(prompt: str):
        for chunk in openai_client.chat.completions.create(..., stream=True):
            yield chunk.choices[0].delta.content or ""

Context manager usage (when you don't own the function):

    with cache_context(prompt, config=config) as ctx:
        if ctx.hit:
            return ctx.response
        response = call_real_api(prompt)
        ctx.store(response)
        return response
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Generator, Iterator

from .engine import CacheConfig, CacheEngine, CacheResult

__all__ = [
    "cache",
    "cache_context",
    "CacheConfig",
    "CacheResult",
    "CacheContext",
]

# ---------------------------------------------------------------------------
# Module-level engine registry
# ---------------------------------------------------------------------------
# One CacheEngine per (cache_dir, model) pair so we never re-initialise
# ChromaDB for the same store. Keys are (str(cache_dir), model).

_engines: dict[tuple[str, str], CacheEngine] = {}


def _get_engine(config: CacheConfig) -> CacheEngine:
    key = (str(config.cache_dir), config.model)
    if key not in _engines:
        _engines[key] = CacheEngine(config)
    return _engines[key]


def _flush_engines() -> None:
    """Close all cached engines. Primarily for testing."""
    for engine in _engines.values():
        engine.close()
    _engines.clear()


# ---------------------------------------------------------------------------
# Context object
# ---------------------------------------------------------------------------


class CacheContext:
    """
    Holds lookup state during a cache_context() block.

    Attributes:
        hit (bool): True if a cached response was found.
        hit_type (str): 'exact', 'semantic', or 'miss'.
        response (str | None): Cached response if hit=True.
        similarity (float): Similarity score (1.0 for exact, 0.0 for miss).
        result (CacheResult): Full result object with latency etc.
    """

    def __init__(self, prompt: str, result: CacheResult, engine: CacheEngine) -> None:
        self._prompt = prompt
        self._engine = engine
        self.result = result
        self.hit = result.hit
        self.hit_type = result.hit_type
        self.response = result.response
        self.similarity = result.similarity

    def store(self, response: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Persist the response from a real API call.

        Call this inside the cache_context() block when ctx.hit is False
        and you have the real API response in hand.
        """
        self._engine.store(self._prompt, response, metadata=metadata)

    def stream(self) -> Iterator[str]:
        """
        Yield the cached response as a stream of chunks.

        Useful when your calling code expects a streaming interface:

            with cache_context(prompt, config=config) as ctx:
                if ctx.hit:
                    for chunk in ctx.stream():
                        yield chunk
                    return
                ...
        """
        if self.response is None:
            return iter([])
        return self._engine.stream_cached(self.response)


# ---------------------------------------------------------------------------
# @cache decorator
# ---------------------------------------------------------------------------


def cache(
    config: CacheConfig | None = None,
    *,
    prompt_arg: str = "prompt",
    streaming: bool = False,
    metadata_fn: Callable[..., dict[str, Any]] | None = None,
) -> Callable:
    """
    Decorator that wraps an LLM-calling function with semantic caching.

    Args:
        config: CacheConfig instance. If None, uses global defaults
                (cache_dir=~/.cache/promptcache, threshold=0.85).
        prompt_arg: Name of the function argument that contains the
                    prompt string. Default: 'prompt'.
        streaming: Set True if the wrapped function returns a Generator
                   (yields chunks). The decorator will collect the full
                   response on a cache miss, cache it, then reconstitute
                   as a generator on subsequent hits.
        metadata_fn: Optional callable that receives the same args/kwargs
                     as the wrapped function and returns a dict of extra
                     metadata to store with the cache entry.

    Examples::

        # Basic usage
        @cache(config=CacheConfig(model="gpt-4o"))
        def ask(prompt: str) -> str:
            return client.chat.completions.create(...)...

        # Streaming
        @cache(config=CacheConfig(model="gpt-4o"), streaming=True)
        def ask_stream(prompt: str):
            for chunk in client.chat.completions.create(..., stream=True):
                yield chunk.choices[0].delta.content or ""

        # Custom prompt extraction (e.g. prompt is nested in a dict)
        @cache(config=CacheConfig(model="claude-3-5-sonnet-20241022"),
               prompt_arg="messages")
        def ask(messages: list[dict]) -> str:
            ...
    """
    _config = config or CacheConfig()

    def decorator(fn: Callable) -> Callable:
        engine = _get_engine(_config)
        sig = inspect.signature(fn)

        def _extract_prompt(*args, **kwargs) -> str:
            # Try keyword arg first
            if prompt_arg in kwargs:
                val = kwargs[prompt_arg]
            else:
                # Try positional
                params = list(sig.parameters.keys())
                if prompt_arg in params:
                    idx = params.index(prompt_arg)
                    if idx < len(args):
                        val = args[idx]
                    else:
                        raise ValueError(
                            f"Could not find prompt argument '{prompt_arg}' "
                            f"in call to {fn.__name__}"
                        )
                else:
                    raise ValueError(
                        f"Function {fn.__name__} has no parameter named "
                        f"'{prompt_arg}'. Set prompt_arg= to the correct name."
                    )

            # If prompt_arg points to a list (chat messages), stringify it
            if isinstance(val, list):
                import json as _json

                return _json.dumps(val, ensure_ascii=False)
            return str(val)

        if streaming:
            # ── Streaming wrapper ──────────────────────────────────────────
            @functools.wraps(fn)
            def streaming_wrapper(*args, **kwargs) -> Iterator[str]:
                prompt = _extract_prompt(*args, **kwargs)
                result = engine.lookup(prompt)

                if result.hit:
                    yield from engine.stream_cached(result.response)
                    return

                # Cache miss: collect the real stream and cache the full response
                chunks = []
                for chunk in fn(*args, **kwargs):
                    chunks.append(chunk)
                    yield chunk

                full_response = "".join(chunks)
                metadata = metadata_fn(*args, **kwargs) if metadata_fn else None
                engine.store(prompt, full_response, metadata=metadata)

            return streaming_wrapper

        elif inspect.iscoroutinefunction(fn):
            # ── Async wrapper ──────────────────────────────────────────────
            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs) -> Any:
                prompt = _extract_prompt(*args, **kwargs)
                result = engine.lookup(prompt)

                if result.hit:
                    return result.response

                response = await fn(*args, **kwargs)
                response_str = str(response)
                metadata = metadata_fn(*args, **kwargs) if metadata_fn else None
                engine.store(prompt, response_str, metadata=metadata)
                return response

            return async_wrapper

        else:
            # ── Sync wrapper ───────────────────────────────────────────────
            @functools.wraps(fn)
            def sync_wrapper(*args, **kwargs) -> Any:
                prompt = _extract_prompt(*args, **kwargs)
                result = engine.lookup(prompt)

                if result.hit:
                    return result.response

                response = fn(*args, **kwargs)
                response_str = str(response)
                metadata = metadata_fn(*args, **kwargs) if metadata_fn else None
                engine.store(prompt, response_str, metadata=metadata)
                return response

            return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# cache_context() context manager
# ---------------------------------------------------------------------------


@contextmanager
def cache_context(
    prompt: str,
    config: CacheConfig | None = None,
) -> Generator[CacheContext, None, None]:
    """
    Context manager for caching when you can't use the decorator.

    Useful when the LLM call is buried in existing code you don't want
    to restructure, or when you need fine-grained control over what
    gets stored.

    Usage::

        config = CacheConfig(model="gpt-4o")

        with cache_context(prompt, config=config) as ctx:
            if ctx.hit:
                return ctx.response  # or: yield from ctx.stream()

            response = call_my_llm(prompt)
            ctx.store(response)
            return response

    The context yields a CacheContext. You are responsible for calling
    ctx.store(response) on a cache miss — the context manager does not
    do this automatically because it doesn't know your response.
    """
    _config = config or CacheConfig()
    engine = _get_engine(_config)
    result = engine.lookup(prompt)
    ctx = CacheContext(prompt=prompt, result=result, engine=engine)
    yield ctx
