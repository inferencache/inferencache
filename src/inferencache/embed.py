"""
embed.py — Embedding interface and implementations.

The Embedder protocol defines a single-method contract so swapping
providers (sentence-transformers → OpenAI → custom) is a one-liner.

Default model: BAAI/bge-small-en-v1.5 (384d, higher MTEB than MiniLM).
Power model:   Qwen3-Embedding-0.6B via Qwen3Embedder (1024d, best-in-class).

All sentence-transformers embedders are lazy: the model loads on the
first call to embed(), not at import time.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "Qwen3Embedder",
    "get_embedder",
    "get_default_embedder",
]

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@runtime_checkable
class Embedder(Protocol):
    """
    Minimal interface all embedder implementations must satisfy.

    Any callable object that has an embed(text) -> list[float] method
    qualifies — no subclassing required.
    """

    def embed(self, text: str) -> list[float]:
        """Return a fixed-length float vector for the given text."""
        ...

    def model_id(self) -> str:
        """
        Return a stable string identifying this embedder + model.

        Used as part of the Qdrant collection name so embeddings
        produced by different models never collide in the same store.
        """
        ...


class SentenceTransformerEmbedder:
    """
    Default embedder backed by sentence-transformers.

    The underlying SentenceTransformer model is not loaded until the
    first call to embed(). Subsequent calls reuse the same instance.

    Args:
        model_name: Any model name accepted by sentence-transformers.
                    Defaults to 'BAAI/bge-small-en-v1.5' (384d,
                    higher MTEB score than MiniLM at the same cost).
    """

    dimension: int = 384

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        """Embed text, loading the model on first call."""
        model = self._get_model()
        vector = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        return vector.tolist()

    def model_id(self) -> str:
        """
        Stable identifier used to namespace Qdrant collections.

        We hash the model name so collection names stay filesystem-safe
        regardless of what model string is passed in.
        """
        slug = self._model_name.replace("/", "-").replace(" ", "_")
        return f"st-{slug}"

    def _get_model(self):
        """Load and cache the SentenceTransformer model (lazy)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for the default embedder. "
                    "Install it with: pip install inferencache[embed]"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"SentenceTransformerEmbedder(model={self._model_name!r}, {loaded})"


class OpenAIEmbedder:
    """
    Optional embedder backed by the OpenAI embeddings API.

    Args:
        model: OpenAI embedding model name. Defaults to
               'text-embedding-3-small' (best cost/quality tradeoff).
        api_key: OpenAI API key. If None, falls back to the
                 OPENAI_API_KEY environment variable.
        dimensions: Output vector size. Defaults to 1536 for
                    text-embedding-3-small.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._client = None

    @property
    def dimension(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model=self._model,
            dimensions=self._dimensions,
        )
        return response.data[0].embedding

    def model_id(self) -> str:
        slug = self._model.replace("/", "-").replace(" ", "_")
        return f"openai-{slug}"

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAIEmbedder. "
                    "Install it with: pip install openai"
                ) from exc
            kwargs = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = OpenAI(**kwargs)
        return self._client

    def __repr__(self) -> str:
        return f"OpenAIEmbedder(model={self._model!r})"


class Qwen3Embedder:
    """
    High-accuracy embedder using Qwen3-Embedding-0.6B (Apache 2.0).

    Best MTEB score in the sub-1B parameter class. Requires ~1.2 GB RAM.
    Use when semantic accuracy matters more than embedding latency.

    Args:
        model_name: Defaults to 'Alibaba-NLP/gte-Qwen3-0.6B-embedding'.
    """

    dimension: int = 1024

    def __init__(self, model_name: str = "Alibaba-NLP/gte-Qwen3-0.6B-embedding") -> None:
        self._model_name = model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for Qwen3Embedder. "
                    "Install it with: pip install inferencache[embed]"
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def model_id(self) -> str:
        return "qwen3-0.6b"

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "not loaded"
        return f"Qwen3Embedder(model={self._model_name!r}, {loaded})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_PRESETS = {
    "fast":     lambda: SentenceTransformerEmbedder("all-MiniLM-L6-v2"),
    "balanced": lambda: SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5"),
    "accurate": lambda: Qwen3Embedder(),
}


def get_embedder(preset: str = "balanced") -> SentenceTransformerEmbedder | OpenAIEmbedder | Qwen3Embedder:
    """
    Return an embedder instance by preset name.

    Presets:
        'fast'     — all-MiniLM-L6-v2, 384d, fastest CPU inference
        'balanced' — bge-small-en-v1.5, 384d, default, best cost/quality
        'accurate' — Qwen3-Embedding-0.6B, 1024d, highest MTEB score

    Raises:
        ValueError: If the preset name is not recognised.
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose: {list(_PRESETS)}")
    return _PRESETS[preset]()


# ---------------------------------------------------------------------------
# Module-level singleton (backward-compatible default)
# ---------------------------------------------------------------------------

_default_embedder: SentenceTransformerEmbedder | None = None


def get_default_embedder() -> SentenceTransformerEmbedder:
    """
    Return the module-level default embedder instance (bge-small-en-v1.5).

    The same instance is reused across the process so the model is
    only loaded once even if multiple CacheConfig objects are created
    without an explicit embedder.
    """
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformerEmbedder()
    return _default_embedder
