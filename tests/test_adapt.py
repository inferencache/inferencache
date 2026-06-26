"""Unit tests for AdaptationEngine."""

from __future__ import annotations

import pytest

from inferencache.adapt import AdaptationEngine


class MockClient:
    def __init__(
        self,
        response: str = "",
        *,
        raises: Exception | None = None,
    ) -> None:
        self._response = response
        self._raises = raises
        self._model = "gpt-4o-mini"

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        if self._raises is not None:
            raise self._raises
        return self._response, 50, 25


def test_adapt_csv_to_tsv():
    mock_client = MockClient(response="def parse_tsv(path):\n    ...")
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt(
        cached_prompt="Write a Python function to parse a CSV file",
        cached_response="def parse_csv(path):\n    import csv\n    ...",
        new_prompt="Write a Python function to parse a TSV file",
    )
    assert result is not None
    assert "tsv" in result.response.lower()


def test_adapt_returns_none_on_sentinel():
    mock_client = MockClient(response="ADAPTATION_FAILED")
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt("prompt A", "response A", "prompt B")
    assert result is None


def test_adapt_returns_none_on_client_exception():
    mock_client = MockClient(raises=RuntimeError("timeout"))
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt("prompt A", "response A", "prompt B")
    assert result is None


def test_adapt_truncates_long_cached_response():
    long_response = "x" * 5000
    mock_client = MockClient(response="adapted")
    engine = AdaptationEngine(
        client=mock_client,
        model="gpt-4o-mini",
        max_cached_response_chars=100,
    )
    result = engine.adapt("a", long_response, "b")
    assert result is not None
    assert result.response == "adapted"
