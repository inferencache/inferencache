"""
tests/test_proxy_stream.py

Tests for streaming write-back and SSE hit reconstruction.
"""

from __future__ import annotations

import json

from inferencache.proxy.intercept import (
    InterceptResult,
    _extract_text_from_sse,
    build_sse_stream_anthropic,
    build_sse_stream_openai,
)


def _parse_sse_chunks(chunks: list[bytes]) -> list[dict]:
    events = []
    for chunk in chunks:
        for line in chunk.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data:"):
                data = line[5:].strip()
                if data and data != b"[DONE]":
                    events.append(json.loads(data))
    return events


def _make_cached_anthropic(text: str = "Hello world") -> dict:
    return {
        "id": "cache-hit",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-haiku-20241022",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def test_sse_anthropic_event_sequence():
    chunks = build_sse_stream_anthropic(_make_cached_anthropic())
    events = _parse_sse_chunks(chunks)
    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_delta" in types
    assert "message_stop" in types


def test_sse_anthropic_text_preserved():
    text = "The answer is 42."
    chunks = build_sse_stream_anthropic(_make_cached_anthropic(text))
    events = _parse_sse_chunks(chunks)
    deltas = [
        e["delta"]["text"]
        for e in events
        if e.get("type") == "content_block_delta"
    ]
    assert "".join(deltas) == text


def test_sse_anthropic_roundtrip():
    text = "Roundtrip test — special chars: ñ 你好 🎉"
    chunks = build_sse_stream_anthropic(_make_cached_anthropic(text))
    raw = b"".join(chunks)
    recovered = _extract_text_from_sse(raw, is_anthropic=True)
    assert recovered == text


def _make_cached_openai(text: str = "Hello world") -> dict:
    return {
        "id": "cache-hit",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_sse_openai_ends_with_done():
    chunks = build_sse_stream_openai(_make_cached_openai())
    assert b"[DONE]" in b"".join(chunks)


def test_sse_openai_roundtrip():
    text = "OpenAI roundtrip."
    chunks = build_sse_stream_openai(_make_cached_openai(text))
    raw = b"".join(chunks)
    recovered = _extract_text_from_sse(raw, is_anthropic=False)
    assert recovered == text


def test_extract_ignores_non_data_lines():
    raw = b"event: ping\ndata: {}\n\ndata: [DONE]\n\n"
    result = _extract_text_from_sse(raw, is_anthropic=True)
    assert result == ""


def test_extract_handles_malformed_json_gracefully():
    raw = b"data: {bad json}\n\ndata: [DONE]\n\n"
    result = _extract_text_from_sse(raw, is_anthropic=True)
    assert result == ""


def test_extract_multi_delta_anthropic():
    deltas = ["Hello ", "world", "!"]
    lines = []
    for d in deltas:
        event = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": d},
        }
        lines.append(f"data: {json.dumps(event)}\n\n".encode())
    lines.append(b"data: [DONE]\n\n")
    raw = b"".join(lines)
    result = _extract_text_from_sse(raw, is_anthropic=True)
    assert result == "Hello world!"


def test_intercept_result_is_streaming_default():
    r = InterceptResult(
        hit=False, hit_type="miss", cached_response=None,
        model="m", prompt="p", similarity=0.0, latency_ms=0.0,
    )
    assert r.is_streaming is False


def test_intercept_result_is_streaming_set():
    r = InterceptResult(
        hit=True, hit_type="exact", cached_response={},
        model="m", prompt="p", similarity=1.0, latency_ms=1.0,
        is_streaming=True,
    )
    assert r.is_streaming is True
