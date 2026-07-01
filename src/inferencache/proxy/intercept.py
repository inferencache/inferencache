"""
intercept.py — Cache interception layer for proxied LLM requests.
"""

from __future__ import annotations

import json
import time
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine import CacheResult
from .state import get_engine_for_model


def _infer_provider(model: str, path: str) -> str:
    if "/messages" in path or model.startswith("claude"):
        return "anthropic"
    return "openai"


def _extract_prompt_anthropic(body: dict[str, Any]) -> str | None:
    parts: list[str] = []
    system = body.get("system")
    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else None


def _extract_prompt_openai(body: dict[str, Any]) -> str | None:
    parts: list[str] = []
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else None


def _extract_response_text_anthropic(body: dict[str, Any]) -> str | None:
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text")
    return None


def _extract_response_text_openai(body: dict[str, Any]) -> str | None:
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _build_cached_response_anthropic(
    cached_text: str, original_body: dict[str, Any]
) -> dict[str, Any]:
    model = original_body.get("model", "unknown")
    return {
        "id": "cache-hit",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": cached_text}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": len(cached_text.split()),
        },
    }


def _build_cached_response_openai(
    cached_text: str, original_body: dict[str, Any]
) -> dict[str, Any]:
    model = original_body.get("model", "unknown")
    return {
        "id": "cache-hit",
        "object": "chat.completion",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": cached_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def build_sse_stream_anthropic(
    cached_response: dict[str, Any],
) -> list[bytes]:
    """
    Convert a cached Anthropic JSON response into SSE bytes.

    Returns a list of b"data: {...}\\n\\n" byte strings in the exact order
    Anthropic's streaming API emits them.
    """
    msg_id = f"msg_{_uuid.uuid4().hex[:24]}"
    model = cached_response.get("model", "unknown")
    role = cached_response.get("role", "assistant")

    text = ""
    for block in cached_response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            break

    usage = cached_response.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", max(1, len(text) // 4))

    def sse(event_type: str, data: dict) -> bytes:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n".encode()

    return [
        sse("message_start", {
            "message": {
                "id": msg_id,
                "type": "message",
                "role": role,
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        }),
        sse("content_block_start", {
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        sse("ping", {}),
        sse("content_block_delta", {
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        }),
        sse("content_block_stop", {"index": 0}),
        sse("message_delta", {
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }),
        sse("message_stop", {}),
    ]


def build_sse_stream_openai(
    cached_response: dict[str, Any],
) -> list[bytes]:
    """Convert a cached OpenAI chat.completion response into SSE bytes."""
    completion_id = f"chatcmpl-{_uuid.uuid4().hex[:24]}"
    model = cached_response.get("model", "unknown")

    text = ""
    try:
        text = cached_response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        pass

    def sse(data: dict) -> bytes:
        payload = json.dumps(data)
        return f"data: {payload}\n\n".encode()

    base = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "model": model,
        "system_fingerprint": None,
    }

    return [
        sse({
            **base,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }),
        sse({
            **base,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }),
        sse({
            **base,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }),
        b"data: [DONE]\n\n",
    ]


def _extract_text_from_sse(raw: bytes, is_anthropic: bool) -> str:
    """
    Parse raw SSE bytes and reconstruct the full assistant text.

    For Anthropic: accumulate text_delta values from content_block_delta events.
    For OpenAI: accumulate delta.content values from chat.completion.chunk events.
    """
    text_parts: list[str] = []

    for line in raw.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:"):].strip()
        if data == b"[DONE]":
            break
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if is_anthropic:
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))
        else:
            try:
                content = event["choices"][0]["delta"].get("content") or ""
                text_parts.append(content)
            except (KeyError, IndexError, TypeError):
                pass

    return "".join(text_parts)


@dataclass
class InterceptResult:
    hit: bool
    hit_type: str
    cached_response: dict[str, Any] | None
    model: str
    prompt: str
    similarity: float
    latency_ms: float
    is_streaming: bool = False
    call_id: int | None = None
    best_similarity: float = 0.0
    matched_prompt: str | None = None


def intercept(path: str, body_bytes: bytes, cache_dir: Path) -> InterceptResult:
    """Run cache lookup for an incoming proxied LLM request."""
    del cache_dir  # engines use state.get_cache_dir()
    t0 = time.perf_counter()

    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return InterceptResult(
            hit=False, hit_type="miss", cached_response=None,
            model="unknown", prompt="", similarity=0.0, latency_ms=0.0,
        )

    is_anthropic = "/messages" in path
    model = body.get("model", "unknown")
    prompt = _extract_prompt_anthropic(body) if is_anthropic else _extract_prompt_openai(body)

    if not prompt:
        return InterceptResult(
            hit=False, hit_type="miss", cached_response=None,
            model=model, prompt="", similarity=0.0, latency_ms=0.0,
        )

    is_streaming = bool(body.get("stream", False))

    engine = get_engine_for_model(model, path)
    result: CacheResult = engine.lookup(prompt, endpoint="proxy")
    latency_ms = (time.perf_counter() - t0) * 1000

    if result.hit:
        cached_response = (
            _build_cached_response_anthropic(result.response or "", body)
            if is_anthropic
            else _build_cached_response_openai(result.response or "", body)
        )
        matched = None
        if result.hit_type == "semantic" and result.entry:
            matched = result.entry.prompt
        return InterceptResult(
            hit=True,
            hit_type=result.hit_type,
            cached_response=cached_response,
            model=model,
            prompt=prompt,
            similarity=result.similarity,
            latency_ms=latency_ms,
            is_streaming=is_streaming,
            call_id=result.call_id,
            best_similarity=result.best_similarity,
            matched_prompt=matched,
        )

    return InterceptResult(
        hit=False, hit_type="miss", cached_response=None,
        model=model, prompt=prompt, similarity=0.0,
        latency_ms=latency_ms, best_similarity=result.best_similarity,
        is_streaming=is_streaming,
    )


def write_back(
    path: str,
    prompt: str,
    response_bytes: bytes,
    cache_dir: Path,
    model: str,
) -> None:
    """Store upstream response text after a cache miss.

    Handles both JSON (non-streaming) and SSE (streaming) response bytes.
    """
    del cache_dir
    is_anthropic = "/messages" in path

    try:
        body = json.loads(response_bytes)
        text = (
            _extract_response_text_anthropic(body)
            if is_anthropic
            else _extract_response_text_openai(body)
        )
        if text:
            engine = get_engine_for_model(model, path)
            usage = body.get("usage") or {}
            tokens_input = usage.get("input_tokens") or usage.get("prompt_tokens")
            tokens_output = usage.get("output_tokens") or usage.get("completion_tokens")
            engine.store(
                prompt,
                text,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                endpoint="proxy",
            )
        return
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    text = _extract_text_from_sse(response_bytes, is_anthropic)
    if text:
        get_engine_for_model(model, path).store(
            prompt,
            text,
            tokens_input=None,
            tokens_output=None,
            endpoint="proxy",
        )
