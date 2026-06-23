"""
forward.py — Forward requests to upstream Anthropic/OpenAI APIs.

On a cache miss, the proxy passes the request through unchanged and
returns the upstream response to the caller.
"""

from __future__ import annotations

import httpx
from fastapi import Request
from fastapi.responses import Response, StreamingResponse

ANTHROPIC_BASE = "https://api.anthropic.com"
OPENAI_BASE = "https://api.openai.com"

_STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}

_STRIP_RESPONSE_HEADERS = {
    "transfer-encoding",
    "connection",
    "keep-alive",
}

_client = httpx.AsyncClient(timeout=120.0)


def _upstream_base(path: str) -> str:
    if path.startswith("/v1/messages"):
        return ANTHROPIC_BASE
    return OPENAI_BASE


async def forward_request(request: Request, body: bytes | None = None) -> Response:
    """Forward request to upstream API; handles streaming and non-streaming."""
    base = _upstream_base(request.url.path)
    url = f"{base}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in _STRIP_REQUEST_HEADERS
    }
    if body is None:
        body = await request.body()

    upstream = await _client.send(
        httpx.Request(method=request.method, url=url, headers=headers, content=body),
        stream=True,
    )

    content_type = upstream.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    response_headers = {
        k: v for k, v in upstream.headers.items() if k.lower() not in _STRIP_RESPONSE_HEADERS
    }

    if is_stream:

        async def _stream():
            async for chunk in upstream.aiter_bytes():
                yield chunk

        return StreamingResponse(
            _stream(),
            status_code=upstream.status_code,
            headers=response_headers,
            media_type="text/event-stream",
        )

    content = await upstream.aread()
    return Response(content=content, status_code=upstream.status_code, headers=response_headers)
