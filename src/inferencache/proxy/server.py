"""
server.py — Main ASGI application for inferencache serve.

Routing:
  /v1/messages          → Anthropic API intercept
  /v1/chat/completions  → OpenAI API intercept
  /api/*                → Dashboard control REST API
  /                     → Next.js static export (landing + dashboard)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from inferencache.paths import default_cache_dir

from .control import db as control_db
from .control.router import router as control_router
from .forward import forward_and_capture, forward_request
from .intercept import (
    build_sse_stream_anthropic,
    build_sse_stream_openai,
    intercept,
    write_back,
)
from .state import broadcast_sse, init_state, prune_expired_on_startup

_log = logging.getLogger(__name__)

_SITE_DIR = Path(__file__).parent / "site"


def site_dir() -> Path:
    """Return the embedded static site directory."""
    return _SITE_DIR


def site_is_built(site_dir: Path | None = None) -> bool:
    """True when a Next.js static export is present (not just an empty placeholder dir)."""
    root = site_dir if site_dir is not None else _SITE_DIR
    return (root / "index.html").is_file()


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cache_headers(result) -> dict[str, str]:
    """Headers that prove intercept() ran, on both hits and misses."""
    return {
        "X-Cache": result.hit_type,
        "X-Cache-Similarity": str(
            round(result.similarity if result.hit else result.best_similarity, 4)
        ),
        "X-Cache-Latency-Ms": str(round(result.latency_ms, 2)),
    }


def _with_cache_headers(response: Response, cache_headers: dict[str, str]) -> Response:
    """Merge cache verdict headers into a forwarded upstream response."""
    merged = {k: v for k, v in response.headers.items()}
    merged.update(cache_headers)
    return Response(
        content=response.body,
        status_code=response.status_code,
        headers=merged,
        media_type=response.media_type,
    )


async def _emit_proxy_call(result, path: str) -> None:
    """Broadcast proxy intercept to SSE so the Live tab updates for agent traffic."""
    preview = result.prompt[:80] if result.prompt else ""
    response_preview = ""
    if result.cached_response:
        if "/messages" in path:
            for block in result.cached_response.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    response_preview = (block.get("text") or "")[:120]
                    break
        else:
            try:
                response_preview = (
                    result.cached_response["choices"][0]["message"]["content"] or ""
                )[:120]
            except (KeyError, IndexError, TypeError):
                pass

    event = {
        "event_type": "call",
        "run_id": "proxy",
        "prompt_index": 0,
        "total_prompts": 0,
        "prompt_preview": preview,
        "hit": result.hit,
        "hit_type": result.hit_type,
        "similarity": result.similarity,
        "best_similarity": result.best_similarity,
        "latency_ms": round(result.latency_ms, 1),
        "tokens_used": _count_tokens(response_preview) if result.hit else 0,
        "cost_usd": 0.0 if result.hit else None,
        "model": result.model,
        "response_preview": response_preview,
        "call_id": result.call_id,
        "endpoint": "proxy",
        "session_id": "proxy",
        "matched_prompt": result.matched_prompt,
        "tier1_hit": result.hit,
    }
    await broadcast_sse(json.dumps(event))


def create_app(
    cache_dir: Path | None = None,
    serve_site: bool = True,
) -> FastAPI:
    if cache_dir is None:
        cache_dir = default_cache_dir()
    init_state(cache_dir)

    app = FastAPI(title="inferencache", version="0.1.0", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        control_db.ensure_schema()
        pruned = prune_expired_on_startup()
        if pruned:
            _log.info("Pruned %d expired cache entries on startup", pruned)

    app.include_router(control_router)

    @app.api_route("/v1/messages", methods=["POST"], include_in_schema=False)
    @app.api_route("/v1/chat/completions", methods=["POST"], include_in_schema=False)
    async def proxy_llm(request: Request):
        body_bytes = await request.body()
        result = intercept(
            path=request.url.path,
            body_bytes=body_bytes,
            cache_dir=cache_dir,
        )

        await _emit_proxy_call(result, request.url.path)

        if result.hit:
            hit_headers = _cache_headers(result)

            if result.is_streaming:
                is_anthropic = "/messages" in request.url.path
                if is_anthropic:
                    chunks = build_sse_stream_anthropic(result.cached_response)
                else:
                    chunks = build_sse_stream_openai(result.cached_response)

                async def _cached_stream():
                    for chunk in chunks:
                        yield chunk

                return StreamingResponse(
                    _cached_stream(),
                    status_code=200,
                    headers=hit_headers,
                    media_type="text/event-stream",
                )

            return JSONResponse(
                content=result.cached_response,
                headers=hit_headers,
            )

        miss_headers = _cache_headers(result)

        if result.is_streaming:

            async def _on_complete(raw: bytes) -> None:
                write_back(
                    path=request.url.path,
                    prompt=result.prompt,
                    response_bytes=raw,
                    cache_dir=cache_dir,
                    model=result.model,
                )

            stream_gen = await forward_and_capture(request, body_bytes, _on_complete)
            return StreamingResponse(
                stream_gen,
                media_type="text/event-stream",
                headers=miss_headers,
            )

        upstream_response = await forward_request(request, body=body_bytes)

        if (
            upstream_response.status_code == 200
            and hasattr(upstream_response, "body")
            and upstream_response.body
        ):
            write_back(
                path=request.url.path,
                prompt=result.prompt,
                response_bytes=upstream_response.body,
                cache_dir=cache_dir,
                model=result.model,
            )

        return _with_cache_headers(upstream_response, miss_headers)

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def proxy_other(request: Request, path: str):
        del path
        return await forward_request(request)

    if serve_site and site_is_built():
        app.mount(
            "/",
            StaticFiles(directory=str(_SITE_DIR), html=True),
            name="site",
        )
    elif serve_site:

        @app.get("/")
        async def site_not_built():
            return JSONResponse(
                {
                    "error": (
                        "Site not built. Run: ./scripts/build-dashboard.sh "
                        "or npm run build in inferencache-dashboard/frontend-next"
                    )
                },
                status_code=503,
            )

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    print("\n  inferencache proxy  →  http://127.0.0.1:8080")
    print("  landing             →  http://127.0.0.1:8080/")
    print("  dashboard           →  http://127.0.0.1:8080/dashboard/\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
