"""
server.py — Main ASGI application for promptcache serve.

Routing:
  /v1/messages          → Anthropic API intercept
  /v1/chat/completions  → OpenAI API intercept
  /api/*                → Dashboard control REST API
  /dashboard            → Next.js static build
  /                     → redirect to /dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .control import db as control_db
from .control.router import router as control_router
from .forward import forward_request
from .intercept import intercept, write_back
from .state import broadcast_sse, init_state

_DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def _count_tokens(text: str) -> int:
    return max(1, len(text) // 4)


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
    serve_dashboard: bool = True,
) -> FastAPI:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "promptcache"
    init_state(cache_dir)

    app = FastAPI(title="promptcache", version="0.1.0", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        control_db.ensure_schema()

    app.include_router(control_router)

    if serve_dashboard and _DASHBOARD_DIR.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(_DASHBOARD_DIR), html=True),
            name="dashboard",
        )
    elif serve_dashboard:

        @app.get("/dashboard")
        async def dashboard_not_built():
            return JSONResponse(
                {
                    "error": (
                        "Dashboard not built. Run: ./scripts/build-dashboard.sh "
                        "or npm run build in promptcache-ui/frontend-next"
                    )
                },
                status_code=503,
            )

    @app.get("/")
    async def root():
        return RedirectResponse(url="/dashboard")

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
            return JSONResponse(
                content=result.cached_response,
                headers={
                    "X-Cache": result.hit_type,
                    "X-Cache-Similarity": str(round(result.similarity, 4)),
                    "X-Cache-Latency-Ms": str(round(result.latency_ms, 2)),
                },
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

        return upstream_response

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def proxy_other(request: Request, path: str):
        del path
        return await forward_request(request)

    return app


def main() -> None:
    import uvicorn

    app = create_app()
    print("\n  promptcache proxy  →  http://127.0.0.1:8080")
    print("  dashboard          →  http://127.0.0.1:8080/dashboard\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
