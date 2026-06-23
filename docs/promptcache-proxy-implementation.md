# promptcache — Proxy Restructure Implementation

## Goal

Restructure two repos into a clean open source product. One command starts everything.
User experience after this is done:

```bash
pip install promptcache
promptcache serve
# proxy:     http://localhost:8080  (Anthropic + OpenAI compatible)
# dashboard: http://localhost:8080/dashboard  (auto-opens in browser)
```

Agent config change (Cursor, Claude Code, Codex, etc.):
```
ANTHROPIC_BASE_URL=http://localhost:8080
```

---

## Repo Structure After This Work

```
promptcache/                        ← existing library repo (mostly unchanged)
  src/promptcache/
    __init__.py
    api.py
    engine.py
    store.py
    embed.py
    cli.py                          ← ADD: `promptcache serve` command here
    proxy/                          ← NEW package
      __init__.py
      server.py                     ← ASGI app (FastAPI), all routing
      intercept.py                  ← cache logic wired to incoming requests
      forward.py                    ← httpx client that forwards misses
      control.py                    ← /api/* REST endpoints for dashboard
      dashboard/                    ← NEW: Next.js static build output lives here
        (populated by build step, see Section 5)
    mcp/
      ...                           ← unchanged
  pyproject.toml                    ← ADD proxy deps, ADD `promptcache serve` script

promptcache-dashboard/              ← existing dashboard repo
  frontend-next/
    next.config.js                  ← CHANGE: output: 'export', basePath: '/dashboard'
    src/lib/api.ts                  ← CHANGE: BASE_URL points to /api (same origin)
    ...                             ← everything else unchanged
  backend/
    main.py                         ← KEEP for local dev only, not used in production
```

---

## Section 1 — `pyproject.toml` changes

In `promptcache/pyproject.toml`, make these changes:

**Add proxy dependencies** to `[project.dependencies]`:
```toml
"fastapi>=0.110.0",
"uvicorn[standard]>=0.29.0",
"httpx>=0.27.0",
"python-multipart>=0.0.9",
```

**Add the `serve` CLI entrypoint** in `[project.scripts]`:
```toml
[project.scripts]
promptcache = "promptcache.cli:main"
promptcache-serve = "promptcache.proxy.server:main"
```

Note: keep `promptcache` pointing at the existing `cli.py:main`. The proxy server
gets its own entrypoint. The existing `promptcache stats`, `promptcache clear`,
`promptcache config` commands are unchanged.

**Add `serve` subcommand to `cli.py`** — see Section 2.

---

## Section 2 — `cli.py` changes

Add a `serve` subcommand to the existing `build_parser()` function in `src/promptcache/cli.py`.

Add this block inside `build_parser()` after the existing subcommands:

```python
# ── serve ──
serve_p = sub.add_parser("serve", help="Start the proxy server + dashboard")
serve_p.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Port to bind (default: 8080)",
)
serve_p.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host to bind (default: 127.0.0.1)",
)
serve_p.add_argument(
    "--no-browser",
    action="store_true",
    help="Don't auto-open the dashboard in a browser",
)
serve_p.add_argument(
    "--no-dashboard",
    action="store_true",
    help="Start proxy only, no dashboard",
)
serve_p.set_defaults(func=cmd_serve)
```

Add the `cmd_serve` function near the top of `cli.py` with the other `cmd_*` functions:

```python
def cmd_serve(args: argparse.Namespace) -> int:
    """Start proxy + dashboard."""
    import webbrowser
    import threading
    import uvicorn
    from .proxy.server import create_app

    app = create_app(
        cache_dir=Path(args.cache_dir),
        serve_dashboard=not args.no_dashboard,
    )

    dashboard_url = f"http://{args.host}:{args.port}/dashboard"
    proxy_url = f"http://{args.host}:{args.port}"

    print(f"\n  promptcache proxy  →  {proxy_url}")
    if not args.no_dashboard:
        print(f"  dashboard          →  {dashboard_url}\n")

    if not args.no_browser and not args.no_dashboard:
        # Open after a short delay to let the server bind
        def _open():
            import time
            time.sleep(1.2)
            webbrowser.open(dashboard_url)
        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0
```

---

## Section 3 — Proxy package: `src/promptcache/proxy/`

Create all files below. This is a new package.

---

### `src/promptcache/proxy/__init__.py`

```python
"""promptcache proxy — local ASGI server that intercepts LLM API calls."""
```

---

### `src/promptcache/proxy/forward.py`

Handles forwarding cache misses to the real API.

```python
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
OPENAI_BASE    = "https://api.openai.com"

# Headers we strip before forwarding (hop-by-hop or proxy-internal)
_STRIP_REQUEST_HEADERS = {
    "host", "content-length", "transfer-encoding",
    "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "upgrade",
}

_STRIP_RESPONSE_HEADERS = {
    "transfer-encoding", "connection", "keep-alive",
}

_client = httpx.AsyncClient(timeout=120.0)


def _upstream_base(path: str) -> str:
    """Pick Anthropic or OpenAI base URL from the request path."""
    if path.startswith("/v1/messages"):
        return ANTHROPIC_BASE
    return OPENAI_BASE


async def forward_request(request: Request) -> Response:
    """
    Forward `request` to the appropriate upstream API and return the response.
    Handles both streaming (SSE) and non-streaming responses.
    """
    base = _upstream_base(request.url.path)
    url  = f"{base}{request.url.path}"
    if request.url.query:
        url += f"?{request.url.query}"

    # Build forwarded headers
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _STRIP_REQUEST_HEADERS
    }

    body = await request.body()

    upstream = await _client.send(
        httpx.Request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        ),
        stream=True,
    )

    # Determine if this is a streaming response
    content_type = upstream.headers.get("content-type", "")
    is_stream = "text/event-stream" in content_type

    response_headers = {
        k: v for k, v in upstream.headers.items()
        if k.lower() not in _STRIP_RESPONSE_HEADERS
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
    else:
        content = await upstream.aread()
        return Response(
            content=content,
            status_code=upstream.status_code,
            headers=response_headers,
        )
```

---

### `src/promptcache/proxy/intercept.py`

Cache lookup + write-back logic, wired to incoming HTTP requests.

```python
"""
intercept.py — Cache interception layer.

Extracts the prompt from an incoming Anthropic or OpenAI request body,
runs it through CacheEngine, and returns either a cached response or
a signal to forward to upstream.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..engine import CacheConfig, CacheEngine, CacheResult

# One engine per (cache_dir, model) pair — same pattern as api.py
_engines: dict[tuple[str, str], CacheEngine] = {}


def _get_engine(cache_dir: Path, model: str) -> CacheEngine:
    key = (str(cache_dir), model)
    if key not in _engines:
        config = CacheConfig(
            cache_dir=cache_dir,
            model=model,
            provider=_infer_provider(model),
        )
        _engines[key] = CacheEngine(config)
    return _engines[key]


def _infer_provider(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def _extract_prompt_anthropic(body: dict) -> str | None:
    """
    Extract a cache key string from an Anthropic /v1/messages request body.
    Concatenates system prompt + all message contents into one string.
    This is the semantic unit we cache on.
    """
    parts = []

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


def _extract_prompt_openai(body: dict) -> str | None:
    """
    Extract a cache key string from an OpenAI /v1/chat/completions request body.
    """
    parts = []
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return "\n".join(parts) if parts else None


def _extract_response_text_anthropic(body: dict) -> str | None:
    """Extract the assistant text from an Anthropic response body."""
    for block in body.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text")
    return None


def _extract_response_text_openai(body: dict) -> str | None:
    """Extract the assistant text from an OpenAI response body."""
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def _build_cached_response_anthropic(cached_text: str, original_body: dict) -> dict:
    """Wrap a cached text string in a valid Anthropic response envelope."""
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


def _build_cached_response_openai(cached_text: str, original_body: dict) -> dict:
    """Wrap a cached text string in a valid OpenAI response envelope."""
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


@dataclass
class InterceptResult:
    hit: bool
    hit_type: str                    # 'exact' | 'semantic' | 'miss'
    cached_response: dict | None     # JSON body to return, or None on miss
    model: str
    prompt: str
    similarity: float
    latency_ms: float


def intercept(
    path: str,
    body_bytes: bytes,
    cache_dir: Path,
) -> InterceptResult:
    """
    Main interception entry point. Called for every proxied LLM request.

    Returns InterceptResult. If result.hit is True, return result.cached_response
    directly. If False, forward to upstream then call write_back().
    """
    t0 = time.perf_counter()

    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return InterceptResult(
            hit=False, hit_type="miss", cached_response=None,
            model="unknown", prompt="", similarity=0.0,
            latency_ms=0.0,
        )

    is_anthropic = "/messages" in path
    model = body.get("model", "unknown")

    prompt = (
        _extract_prompt_anthropic(body) if is_anthropic
        else _extract_prompt_openai(body)
    )

    if not prompt:
        return InterceptResult(
            hit=False, hit_type="miss", cached_response=None,
            model=model, prompt="", similarity=0.0, latency_ms=0.0,
        )

    # Don't cache streaming requests for now — forward them directly
    # Streaming cache support can be added in a future iteration
    if body.get("stream", False):
        return InterceptResult(
            hit=False, hit_type="miss", cached_response=None,
            model=model, prompt=prompt, similarity=0.0, latency_ms=0.0,
        )

    engine = _get_engine(cache_dir, model)
    result: CacheResult = engine.lookup(prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    if result.hit:
        cached_response = (
            _build_cached_response_anthropic(result.response, body) if is_anthropic
            else _build_cached_response_openai(result.response, body)
        )
        return InterceptResult(
            hit=True,
            hit_type=result.hit_type,
            cached_response=cached_response,
            model=model,
            prompt=prompt,
            similarity=result.similarity,
            latency_ms=latency_ms,
        )

    return InterceptResult(
        hit=False, hit_type="miss", cached_response=None,
        model=model, prompt=prompt, similarity=0.0, latency_ms=latency_ms,
    )


def write_back(
    path: str,
    prompt: str,
    response_bytes: bytes,
    cache_dir: Path,
    model: str,
) -> None:
    """
    Called after a successful upstream response on a cache miss.
    Stores the response text in the cache.
    """
    try:
        body = json.loads(response_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return

    is_anthropic = "/messages" in path
    text = (
        _extract_response_text_anthropic(body) if is_anthropic
        else _extract_response_text_openai(body)
    )

    if not text:
        return

    engine = _get_engine(cache_dir, model)
    engine.store(prompt, text)
```

---

### `src/promptcache/proxy/control.py`

REST API the dashboard consumes. Replaces `backend/main.py`.

```python
"""
control.py — /api/* REST endpoints for the dashboard.

Mounted at /api on the main ASGI app. The dashboard frontend calls these
instead of the old FastAPI backend. All state lives in the shared CacheEngine
instances from intercept.py.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from ..engine import CacheConfig, CacheEngine
from ..store import CacheStore
from .intercept import _engines, _get_engine

router = APIRouter(prefix="/api")

_cache_dir: Path | None = None


def init(cache_dir: Path) -> None:
    """Called by server.py at startup to set the shared cache directory."""
    global _cache_dir
    _cache_dir = cache_dir


def _default_engine() -> CacheEngine | None:
    """Return any active engine, or None if none have been used yet."""
    if not _engines:
        return None
    return next(iter(_engines.values()))


# ── Models ────────────────────────────────────────────────────────────────────

class ThresholdUpdate(BaseModel):
    model: str = "unknown"
    threshold: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    return {
        "status": "ok",
        "cache_dir": str(_cache_dir),
        "active_models": list({k[1] for k in _engines.keys()}),
    }


@router.get("/stats")
async def stats(model: str | None = None):
    """
    Return aggregate cache stats. If model is specified, filter to that model.
    If no engines are active yet, return zeroed stats.
    """
    if not _engines:
        return {
            "total_entries": 0,
            "total_hits": 0,
            "exact_hits": 0,
            "semantic_hits": 0,
            "hit_rate": 0.0,
            "top_entries": [],
            "active_models": [],
        }

    # If model specified, get that engine; otherwise use first available
    if model:
        engine = _get_engine(_cache_dir, model)
    else:
        engine = _default_engine()

    s = engine.cache_store.stats(top_n=10)
    return {
        "total_entries": s.total_entries,
        "total_hits": s.total_hits,
        "exact_hits": s.exact_hits,
        "semantic_hits": s.semantic_hits,
        "hit_rate": s.hit_rate,
        "top_entries": s.top_entries,
        "active_models": list({k[1] for k in _engines.keys()}),
    }


@router.get("/calls")
async def get_calls(limit: int = 100, model: str | None = None):
    """
    Return recent call events from the calls table.
    Used by the dashboard Live tab call log.
    """
    engine = (
        _get_engine(_cache_dir, model) if model
        else _default_engine()
    )
    if not engine:
        return {"data": []}

    import sqlite3
    db_path = engine.cache_store._db_path
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, prompt_hash, model, hit_type, similarity,
               latency_ms, tokens_input, tokens_output, cost_usd,
               false_positive, timestamp
        FROM calls
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return {"data": [dict(r) for r in rows]}


@router.post("/clear")
async def clear_cache(model: str | None = None):
    """Flush cached entries. Optionally scoped to a model."""
    if not _engines:
        return {"deleted": 0}
    engine = (
        _get_engine(_cache_dir, model) if model
        else _default_engine()
    )
    deleted = engine.cache_store.clear(model=model)
    return {"deleted": deleted}


@router.post("/threshold")
async def set_threshold(update: ThresholdUpdate):
    """Update similarity threshold for a model's engine."""
    engine = _get_engine(_cache_dir, update.model)
    engine.set_threshold(update.threshold)
    return {"threshold": update.threshold, "model": update.model}


@router.get("/analytics/hit-rate")
async def hit_rate(model: str = "unknown", window_hours: int = 24):
    from ..analytics import CacheAnalytics
    engine = _get_engine(_cache_dir, model)
    analytics = CacheAnalytics(engine.cache_store._db_path)
    data = analytics.hit_rate_over_time(model, window_hours, bucket_minutes=30)
    return {"data": data}


@router.get("/analytics/cost-saved")
async def cost_saved(model: str = "unknown", window_hours: int = 24):
    from ..analytics import CacheAnalytics
    engine = _get_engine(_cache_dir, model)
    analytics = CacheAnalytics(engine.cache_store._db_path)
    data = analytics.cost_saved_cumulative(model, window_hours)
    return {"data": data}
```

---

### `src/promptcache/proxy/server.py`

The ASGI app. Routes LLM calls through the cache, serves dashboard as static files, mounts the control API.

```python
"""
server.py — Main ASGI application.

Routing:
  /v1/messages          → Anthropic API intercept
  /v1/chat/completions  → OpenAI API intercept
  /api/*                → Dashboard control REST API
  /dashboard            → Next.js static build (served as files)
  /dashboard/*          → Next.js static build (served as files)
  /                     → redirect to /dashboard

All routes except /api/* and /dashboard are treated as LLM proxy routes.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import control
from .forward import forward_request
from .intercept import intercept, write_back


_PROXY_PATHS = {"/v1/messages", "/v1/chat/completions"}

# Path to the compiled Next.js static export
_DASHBOARD_DIR = Path(__file__).parent / "dashboard"


def create_app(
    cache_dir: Path | None = None,
    serve_dashboard: bool = True,
) -> FastAPI:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "promptcache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="promptcache", version="0.1.0", docs_url=None, redoc_url=None)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Init control API ──────────────────────────────────────────────────────
    control.init(cache_dir)
    app.include_router(control.router)

    # ── Dashboard static files ────────────────────────────────────────────────
    if serve_dashboard and _DASHBOARD_DIR.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(_DASHBOARD_DIR), html=True),
            name="dashboard",
        )
    elif serve_dashboard and not _DASHBOARD_DIR.exists():
        @app.get("/dashboard")
        async def dashboard_not_built():
            return JSONResponse(
                {"error": "Dashboard not built. Run: cd dashboard && npm run build"},
                status_code=503,
            )

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/")
    async def root():
        return RedirectResponse(url="/dashboard")

    # ── LLM proxy routes ──────────────────────────────────────────────────────
    @app.api_route(
        "/v1/messages",
        methods=["POST"],
        include_in_schema=False,
    )
    @app.api_route(
        "/v1/chat/completions",
        methods=["POST"],
        include_in_schema=False,
    )
    async def proxy_llm(request: Request):
        body_bytes = await request.body()

        result = intercept(
            path=request.url.path,
            body_bytes=body_bytes,
            cache_dir=cache_dir,
        )

        if result.hit:
            # Return cached response immediately — no upstream call
            return JSONResponse(
                content=result.cached_response,
                headers={
                    "X-Cache": result.hit_type,
                    "X-Cache-Similarity": str(round(result.similarity, 4)),
                    "X-Cache-Latency-Ms": str(round(result.latency_ms, 2)),
                },
            )

        # Cache miss — forward to upstream
        upstream_response = await forward_request(request)

        # Write back to cache if it's a clean JSON response
        if (
            hasattr(upstream_response, "body")
            and upstream_response.status_code == 200
        ):
            write_back(
                path=request.url.path,
                prompt=result.prompt,
                response_bytes=upstream_response.body,
                cache_dir=cache_dir,
                model=result.model,
            )

        return upstream_response

    # ── Catch-all: also proxy any other /v1/* paths ───────────────────────────
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def proxy_other(request: Request, path: str):
        return await forward_request(request)

    return app


def main() -> None:
    """Entry point for `promptcache-serve` script."""
    import uvicorn
    app = create_app()
    print("\n  promptcache proxy  →  http://127.0.0.1:8080")
    print("  dashboard          →  http://127.0.0.1:8080/dashboard\n")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")


if __name__ == "__main__":
    main()
```

---

## Section 4 — Dashboard frontend changes

Two changes to `promptcache-dashboard/frontend-next/`:

### 4a — `next.config.js`

Replace existing config with:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',          // static HTML/JS/CSS export
  basePath: '/dashboard',    // matches where the proxy serves it
  trailingSlash: true,       // required for static export routing
  images: {
    unoptimized: true,       // required for static export
  },
}

module.exports = nextConfig
```

### 4b — `src/lib/api.ts`

Change the `BASE` URL constant so the dashboard calls the proxy's `/api` instead of the old backend port.

Find this line (approximately):
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
```

Replace with:
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";
```

This makes all dashboard API calls go to `/api/*` on the same origin as the dashboard itself — no CORS, no hardcoded ports.

Also update `.env.local` (and remove it from `.gitignore` if it's there — it's no longer secret):
```
NEXT_PUBLIC_API_URL=/api
```

---

## Section 5 — Build step and packaging

After making all code changes, the Next.js dashboard needs to be built and the output copied into the Python package so it gets included in the pip install.

### Build the dashboard

```bash
cd promptcache-dashboard/frontend-next
npm install
npm run build
# Output goes to: out/
```

### Copy output into Python package

```bash
cp -r promptcache-dashboard/frontend-next/out/* \
      promptcache/src/promptcache/proxy/dashboard/
```

Create the dashboard directory first:
```bash
mkdir -p promptcache/src/promptcache/proxy/dashboard
```

### Include it in the package

In `promptcache/pyproject.toml`, ensure package data is included:

```toml
[tool.setuptools.package-data]
"promptcache.proxy" = ["dashboard/**/*"]
```

Or if using `[tool.setuptools.packages.find]`:
```toml
[tool.setuptools]
package-dir = {"" = "src"}
include-package-data = true
```

And add a `MANIFEST.in` at the repo root:
```
recursive-include src/promptcache/proxy/dashboard *
```

---

## Section 6 — Response body access fix

There's a subtle issue in `server.py`: FastAPI's `StreamingResponse` and some `Response` objects don't expose `.body` after forwarding. The `write_back()` call needs the raw bytes.

In `forward.py`, for non-streaming responses, store the content before returning:

The current `forward_request` function already reads `content = await upstream.aread()` for non-streaming responses. The `Response(content=content, ...)` object does have `.body`. This works as-is.

For streaming responses, we skip write-back entirely (streaming cache is a future feature). The `if hasattr(upstream_response, "body")` guard in `server.py` handles this correctly.

---

## Section 7 — Verification steps

After implementation, run these in order:

```bash
# 1. Install the updated library in dev mode
cd promptcache
pip install -e ".[embed]"

# 2. Build the dashboard
cd ../promptcache-dashboard/frontend-next
npm install && npm run build
cp -r out/* ../../promptcache/src/promptcache/proxy/dashboard/

# 3. Start the proxy
cd ../../promptcache
promptcache serve

# Expected output:
#   promptcache proxy  →  http://127.0.0.1:8080
#   dashboard          →  http://127.0.0.1:8080/dashboard
# Browser opens automatically to dashboard

# 4. Test proxy interception
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'

# 5. Second call should hit cache — check for X-Cache header
# Should see: X-Cache: exact or X-Cache: semantic

# 6. Check dashboard API
curl http://localhost:8080/api/health
curl http://localhost:8080/api/stats
```

---

## Section 8 — Agent configuration reference

Include this in the README after implementation:

```bash
# Cursor
# Settings → Features → Claude API → Base URL
http://localhost:8080

# Claude Code
export ANTHROPIC_BASE_URL=http://localhost:8080
claude

# OpenAI SDK (Codex, Copilot custom setups)
export OPENAI_BASE_URL=http://localhost:8080

# Any httpx/requests call
client = anthropic.Anthropic(base_url="http://localhost:8080")
```

---

## What's intentionally out of scope for this pass

- Streaming cache (stream=true requests are forwarded as-is, not cached)
- Multi-user / shared cache (all requests hit the same local cache dir)
- MCP server changes (leave as-is in the library)
- Old dashboard backend (`promptcache-dashboard/backend/main.py`) — keep for local dev reference, don't delete yet
- The old `promptcache serve` CLI commands (stats, clear, config) — all unchanged

These are follow-on work, not blockers for the first open source release.
