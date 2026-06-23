# Contributing to promptcache

## Repository layout

promptcache is split across two repos:

| Repo | Purpose |
|------|---------|
| [lavondev/promptcache](https://github.com/lavondev/promptcache) | Python library, proxy server, embedded dashboard assets |
| [lavondev/promptcache-ui](https://github.com/lavondev/promptcache-ui) | Next.js dashboard frontend |

Clone both as siblings:

```
your-workspace/
  promptcache/           ← this repo
  promptcache-dashboard/ ← promptcache-ui clone (any folder name works for dev)
```

## Development setup

```bash
# Python library + proxy
cd promptcache
pip install -e ".[embed,serve,dev]"
pytest

# Dashboard frontend (hot reload)
cd ../promptcache-dashboard/frontend-next
cp .env.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000/api
npm install && npm run dev

# Dev backend (in another terminal)
cd ../promptcache-dashboard/backend
pip install -e ../../promptcache[embed,serve]
./run.sh
```

## Building the embedded dashboard

Before a release, build the static frontend into the Python package:

```bash
cd promptcache
./scripts/build-dashboard.sh
```

Override paths if needed:

```bash
DASHBOARD_REPO=/path/to/promptcache-ui ./scripts/build-dashboard.sh
```

## Verification checklist

```bash
# 1. Dev install
pip install -e ".[embed,serve,dev]"

# 2. Build + embed dashboard
./scripts/build-dashboard.sh

# 3. Start unified server
promptcache serve

# 4. API health
curl http://localhost:8080/api/health
curl http://localhost:8080/api/suites

# 5. Proxy (repeat call should return X-Cache header)
curl -X POST http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"claude-haiku-4-5-20251001","max_tokens":100,"messages":[{"role":"user","content":"What is 2+2?"}]}'
```

## Architecture

- `src/promptcache/proxy/server.py` — ASGI app (`promptcache serve`)
- `src/promptcache/proxy/intercept.py` — cache lookup for `/v1/messages` and `/v1/chat/completions`
- `src/promptcache/proxy/control/` — dashboard REST API + test suite runner
- `src/promptcache/proxy/dashboard/` — static Next.js build (generated, not hand-edited)

The dashboard backend in `promptcache-ui/backend/main.py` is a thin dev shim that imports the shared control router from this package.

## Pull requests

- Run `pytest` and `ruff check src tests` before opening a PR
- Frontend changes: ensure `npm run build` passes (static export with `basePath: /dashboard`)
- Keep cache data under `~/.cache/promptcache` (unified across proxy, CLI, and dashboard)
