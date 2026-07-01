# inferencache

**Multi-tier semantic caching for LLM APIs. Stop paying for the same prompt twice.**

```bash
pip install "inferencache[embed,serve]"
export ANTHROPIC_API_KEY=sk-ant-...
inferencache serve
# landing:   http://localhost:8080/
# dashboard: http://localhost:8080/dashboard/
# proxy:     http://localhost:8080/v1/messages
```

Point Cursor or Claude Code at `http://localhost:8080` — no code changes required.

## Dashboard

PyPI releases and tagged GitHub releases bundle the embedded dashboard automatically.

When installing from source, build the static site before `inferencache serve` if you want the UI:

```bash
./scripts/build-dashboard.sh
```

The proxy and control API work without the dashboard. Use `inferencache serve --no-dashboard` to skip site routes entirely.

## Limitations (v0.1)

**Streaming** — Clients that send `stream: true` (Cursor and Claude Code default to streaming) are forwarded correctly but are **not cached** in v0.1. Identical prompts sent with `stream: false` will hit the cache. Streaming cache support is planned for v0.2.

**Generative reuse** — Available in the library when you set `generative_reuse_enabled=True` and provide an `adaptation_client`. Off by default in the proxy.

**Temporal validity** — Active in library and proxy lookup; expired or session-bound entries return `stale_miss` instead of stale answers.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
