# promptcache

**Multi-tier semantic caching for LLM APIs. Most teams waste 30–60% of their token budget on repeated and near-identical calls. promptcache stops that.**

```python
from promptcache import cache, CacheConfig

@cache(config=CacheConfig(model="gpt-4o"))
def ask(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

ask("What is the capital of France?")  # → real API call, response cached
ask("What is the capital of France?")  # → 2ms, from cache, $0.000000
ask("Capital of France?")              # → 18ms, semantic hit, $0.000000
```

Three lines of code. Every LLM call now checks the cache before it leaves your machine.

---

## Quick start (proxy + dashboard)

One command starts a local proxy and web dashboard. Point your AI coding assistant at it — no code changes required.

```bash
pip install "promptcache[embed,serve]"
export ANTHROPIC_API_KEY=sk-ant-...
promptcache serve
# proxy:     http://localhost:8080
# dashboard: http://localhost:8080/dashboard
```

**Agent configuration:**

| Tool | Setting |
|------|---------|
| Cursor | Settings → Claude API → Base URL → `http://localhost:8080` |
| Claude Code | `export ANTHROPIC_BASE_URL=http://localhost:8080` |
| OpenAI SDK | `export OPENAI_BASE_URL=http://localhost:8080` |

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8080")
```

Repeat calls hit the cache automatically. Open the dashboard to watch live stats, run test suites, and tune your similarity threshold.

---

## How it works

promptcache runs three checks on every call, in order:

**Tier 1 — Client-side cache (your machine, free)**

- **Exact match** — SHA-256 of `(prompt, model)` in SQLite. Sub-millisecond. Zero tokens.
- **Semantic match** — embed the prompt, query Qdrant for nearest neighbors. Catches paraphrases and near-duplicates. Adaptive threshold by prompt type: code queries use 0.92, factual lookups 0.95, conversational 0.82.

If Tier 1 hits, the API never sees the request. Zero cost, zero latency beyond the lookup.

**Tier 2 — Prefix cache optimization (provider-side discount)**

On a Tier 1 miss, promptcache restructures your prompt before sending it — injecting `cache_control` markers for Anthropic (90% discount on cached input tokens) and ensuring stable prefix ordering for OpenAI (50% discount). You pay less even when Tier 1 misses.

**Tier 3 — Inference cache attribution**

Reads `usage.cached_tokens` from provider responses and tracks when the provider served the response from its own inference cache. Attributed separately in the dashboard so you know exactly what each tier is saving.

All three tiers log to a local SQLite event store. The dashboard reads it.

---

## Install

```bash
# Proxy + dashboard (recommended for Cursor / Claude Code users)
pip install "promptcache[embed,serve]"

# Library only + default embedder (bge-small-en-v1.5)
pip install "promptcache[embed]"

# With MCP server for Cursor / Windsurf / Claude Code
pip install "promptcache[embed,mcp]"
```

Python 3.10+. No Docker, no external services. Cache data lives in `~/.cache/promptcache`.

## Usage

### Decorator (default — Tier 1 only)

```python
from promptcache import cache, CacheConfig

@cache(config=CacheConfig(model="gpt-4o", threshold=0.88))
def ask(prompt: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content
```

### Multi-tier (Tier 1 + 2 + 3)

```python
@cache(config=CacheConfig(
    model="claude-3-5-sonnet-20241022",
    provider="anthropic",
    tier="auto",          # enables all three tiers
    session_aware=True,   # prevents false hits across conversation contexts
    prefix_ttl="1hr",     # Anthropic cache TTL preference
))
def ask(prompt: str, system_prompt: str, history: list[dict]) -> str:
    return anthropic_client.messages.create(...).content[0].text
```

### Streaming

```python
@cache(config=config, streaming=True)
def ask_stream(prompt: str):
    for chunk in client.chat.completions.create(..., stream=True):
        yield chunk.choices[0].delta.content or ""

# Cache miss: streams from API, caches full response.
# Cache hit: reconstitutes a generator. Callers see no difference.
for chunk in ask_stream("Explain quantum entanglement"):
    print(chunk, end="", flush=True)
```

### Context manager

```python
from promptcache import cache_context, CacheConfig

with cache_context(prompt, config=config) as ctx:
    if ctx.hit:
        return ctx.response
    response = call_my_llm(prompt)
    ctx.store(response)
    return response
```

### Embedder presets

```python
from promptcache import CacheConfig

# fast — all-MiniLM-L6-v2, 384d, quickest CPU inference
# balanced — bge-small-en-v1.5, 384d (default, best cost/quality)
# accurate — Qwen3-Embedding-0.6B, 1024d, highest MTEB score

config = CacheConfig(model="gpt-4o", embedder_preset="accurate")
```

### Any provider

```python
CacheConfig(model="claude-sonnet-4-20250514", provider="anthropic")
CacheConfig(model="llama-3.1-70b-versatile", provider="groq")
CacheConfig(model="llama3.2", provider="ollama")  # local
```

---

## CLI

```bash
promptcache serve                    # start proxy + dashboard
promptcache serve --no-browser       # don't auto-open browser
promptcache serve --no-dashboard     # proxy only
promptcache stats --model gpt-4o
promptcache stats --json | jq .hit_rate
promptcache clear --model gpt-4o -y
promptcache config
```

```
  promptcache stats
  ──────────────────────────────────────────
  Cache dir        /Users/you/.cache/promptcache
  Entries stored   1,243

  Hit rate         67.3%
    exact          834  (72%)
    semantic       188  (16%)

  Tier 2 prefix    41,200 tokens saved
  Tier 3 inference 8 hits

  Est. tokens saved   203,400
  Est. cost saved     $1.0170

  Top cached prompts:
   1. [ 234×]  'Summarize this document in three sentences...'
   2. [  89×]  'What is the sentiment of the following text...'
```

---

## MCP server

Connect promptcache to your AI coding assistant so it can see your cache stats, check if a prompt is cached, and tune the threshold — without leaving the editor.

```bash
pip install promptcache[mcp]
```

Add to your MCP config (`~/.cursor/mcp.json` or `.claude/config.json`):

```json
{
  "mcpServers": {
    "promptcache": {
      "command": "promptcache-mcp",
      "args": ["--cache-dir", "~/.cache/promptcache"]
    }
  }
}
```

Available tools:

| Tool | What it does |
|---|---|
| `get_stats` | Hit rate, cost saved, exact/semantic/prefix breakdown |
| `list_recent` | Most recently cached prompts with hit counts |
| `get_cached_entry` | Check if a specific prompt is in the cache |
| `set_threshold` | Update similarity threshold at runtime |
| `clear_cache` | Flush entries, optionally by model |

Ask your assistant: *"How much have I saved with promptcache this week?"*

---

## Testing dashboard

The dashboard is embedded in `promptcache serve` at `http://localhost:8080/dashboard`. It pressure-tests your cache with real API calls, streams results live, and breaks down savings by tier.

**End users:** use `promptcache serve` — no separate setup needed.

**UI contributors:** clone [promptcache-ui](https://github.com/lavondev/promptcache-ui) alongside this repo and run the frontend in dev mode:

```bash
# From a parent directory:
git clone https://github.com/lavondev/promptcache.git
git clone https://github.com/lavondev/promptcache-ui.git promptcache-dashboard

# Terminal 1 — dev backend (shared control API from this package)
cd promptcache-dashboard/backend
pip install -e ../../promptcache[embed,serve]
./run.sh
# → http://localhost:8000/api

# Terminal 2 — frontend hot reload
cd promptcache-dashboard/frontend-next
cp .env.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000/api in .env.local
npm install && npm run dev
# → http://localhost:3000
```

To embed a production build into the wheel:

```bash
cd promptcache
./scripts/build-dashboard.sh
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full two-repo layout.

---

## Configuration reference

```python
CacheConfig(
    # Core
    cache_dir="~/.cache/promptcache",  # where SQLite + Qdrant data live
    model="gpt-4o",                    # LLM model — part of the cache key
    provider="openai",                 # "openai" | "anthropic" | "groq" | ...
    threshold=0.85,                    # semantic similarity floor (0.0–1.0)

    # Embedder
    embedder=None,                     # custom Embedder instance
    embedder_preset="balanced",        # "fast" | "balanced" | "accurate"

    # Multi-tier (opt-in)
    tier=None,                         # None = Tier 1 only; "auto" = all three
    session_aware=False,               # prevent false hits across conversation turns
    prefix_ttl="5min",                 # "5min" | "1hr" — Anthropic prefix cache TTL

    # Limits
    max_response_tokens=8192,          # skip caching very long responses
    stream_chunk_size=32,              # chars per chunk on stream cache hits
    stream_delay=0.0,                  # artificial delay between chunks (0 = instant)
    enabled=True,                      # master switch — set False to bypass entirely
)
```

---

## What stays out of scope

No mandatory gateway — use the library decorator directly, or `promptcache serve` if you want a drop-in proxy. No sidecar. No config files beyond what you pass to `CacheConfig`. No framework lock-in. The only things that run are SQLite (already on your system) and Qdrant (embedded, file-based).

---

## Why session_aware matters for Cursor / Claude Code users

Without `session_aware=True`, a prompt like *"fix this function"* in two different coding sessions can produce a false semantic hit — the embeddings are similar but the context is completely different. With `session_aware=True`, promptcache hashes the last three turns of conversation history and uses that as part of the cache key. Stateless prompts (factual lookups, definitions, boilerplate generation) bypass the session check and cache freely across sessions.

Cursor and Claude Code users burning 50M+ tokens/month see the largest relative savings from this combination: high volume, repetitive patterns, and session context that makes naive caching unreliable.

---

## License

MIT