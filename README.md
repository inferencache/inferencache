# promptcache

Semantic LLM response caching. Stop paying for the same call twice.

```python
from promptcache import cache, CacheConfig

@cache(config=CacheConfig(model="gpt-4o"))
def ask(prompt: str) -> str:
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content

ask("What is the capital of France?")  # → real API call, cached
ask("What is the capital of France?")  # → instant, from cache
ask("Capital of France?")              # → semantic hit, from cache
```

---

## How it works

Two checks before a request leaves your machine:

1. **Exact match** — SHA-256 of `(prompt, model)` in SQLite. Sub-millisecond.
2. **Semantic match** — embed the prompt, query Qdrant for nearest neighbors. Returns if cosine similarity clears your threshold.

Only if both miss does the real API call go out. On miss, the response is stored and embedded for future semantic lookups.

---

## Install

```bash
# Core library + default embedder
pip install promptcache[embed]

# With MCP server for Cursor/Windsurf/Claude Code
pip install promptcache[embed,mcp]
```

---

## Usage

### Decorator

```python
from promptcache import cache, CacheConfig

config = CacheConfig(
    model="gpt-4o",     # cache key includes model — no cross-model collisions
    threshold=0.88,     # cosine similarity threshold (0.0–1.0)
    cache_dir="~/.cache/promptcache",  # default
)

@cache(config=config)
def ask(prompt: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content
```

### Streaming

```python
@cache(config=config, streaming=True)
def ask_stream(prompt: str):
    for chunk in client.chat.completions.create(..., stream=True):
        yield chunk.choices[0].delta.content or ""

# Cache miss: streams from API and caches the full response.
# Cache hit: reconstitutes a generator from the stored string.
for chunk in ask_stream("Explain quantum entanglement"):
    print(chunk, end="", flush=True)
```

### Context manager

```python
from promptcache import cache_context, CacheConfig

with cache_context(prompt, config=config) as ctx:
    if ctx.hit:
        return ctx.response  # or: yield from ctx.stream()
    response = call_my_llm(prompt)
    ctx.store(response)
    return response
```

### Custom embedder

```python
from promptcache import CacheConfig
from promptcache.embed import OpenAIEmbedder

config = CacheConfig(
    model="gpt-4o",
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
)
```

### Any provider

Works with any OpenAI-compatible API:

```python
# Anthropic
config = CacheConfig(model="claude-sonnet-4-20250514")

# Groq
config = CacheConfig(model="llama-3.1-70b-versatile")

# Local (Ollama, vLLM, etc.)
config = CacheConfig(model="llama3.2")
```

---

## CLI

```bash
# Cache statistics
promptcache stats --model gpt-4o

#   promptcache stats
#   ──────────────────────────────────────────
#   Cache dir        /Users/you/.cache/promptcache
#   Entries stored   1,243
#
#   Hit rate         67.3%
#     exact          834  (72%)
#     semantic       188  (16%)
#
#   Est. tokens saved   203,400
#   Est. cost saved     $1.0170
#
#   Top 10 cached prompts:
#    1. [ 234×]  'Summarize this document in three sentences...'
#    2. [  89×]  'What is the sentiment of the following text...'

# JSON output
promptcache stats --json | jq .hit_rate

# Clear everything
promptcache clear -y

# Clear one model
promptcache clear --model gpt-4o
```

---

## MCP server (for Cursor, Windsurf, Claude Code)

```bash
pip install promptcache[mcp]
```

Add to your MCP config:

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

Your AI assistant can now:
- `get_stats` — "How much am I saving with promptcache?"
- `list_recent` — "What prompts have I cached recently?"
- `get_cached_entry` — "Is this prompt in my cache?"
- `set_threshold` — "Raise the similarity threshold to 0.92"
- `clear_cache` — "Clear all gpt-4o cache entries"

---

## Configuration reference

```python
CacheConfig(
    cache_dir="~/.cache/promptcache",   # storage location
    model="gpt-4o",                     # LLM model (part of cache key)
    threshold=0.85,                     # semantic similarity threshold
    embedder=None,                      # defaults to SentenceTransformerEmbedder
    max_response_tokens=8192,           # skip caching responses over this size
    stream_chunk_size=32,               # chars per yielded chunk on stream hits
    stream_delay=0.0,                   # seconds between chunks (0 = instant)
    enabled=True,                       # master switch
)
```

---

## What's deliberately excluded

- No gateway. No Docker. No config files. No framework dependency.
- `pip install promptcache[embed]`, wrap your function, done.

---

## License

MIT
