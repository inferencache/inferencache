# Generative Reuse — Implementation Spec
## inferencache · v0.3 increment

---

## The problem this solves

The current lookup is a cliff:

```
similarity ≥ threshold  →  verbatim cache hit   ($0.00, ~3ms)
similarity < threshold  →  full API call         ($0.003+, ~800ms)
```

A prompt at similarity 0.89 against a CODE threshold of 0.92 is
treated identically to a prompt at 0.40. Both are full API calls.
But the 0.89 prompt is almost certainly doing the same structural
work as the cached one — same operation, different variable names,
different file paths, different parameters.

Generative reuse captures that middle band. Instead of a full
regeneration, it uses a cheap fast model to *adapt* the cached
response to the new prompt. The target model (GPT-4o, Claude Sonnet)
never gets called. The adaptation model (GPT-4o-mini, Haiku) costs
~5-10% as much and returns in ~200-400ms.

---

## The three-zone lookup model

```
                    FLOOR          THRESHOLD
                      |                |
  0.0 ───────────────[0.78]──────────[0.92]──── 1.0
        FULL MISS      |  GENERATIVE  |  VERBATIM
                       |     ZONE     |    HIT
```

- **Below floor (< 0.78):** prompts are different enough that
  adaptation would cost more than fresh generation. Full miss.

- **Generative zone (floor ≤ similarity < threshold):** structurally
  similar. Adaptation with a cheap model is cheaper and faster than
  full regeneration.

- **Above threshold (≥ 0.92):** semantically identical. Return
  verbatim. No model call at all.

All three bounds are configurable per prompt type, same as the
existing adaptive threshold system.

---

## Default zone bounds by prompt type

These mirror the existing `threshold` values and add a `gen_floor`:

```python
GENERATIVE_ZONES = {
    "CODE":          {"gen_floor": 0.78, "threshold": 0.92},
    "DETERMINISTIC": {"gen_floor": 0.82, "threshold": 0.95},
    "RAG":           {"gen_floor": 0.72, "threshold": 0.88},
    "CONVERSATIONAL":{"gen_floor": 0.68, "threshold": 0.82},
}
```

CODE has the tightest floor (0.78) because code adaptations are
highest risk — a wrong variable name or type signature is a silent
bug. CONVERSATIONAL has the lowest floor because tone and phrasing
adaptations are lowest risk — wrong here just means a slightly off
answer, not broken code.

---

## Architecture

### New module: `adapt.py`

Single responsibility: given a cached prompt, cached response, and
new prompt, call a fast cheap model and return the adapted response.
No knowledge of the cache, no knowledge of thresholds.

```python
# src/inferencache/adapt.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


class AdaptationClient(Protocol):
    """
    Minimal interface for the adaptation model client.
    Decoupled from any specific provider SDK.
    """
    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        """
        Returns (response_text, tokens_in, tokens_out).
        """
        ...


@dataclass
class AdaptationResult:
    response:      str
    tokens_input:  int
    tokens_output: int
    latency_ms:    float
    model:         str   # the adaptation model used, e.g. "gpt-4o-mini"


ADAPTATION_SYSTEM_PROMPT = """You are a response adaptation assistant.
You will be given:
1. An original request
2. A cached response to that request
3. A new request that is structurally similar but differs in specifics

Your job is to adapt the cached response to fit the new request.

Rules:
- Change ONLY what is necessary to address the differences in the new request
- Preserve the structure, depth, format, and style of the cached response
- If the cached response is code, change variable names / types / logic
  only where the new request differs — do not rewrite working sections
- If you cannot adapt confidently (the requests are too different in intent),
  respond with exactly: ADAPTATION_FAILED
- Do not explain what you changed. Return only the adapted response."""


def build_adaptation_prompt(
    cached_prompt:   str,
    cached_response: str,
    new_prompt:      str,
) -> str:
    return f"""ORIGINAL REQUEST:
{cached_prompt}

CACHED RESPONSE:
{cached_response}

NEW REQUEST:
{new_prompt}

Adapted response:"""


class AdaptationEngine:
    """
    Calls a cheap fast model to adapt a cached response to a new prompt.
    Provider-agnostic via the AdaptationClient protocol.
    """

    FAILURE_SENTINEL = "ADAPTATION_FAILED"

    def __init__(
        self,
        client: AdaptationClient,
        model: str,                    # e.g. "gpt-4o-mini", "claude-haiku-4-5"
        max_cached_response_chars: int = 4000,  # truncate very long cached responses
    ) -> None:
        self._client = client
        self._model  = model
        self._max_chars = max_cached_response_chars

    def adapt(
        self,
        cached_prompt:   str,
        cached_response: str,
        new_prompt:      str,
    ) -> AdaptationResult | None:
        """
        Returns AdaptationResult on success, None on ADAPTATION_FAILED
        or if the adaptation model signals it cannot adapt confidently.
        """
        import time

        # Truncate very long cached responses to keep adaptation cost bounded.
        # If the cached response is huge, adaptation may not be worth it anyway.
        truncated_response = cached_response[:self._max_chars]
        if len(cached_response) > self._max_chars:
            truncated_response += "\n[... response truncated for adaptation ...]"

        user_prompt = build_adaptation_prompt(
            cached_prompt, truncated_response, new_prompt
        )

        t0 = time.perf_counter()
        try:
            text, tok_in, tok_out = self._client.complete(
                system=ADAPTATION_SYSTEM_PROMPT,
                user=user_prompt,
            )
        except Exception:
            # If the adaptation call fails for any reason, fall through
            # to a full miss — never surface an error to the caller.
            return None
        latency_ms = (time.perf_counter() - t0) * 1000

        # Check for explicit failure sentinel
        if text.strip() == self.FAILURE_SENTINEL:
            return None

        return AdaptationResult(
            response=text,
            tokens_input=tok_in,
            tokens_output=tok_out,
            latency_ms=latency_ms,
            model=self._model,
        )
```

---

### Provider client implementations

Two thin wrappers, one per provider. Kept outside `adapt.py` to
avoid importing provider SDKs in the core library.

```python
# src/inferencache/adapt_clients.py

class OpenAIAdaptationClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model  = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=2048,
            temperature=0.1,   # low temp — adaptation should be deterministic
        )
        msg = resp.choices[0].message.content or ""
        return msg, resp.usage.prompt_tokens, resp.usage.completion_tokens


class AnthropicAdaptationClient:
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5") -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model  = model

    def complete(self, system: str, user: str) -> tuple[str, int, int]:
        resp = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            temperature=0.1,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text if resp.content else ""
        return text, resp.usage.input_tokens, resp.usage.output_tokens
```

---

### Changes to `CacheConfig`

Three new fields:

```python
@dataclass
class CacheConfig:
    # ... existing fields unchanged ...

    # Generative reuse
    generative_reuse_enabled: bool = False
    # Off by default — requires the user to provide an adaptation client,
    # so we keep it opt-in rather than failing silently if no client is set.

    generative_reuse_floor: float = 0.78
    # Lower bound of the generative zone. Below this = full miss.
    # Overridden per-prompt-type by the adaptive threshold system.

    adaptation_client: Any | None = None
    # An AdaptationClient instance. Required when generative_reuse_enabled=True.
    # If None and enabled=True, logs a warning and falls through to full miss.
```

---

### Changes to `CacheEngine.lookup()`

The lookup now has four outcomes instead of three:

```python
def lookup(self, prompt: str, ...) -> CacheResult:

    # 1. Exact match (unchanged)
    exact = self._store.get_exact(prompt, self._config.model)
    if exact is not None:
        if not self._is_valid(exact, session_id):          # from TTL spec
            pass  # fall through to semantic
        else:
            return CacheResult(hit=True, hit_type="exact", ...)

    # 2. Semantic search (unchanged — gets top_k candidates)
    embedding  = self._config.embedder.embed(prompt)
    candidates = self._store.query_semantic(embedding, model, threshold=0.0, top_k=5)
    # Note: query with 0.0 floor so we see ALL candidates including
    # those in the generative zone. We apply zone logic ourselves.

    if not candidates:
        return CacheResult(hit=False, hit_type="miss", ...)

    best_entry, best_score = candidates[0]

    # 3. Verbatim hit (above threshold)
    if best_score >= self._get_threshold(prompt):
        if not self._is_valid(best_entry, session_id):
            return CacheResult(hit=False, hit_type="stale_miss", ...)
        return CacheResult(hit=True, hit_type="semantic", similarity=best_score, ...)

    # 4. Generative zone
    gen_floor = self._get_gen_floor(prompt)
    if (
        self._config.generative_reuse_enabled
        and self._config.adaptation_client is not None
        and best_score >= gen_floor
    ):
        result = self._try_generative_reuse(prompt, best_entry, best_score)
        if result is not None:
            return result
        # If adaptation failed, fall through to full miss

    # 5. Full miss
    return CacheResult(
        hit=False,
        hit_type="miss",
        best_similarity=best_score,
        ...
    )


def _try_generative_reuse(
    self,
    new_prompt:  str,
    cached_entry: CacheEntry,
    similarity:  float,
) -> CacheResult | None:
    """
    Attempt generative reuse. Returns CacheResult on success, None on failure.
    Failure falls through to full miss — never raises.
    """
    engine = AdaptationEngine(
        client=self._config.adaptation_client,
        model=self._config.adaptation_client._model,
    )
    adaptation = engine.adapt(
        cached_prompt=cached_entry.prompt,
        cached_response=cached_entry.response,
        new_prompt=new_prompt,
    )

    if adaptation is None:
        return None  # model said ADAPTATION_FAILED

    return CacheResult(
        hit=True,
        hit_type="generative",          # new hit_type
        response=adaptation.response,
        similarity=similarity,
        entry=cached_entry,
        latency_ms=adaptation.latency_ms,
        adaptation_model=adaptation.model,
        adaptation_tokens_in=adaptation.tokens_input,
        adaptation_tokens_out=adaptation.tokens_output,
    )
```

---

### New fields on `CacheResult`

```python
@dataclass
class CacheResult:
    # ... existing fields ...
    hit_type: str  # now: 'exact' | 'semantic' | 'generative' | 'miss' | 'stale_miss'

    # Populated only for generative hits
    adaptation_model:       str | None = None
    adaptation_tokens_in:   int | None = None
    adaptation_tokens_out:  int | None = None
```

---

### Schema: new columns on `calls`

```sql
ALTER TABLE calls ADD COLUMN adaptation_model      TEXT;
ALTER TABLE calls ADD COLUMN adaptation_tokens_in  INTEGER;
ALTER TABLE calls ADD COLUMN adaptation_tokens_out INTEGER;
ALTER TABLE calls ADD COLUMN adaptation_cost_usd   REAL;
```

These are NULL for exact/semantic hits and misses. Populated only for
`hit_type = 'generative'`. This lets analytics track the true cost of
generative hits (adaptation model cost) separately from saved cost
(target model cost avoided).

---

### Cost accounting for generative hits

This is important and easy to get wrong. A generative hit has two
cost figures:

```
cost_avoided  = what the target model (GPT-4o) would have cost
cost_incurred = what the adaptation model (GPT-4o-mini) actually cost

net_saved     = cost_avoided - cost_incurred
```

The analytics layer should show `net_saved` for generative hits, not
the raw `cost_avoided`. Otherwise the dashboard overstates savings.

```python
# In analytics.py — cost_saved_cumulative()
# For 'generative' hit_type rows:
#   net_saved = model_cost_per_token(model) * estimated_tokens
#             - model_cost_per_token(adaptation_model) * adaptation_tokens_out
```

---

### Public API: how developers enable this

```python
from inferencache import cache, CacheConfig
from inferencache.adapt_clients import OpenAIAdaptationClient

config = CacheConfig(
    model="gpt-4o",
    threshold=0.92,
    generative_reuse_enabled=True,
    generative_reuse_floor=0.78,
    adaptation_client=OpenAIAdaptationClient(
        api_key=os.environ["OPENAI_API_KEY"],
        model="gpt-4o-mini",
    ),
)

@cache(config=config)
def ask(prompt: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content
```

Developer opts in explicitly. Off by default. No surprises.

---

### `_get_gen_floor()` — per-prompt-type floor

Mirrors the existing `_get_threshold()` pattern. Reads prompt type
from the TTL classifier (already being built) and returns the
appropriate floor:

```python
def _get_gen_floor(self, prompt: str) -> float:
    if not hasattr(self, '_ttl_classifier'):
        return self._config.generative_reuse_floor
    prompt_type = self._ttl_classifier.classify_type(prompt)
    return GENERATIVE_ZONES.get(prompt_type, {}).get(
        "gen_floor", self._config.generative_reuse_floor
    )
```

Note: this requires a minor addition to `TTLClassifier` from the
previous spec — a `classify_type()` method that returns the prompt
category string (CODE, DETERMINISTIC, etc.) separate from the TTL
class. These are related but distinct classifications. One classifier,
two output modes.

---

### Dashboard: generative hit tracking

The Tuning tab similarity histogram should render generative zone
hits in a distinct color (e.g. amber, different from green for
verbatim semantic hits). The stats strip should show:

```
Exact hits: 234   Semantic hits: 89   Generative hits: 41   Misses: 31
```

And the cost saved hero number should use `net_saved` for generative
hits, not `cost_avoided`.

The Call Drawer for generative hits should show:
- Hit type: `GENERATIVE`
- Similarity score (to the cached prompt it adapted from)
- Adaptation model used
- Adaptation tokens consumed
- Net cost saved (avoided - incurred)

---

## Implementation order for Cursor

1. **`adapt.py`** — `AdaptationEngine`, `AdaptationResult`,
   `ADAPTATION_SYSTEM_PROMPT`, `build_adaptation_prompt()`.
   No external dependencies yet — pure logic.

2. **`adapt_clients.py`** — `OpenAIAdaptationClient`,
   `AnthropicAdaptationClient`. These are thin wrappers, trivially
   testable with a mock that returns a fixed tuple.

3. **`CacheConfig`** — add three new fields with defaults. No behavior
   change yet — just configuration surface.

4. **`CacheResult`** — add three new optional fields. Backward
   compatible; existing callers see no change.

5. **`GENERATIVE_ZONES` dict** — add to `engine.py` or a new
   `zones.py`. Wire `_get_gen_floor()` into `CacheEngine`.

6. **`CacheEngine.lookup()`** — add the generative zone branch between
   the semantic hit check and the full miss return. Call
   `_try_generative_reuse()` only when enabled and client is set.

7. **Schema migration** — four new nullable columns on `calls` via
   `_migrate_sqlite_schema()`. No breaking change to existing rows.

8. **Analytics** — update `cost_saved_cumulative()` to use `net_saved`
   for generative hits. Add `generative_hit_rate()` method.

9. **Dashboard** — update stats strip, histogram colors, Call Drawer
   for `hit_type='generative'`.

---

## Test cases

```python
# Unit tests for AdaptationEngine

def test_adapt_csv_to_tsv():
    mock_client = MockClient(response="def parse_tsv(path):\n    ...")
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt(
        cached_prompt="Write a Python function to parse a CSV file",
        cached_response="def parse_csv(path):\n    import csv\n    ...",
        new_prompt="Write a Python function to parse a TSV file",
    )
    assert result is not None
    assert "tsv" in result.response.lower()


def test_adapt_returns_none_on_sentinel():
    mock_client = MockClient(response="ADAPTATION_FAILED")
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt("prompt A", "response A", "prompt B")
    assert result is None


def test_adapt_returns_none_on_client_exception():
    mock_client = MockClient(raises=RuntimeError("timeout"))
    engine = AdaptationEngine(client=mock_client, model="gpt-4o-mini")
    result = engine.adapt("prompt A", "response A", "prompt B")
    assert result is None   # never raises


# Integration: lookup() returns generative hit in the zone
def test_lookup_generative_zone():
    config = CacheConfig(
        generative_reuse_enabled=True,
        generative_reuse_floor=0.78,
        threshold=0.92,
        adaptation_client=MockAdaptationClient(),
    )
    engine = CacheEngine(config)

    # Store a response
    engine.store("Parse a CSV file into dicts", "def parse_csv(): ...")

    # Lookup at ~0.85 similarity (in generative zone)
    result = engine.lookup("Parse a TSV file into dicts")
    assert result.hit is True
    assert result.hit_type == "generative"
    assert result.adaptation_model is not None


# Below floor → full miss
def test_lookup_below_floor_is_miss():
    config = CacheConfig(
        generative_reuse_enabled=True,
        generative_reuse_floor=0.78,
        threshold=0.92,
        adaptation_client=MockAdaptationClient(),
    )
    engine = CacheEngine(config)
    engine.store("Parse a CSV file into dicts", "def parse_csv(): ...")

    result = engine.lookup("Write a poem about autumn")  # clearly different
    assert result.hit is False
    assert result.hit_type == "miss"
```

---

## Why `ADAPTATION_FAILED` instead of a confidence score

The adaptation model self-reports failure via a sentinel string rather
than having the caller evaluate output quality. This is intentional:

1. Quality evaluation requires another model call or a classifier —
   adding latency and cost to every generative hit.

2. The adaptation model has full context (both prompts + cached
   response) and is better positioned to know if the adaptation is
   safe than any external judge.

3. The sentinel is deterministic and cheap to check — one string
   comparison.

The tradeoff: the model may occasionally return a bad adaptation
without flagging it. This is why generative hits are tracked as a
distinct `hit_type` in the calls log — developers can flag bad
generative hits in the Call Drawer the same way they flag false
positives today, and that signal can train better prompts over time.

---

## What this changes about inferencache's positioning

With this implemented, the full lookup hierarchy becomes:

```
1. Exact match (SQLite SHA-256)      → 0ms,   $0.00
2. Verbatim semantic hit (Qdrant)    → ~3ms,  $0.00
3. Generative reuse (cheap model)    → ~300ms, ~$0.0003
4. Full miss (target model)          → ~800ms, ~$0.003
```

That's a 4-tier cost hierarchy. No other open source semantic caching
project has this. The 3x latency difference between tier 3 and tier 4
is still fast enough for interactive agentic use, and the 10x cost
difference makes it worth doing.

The hit rate improvement is the real story: from ~40% to ~80%+ on
agentic workflows where agents repeatedly do structurally similar work
with varying parameters. That's the number that gets developers to
talk about inferencache.
