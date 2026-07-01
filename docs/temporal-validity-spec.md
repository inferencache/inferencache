# Temporal Validity Metadata — Implementation Spec
## inferencache · v0.2 increment

---

## What temporal validity actually is

Right now, every cache entry lives forever. When you store a response, you
store a `created_at` timestamp, but nothing ever *reads* that timestamp to
decide whether the response is still trustworthy. The cache treats a
three-week-old answer to "what files did I change recently?" identically to
a three-second-old one.

Temporal validity is the idea that a cached response has a *decay class* —
a policy that describes how long it can be trusted. Different prompts decay
at completely different rates:

- `"What is the formula for cosine similarity?"` → never expires
- `"Summarize this codebase structure"` → expires when files change (session-scoped)
- `"What are the latest LLM benchmarks?"` → expires in hours or days
- `"Which PRs are open right now?"` → expires in minutes

Without encoding this, inferencache silently returns stale answers. That's
worse than a cache miss, because a miss just costs money — a stale hit
*corrupts* the agent's world model. For coding agents in particular, this
is a real failure mode: a cached answer about "current" state can send the
agent down entirely the wrong path.

---

## The four TTL classes

These four classes cover the practical space for agentic developer workflows:

```
PERMANENT      No expiry. Facts, definitions, math, code patterns that
               don't change with environment.
               Examples: "explain recursion", "what does --no-ff mean",
               "regex for email validation"

SESSION        Valid for the duration of a single agentic session only.
               Expires when the session context changes or ends.
               Examples: "summarize this repo", "what tests are failing",
               "what's in this file"

TIME_WINDOWED  Valid for a configured duration (default 24h). Good for
               things that change on a daily or hourly cadence.
               Examples: "latest news about X", "benchmark comparisons",
               "current best practice for Y"

EPHEMERAL      Very short-lived (default 5 minutes). For near-real-time
               queries where even a recent answer may be wrong.
               Examples: "open PRs", "current build status", "active errors"
```

---

## Architecture: what changes and where

### 1. New module: `ttl.py`

A single new file handles all TTL logic. No changes to `engine.py` lookup
path until the very end — the TTL check is a final gate before returning a
hit.

```python
# src/inferencache/ttl.py

from enum import Enum
from dataclasses import dataclass
import time
import re

class TTLClass(str, Enum):
    PERMANENT     = "permanent"
    SESSION       = "session"
    TIME_WINDOWED = "time_windowed"
    EPHEMERAL     = "ephemeral"

@dataclass
class TTLPolicy:
    ttl_class:     TTLClass
    max_age_secs:  float | None   # None = no time limit (PERMANENT/SESSION)
    session_bound: bool           # True = invalidated when session_id changes

# Default policies per class
DEFAULT_POLICIES: dict[TTLClass, TTLPolicy] = {
    TTLClass.PERMANENT:     TTLPolicy(TTLClass.PERMANENT,     None,    False),
    TTLClass.SESSION:       TTLPolicy(TTLClass.SESSION,       None,    True),
    TTLClass.TIME_WINDOWED: TTLPolicy(TTLClass.TIME_WINDOWED, 86400.0, False),  # 24h
    TTLClass.EPHEMERAL:     TTLPolicy(TTLClass.EPHEMERAL,     300.0,   False),  # 5m
}
```

---

### 2. TTL classifier: `TTLClassifier`

The classifier looks at the prompt text and returns a `TTLClass`. This
runs at *write time* — when a new cache entry is stored — so there's zero
overhead on cache hits (the class is already stored in metadata).

```python
class TTLClassifier:
    """
    Lightweight rule-based classifier. No LLM call, no embedding.
    Runs in microseconds.
    """

    # Signal patterns → TTL class (evaluated in order, first match wins)
    _RULES: list[tuple[TTLClass, list[str]]] = [

        # EPHEMERAL: real-time queries
        (TTLClass.EPHEMERAL, [
            r"\b(open|current|active|right now|at the moment)\b.*(pr|pull request|issue|bug|error|build|deploy)",
            r"\b(build|pipeline|ci)\b.*(status|passing|failing|running)",
            r"\bwhat('s| is) (running|active|live|deployed)\b",
            r"\b(latest|recent) (error|exception|log|trace)\b",
        ]),

        # SESSION: file/repo/codebase-specific context
        (TTLClass.SESSION, [
            r"\b(this|the) (file|repo|codebase|project|module|function|class)\b",
            r"\b(summarize|explain|describe).*(file|code|implementation|function)",
            r"\bwhat (does this|is this|is in this)\b",
            r"\b(failing|passing) (test|spec|suite)\b",
            r"\b(my|our) (code|implementation|approach)\b",
        ]),

        # TIME_WINDOWED: current-state queries without real-time urgency
        (TTLClass.TIME_WINDOWED, [
            r"\b(latest|current|recent|new|updated)\b.*(model|library|version|release|benchmark|paper)",
            r"\bbest practice\b",
            r"\b(what|which).*(recommend|use|prefer)\b",
            r"\b(news|update|change).*(about|in|for)\b",
            r"\b(how|what).*(in \d{4}|today|this (year|month|week))\b",
        ]),

        # PERMANENT: facts, math, concepts, syntax
        (TTLClass.PERMANENT, [
            r"\b(what is|explain|define|how does)\b.*(algorithm|concept|formula|theorem|syntax)",
            r"\b(regex|pattern) for\b",
            r"\bhow to\b.*(configure|install|set up)\b",   # stable enough
            r"\b(git|bash|sql|python|javascript)\b.*(command|syntax|flag|option)",
        ]),
    ]

    def classify(self, prompt: str) -> TTLClass:
        lowered = prompt.lower()
        for ttl_class, patterns in self._RULES:
            for pattern in patterns:
                if re.search(pattern, lowered):
                    return ttl_class
        # Default: PERMANENT (conservative — better to over-cache facts
        # than to under-cache operational state)
        return TTLClass.PERMANENT
```

**Why rule-based, not ML?** Speed. The classifier runs on every store
operation. An ML classifier would add 50-200ms per write. Rules run in
<1ms. The patterns cover the 80% case; the remaining 20% defaults to
PERMANENT, which is the safe failure mode.

**Why classify at write time?** Two reasons. First, it's free — you're
already computing the embedding and doing a DB write. Second, it means
the TTL class is stored in the entry, so lookup is a simple column
comparison, not a re-classification on every hit.

---

### 3. Schema change: two new columns on `entries`

```sql
-- Migration (guarded with PRAGMA table_info, same pattern as existing migrations)
ALTER TABLE entries ADD COLUMN ttl_class    TEXT NOT NULL DEFAULT 'permanent';
ALTER TABLE entries ADD COLUMN expires_at   REAL;          -- Unix timestamp, NULL = no expiry
```

`expires_at` is computed at write time:
```
expires_at = created_at + policy.max_age_secs   (if max_age_secs is not None)
expires_at = NULL                                (PERMANENT, SESSION)
```

SESSION entries get no `expires_at` because they're invalidated by
session context, not time. The lookup handles this separately.

---

### 4. Write path change: `store.py` + `engine.py`

**`CacheEntry` gets two new fields:**

```python
@dataclass
class CacheEntry:
    # ... existing fields ...
    ttl_class:  str = "permanent"   # TTLClass value
    expires_at: float | None = None # Unix timestamp, None = no expiry
```

**`engine.store()` calls the classifier:**

```python
def store(self, prompt: str, response: str, ..., session_id: str | None = None) -> None:
    ttl_class = self._ttl_classifier.classify(prompt)
    policy    = DEFAULT_POLICIES[ttl_class]
    expires_at = (
        time.time() + policy.max_age_secs
        if policy.max_age_secs is not None
        else None
    )

    entry = CacheEntry(
        prompt=prompt,
        model=self._config.model,
        response=response,
        created_at=time.time(),
        ttl_class=ttl_class.value,
        expires_at=expires_at,
        metadata=metadata,
    )
    # ... rest of store unchanged ...
```

**`_sqlite_write()` persists the new fields:**

```python
self._conn.execute(
    """
    INSERT OR REPLACE INTO entries
        (prompt_hash, prompt, model, response, created_at, hit_count,
         metadata, ttl_class, expires_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    (..., entry.ttl_class, entry.expires_at),
)
```

---

### 5. Lookup path change: the validity gate

This is the most important part. After a semantic or exact hit is found,
before returning it, the engine checks validity:

```python
# engine.py — inside lookup(), after hit is found

def _is_valid(
    self,
    entry: CacheEntry,
    current_session_id: str | None,
) -> bool:
    """
    Returns False if the entry should be treated as a miss
    due to expiry or session mismatch.
    """
    now = time.time()

    # Time-based expiry (TIME_WINDOWED, EPHEMERAL)
    if entry.expires_at is not None and now > entry.expires_at:
        return False

    # Session-bound expiry (SESSION class)
    # If the entry was cached in a different session, it's stale
    if entry.ttl_class == TTLClass.SESSION.value:
        entry_session = (entry.metadata or {}).get("session_id")
        if entry_session and current_session_id:
            if entry_session != current_session_id:
                return False

    return True
```

```python
# In lookup():
result = self._store.get_exact(prompt, self._config.model)
if result is not None:
    if not self._is_valid(result, session_id):
        # Treat as miss — don't return stale entry
        # Still write a call event with hit_type="stale_miss"
        return CacheResult(hit=False, hit_type="stale_miss", ...)
    # ... return hit as before
```

**New hit type: `"stale_miss"`** — this is separate from a regular miss
so the analytics layer can distinguish "never seen before" from "seen but
expired." This is a key observability win.

---

### 6. SQLite index for efficient expiry queries

```sql
CREATE INDEX IF NOT EXISTS idx_entries_expires_at
    ON entries (expires_at)
    WHERE expires_at IS NOT NULL;
```

This makes a future `prune_expired()` method fast — a single indexed
scan instead of a full table scan.

---

### 7. Periodic pruning: `prune_expired()`

Optional method on `CacheStore`. The proxy can call this on startup or
on a background thread every N minutes.

```python
def prune_expired(self) -> int:
    """
    Delete entries past their expires_at. Returns count deleted.
    Also removes corresponding Qdrant vectors.
    """
    now = time.time()
    expired_rows = self._conn.execute(
        "SELECT prompt_hash FROM entries WHERE expires_at IS NOT NULL AND expires_at < ?",
        (now,)
    ).fetchall()

    if not expired_rows:
        return 0

    hashes = [row[0] for row in expired_rows]
    with self._conn:
        self._conn.execute(
            f"DELETE FROM entries WHERE prompt_hash IN ({','.join('?' * len(hashes))})",
            hashes
        )

    # Remove from Qdrant too
    client = self._get_qdrant_client()
    client.delete(
        collection_name=self._collection_name,
        points_selector=hashes,
    )

    return len(hashes)
```

---

### 8. Analytics: new `stale_miss` tracking

The DuckDB analytics layer in `analytics.py` should expose:

```python
def stale_miss_rate(self, model: str, window_hours: int = 24) -> dict:
    """
    Returns counts of stale_miss events vs regular misses.
    Useful for tuning TTL windows — if stale_miss rate is high,
    TIME_WINDOWED entries are expiring too fast.
    """
```

The dashboard Tuning tab can surface stale miss rate as a new metric,
letting developers tune their TTL windows the same way they tune
similarity thresholds today.

---

### 9. Manual TTL override (developer escape hatch)

The `cache()` decorator and `cache_context()` should accept an optional
`ttl_override` parameter so developers can bypass classifier inference:

```python
@cache(config=config, ttl_override=TTLClass.EPHEMERAL)
def get_open_prs(prompt: str) -> str:
    ...
```

This is important for correctness in cases where the developer knows the
query touches live state but the classifier might not catch it.

---

## What this looks like end-to-end

```
Developer calls: "What tests are currently failing in this repo?"

1. Exact lookup → miss
2. Semantic lookup → hit (similarity 0.94)
3. _is_valid() check:
   - ttl_class = "session" (correctly classified at write time)
   - entry.metadata["session_id"] = "session-abc"
   - current session_id = "session-xyz"  ← different session
   - returns False → STALE MISS
4. Real API call goes out
5. store() runs:
   - TTLClassifier.classify() → SESSION
   - expires_at = None (session-bound, not time-bound)
   - metadata["session_id"] = "session-xyz" stored
6. Next call in same session → valid SESSION hit ✓
```

---

## Implementation order for Cursor

Do these in sequence. Each step is independently testable.

1. **`ttl.py`** — `TTLClass` enum, `TTLPolicy` dataclass, `DEFAULT_POLICIES`,
   `TTLClassifier.classify()` with the rule set above. Write unit tests
   against the 20-30 representative prompts (one per category).

2. **Schema migration** — add `ttl_class` and `expires_at` to `entries` in
   `_init_sqlite()` and `_migrate_sqlite_schema()`. Update `_sqlite_write()`
   and `_row_to_entry()` in `store.py`. Update `CacheEntry` dataclass.

3. **Write path** — instantiate `TTLClassifier` in `CacheEngine.__init__()`.
   Call `classify()` in `engine.store()`. Pass `ttl_class` and `expires_at`
   into the `CacheEntry`.

4. **Lookup gate** — add `_is_valid()` to `CacheEngine`. Call it after any
   hit is found in `lookup()`. Add `"stale_miss"` as a valid `hit_type`
   string in `CacheResult`.

5. **`prune_expired()`** — add to `CacheStore`. Call it in the proxy's
   startup sequence.

6. **Analytics** — add `stale_miss_rate()` to `analytics.py`. Surface in
   dashboard Tuning tab.

7. **`ttl_override`** — add optional param to `@cache` decorator and
   `cache_context()`.

---

## Test cases to write for step 1

```python
classifier = TTLClassifier()

# EPHEMERAL
assert classifier.classify("What PRs are open right now?") == TTLClass.EPHEMERAL
assert classifier.classify("Is the build passing?") == TTLClass.EPHEMERAL
assert classifier.classify("What's the current build status?") == TTLClass.EPHEMERAL

# SESSION
assert classifier.classify("Summarize this file") == TTLClass.SESSION
assert classifier.classify("What does this function do?") == TTLClass.SESSION
assert classifier.classify("Explain the implementation in this module") == TTLClass.SESSION
assert classifier.classify("What tests are failing in this repo?") == TTLClass.SESSION

# TIME_WINDOWED
assert classifier.classify("What's the latest version of langchain?") == TTLClass.TIME_WINDOWED
assert classifier.classify("What are current best practices for RAG?") == TTLClass.TIME_WINDOWED
assert classifier.classify("Latest benchmark results for GPT-4o") == TTLClass.TIME_WINDOWED

# PERMANENT
assert classifier.classify("What is cosine similarity?") == TTLClass.PERMANENT
assert classifier.classify("Regex for validating an email address") == TTLClass.PERMANENT
assert classifier.classify("How does backpropagation work?") == TTLClass.PERMANENT
assert classifier.classify("Git command to undo last commit") == TTLClass.PERMANENT
```

---

## Why this is the right first move

Every other semantic caching project silently serves stale answers. Adding
temporal validity with the `stale_miss` hit type gives inferencache
something no competitor has: **honest observability about why a hit was
rejected**. That's a differentiated story developers can immediately
understand and trust.
