Let me get current on what prefix and inference caching actually look like at the provider level before designing this.Good. I now have a clear picture of exactly how each tier works at the provider level. Let me design the full architecture.

---

## What the three tiers actually are

Before the design, be precise about what each tier does — because they operate at completely different levels and most people conflate them.

**Tier 1 — Semantic cache (your layer).** Sits entirely on the client side, before any API call. Intercepts the full prompt, checks exact hash then embedding similarity, returns a stored response if close enough. The API never sees the request. Zero tokens consumed. This is what promptcache does today.

**Tier 2 — Prefix cache (provider's layer).** Happens server-side at OpenAI or Anthropic. OpenAI routes API requests to servers that recently processed the same prompt, making it cheaper and faster than processing a prompt from scratch. Cache hits are only possible for exact prefix matches within a prompt. The API caches the longest prefix of a prompt that has been previously computed, starting at 1,024 tokens and increasing in 128-token increments. This is about the *structure* of the prompt you send — stable content up front, dynamic content at the end. The request still goes to the API, tokens are still consumed, but input tokens in the cached prefix are billed at 50–90% discount.

**Tier 3 — Inference cache (response reuse).** Some requests deserve identical responses — repeated questions, deterministic queries, or lookups that don't require regeneration. This is exact-match response caching at the provider level — if you send the exact same tokens, the provider returns a stored response without running inference. OpenAI does this automatically. Anthropic's `cache_control: ephemeral` marker enables it with a 5-minute TTL, extendable to 1 hour at additional cost.

The critical insight: **these three tiers are independent and additive.** A single LLM call can benefit from all three simultaneously, or from none. The reason nobody has built a product around this is that Tier 1 is your code, Tier 2 requires prompt structure discipline, and Tier 3 requires provider-specific API parameters. Orchestrating all three and making the savings visible from each tier is the product.

---

## System architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your application                         │
│                                                             │
│   @cache(config=config, tier="auto")                        │
│   def ask(prompt, system_prompt, context):                  │
│       return llm_call(...)                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              promptcache Orchestration Layer                 │
│                                                             │
│   1. Route decision engine                                  │
│      - Classify prompt type (code / conversational / RAG)   │
│      - Estimate tier suitability                            │
│      - Check session context (multi-turn aware)             │
│                                                             │
│   2. Tier 1: Semantic cache check                           │
│      - SHA-256 exact match (SQLite)                         │
│      - Embedding similarity search (Qdrant)                 │
│      - Session-aware context hash                           │
│      → HIT: return stored response, log savings             │
│                                                             │
│   3. Tier 2: Prefix optimizer (on cache miss)               │
│      - Analyze prompt structure                             │
│      - Inject cache_control markers (Anthropic)             │
│      - Reorder stable vs dynamic content                    │
│      - Set prompt_cache=True (OpenAI automatic)             │
│                                                             │
│   4. Tier 3: Inference cache (pass-through)                 │
│      - Provider handles automatically on repeated tokens    │
│      - Read cached_tokens from response.usage               │
│                                                             │
│   5. Write-back + event logging                             │
│      - Store response in Tier 1 cache                       │
│      - Log: which tiers fired, tokens saved per tier        │
│      - Update savings estimates                             │
└──────────────────────────┬──────────────────────────────────┘
                           │ (only if Tier 1 misses)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    LLM Provider API                         │
│                                                             │
│   Anthropic                      OpenAI                     │
│   ├─ cache_control: ephemeral    ├─ Automatic prefix cache  │
│   ├─ 5min / 1hr TTL             ├─ 50% discount on hits    │
│   ├─ 90% discount on reads      └─ usage.cached_tokens      │
│   └─ 25% write premium                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## The route decision engine

This is the new brain that doesn't exist in the current promptcache. It runs before any cache check and answers: which tiers are worth attempting for this prompt?

```python
class TierRouter:
    """
    Classifies incoming prompts and decides which caching tiers
    to attempt and in what configuration.
    """

    def route(self, prompt: str, context: CallContext) -> TierDecision:
        prompt_type   = self._classify(prompt)
        session_depth = context.turn_count
        prompt_tokens = self._estimate_tokens(prompt, context)

        return TierDecision(
            # Tier 1: Always attempt semantic cache
            semantic=SemanticConfig(
                enabled=True,
                threshold=self._threshold_for_type(prompt_type),
                session_aware=(session_depth > 1),
            ),

            # Tier 2: Prefix optimization
            # Only valuable if prompt is long enough and has stable prefix
            prefix=PrefixConfig(
                enabled=prompt_tokens > 1024,
                stable_prefix=context.system_prompt,
                inject_cache_control=(context.provider == "anthropic"),
                ttl="1hr" if context.session_frequency == "high" else "5min",
            ),

            # Tier 3: Inference cache
            # Provider handles automatically, we just read the signal
            inference=InferenceConfig(
                enabled=True,
                read_cached_tokens=True,
            ),
        )

    def _classify(self, prompt: str) -> PromptType:
        # Code queries: dense embedding clusters, high cache hit potential
        # Research shows 40-60% hit rate vs 5-15% for conversational
        if self._is_code_query(prompt):
            return PromptType.CODE

        # Deterministic lookups: exact answers, high confidence in reuse
        elif self._is_deterministic(prompt):
            return PromptType.DETERMINISTIC

        # RAG queries with retrieved context: prefix cache valuable
        elif self._has_long_context(prompt):
            return PromptType.RAG

        # Conversational: lower semantic hit expectation, context matters
        else:
            return PromptType.CONVERSATIONAL

    def _threshold_for_type(self, prompt_type: PromptType) -> float:
        # VectorQ insight: per-category thresholds outperform single global threshold
        return {
            PromptType.CODE:            0.92,  # code is precise, high bar
            PromptType.DETERMINISTIC:   0.95,  # facts need exact match
            PromptType.RAG:             0.88,  # context-dependent
            PromptType.CONVERSATIONAL:  0.82,  # more flexible
        }[prompt_type]
```

The per-type threshold is the VectorQ insight applied practically. Code queries cluster tightly in embedding space so a higher threshold is safe and reduces false positives. Conversational queries need a lower threshold to catch paraphrases.

---

## The prefix optimizer (Tier 2)

This is what promptcache doesn't do today but what unlocks the 50–90% input token savings even when Tier 1 misses. It restructures the prompt before sending it to the provider.

```python
class PrefixOptimizer:
    """
    Restructures prompts to maximize provider-side prefix cache hit rate.

    Rule: stable content first, dynamic content last.
    The provider caches the longest matching prefix — any change in the
    prefix invalidates the entire cache entry.
    """

    def optimize(
        self,
        prompt: str,
        system_prompt: str,
        context: list[Message],
        config: PrefixConfig,
    ) -> OptimizedRequest:

        if config.provider == "anthropic":
            return self._optimize_anthropic(system_prompt, context, config)
        elif config.provider == "openai":
            return self._optimize_openai(system_prompt, context)
        else:
            return OptimizedRequest(messages=context, modified=False)

    def _optimize_anthropic(self, system_prompt, context, config):
        """
        Anthropic requires explicit cache_control markers.
        Structure: tools → system (cached) → conversation → user query
        """
        system_with_cache = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {
                    "type": "ephemeral",
                    # Use 1-hour TTL for high-frequency sessions
                    # Default 5-minute TTL for sporadic calls
                    **({"ttl": 3600} if config.ttl == "1hr" else {})
                }
            }
        ]

        # Mark the conversation history as cacheable prefix too
        # (stable across turns, changes only at the end)
        optimized_messages = []
        for i, msg in enumerate(context[:-1]):  # all but last message
            if i == len(context) - 2:  # mark last historical message
                optimized_messages.append({
                    **msg,
                    "cache_control": {"type": "ephemeral"}
                })
            else:
                optimized_messages.append(msg)

        # Current user message — this is always dynamic, never cached
        optimized_messages.append(context[-1])

        return OptimizedRequest(
            system=system_with_cache,
            messages=optimized_messages,
            modified=True,
            expected_savings_tier="prefix",
        )

    def _optimize_openai(self, system_prompt, context):
        """
        OpenAI caching is automatic — no markers needed.
        Strategy: ensure system prompt is identical across requests
        (even small changes break the cache) and push dynamic content
        to the end of the message array.
        """
        # Warn if system prompt contains dynamic content
        dynamic_indicators = ["{user}", "{date}", "{session_id}", str(datetime.now().year)]
        has_dynamic = any(d in system_prompt for d in dynamic_indicators)

        return OptimizedRequest(
            messages=context,
            modified=False,
            warnings=["System prompt contains dynamic content — will reduce prefix cache hits"]
            if has_dynamic else [],
        )
```

---

## Session-aware Tier 1 (ContextCache insight)

The current promptcache matches single prompts. For Cursor users this is wrong — "fix this function" means something completely different in two different sessions even though the embeddings are similar. The fix is a session context hash as part of the cache key.

```python
class SessionAwareLookup:
    """
    Extends exact + semantic matching with session context awareness.
    Prevents false hits where identical prompts appear in different
    conversation contexts (critical for Cursor / Claude Code users).
    """

    def lookup(
        self,
        prompt: str,
        session_history: list[str],
        model: str,
        threshold: float,
    ) -> CacheResult:

        # Build session context signature
        # Hash of the last N turns — similar to ContextCache's two-stage approach
        session_hash = self._session_hash(session_history, window=3)

        # Check 1: exact match with session context
        result = self.store.get_exact(
            prompt=prompt,
            model=model,
            session_hash=session_hash,
        )
        if result:
            return CacheResult(hit=True, hit_type="exact", source="tier1")

        # Check 2: exact match ignoring session (for session-agnostic prompts)
        # Only use if prompt is classified as stateless (e.g., factual lookup)
        if self._is_stateless(prompt):
            result = self.store.get_exact(prompt=prompt, model=model)
            if result:
                return CacheResult(hit=True, hit_type="exact", source="tier1_stateless")

        # Check 3: semantic match with session context filtering
        embedding = self.embedder.embed(prompt)
        hits = self.store.query_semantic(
            embedding=embedding,
            model=model,
            threshold=threshold,
            session_hash=session_hash,  # filter to same session context
        )
        if hits:
            return CacheResult(hit=True, hit_type="semantic", source="tier1_session")

        return CacheResult(hit=False, hit_type="miss")

    def _session_hash(self, history: list[str], window: int) -> str:
        """
        Hash the last N messages to create a session context fingerprint.
        Window of 3 captures enough context without being too specific.
        """
        recent = history[-window:] if len(history) >= window else history
        combined = "|||".join(recent)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _is_stateless(self, prompt: str) -> bool:
        """
        Stateless prompts: factual questions, definitions, code snippets
        with no reference to prior context. Safe to match across sessions.
        """
        stateless_patterns = [
            r"^what is ",
            r"^explain ",
            r"^define ",
            r"^write a .* function",
            r"^how do(es)? ",
        ]
        return any(re.match(p, prompt.lower()) for p in stateless_patterns)
```

---

## Savings attribution per tier

This is the dashboard feature that doesn't exist anywhere else. Every call event gets tagged with which tier saved money and how much. The analytics tab can then break down savings by tier, which tells a user exactly what they're getting from each layer.

```python
@dataclass
class SavingsEvent:
    """Written to the calls table on every LLM call."""
    call_id:              int
    prompt_hash:          str
    model:                str
    provider:             str
    timestamp:            float

    # Tier 1 — semantic cache
    tier1_hit:            bool
    tier1_hit_type:       str    # 'exact' | 'semantic' | 'miss'
    tier1_similarity:     float | None
    tier1_tokens_saved:   int    # full response tokens if hit, else 0
    tier1_cost_saved:     float

    # Tier 2 — prefix cache (only populated when Tier 1 misses)
    tier2_cached_input_tokens:  int    # from provider response.usage
    tier2_fresh_input_tokens:   int
    tier2_cost_saved:           float  # (cached_tokens * fresh_rate) - (cached_tokens * cached_rate)

    # Tier 3 — inference cache (provider handles)
    tier3_hit:            bool    # True if response.usage.cached_tokens == total_tokens
    tier3_tokens_saved:   int
    tier3_cost_saved:     float

    # Totals
    total_cost_saved:     float
    total_tokens_saved:   int
    total_latency_ms:     float
    tier1_latency_ms:     float  # time spent in semantic lookup
    api_latency_ms:       float  # time spent waiting for provider (0 on Tier 1 hit)
```

---

## The new public API surface

The key design decision: the existing `@cache` decorator keeps working unchanged. The multi-tier system is opt-in with `tier="auto"`.

```python
# Existing usage — unchanged, still works
@cache(config=CacheConfig(model="gpt-4o"))
def ask(prompt: str) -> str:
    return llm_call(prompt)

# Multi-tier — opt in with tier="auto"
@cache(
    config=CacheConfig(
        model="claude-3-5-sonnet-20241022",
        provider="anthropic",
        tier="auto",                 # enables all three tiers
        session_aware=True,          # enables ContextCache-style matching
        prefix_ttl="1hr",            # Anthropic cache TTL preference
    )
)
def ask(
    prompt: str,
    system_prompt: str,
    history: list[dict],
) -> str:
    return anthropic_call(system_prompt, history, prompt)

# Result always has savings breakdown
result = ask.last_result  # CacheResult with tier1/2/3 attribution
```

---

## New module structure

```
src/promptcache/
├── __init__.py
├── embed.py          # unchanged
├── store.py          # + session_hash column in calls table
├── engine.py         # + TierDecision, multi-tier orchestration
├── api.py            # + tier= param, session_aware= param
├── analytics.py      # + per-tier savings breakdown queries
├── cli.py            # + tier breakdown in stats output
├── router.py         # NEW — TierRouter, prompt classifier
├── prefix.py         # NEW — PrefixOptimizer per provider
├── session.py        # NEW — SessionAwareLookup, context hashing
└── mcp/
    ├── server.py
    └── tools.py      # + get_tier_breakdown tool
```

---

## What this means for the dashboard

The Analytics tab gets a new panel: **Savings by tier**. Three rows — Tier 1 (semantic, your savings), Tier 2 (prefix, provider discount you engineered), Tier 3 (inference, provider discount you got automatically). Each row shows tokens saved and dollars saved for the selected time window.

This panel is the product story in one table. It shows users that promptcache is doing something none of their existing tools are doing — actively orchestrating all three layers and making the combined savings visible. The Tier 2 row in particular is new value — most developers have no idea how many tokens they're saving from prefix cache or whether their prompt structure is optimal for it. You're showing them that number and giving them the tools to improve it.

The Tuning tab gets a **Prefix optimizer** section: shows the stability score of the user's system prompt across requests (how often does it change?), flags dynamic content that's breaking prefix cache hits, and suggests restructuring. For a Cursor user whose system prompt includes the current file path, this would immediately flag "your system prompt changes every request — move the file path to the user message."

---

## Implementation priority

Build in this order because each tier compounds on the one before:

**Phase 1** — attribution infrastructure. Add `tier1_*`, `tier2_*`, `tier3_*` columns to the `calls` table and read `usage.cached_tokens` from provider responses. This unlocks the per-tier dashboard immediately without any routing logic.

**Phase 2** — prefix optimizer. Implement `PrefixOptimizer` for Anthropic and OpenAI. This is the highest-leverage Tier 2 improvement — most developers aren't structuring prompts correctly for prefix caching and this fixes it automatically.

**Phase 3** — session-aware Tier 1. Add session context hashing to the existing cache lookup. This is the most important fix for Cursor users and the thing that makes Tier 1 reliable in multi-turn contexts.

**Phase 4** — route decision engine. Once you have real usage data from Phases 1–3, the classifier can be trained on actual prompt patterns rather than heuristics. Build heuristics first, replace with learned behavior once you have data.