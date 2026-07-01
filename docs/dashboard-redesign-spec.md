# Dashboard redesign spec — inferencache-ui

## What this replaces

The current dashboard has four tabs: Cache testing (primary), Analytics,
Tuning, Saved runs. The primary mental model is wrong — it's a research
tool, not a monitoring tool. This spec replaces the entire tab structure
and visual design with a product-first dashboard whose hero metric is
money saved.

All existing API endpoints in `control/router.py` are kept exactly as-is.
This is a pure frontend change.

---

## New tab structure

| Tab | Route | Purpose |
|-----|-------|---------|
| Overview | `/dashboard/` | Hero metric, cache map, live feed strip, top prompts, config summary |
| Cache map | `/dashboard/?tab=map` | Full-screen 2D embedding scatter with hover/click |
| Live | `/dashboard/?tab=live` | Full real-time call log (current Live tab content) |
| Tuning | `/dashboard/?tab=tuning` | Existing Tuning tab content, unchanged |

**Dev mode toggle** — small button in top-right nav. When toggled on,
adds a fifth tab "Dev tools" that contains the existing Cache testing
content. Hidden by default. State stored in `localStorage('devMode')`.

**Removed tabs:** Analytics (absorbed into Overview), Saved runs (moved
to a link inside Dev tools).

---

## Design tokens — dark theme

Hard-code these CSS variables in a root `:root` block in `globals.css`
(or a new `dashboard-theme.css`). These override Tailwind/existing theme:

```css
:root {
  --bg: #0e0f11;
  --surface: #16181c;
  --surface2: #1e2026;
  --border: #2a2d35;
  --text: #e8eaf0;
  --muted: #6b7280;
  --dim: #3d4148;

  --green: #22c55e;
  --green-dim: #0d2b18;
  --green-border: #1a4a2a;
  --green-text: #4ade80;

  --amber: #f59e0b;
  --amber-dim: #2a1f07;
  --amber-border: #3a2a07;
  --amber-text: #fbbf24;

  --blue: #3b82f6;
  --blue-dim: #0d1f3c;

  --purple: #a78bfa;
  --red: #ef4444;
  --red-dim: #2a0f0f;
}
```

---

## Section 1 — Nav / shell (`Navbar.tsx` or equivalent)

Replace the current navbar entirely.

```tsx
<nav> /* h-[44px], bg-[--surface], border-b border-[--border] */
  <div className="logo"> /* "inferencache" + version pill */
  <div className="tabs"> /* Overview | Cache map | Live | Tuning */
  <div className="nav-right"> /* live dot + "localhost:8080" + Dev mode btn */
</nav>
```

**Tab active state:** bottom border `2px solid var(--green)`, text
`var(--text)`. Inactive: text `var(--muted)`, no border.

**Dev mode button:** `font-size: 11px`, `color: var(--dim)`,
`border: 0.5px solid var(--border)`, `border-radius: 4px`,
`background: transparent`. Clicking toggles a `devMode` boolean in
React context + localStorage.

**Live indicator dot:** `6px` circle, `background: var(--green)`.
Pulses with a CSS `@keyframes` opacity animation (0.4 → 1.0, 2s ease).

---

## Section 2 — Overview tab

### 2a — Hero row

Four metric cards in a CSS grid: `grid-template-columns: 2fr 1fr 1fr 1fr`,
`gap: 10px`.

**Card 1 — Money saved (hero):**
```
background: var(--green-dim)
border: 0.5px solid var(--green-border)
border-radius: 12px
padding: 18px 20px
```
- Label: `"Money saved"` — `10px`, `letter-spacing: 0.08em`, `color: #4ade80`, uppercase
- Value: total cumulative cost saved — `38px`, `font-weight: 600`, `color: var(--green-text)`, `letter-spacing: -0.03em`
- Sub: `"vs. paying full API price · this month"` — `11px`, `color: #2d7a45`
- Data source: `GET /api/analytics/cost-saved` → last row's `cumulative_saved`, formatted as `$X.XX`

**Cards 2–4 — Hit rate / Tokens saved / Cache entries:**
```
background: var(--surface)
border: 0.5px solid var(--border)
border-radius: 12px
padding: 14px 16px
```
- Label: `10px`, uppercase, `letter-spacing: 0.07em`, `color: var(--muted)`
- Value: `24px`, `font-weight: 500`, `color: var(--text)`, `letter-spacing: -0.02em`
- Sub: `11px`, `color: var(--muted)`

Data sources:
- Hit rate: `GET /api/stats` → `hit_rate` (format as `XX%`)
- Hit rate sub: `"{hits} of {total} calls"`
- Tokens saved: `GET /api/analytics/cost-saved` → sum of `cost_saved` → derive tokens as `Math.round(cost_saved / 0.000003)`, format as `XXXk`
- Cache entries: `GET /api/stats` → `total_entries`, sub: `"XX MB stored"` (omit MB if unavailable)

Poll all three endpoints on mount and every 10s. Show `—` while loading.

### 2b — Mid row

Two-column grid: `grid-template-columns: 1fr 1fr`, `gap: 14px`.

**Left — Cache map (mini):**
Card with header `"Cache map"` + meta `"X prompts · hover to preview"`.

Inner canvas div: `height: 160px`, `background: var(--bg)`, `border-radius: 8px`, `overflow: hidden`, `position: relative`.

Render dots with JavaScript (see Section 4 for the full Cache map tab).
For the mini version: use the same dot-rendering logic but skip labels,
cap dot count at 200 for performance, and disable click-to-expand.

Legend row below canvas: four colored dots + labels (Code, Explanation,
Q&A, Other). Colors: `#3b82f6`, `#a78bfa`, `#f59e0b`, `#4b5563`.

Data source: `GET /api/stats` → `top_entries` for prompt text.
Dot positions: generate deterministically from `prompt_hash` using a
seeded pseudo-random function (no UMAP required for v0.1 — cluster by
prompt type using simple keyword matching, generate positions within
clusters). This gives a convincing spatial layout without a compute step.

**Right — Top cached prompts:**
Card with header `"Top cached prompts"` + meta `"by hit count"`.

Render 6 bar rows. Each row:
```
[label 88px right-aligned muted] [track flex-1 h-[5px] bg-surface2] [count 22px dim]
```
Bar fill color: blue for code prompts, purple for explanation, amber for Q&A.
Width: `(count / maxCount * 100)%`.

Data source: `GET /api/stats` → `top_entries` array (already sorted by
hit count). Truncate label to 14 chars with ellipsis.

### 2c — Live feed strip

Full-width card below the mid row. Header: `"Live feed"` with green
pulse dot + filter buttons (All / Exact / Semantic / Miss).

Show last 6 calls. Each row:
```
[badge] [prompt preview flex-1 truncated] [latency right-aligned]
```

Badge variants:
- `EXACT` — `background: var(--green-dim)`, `color: var(--green-text)`, `border: 0.5px solid var(--green-border)`
- `SEM X.XX` — `background: var(--amber-dim)`, `color: var(--amber-text)`, `border: 0.5px solid var(--amber-border)`
- `MISS` — `background: var(--surface2)`, `color: var(--dim)`, `border: 0.5px solid var(--border)`

Latency: green for `< 20ms`, muted for `>= 20ms`.

Data source: SSE stream from `GET /api/events`. Existing SSE connection
logic from the current Live tab — reuse the hook, just render the last 6
events here instead of all of them.

Filter buttons toggle a `filter` state string. Filter client-side on the
buffered events array.

### 2d — Bottom row

Two-column grid: `grid-template-columns: 1fr 1fr`, `gap: 14px`.

**Left — Config:**
Card with header `"Config"`. Show 5 config rows + threshold pills.

Each config row: `display: flex; justify-content: space-between`,
`padding: 8px 0`, `border-bottom: 0.5px solid var(--border)`.
Key: `12px`, `color: var(--muted)`. Value: `11px`, monospace,
`color: var(--text)`.

Rows:
1. Similarity threshold → `GET /api/config` or hardcoded `0.85` initially
2. Semantic cache → toggle (on/off)
3. Generative reuse → toggle (on/off)
4. TTL class → value string
5. Embedding model → value string

Toggles: `30px × 16px`, `border-radius: 8px`. On: `background: #16632e`,
thumb right. Off: `background: var(--dim)`, thumb left. Clicking calls
`POST /api/threshold` or appropriate config endpoint.

Below config rows, render threshold pills in a flex-wrap row:
```
CODE 0.92 | DETERMINISTIC 0.95 | RAG 0.88 | CONVERSATIONAL 0.82
```
Each pill: `10px`, `padding: 3px 8px`, `border-radius: 20px`,
`background: var(--surface2)`, `border: 0.5px solid var(--border)`,
`color: var(--muted)`. The number inside: `color: var(--text)`,
`font-weight: 500`.

**Right — Savings breakdown:**
Card with header `"Savings breakdown"`.

Three bar rows (Exact match / Semantic / Generative) using same bar
component as top prompts. Colors: green / amber / purple.
Values from `GET /api/analytics/cost-saved` tier breakdown.

Below bars, a divider then three summary rows:
- Total cost avoided → green, `13px`, `font-weight: 500`
- Cost incurred (generative) → muted
- Net saved → green, `13px`, `font-weight: 500`

---

## Section 3 — Live tab

Keep the existing Live tab content exactly as-is, just re-skin to match
the new dark theme tokens above. Specifically:

- Background: `var(--bg)`
- Cards: `var(--surface)` + `0.5px solid var(--border)`
- Text: `var(--text)` / `var(--muted)` / `var(--dim)`
- Hit badges: same as Section 2c
- Remove the "Cache testing" label — this tab is just "Live"

No functional changes to the SSE connection, call log rows, or drawer.

---

## Section 4 — Cache map tab

Full-page scatter plot of cached prompts projected to 2D.

### Layout
```
[toolbar: search input | color-by dropdown | zoom controls]
[canvas: full remaining height, relative positioned]
[tooltip: absolute, follows cursor]
```

### Dot rendering

For v0.1: generate 2D positions deterministically from prompt content.
Do not call UMAP or any external library — this is a visual approximation
that gives spatial meaning without a compute step.

```ts
function promptToXY(prompt: string, canvasW: number, canvasH: number) {
  // Hash prompt to get stable x/y position
  let h1 = 0, h2 = 0;
  for (let i = 0; i < prompt.length; i++) {
    h1 = (Math.imul(31, h1) + prompt.charCodeAt(i)) | 0;
    h2 = (Math.imul(37, h2) + prompt.charCodeAt(i + 1 || 0)) | 0;
  }
  // Cluster by detected type — pull toward a cluster center
  const type = detectPromptType(prompt); // returns 'code'|'explanation'|'qa'|'other'
  const centers = {
    code:        { x: 0.25, y: 0.35 },
    explanation: { x: 0.55, y: 0.48 },
    qa:          { x: 0.75, y: 0.28 },
    other:       { x: 0.62, y: 0.70 },
  };
  const c = centers[type];
  const noise = 0.14;
  const nx = ((h1 >>> 0) / 0xffffffff) * noise * 2 - noise;
  const ny = ((h2 >>> 0) / 0xffffffff) * noise * 2 - noise;
  return {
    x: Math.max(8, Math.min(canvasW - 8, (c.x + nx) * canvasW)),
    y: Math.max(8, Math.min(canvasH - 8, (c.y + ny) * canvasH)),
  };
}

function detectPromptType(prompt: string): 'code' | 'explanation' | 'qa' | 'other' {
  const p = prompt.toLowerCase();
  if (/\b(function|class|import|def |const |var |git |test|debug|fix|refactor|implement)\b/.test(p)) return 'code';
  if (/\b(explain|what is|how does|describe|summarize|why)\b/.test(p)) return 'explanation';
  if (/\b(what|when|where|who|which|how many|how much|\?)\b/.test(p)) return 'qa';
  return 'other';
}
```

### Canvas rendering

Use a `<canvas>` element, not DOM divs (performance for 1000+ dots).

```ts
function renderMap(ctx: CanvasRenderingContext2D, entries: CacheEntry[], W: number, H: number) {
  ctx.clearRect(0, 0, W, H);
  const colors = { code: '#3b82f6', explanation: '#a78bfa', qa: '#f59e0b', other: '#4b5563' };

  for (const entry of entries) {
    const { x, y } = promptToXY(entry.prompt, W, H);
    const type = detectPromptType(entry.prompt);
    const radius = 3 + Math.min(entry.hit_count / 5, 4); // bigger = more hits
    const alpha = 0.5 + Math.min(entry.hit_count / 20, 0.5);

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = colors[type];
    ctx.globalAlpha = alpha;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}
```

### Hover tooltip

On `mousemove` over canvas:
1. Find nearest dot within 20px radius
2. Show tooltip `div` (absolute positioned) with:
   - Prompt text (first 120 chars)
   - Hit type badge
   - Hit count
   - `color: var(--text)`, `background: var(--surface2)`,
     `border: 0.5px solid var(--border)`, `border-radius: 8px`,
     `padding: 8px 12px`, `font-size: 12px`, `max-width: 280px`

### Click to expand

Clicking a dot opens a slide-in panel (right side, `width: 320px`) showing:
- Full prompt text
- Full cached response
- Hit count, last hit timestamp, similarity score
- "Clear this entry" button → `POST /api/clear` with prompt hash

### Data source

`GET /api/stats` → `top_entries` for the initial set.

For the full map, add a new endpoint to `control/router.py`:

```python
@router.get("/api/entries")
async def get_entries(model: str = "gpt-4o-mini", limit: int = 500):
    """Return cache entries for the map view."""
    engine = get_engine(model)
    entries = engine.cache_store.list_entries(limit=limit)
    return {"entries": entries}
```

And add `list_entries()` to `CacheStore` in `store.py`:

```python
def list_entries(self, limit: int = 500) -> list[dict]:
    conn = sqlite3.connect(str(self._db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT prompt_hash, prompt, model, created_at,
               hit_count, last_hit_at
        FROM entries
        ORDER BY hit_count DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
```

---

## Section 5 — Tuning tab

Keep all existing content. Re-skin only:

- Wrap in `background: var(--bg)`
- Cards: `background: var(--surface)`, `border: 0.5px solid var(--border)`
- Orange/amber accents → keep (they're intentional for warning/threshold UI)
- The "Apply" button stays amber — it's a destructive config change
- Text colors → `var(--text)` / `var(--muted)`

No functional changes.

---

## Section 6 — Dev mode / Dev tools tab

When `devMode === true` (localStorage + React context):

1. A fifth tab "Dev tools" appears in the nav
2. Its content = the current "Cache testing" tab, unchanged
3. "Saved runs" link appears as a sub-nav item inside Dev tools

This preserves the existing research workflow without exposing it to
normal users.

---

## Section 7 — Files to change

| File | Change |
|------|--------|
| `frontend-next/src/lib/dashboardNav.tsx` | Replace NAV_ITEMS with new 4-tab structure; add dev mode logic |
| `frontend-next/src/app/dashboard/page.tsx` | Replace with Overview tab layout (Sections 2a–2d) |
| `frontend-next/src/app/dashboard/layout.tsx` | Replace Navbar with new shell (Section 1) |
| `frontend-next/src/components/LiveFeed.tsx` | New component — live feed strip for Overview (Section 2c) |
| `frontend-next/src/components/CacheMapMini.tsx` | New component — mini dot scatter for Overview (Section 2b) |
| `frontend-next/src/components/CacheMap.tsx` | New component — full canvas scatter for Cache map tab (Section 4) |
| `frontend-next/src/components/MetricCard.tsx` | New component — hero + regular metric cards |
| `frontend-next/src/components/SavingsBreakdown.tsx` | New component — breakdown bars + summary rows |
| `frontend-next/src/styles/dashboard-theme.css` | New file — CSS variables (Section 0) |
| `frontend-next/src/app/dashboard/globals.css` | Import dashboard-theme.css; remove conflicting light-mode overrides |
| `src/inferencache/proxy/control/router.py` | Add `GET /api/entries` endpoint (Section 4) |
| `src/inferencache/store.py` | Add `list_entries()` method (Section 4) |

**Do not change:**
- `control/runner.py` — test suite runner logic untouched
- `control/db.py` — saved runs DB untouched
- `analytics.py` — analytics queries untouched
- `intercept.py`, `forward.py`, `server.py` — proxy untouched
- Any existing test files

---

## Section 8 — Verification

After implementation:

```bash
# 1. Build dashboard
cd frontend-next
npm run build

# 2. Copy to proxy site dir
cp -r out/* ../src/inferencache/proxy/site/

# 3. Start proxy
cd ../../
inferencache serve

# 4. Open http://localhost:8080/dashboard
# Confirm:
# - Dark background (#0e0f11)
# - "Money saved" hero card is green, shows $0.00 initially
# - Four tabs visible: Overview | Cache map | Live | Tuning
# - No "Cache testing" or "Saved runs" tabs visible
# - "Dev mode" button in top right
# - Clicking "Dev mode" reveals a 5th "Dev tools" tab

# 5. Run cache_tester.py --script
# Confirm Overview updates live:
# - Money saved ticks up after hits
# - Live feed strip shows rows appearing
# - Hit rate % updates

# 6. Click "Cache map" tab
# Confirm dots render on canvas
# Hover a dot → tooltip shows prompt text
# Click a dot → side panel opens with full prompt + response
```
