# Repo organization playbook

## Context — what exists and what's broken

Two repos side by side on disk:
```
inferencache/          ← Python library + proxy (runs on :8080)
inferencache-ui/       ← Next.js dashboard + dead Python backend
  frontend-next/       ← the actual dashboard UI
  backend/             ← OLD standalone backend (:8000) — ignore this entirely
```

The goal is one command that starts everything:
```bash
inferencache serve
# proxy:     http://localhost:8080
# dashboard: http://localhost:8080/dashboard
```

Why it's broken right now:
1. `frontend-next/.env.local` is gitignored and missing — so API calls go nowhere
2. `next.config.mjs` has no `basePath: '/dashboard'` — so built assets serve from wrong paths
3. `inferencache/src/inferencache/proxy/site/` is empty — proxy has no HTML to serve
4. The old `backend/` Python server on :8000 is dead — never run it

---

## Step 1 — Fix next.config.mjs

File: `inferencache-ui/frontend-next/next.config.mjs`

Replace the entire file with:

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  basePath: "/dashboard",
  trailingSlash: true,
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
```

The `basePath: "/dashboard"` is the critical addition. Without it, the
built assets reference paths like `/_next/static/...` instead of
`/dashboard/_next/static/...`, so the proxy serves broken HTML.

---

## Step 2 — Fix the API base URL

File: `inferencache-ui/frontend-next/src/lib/api.ts`

Find the line that sets `BASE`. It looks like:
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
```
or similar. Replace the fallback value:
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080/api";
```

This makes the dashboard talk to the proxy's `/api` when no env var is set,
instead of the dead :8000 backend.

---

## Step 3 — Create .env.local for local dev

Create file: `inferencache-ui/frontend-next/.env.local`

```
NEXT_PUBLIC_API_URL=http://localhost:8080/api
```

This file is gitignored (correct — don't remove it from .gitignore).
It just needs to exist locally so `npm run dev` works against the real proxy.

---

## Step 4 — Fix pyproject.toml URLs

File: `inferencache/pyproject.toml`

Find the `[project.urls]` section and update both URLs:
```toml
[project.urls]
Homepage = "https://github.com/inferencache/inferencache"
Repository = "https://github.com/inferencache/inferencache"
Issues = "https://github.com/inferencache/inferencache/issues"
```

---

## Step 5 — Build the dashboard and copy into proxy

Run these commands in order from the terminal. Do not use Cursor agent for
this step — run them manually in a shell.

```bash
# 1. Go into the frontend
cd inferencache-ui/frontend-next

# 2. Install deps if needed
npm install

# 3. Build static export
npm run build
# This creates inferencache-ui/frontend-next/out/

# 4. Wipe and repopulate the proxy site dir
rm -rf ../../inferencache/src/inferencache/proxy/site/*
cp -r out/* ../../inferencache/src/inferencache/proxy/site/

# 5. Confirm it populated
ls ../../inferencache/src/inferencache/proxy/site/
# Should show: _next/  dashboard/  index.html  (and other files)
```

---

## Step 6 — Verify the proxy serves the dashboard

```bash
# Start the proxy
cd ../../inferencache
inferencache serve
```

Expected output:
```
inferencache proxy  →  http://127.0.0.1:8080
dashboard          →  http://127.0.0.1:8080/dashboard
```

Open `http://localhost:8080/dashboard` in a browser.

You should see the dashboard UI load (not a blank page, not a 404).

Then hit the health endpoint to confirm the API is reachable:
```bash
curl http://localhost:8080/api/health
# Expected: {"status": "ok", "cache_dir": "..."}
```

---

## Step 7 — Smoke test the full flow

With `inferencache serve` still running, open a second terminal:

```bash
# Send a request through the proxy
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  -i | grep -E "X-Cache|HTTP/"
```

First call: `HTTP/1.1 200` with no `X-Cache` header (miss, forwarded to OpenAI).

Send the exact same request again:
```bash
curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"say hi"}],"stream":false}' \
  -i | grep -E "X-Cache|HTTP/"
```

Second call: should show `X-Cache: exact`.

Open the dashboard at `http://localhost:8080/dashboard` — the Analytics tab
should show a non-zero hit rate and the Live tab should show the two calls.

---

## Step 8 — Update the build script

File: `inferencache/scripts/build-dashboard.sh`

The existing script references `inferencache-dashboard` but the repo is now
`inferencache-ui`. Update the `DASHBOARD_REPO` default:

Find:
```bash
DASHBOARD_REPO="${DASHBOARD_REPO:-$(cd "$INFERENCACHE_REPO/../inferencache-dashboard" && pwd)}"
```

Replace with:
```bash
DASHBOARD_REPO="${DASHBOARD_REPO:-$(cd "$INFERENCACHE_REPO/../inferencache-ui" && pwd)}"
```

---

## What NOT to touch

- `inferencache-ui/backend/` — dead code, leave it, don't run it, don't delete it yet
- `inferencache/src/inferencache/proxy/site/.gitignore` — leave as-is (site/* is gitignored by design, populated at build time)
- Any Python files in `inferencache/src/` — this playbook is infrastructure only
- Existing tests — nothing here affects them

---

## How to develop going forward

**Changing Python proxy code:**
```bash
cd inferencache
# edit src/inferencache/proxy/...
inferencache serve   # restart to pick up changes
```

**Changing dashboard UI:**
```bash
cd inferencache-ui/frontend-next
npm run dev          # dev server on :3000, hits proxy at :8080
# edit components, pages, etc.
# when done:
npm run build
cp -r out/* ../../inferencache/src/inferencache/proxy/site/
# restart inferencache serve
```

**The only port that matters for end users: :8080.**
Port :3000 is only for dashboard development.
Port :8000 is dead — ignore it.
