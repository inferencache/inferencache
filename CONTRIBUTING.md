# Contributing to inferencache

## Repos

| Repo | Purpose |
|------|---------|
| [inferencache/inferencache](https://github.com/inferencache/inferencache) | Python library, proxy server, embedded site assets |
| [inferencache/inferencache-ui](https://github.com/inferencache/inferencache-ui) | Next.js dashboard + landing (dev clone: `inferencache-dashboard`) |

## Local layout

```
SEMANTIC/
  inferencache/           ← Python package
  inferencache-dashboard/ ← Next.js frontend
```

## Build embedded site

```bash
./inferencache/scripts/build-dashboard.sh
```

Copies the Next static export to `src/inferencache/proxy/site/` (gitignored). Tagged releases run this before packaging and bundle the dashboard in the wheel. Local dev can skip it — `inferencache serve` prints a warning and `/` returns 503 until the site is built.

## CLI

```bash
./inferencache/scripts/ensure-site-dir.sh   # empty dir required for pip editable install
pip install -e ".[embed,serve,dev]"
inferencache serve
```

Use `inferencache serve --no-dashboard` for proxy-only mode without site routes.
