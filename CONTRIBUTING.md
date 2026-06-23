# Contributing to inferencache

## Repos

| Repo | Purpose |
|------|---------|
| [lavondev/inferencache](https://github.com/lavondev/inferencache) | Python library, proxy server, embedded site assets |
| [lavondev/inferencache-ui](https://github.com/lavondev/inferencache) | Next.js dashboard + landing (dev clone: `inferencache-dashboard`) |

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

Copies the Next static export to `src/inferencache/proxy/site/` (gitignored). Release builds run this before packaging; local dev can skip it — the proxy serves a JSON hint until the site is built.

## CLI

```bash
./inferencache/scripts/ensure-site-dir.sh   # empty dir required for pip editable install
pip install -e ".[embed,serve,dev]"
inferencache serve
```
