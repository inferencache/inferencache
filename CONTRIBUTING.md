# Contributing to inferencache

## Repos

| Repo | Purpose |
|------|---------|
| [lavondev/inferencache](https://github.com/lavondev/inferencache) | Python library, proxy server, embedded site assets |
| [lavondev/inferencache-ui](https://github.com/lavondev/inferencache) | Next.js dashboard + landing (dev clone: `promptcache-dashboard`) |

## Local layout

```
SEMANTIC/
  promptcache/           ← Python package (inferencache on PyPI)
  promptcache-dashboard/ ← Next.js frontend
```

## Build embedded site

```bash
./promptcache/scripts/build-dashboard.sh
```

Copies the Next static export to `src/inferencache/proxy/site/`.

## CLI

```bash
pip install -e ".[embed,serve]"
inferencache serve
```
