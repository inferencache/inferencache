# inferencache

**Multi-tier semantic caching for LLM APIs. Stop paying for the same prompt twice.**

```bash
pip install "inferencache[embed,serve]"
export ANTHROPIC_API_KEY=sk-ant-...
inferencache serve
# landing:   http://localhost:8080/
# dashboard: http://localhost:8080/dashboard/
# proxy:     http://localhost:8080/v1/messages
```

Point Cursor or Claude Code at `http://localhost:8080` — no code changes required.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.
