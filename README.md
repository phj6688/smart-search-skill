# smart-search-skill

> Privacy-first hybrid web search for agents and workflows — no commercial search API required.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/search-workflow)](https://pypi.org/project/search-workflow/)

---

## Features

- 🔍 **Parallel hybrid search** — SearXNG + DuckDuckGo run simultaneously via `asyncio.gather`
- 🛡️ **Privacy-first** — self-hosted SearXNG, no queries sent to commercial APIs
- ⚙️ **Configurable categories** — `general`, `news`, `it`, or combined per query
- 🔄 **Resilient fallback** — SearXNG down? DuckDuckGo fills automatically
- 🤖 **AI-evaluated results** — LLM ranks and filters results for relevance
- 📦 **LangGraph-compatible** — drop-in tool for any LangGraph agent
- 🔌 **OpenClaw skill** — invoke directly from any OpenClaw agent via `SKILL.md`

---

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` — for AI result evaluation
- SearXNG (optional) — runs via Docker; DuckDuckGo is the automatic fallback

---

## Installation

```bash
pip install search-workflow
# or
uv add search-workflow
```

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | OpenAI API key for AI result evaluation |
| `SEARXNG_SECRET` | ✅ Yes (Docker) | Random secret for SearXNG session signing |
| `SEARXNG_URL` | No | SearXNG instance URL (default: `http://localhost:9090`) |
| `MAX_SEARCH_RESULTS_TOOL` | No | Max results fetched per engine (default: `10`) |
| `MAX_SEARCH_RESULTS_EVALUATOR` | No | Max results returned after AI eval (default: `5`) |

Generate a `SEARXNG_SECRET`:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

---

## Quick Start

### Python Package

```python
import asyncio
from search_workflow import run_workflow

async def main():
    results = await run_workflow("FastAPI authentication JWT tutorial")
    for r in results:
        print(f"• {r['title']}")
        print(f"  {r['link']}")

asyncio.run(main())
```

### CLI

```bash
# Basic query (JSON output)
python -m search_workflow "neo4j python driver documentation"

# Limit results
python -m search_workflow "docker networking" --max-results 3

# Human-readable output
python -m search_workflow "LLM benchmarks" --format text

# Time-filtered (d=day, w=week, m=month, y=year)
python -m search_workflow "AI news" --timelimit w
```

---

## OpenClaw Skill

The `SKILL.md` at the repo root makes this an installable OpenClaw skill.
Any OpenClaw agent (AXIOM, LUMO, etc.) can invoke it directly.

### Install

```bash
# From the repo root — publishes or installs locally
clawhub install .
```

### Invoke from any OpenClaw agent

Tell any agent:
> "Search for: FastAPI rate limiting best practices"

The agent reads `SKILL.md`, invokes the CLI, and returns structured results.

### Direct CLI invocation

```bash
python -m search_workflow "your query" --max-results 5 --format json
```

Output:
```json
[
  {
    "title": "Result title",
    "link": "https://example.com",
    "snippet": "Short description..."
  }
]
```

---

## LangGraph Integration

### Drop-in Tool

```python
from search_workflow.tools import TOOLS  # list containing [search]
from search_workflow.graph import run_workflow

config = {
    "configurable": {
        "max_search_results_tool": 10,
        "max_search_results_evaluator": 5,
        "searxng_url": "http://localhost:9090",
        "model": "gpt-4o-mini",
    }
}
results = await run_workflow("Python asyncio best practices", config=config)
```

### Direct Search (no LangGraph)

```python
from search_workflow.tools import search_direct

results = await search_direct(
    query="neo4j python driver",
    max_results=10,
    searxng_url="http://localhost:9090",
    categories="it",
)
```

---

## Docker / SearXNG Setup

SearXNG runs as a local Docker container — no external service needed.

```bash
cd docker
cp .env.example .env          # fill in SEARXNG_SECRET and OPENAI_API_KEY
docker compose up -d
# SearXNG available at http://localhost:9090
```

### Recommended SearXNG tuning (already applied in `docker/config/settings.yml`)

```yaml
search:
  suspended_times:
    SearxEngineAccessDenied: 900     # 15 min instead of 24h
    SearxEngineCaptcha: 900
    SearxEngineTooManyRequests: 300  # 5 min
    cf_SearxEngineCaptcha: 3600      # 1h instead of 15 days
```

Default SearXNG suspends engines for 24h on rate limits — this config reduces it to 15 minutes.

---

## Architecture

```
query
  │
  ├─► SearXNG (categories=general)  ─┐
  │   asyncio.gather (parallel)      ├─► merge + dedup by URL ──► LLM eval ──► results
  └─► DuckDuckGo (DDGS text)        ─┘
```

Both engines run in parallel. Results are merged and deduplicated by URL.
If SearXNG returns 0 results, DuckDuckGo fills transparently.
The LLM evaluator ranks results for relevance before returning.

---

## Development

```bash
git clone https://github.com/phj6688/smart-search-skill.git
cd smart-search-skill
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run black src/
```

---

## Security

- No API keys stored in code
- SearXNG runs on localhost, not exposed publicly
- `.env` files are gitignored (`chmod 600` recommended)
- Docker container runs as non-root (SearXNG default)

---

## License

MIT License — © 2025 Peyman
