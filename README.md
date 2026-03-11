# search-workflow

A self-hosted, privacy-first search package powered by **SearXNG** with **DuckDuckGo** fallback. Built for technical teams who want fast, reliable search over docs, GitHub, Stack Overflow, and the general web — without sending queries to commercial APIs.

## Features

- 🔍 **Parallel hybrid search** — SearXNG + DuckDuckGo run simultaneously via `asyncio.gather`, results merged and deduplicated
- 🛡️ **Self-hosted** — SearXNG runs in your own Docker container; no external API keys required
- ⚙️ **Configurable categories** — pass `categories="general"`, `"news"`, `"it"`, `"general,news"` per query
- 🔄 **Resilient fallback** — if SearXNG returns 0 results, DDG fills the gap automatically
- 📦 **LangGraph-compatible** — drop-in tool for LangGraph agent workflows
- 🧪 **Tested** — 100% query coverage on 10-query benchmark, avg 10 results/query

## Quick Start

### Installation

```bash
pip install search-workflow
# or
uv add search-workflow
```

### Prerequisites

Run SearXNG locally via Docker:

```bash
cd docker
docker compose up -d
# SearXNG available at http://localhost:9090
```

### Basic Usage

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

### Direct Search (no LangGraph)

```python
from search_workflow.tools import search_direct

results = await search_direct(
    query="neo4j python driver documentation",
    max_results=10,
    searxng_url="http://localhost:9090",
    categories="general",        # or "news", "it", "general,news"
)
```

### LangGraph Tool

```python
from search_workflow.tools import TOOLS  # [search]
from search_workflow.graph import run_workflow

# With custom config
config = {
    "configurable": {
        "max_search_results_tool": 10,
        "searxng_url": "http://localhost:9090",
        "model": "claude-3-5-haiku-20241022",
    }
}
results = await run_workflow("Python asyncio best practices", config=config)
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `searxng_url` | `http://localhost:9090` | SearXNG instance URL |
| `categories` | `"general"` | SearXNG search categories |
| `max_results` | `10` | Max results per query |
| `searxng_timeout` | `12` | SearXNG request timeout (seconds) |
| `language` | `"en"` | Search language |
| `time_range` | `None` | `"d"`, `"w"`, `"m"`, `"y"` |

## SearXNG Setup

The `docker/` directory includes a production-ready SearXNG configuration.

**Recommended `settings.yml` tuning** (already applied in `docker/config/settings.yml`):

```yaml
search:
  suspended_times:
    SearxEngineAccessDenied: 900    # 15 min (not 24h)
    SearxEngineCaptcha: 900         # 15 min
    SearxEngineTooManyRequests: 300 # 5 min
    cf_SearxEngineCaptcha: 3600     # 1h (not 15 days)
```

Default SearXNG suspends engines for 24h on rate limits — this reduces it to 15 minutes for self-hosted instances.

## Architecture

```
query
  │
  ├─► SearXNG (categories=general)  ─┐
  │   asyncio.gather (parallel)      ├─► merge + dedup by link ──► results[:max]
  └─► DuckDuckGo (DDGS text)        ─┘
```

Both engines run in parallel. Results are merged and deduplicated by URL. If SearXNG returns 0 (e.g. engine suspended), DDG results are used transparently.

## Security

- No API keys stored in code
- SearXNG runs on localhost, not exposed publicly
- `.env` file for any secrets (never committed)
- Docker container runs as non-root (SearXNG default)

## Development

```bash
git clone https://github.com/phj6688/search_workflow.git
cd search_workflow
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check src/
uv run black src/
```

## License

MIT License — © 2025 Peyman
