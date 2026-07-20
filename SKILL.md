---
name: smart-search
description: >
  Privacy-first hybrid web search using SearXNG + DuckDuckGo in parallel.
  Returns AI-evaluated, deduplicated results for any query — no commercial
  search API keys required. Use when: any agent needs web search results.
  Requires: Python env with search-workflow installed. SearXNG optional
  (DuckDuckGo runs as automatic fallback).
metadata:
  openclaw:
    emoji: 🔍
---

# smart-search Skill

Parallel hybrid search: SearXNG + DuckDuckGo run simultaneously, results
merged, deduplicated, then selected by index from the fetched set by an LLM evaluator.

## Prerequisites

1. Python environment with the package installed:
   ```bash
   pip install search-workflow
   # or
   uv add search-workflow
   ```

2. Set your OpenAI API key (used for result evaluation):
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

3. (Optional but recommended) SearXNG running locally:
   ```bash
   cd docker && docker compose up -d
   # SearXNG at http://localhost:9090
   ```
   Without SearXNG, DuckDuckGo handles all queries automatically.

## Invocation

### From any agent or shell

```bash
# Basic query — returns JSON results
python -m search_workflow FastAPI authentication JWT tutorial

# Limit results
python -m search_workflow neo4j python driver --max-results 3

# Text output instead of JSON
python -m search_workflow docker compose networking --format text

# With time filter (d=day, w=week, m=month, y=year)
python -m search_workflow LLM benchmarks 2025 --timelimit w
```

### From Python

```python
import asyncio
from search_workflow import run_workflow

outcome = asyncio.run(run_workflow("your query here"))
if outcome["status"] == "ok":
    for r in outcome["results"]:
        print(r['title'], r['link'])
else:
    print("search failed:", outcome["error"]["message"])
```

## Output Format

`run_workflow` returns a discriminated dict. Success:

```json
{
  "status": "ok",
  "results": [
    {"title": "Result title", "link": "https://example.com/page", "snippet": "Short description..."}
  ]
}
```

Failure:

```json
{"status": "error", "error": {"type": "json_parse_error", "message": "..."}}
```

## Configuration

| Env Var | Default | Description |
|---|---|---|
|  | — | Required for AI result evaluation |
|  |  | SearXNG instance URL |
|  |  | Results fetched per engine |
|  |  | Results the evaluator selects by index |

## Fallback Behavior

- SearXNG down or returns 0 results → DuckDuckGo fills automatically
- No  → results returned without evaluator selection (no AI evaluation step)
