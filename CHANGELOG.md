# Changelog

## [Unreleased]

### Added
- `search_workflow.mcp_server`: an SSE MCP server exposing hybrid web search as a single `web_search` tool (query, max_results). FastAPI app with `/health`, MCP mounted at `/mcp/sse`; uvicorn target `search_workflow.mcp_server:app`. Lets LibreChat and other MCP clients call the workflow directly.
- `web_search` LLM-free fallback: when the ranker LLM is unavailable, the tool falls back to `search_direct` (SearXNG + DuckDuckGo, no LLM) so search keeps working on SearXNG alone; AI ranking resumes automatically once the ranker LLM is reachable.
- `docker/Dockerfile.mcp`: container image for the MCP server.
- `[project.optional-dependencies] mcp`: `mcp[cli]`, `fastapi`, `uvicorn` extras for running the server (`pip install '.[mcp]'`).

### Fixed
- `tools.py`: the module-level `searxng_client` now honours `SEARXNG_URL` (was hardcoded to `http://localhost:9090`). Inside a container the default was unreachable, silently forcing the DuckDuckGo-only fallback; the agentic `search` tool now reaches a SearXNG instance by service name.

## [0.3.0] - 2026-03-10

### Changed
- `SearXNGClient.search()`: `categories` is now a parameter (default: `"general"`) instead of hardcoded `"news"`. Callers can pass `"news"`, `"it"`, `"general,news"`, etc.
- `search_direct()`: exposes `categories` parameter, passed through to `SearXNGClient`
- `SearXNGClient.health_check()`: replaced broken `/healthz` probe with real `/search?q=test&format=json` HTTP probe

### Fixed
- DuckDuckGo fallback: changed `backend='news'` to `backend='text'` in both `search()` and `search_direct()` — eliminates `DecodeError` on non-news queries
- Added missing `import re` for DDG result string parsing

### Added
- `_ddg_search()`: extracted standalone DDG search function using `DDGS().text()` from the `ddgs` package — returns up to `max_results` structured dicts (was capped at 4 with LangChain wrapper)
- Parallel search in `search_direct()`: SearXNG + DDG now run simultaneously via `asyncio.gather()`. Results merged and deduplicated by URL. Eliminates sequential fallback latency.

### Infrastructure
- SearXNG `settings.yml`: reduced engine suspension times from 24h → 15min for `AccessDenied`/`CAPTCHA`, 1h → 5min for `TooManyRequests`. Prevents engines being silently banned for extended periods.

### Performance
- Query coverage: 70% → 100% (10/10 queries return results)
- Avg results per query: 3.7 → 10.0
- GitHub, Stack Overflow, and official docs surface correctly for technical queries

## [0.2.0] - 2025-10-14

### Added
- Initial SearXNG integration with DuckDuckGo fallback
- LangGraph workflow with AI evaluation step
- Docker compose setup for SearXNG
- `search_direct()` for eval harness use

## [0.1.0] - 2025-10-01

### Added
- Initial release
- DuckDuckGo-only search via LangChain community tools
