# Changelog

## [Unreleased]

### Fixed
- CLI entry points: the `search-workflow` console script pointed at an async `main`, so it returned an unawaited coroutine and did nothing. `main` is now a sync wrapper around `_async_main` via `asyncio.run`. Behavior note: exceptions the unawaited coroutine silently swallowed now surface; runs that previously exited 0 without doing anything can now print an error and exit 1.
- `python -m search_workflow` failed with "No module named search_workflow.__main__"; added `src/search_workflow/__main__.py`.
- README install commands named the nonexistent `search-workflow` distribution; corrected to `smart-search-skill` (pip, uv, and the PyPI badge).
- `tools.py`: the module-level `searxng_client` now honours `SEARXNG_URL` (was hardcoded to `http://localhost:9090`). Inside a container the default was unreachable, silently forcing the DuckDuckGo-only fallback; the agentic `search` tool now reaches a SearXNG instance by service name.

### Added
- `--version` flag on both CLI entry points, resolved via `importlib.metadata` with a `0.0.0.dev0` fallback for uninstalled source checkouts.
- `tests/test_entry_points.py`: entry-point contract tests, README honesty checks that run the documented non-network commands verbatim, an in-loop `search` tool test, and a wheel-install subprocess gate (marker `wheel_install`).
- `search_workflow.mcp_server`: an SSE MCP server exposing hybrid web search as a single `web_search` tool (query, max_results). FastAPI app with `/health`, MCP mounted at `/mcp/sse`; uvicorn target `search_workflow.mcp_server:app`. Lets LibreChat and other MCP clients call the workflow directly.
- `docker/Dockerfile.mcp`: container image for the MCP server.
- `[project.optional-dependencies] mcp`: `mcp[cli]`, `fastapi`, `uvicorn` extras for running the server (`pip install '.[mcp]'`).
- Per-query search instrumentation in `tools.py`: module-level `METRICS` (`SearchMetrics`) with lock-guarded counters `outbound_search_requests`, `llm_calls`, `cache_hit`, `engines_used`, plus `snapshot()` and `reset()`. `search_direct()` now emits one structured provenance log record per query (logger `search_workflow.tools`) with `n_searxng`, `n_ddg`, `n_after_dedup`, `elapsed_ms`, `ddg_ok`, `fell_back`; `fell_back` and `engines_used` derive from which engine's results were actually returned, not from the executed branch.
- `tests/fixtures_fallback.py`: parametrized five-state `fallback_state` fixture (searxng_ok, searxng_raises, searxng_empty, searxng_ok_ddg_unused, both_fail) mocking the `SearXNGClient`/`_ddg_search` boundary, re-exported via `tests/conftest.py` for later stories. `tests/test_instrumentation.py`: caplog provenance assertions per state and known-count counter tests.

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
