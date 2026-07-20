# Changelog

## [Unreleased]

### Changed
- Evaluator selects results by index instead of re-emitting them. The evaluator now asks the model for a `SelectionResponse` (`selected: list[int]`, `with_structured_output`, `temperature=0`) and joins those indices back to the objects the tool fetched (parsed from the last `ToolMessage`). Returned `title`/`link`/`snippet` are the fetched bytes verbatim, so URLs are no longer lowercased or hallucinated and no similarity score is invented. Out-of-range indices drop, the selection caps at `max_search_results_evaluator`, and the sort-by-invented-similarity padding path is gone. `run_workflow` still wraps the evaluator's list as `{"status": "ok", "results": [...]}`. One output-format sentence in `EVALUATOR_PROMPT` now asks for indices.
- BREAKING (v0.4.0): `run_workflow` no longer returns a bare error string. Its contract is now a discriminated dict: success is `{"status": "ok", "results": [...]}` and failure is `{"status": "error", "error": {"type": <str>, "message": <str>}}`. A `SearchError` (in `search_workflow.errors`) carries the internal failure and serializes to that error shape; a bare string is never returned. Consumer changes: the CLI now prints `error.message` to stderr and exits 1 on `status == "error"`; the Python API, LangGraph TOOLS path, and OpenClaw SKILL.md must read `outcome["status"]` and use `outcome["results"]` instead of iterating the return directly; `mcp_server._coerce_results` reads the discriminated shape natively instead of guessing whether a string was an error. README and SKILL.md snippets updated.
- `tools.py`: both search paths now share one async fetch core (`_fetch_and_merge`). The agentic `search` tool and `search_direct` each call it; the core runs SearXNG and DuckDuckGo in parallel via `asyncio.gather` and returns the deduplicated merge. The `search` tool keeps its SearXNG health probe but no longer gates the fetch on it or falls back to DuckDuckGo sequentially, so the tool path now fetches both engines in parallel like `search_direct`. Deduplication now keys on `normalize_url(link)` rather than the exact link string, so tracking-param and trailing-slash twins collapse. The core makes no LLM call and each request still opens its own `aiohttp.ClientSession`. Behavior note: the tool path now issues one DuckDuckGo request per query even when SearXNG answers; provenance and `METRICS` (outbound_search_requests, engines_used) emit from the shared core on both paths.

### Fixed
- CLI entry points: the `search-workflow` console script pointed at an async `main`, so it returned an unawaited coroutine and did nothing. `main` is now a sync wrapper around `_async_main` via `asyncio.run`. Behavior note: exceptions the unawaited coroutine silently swallowed now surface; runs that previously exited 0 without doing anything can now print an error and exit 1.
- `python -m search_workflow` failed with "No module named search_workflow.__main__"; added `src/search_workflow/__main__.py`.
- README install commands named the nonexistent `search-workflow` distribution; corrected to `smart-search-skill` (pip, uv, and the PyPI badge).
- `tools.py`: the module-level `searxng_client` now honours `SEARXNG_URL` (was hardcoded to `http://localhost:9090`). Inside a container the default was unreachable, silently forcing the DuckDuckGo-only fallback; the agentic `search` tool now reaches a SearXNG instance by service name.
- `tests/conftest.py`: the nested-run recursion guard now also skips `test_egress_canary_passes_under_pytest`. That held-out probe shells out to `pytest tests/ -k egress_canary`, and the `-k` filter re-matched the probe's own name, so each nested run spawned another, recursing until the fork limit. Both self-spawning probes (offline-suite and egress-canary) are now skipped in nested runs, so `pytest tests/` terminates.
- `tests/test_entry_points.py`: the in-loop `search` tool test stubs the engines at the class level instead of on the module-global `searxng_client` instance. Instance-level `monkeypatch.setattr` restored a lingering instance attribute on the shared singleton that shadowed later class-level stubs; it also now stubs `_ddg_search` so the unified tool path stays offline.

### Added
- `--version` flag on both CLI entry points, resolved via `importlib.metadata` with a `0.0.0.dev0` fallback for uninstalled source checkouts.
- `tests/test_entry_points.py`: entry-point contract tests, README honesty checks that run the documented non-network commands verbatim, an in-loop `search` tool test, and a wheel-install subprocess gate (marker `wheel_install`).
- `search_workflow.mcp_server`: an SSE MCP server exposing hybrid web search as a single `web_search` tool (query, max_results). FastAPI app with `/health`, MCP mounted at `/mcp/sse`; uvicorn target `search_workflow.mcp_server:app`. Lets LibreChat and other MCP clients call the workflow directly.
- `docker/Dockerfile.mcp`: container image for the MCP server.
- `[project.optional-dependencies] mcp`: `mcp[cli]`, `fastapi`, `uvicorn` extras for running the server (`pip install '.[mcp]'`).
- Per-query search instrumentation in `tools.py`: module-level `METRICS` (`SearchMetrics`) with lock-guarded counters `outbound_search_requests`, `llm_calls`, `cache_hit`, `engines_used`, plus `snapshot()` and `reset()`. `search_direct()` now emits one structured provenance log record per query (logger `search_workflow.tools`) with `n_searxng`, `n_ddg`, `n_after_dedup`, `elapsed_ms`, `ddg_ok`, `fell_back`; `fell_back` and `engines_used` derive from which engine's results were actually returned, not from the executed branch.
- `tests/fixtures_fallback.py`: parametrized five-state `fallback_state` fixture (searxng_ok, searxng_raises, searxng_empty, searxng_ok_ddg_unused, both_fail) mocking the `SearXNGClient`/`_ddg_search` boundary, re-exported via `tests/conftest.py` for later stories. `tests/test_instrumentation.py`: caplog provenance assertions per state and known-count counter tests.
- Offline VCR test harness (`pytest-recording`): cassettes for SearXNG and DDG under `tests/cassettes/`, replayed by `tests/test_vcr_replay.py` through `SearXNGClient` (aiohttp) and the ddgs client pinned to `backend="duckduckgo"` (the default "auto" backend rides primp, which vcrpy cannot intercept). CI pytest now runs with `--block-network`. A delete-a-cassette negative test reruns the SearXNG replay in a pytest subprocess against a pruned cassette copy and asserts it fails.
- Scheduled live integration canary (`.github/workflows/live-canary.yml`): daily off-minute cron plus `workflow_dispatch`, with a `dev-v04` push trigger for the baseline run (the schedule starts once the file lands on master). Starts a throwaway SearXNG from `.github/searxng-compose.yml` (bound to 127.0.0.1:9090, generated secret), continues DDG-only when startup fails, compares live `/search` response shape against the recorded SearXNG cassette via `scripts/live_canary_checks.py`, asserts the live search leg stays under a 60s ceiling, and runs a live `ddgs` import-path probe. Canary only, never a merge gate; ci.yml stays free of wall-clock assertions. Static definition tests live in `tests/test_live_canary_workflow.py`.
- Autouse OpenAI stub in `tests/conftest.py`: patches `load_chat_model` and the evaluator's `with_structured_output` path with canned responses and deletes `OPENAI_API_KEY`, so the suite passes with no key set. `tests/test_cassette_hygiene.py` parses every cassette and asserts credential headers are scrubbed (vcr `filter_headers`) and no interaction targets api.openai.com.
- `normalize_url()` in `tools.py`: canonicalizes a URL into a dedup key. Lowercases scheme and host, drops a default port (80/443), strips one trailing slash, and removes only tracking params (`utm_*`, `fbclid`, `gclid`). Path and query case, the fragment, and every non-tracking param (for example `?page=2`) are preserved, so distinct pages stay distinct.
- `tests/test_shared_fetch_core.py`: `normalize_url` unit tests (`pytest -k normalize_url`); dedup tests drawn from `tests/fixtures/FIX-MERGE.json` covering COLLAPSE (a tracking-param twin and a trailing-slash twin merge to one) and SURVIVAL (a `?page=2` twin stays distinct); and an end-to-end test driving `run_workflow` through the shared core offline, asserting the `{"status": "ok", "results": [...]}` shape and that both engines were fetched in parallel on the tool path.

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
