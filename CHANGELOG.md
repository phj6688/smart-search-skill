# Changelog

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
