"""Search tools for web scraping and retrieving information from news sources."""

import asyncio
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Iterable
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from .configuration import Configuration

logger = logging.getLogger("search_workflow.tools")


class SearchMetrics:
    """Cumulative counters for the client/search_direct seam.

    Read with snapshot(), zero with reset(). Every mutation takes the lock:
    the DDG call runs in a worker thread, so unlocked increments could race.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._outbound_search_requests = 0
        self._llm_calls = 0
        self._cache_hit = 0
        self._engines_used: dict[str, int] = {}
        # Last per-query provenance record (n_searxng/n_ddg/engines_used/...),
        # kept so result-path consumers can read attribution without a logging
        # handler. snapshot()'s engines_used is post-dedup and cumulative; the
        # per-query record additionally carries the raw n_searxng/n_ddg needed
        # to tell a genuine single-engine fallback from a deduped-away one.
        self._last_provenance: dict[str, Any] | None = None

    def record_outbound_search_requests(self, count: int = 1) -> None:
        with self._lock:
            self._outbound_search_requests += count

    def record_llm_call(self) -> None:
        with self._lock:
            self._llm_calls += 1

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hit += 1

    def record_engines_used(self, engines: Iterable[str]) -> None:
        with self._lock:
            for engine in engines:
                self._engines_used[engine] = self._engines_used.get(engine, 0) + 1

    def record_provenance(self, provenance: dict[str, Any]) -> None:
        with self._lock:
            self._last_provenance = dict(provenance)

    def last_provenance(self) -> dict[str, Any] | None:
        with self._lock:
            return dict(self._last_provenance) if self._last_provenance else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "outbound_search_requests": self._outbound_search_requests,
                "llm_calls": self._llm_calls,
                "cache_hit": self._cache_hit,
                "engines_used": dict(self._engines_used),
            }

    def reset(self) -> None:
        with self._lock:
            self._outbound_search_requests = 0
            self._llm_calls = 0
            self._cache_hit = 0
            self._engines_used = {}
            self._last_provenance = None


METRICS = SearchMetrics()


def _emit_provenance(provenance: dict[str, Any]) -> None:
    """Log the one provenance record a query gets.

    The dict rides on the record as `extra` for machine consumers; the message
    repeats every field as key=value so plain-text logs stay parseable.
    """
    logger.info(
        "search_direct provenance: n_searxng=%d n_ddg=%d "
        "n_after_dedup=%d elapsed_ms=%.2f ddg_ok=%s fell_back=%s engines_used=%s",
        provenance["n_searxng"],
        provenance["n_ddg"],
        provenance["n_after_dedup"],
        provenance["elapsed_ms"],
        provenance["ddg_ok"],
        provenance["fell_back"],
        ",".join(provenance["engines_used"]) or "none",
        extra={"provenance": provenance},
    )


# Query params that identify a marketing/tracking source, not a distinct page.
# utm_* is matched by prefix; the rest by exact name. Stripping them lets two
# links that differ only by campaign tags collapse to one dedup key.
_TRACKING_PARAM_EXACT = frozenset({"fbclid", "gclid"})


def _is_tracking_param(pair: str) -> bool:
    name = pair.split("=", 1)[0]
    return name.startswith("utm_") or name in _TRACKING_PARAM_EXACT


def normalize_url(url: str) -> str:
    """Canonicalize a URL into a dedup key.

    Lowercases scheme and host, drops a default port (80/443), strips one
    trailing slash, and removes only tracking params (utm_*, fbclid, gclid).
    Path and query CASE are left untouched and every non-tracking param is
    kept, so pages that differ by a real param (e.g. ?page=2) stay distinct.
    The fragment is not listed for stripping, so it is preserved. A URL that
    cannot be parsed (e.g. an out-of-range port that makes ``parts.port``
    raise) is returned unchanged, so one malformed result becomes its own
    dedup key instead of aborting the whole merge.
    """
    try:
        parts = urlsplit(url)

        scheme = parts.scheme.lower()

        host = (parts.hostname or "").lower()
        # Rebuild userinfo verbatim; only host case and default ports change.
        userinfo = ""
        if parts.username is not None:
            userinfo = parts.username
            if parts.password is not None:
                userinfo += f":{parts.password}"
            userinfo += "@"
        port = parts.port
        netloc = f"{userinfo}{host}"
        # Drop the port only when it is the default FOR THIS scheme; an
        # explicit :80 on https (or :443 on http) is meaningful and kept.
        default_port = {"http": 80, "https": 443}.get(scheme)
        if port is not None and port != default_port:
            netloc = f"{netloc}:{port}"

        path = parts.path[:-1] if parts.path.endswith("/") else parts.path

        # Filter on the raw pairs so remaining params keep their exact encoding
        # and case; parse_qsl would decode and reorder them.
        kept = [p for p in parts.query.split("&") if p and not _is_tracking_param(p)]
        query = "&".join(kept)

        return urlunsplit((scheme, netloc, path, query, parts.fragment))
    except ValueError:
        # Malformed URL (e.g. out-of-range port). Fall back to the raw string
        # so this one result dedups on itself rather than aborting the merge.
        return url


# Define available region and timeframe options
Region = Literal[
    'us-en', 'uk-en', 'au-en', 'ca-en', 'in-en', 'de-de', 'at-de',
    'ch-de', 'ch-fr', 'ch-it', 'es-es', 'mx-es', 'ar-es', 'ue-es'
]
Timeframe = Literal['d', 'w', 'm', 'y'] | None

class SearXNGClient:
    """SearXNG API client with fallback capabilities"""

    def __init__(self, base_url: str = "http://localhost:9090", timeout: int = 12):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    async def search(
        self,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict]:
        """Search using SearXNG API"""
        params = {
            "q": query,
            "categories": categories,
            "language": language,
            "format": "json",
            "safesearch": "0"
        }

        if time_range:
            # Map DuckDuckGo timeframes to SearXNG
            time_map = {'d': 'day', 'w': 'week', 'm': 'month', 'y': 'year'}
            params["time_range"] = time_map.get(time_range, time_range)

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(f"{self.base_url}/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_results(data.get("results", []), max_results)
                    else:
                        print(f"SearXNG HTTP {response.status}")
                        return []
        except Exception as e:
            print(f"SearXNG error: {e}")
            return []

    def _normalize_results(self, raw_results: list[dict], max_results: int) -> list[dict]:
        """Normalize SearXNG results to DuckDuckGo format"""
        normalized = []
        for result in raw_results[:max_results]:
            normalized_result = {
                "title": result.get("title", ""),
                "link": result.get("url", ""),
                "snippet": result.get("content", ""),
                # Additional SearXNG metadata
                "publishedDate": result.get("publishedDate", ""),
                "engine": result.get("engine", ""),
            }
            normalized.append(normalized_result)
        return normalized

    async def health_check(self) -> bool:
        """Check if SearXNG is available via a real search probe."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.base_url}/search", params={"q": "test", "format": "json"}) as response:
                    return response.status == 200
        except Exception:
            return False

# Initialize SearXNG client. Honour SEARXNG_URL so the agentic `search` tool
# reaches a SearXNG instance by service name inside Docker (default localhost
# is wrong from a container and silently forces the DuckDuckGo-only fallback).
searxng_client = SearXNGClient(base_url=os.getenv("SEARXNG_URL", "http://localhost:9090"))

def _extract_language(region: Region) -> str:
    """Extract language from region code"""
    return region.split('-')[1] if '-' in region else 'en'

async def search(
    query: str,
    region: Region,
    timelimit: Timeframe,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> list[dict[str, Any]] | None:
    """Performs a web search for news-related content using SearXNG with DuckDuckGo fallback.

    Args:
        query: Search query string
        region: Geographic region for localized results
        timelimit: Optional timeframe for filtering results
    """
    configuration = Configuration.from_runnable_config(config)

    # The health probe stays on the tool path (its wiring is owned by the
    # later S09/S10 stories); it no longer gates the fetch. The actual
    # fetch+merge routes through the shared core so both engines run in
    # parallel here, identical to the search_direct path.
    try:
        if not await searxng_client.health_check():
            print("⚠️ SearXNG health probe failed; core still fetches both engines")
    except Exception as e:
        print(f"❌ SearXNG health probe error: {e}")

    language = _extract_language(region)
    return await _fetch_and_merge(
        query,
        client=searxng_client,
        max_results=configuration.max_search_results_tool,
        language=language,
        time_range=timelimit,
        categories="general",
    )

async def _ddg_search(query: str, max_results: int) -> list:
    """Run DDG search using DDGS().text(), returning up to max_results structured dicts.

    Failures propagate: search_direct gathers with return_exceptions=True and
    records ddg_ok=False, so swallowing here would make outages look like
    ordinary empty result sets.
    """
    from ddgs import DDGS
    # DDGS().text() returns list of dicts: {title, href, body}
    # Run in executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        None,
        lambda: list(DDGS().text(query, max_results=max_results))
    )
    # Normalize to {title, link, snippet}
    return [
        {"title": r.get("title", ""), "link": r.get("href", ""), "snippet": r.get("body", "")}
        for r in raw
    ]


async def _fetch_and_merge(
    query: str,
    *,
    client: SearXNGClient,
    max_results: int,
    language: str = "en",
    time_range: str | None = None,
    categories: str = "general",
) -> list[dict[str, Any]]:
    """Shared fetch core for both the `search` tool and `search_direct`.

    Dispatches SearXNG and DuckDuckGo concurrently, merges and deduplicates by
    normalize_url(link), records the outbound/engine counters, and emits the
    one provenance record. Makes no LLM call: ranking lives in the graph, so
    this seam stays runnable with no OPENAI_API_KEY.
    """
    started = time.monotonic()

    searxng_task = client.search(
        query=query,
        language=language,
        time_range=time_range,
        max_results=max_results,
        categories=categories,
    )
    ddg_task = _ddg_search(query, max_results)
    # Both engines are always dispatched from this seam, one request each.
    METRICS.record_outbound_search_requests(2)
    searxng_results, ddg_results = await asyncio.gather(
        searxng_task, ddg_task, return_exceptions=True
    )

    # ddg_ok reflects the call outcome; fell_back and engines_used come from
    # attribution of the returned results below, because the fallback fires on
    # a raise OR an empty response and a branch-keyed record would lie.
    ddg_ok = not isinstance(ddg_results, Exception)
    searxng_list = [] if isinstance(searxng_results, Exception) else list(searxng_results)
    ddg_list = [] if isinstance(ddg_results, Exception) else list(ddg_results)

    # Merge, deduplicate on the normalized link; track each survivor's engine.
    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    merged_sources: list[str] = []
    for source, engine_results in (("searxng", searxng_list), ("ddg", ddg_list)):
        for r in engine_results:
            link = r.get("link", "")
            key = normalize_url(link) if link else ""
            if link and key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(r)
            merged_sources.append(source)

    if not merged:
        print(f"Warning: No results for '{query}' from either engine")
    elif not searxng_list:
        print(f"Warning: SearXNG returned 0 results for '{query}', used DDG only")

    returned = merged[:max_results]
    engines_used = set(merged_sources[: len(returned)])
    fell_back = engines_used == {"ddg"}
    METRICS.record_engines_used(engines_used)
    provenance = {
        "n_searxng": len(searxng_list),
        "n_ddg": len(ddg_list),
        "n_after_dedup": len(merged),
        "elapsed_ms": (time.monotonic() - started) * 1000.0,
        "ddg_ok": ddg_ok,
        "fell_back": fell_back,
        "engines_used": sorted(engines_used),
    }
    METRICS.record_provenance(provenance)
    _emit_provenance(provenance)
    return returned


async def search_direct(
    query: str,
    max_results: int = 10,
    searxng_url: str = "http://localhost:9090",
    language: str = "en",
    time_range: str | None = None,
    searxng_timeout: int = 12,
    categories: str = "general",
) -> list[dict[str, Any]]:
    """Direct search: run SearXNG + DDG in parallel, merge and deduplicate.

    Args:
        categories: SearXNG categories to search (e.g. "general", "news", "general,news", "it")
    """
    client = SearXNGClient(searxng_url, timeout=searxng_timeout)
    return await _fetch_and_merge(
        query,
        client=client,
        max_results=max_results,
        language=language,
        time_range=time_range,
        categories=categories,
    )


# Exported tools for external use
TOOLS: list[Callable[..., Any]] = [search]
