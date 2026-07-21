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

from .configuration import Configuration, default_config

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
        # Set by the evaluator node when the LLM's structured selection comes
        # back malformed and it falls back to the raw fetched results. Lives
        # here, next to the engine attribution, so run_workflow reads one
        # provenance source for the whole degraded/degraded_reason decision.
        # Cleared per query by run_workflow; the engine record overwrites every
        # query but this flag would otherwise carry over.
        self._evaluator_degraded = False

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

    def record_evaluator_degraded(self) -> None:
        with self._lock:
            self._evaluator_degraded = True

    def clear_evaluator_degraded(self) -> None:
        with self._lock:
            self._evaluator_degraded = False

    def evaluator_degraded(self) -> bool:
        with self._lock:
            return self._evaluator_degraded

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
            self._evaluator_degraded = False


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


# --- reciprocal rank fusion merge (HLB-651) --------------------------------

# RRF constant: score(d) = sum over engines of 1 / (RRF_K + rank), rank 0-based.
# A SMALL k is deliberate. The classic k=60 flattens a 10-result set so a single
# engine's rank-0 hit almost always outscores a document both engines rank
# mid-list, which is the opposite of the cross-engine-consensus ordering this
# merge wants. Two engines' contributions overtake one engine's rank-0 hit only
# when k exceeds the consensus rank: 2/(k+r) > 1/k  <=>  k > r. k=5 lets a
# document both engines place at the middle of ten (rank 4) outrank a single
# engine's number-one result, while staying far below 60.
RRF_K = 5

# At most this many results per registrable domain survive the merge, so one
# site cannot fill the list. Applied AFTER RRF ranking (the highest-ranked per
# domain are kept) and BEFORE truncation.
RRF_DOMAIN_CAP = 2

# Second-level public suffixes where the registrable domain is the last THREE
# labels rather than two (a.example.co.uk -> example.co.uk). A documented
# heuristic set, not a full public-suffix list: it covers the common ccTLD
# second levels without pulling in a dependency such as tldextract. Any host
# whose final two labels are not listed falls back to those two labels.
_MULTI_PART_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "ltd.uk", "plc.uk",
        "co.jp", "or.jp", "ne.jp", "go.jp", "ac.jp",
        "com.au", "net.au", "org.au", "edu.au", "gov.au",
        "co.nz", "org.nz", "govt.nz",
        "com.br", "com.cn", "com.mx", "com.tr", "com.sg", "com.hk", "com.tw",
        "co.in", "co.za", "co.kr", "co.il", "co.id", "co.th",
    }
)


def _registrable_domain(host: str) -> str:
    """Return the registrable domain of a host: the key the domain cap groups on.

    Documented heuristic, not a full public-suffix lookup: take the last two
    labels, but when those two are a known multi-part suffix (co.uk, com.au, ...)
    take the last three, so a.example.co.uk and b.example.co.uk both key on
    example.co.uk. A bare hostname split would treat every subdomain as distinct
    (never capping) or collapse everything under co.uk (over-capping). Hosts with
    two or fewer labels are returned unchanged.
    """
    host = host.lower().strip(".")
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_PART_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _rrf_merge(
    searxng_results: list[dict[str, Any]],
    ddg_results: list[dict[str, Any]],
    max_results: int,
) -> tuple[list[dict[str, Any]], set[str], int]:
    """Merge two per-engine ranked lists into one ordered result list.

    Fixed order: normalize_url dedup FIRST, then reciprocal rank fusion, then a
    domain cap of RRF_DOMAIN_CAP per registrable domain, then truncation to
    max_results. Dedup runs before RRF so a URL both engines returned collapses
    to ONE entry whose fused score adds both ranks (1/(RRF_K+rank) per engine)
    instead of the two copies competing.

    Returns (results, engines_used, n_after_dedup):
      * results       the merged, ranked, capped, truncated dicts (the input
                      dict objects, unmodified).
      * engines_used  the set of engines that first-returned a surviving deduped
                      URL, taken over the WHOLE deduped pool rather than the shown
                      slice, so the cap and truncation (which only trim what is
                      shown) cannot make the record disown a contributing engine.
      * n_after_dedup count of distinct URLs after dedup, before cap/truncate.
    """
    order: list[str] = []
    by_key: dict[str, dict[str, Any]] = {}
    nolink = 0
    for engine, results in (("searxng", searxng_results), ("ddg", ddg_results)):
        for rank, r in enumerate(results):
            link = r.get("link", "")
            if link:
                key = normalize_url(link)
            else:
                # A link-less row cannot dedup: give it its own key so several
                # never collapse together (the pre-RRF merge appended each one).
                key = f"\x00nolink\x00{nolink}"
                nolink += 1
            entry = by_key.get(key)
            if entry is None:
                entry = {"result": r, "source": engine, "score": 0.0}
                by_key[key] = entry
                order.append(key)
            # Both engines add to the fused score; the source stays the engine
            # that FIRST returned the URL (result-source attribution, HLB-646).
            entry["score"] += 1.0 / (RRF_K + rank)

    n_after_dedup = len(order)
    engines_used = {by_key[key]["source"] for key in order}

    # Rank by fused score, descending. sorted() is stable, so ties keep
    # first-seen order (SearXNG rows before DDG rows), matching the concat
    # tiebreak the merge used before.
    ranked = sorted(order, key=lambda key: by_key[key]["score"], reverse=True)

    kept: list[dict[str, Any]] = []
    per_domain: dict[str, int] = {}
    for key in ranked:
        result = by_key[key]["result"]
        try:
            host = urlsplit(result.get("link", "")).hostname or ""
        except ValueError:
            host = ""
        domain = _registrable_domain(host)
        # Cap by registrable domain. A result whose host yields no domain
        # (link-less or unparseable) is never capped, so the cap cannot silently
        # drop rows it cannot attribute to a site.
        if domain and per_domain.get(domain, 0) >= RRF_DOMAIN_CAP:
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        kept.append(result)
        if len(kept) >= max_results:
            break

    return kept, engines_used, n_after_dedup


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

    # No pre-fetch health probe. It was a second real GET /search per query on
    # top of the actual search. The shared core dispatches SearXNG and DDG in
    # parallel and already tolerates an empty or raising SearXNG leg (see the
    # return_exceptions gather in _fetch_and_merge), so DDG still merges when
    # SearXNG is down; probing first only bought a redundant round trip.
    language = _extract_language(region)
    # Build the client from Configuration so the per-call url and timeout win.
    # The import-time singleton below cannot see per-call config, and its class
    # default timeout was 12s; routing through Configuration lifts the effective
    # default to searxng_timeout (30).
    client = SearXNGClient(
        base_url=configuration.searxng_url,
        timeout=configuration.searxng_timeout,
    )
    return await _fetch_and_merge(
        query,
        client=client,
        max_results=configuration.max_search_results_tool,
        engine_deadline_s=configuration.engine_deadline_s,
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
    engine_deadline_s: float,
    language: str = "en",
    time_range: str | None = None,
    categories: str = "general",
) -> list[dict[str, Any]]:
    """Shared fetch core for both the `search` tool and `search_direct`.

    Dispatches SearXNG and DuckDuckGo concurrently, merges and deduplicates by
    normalize_url(link), records the outbound/engine counters, and emits the
    one provenance record. Makes no LLM call: ranking lives in the graph, so
    this seam stays runnable with no OPENAI_API_KEY.

    Each engine leg is bounded by engine_deadline_s. The SearXNG request
    timeout can be as high as 30s; without a shorter per-engine deadline a hung
    engine would stall the whole query for that long. asyncio.wait_for cancels
    the slow leg at the deadline and the timeout rides the return_exceptions
    gather below, so a hung engine becomes an absorbed exception and the other
    engine's results are still returned.
    """
    started = time.monotonic()

    searxng_task = asyncio.wait_for(
        client.search(
            query=query,
            language=language,
            time_range=time_range,
            max_results=max_results,
            categories=categories,
        ),
        timeout=engine_deadline_s,
    )
    ddg_task = asyncio.wait_for(
        _ddg_search(query, max_results),
        timeout=engine_deadline_s,
    )
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

    # One merge point for the shared core: dedup on the normalized link, fuse the
    # per-engine ranks with reciprocal rank fusion, cap per registrable domain,
    # then truncate. _rrf_merge applies that fixed order.
    returned, engines_used, n_after_dedup = _rrf_merge(
        searxng_list, ddg_list, max_results
    )

    if n_after_dedup == 0:
        print(f"Warning: No results for '{query}' from either engine")
    elif not searxng_list:
        print(f"Warning: SearXNG returned 0 results for '{query}', used DDG only")

    # engines_used is taken over the deduped candidate pool (not the capped
    # slice), so a domain cap or truncation that trims what is shown cannot make
    # fell_back claim a DDG-only run that did not happen.
    fell_back = engines_used == {"ddg"}
    METRICS.record_engines_used(engines_used)
    provenance = {
        "n_searxng": len(searxng_list),
        "n_ddg": len(ddg_list),
        "n_after_dedup": n_after_dedup,
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
    searxng_url: str | None = None,
    language: str = "en",
    time_range: str | None = None,
    searxng_timeout: int | None = None,
    categories: str = "general",
    engine_deadline_s: float | None = None,
) -> list[dict[str, Any]]:
    """Direct search: run SearXNG + DDG in parallel, merge and deduplicate.

    Args:
        categories: SearXNG categories to search (e.g. "general", "news", "general,news", "it")

    searxng_url, searxng_timeout and engine_deadline_s fall back to the
    Configuration defaults (url from SEARXNG_URL, timeout 30, deadline 4.5) when
    left as None; passing any of them still overrides. The signature stays
    backwards-compatible: callers that pass searxng_timeout keep working. No
    OPENAI_API_KEY is read here, so the zero-key contract holds.
    """
    if searxng_url is None:
        searxng_url = default_config.searxng_url
    if searxng_timeout is None:
        searxng_timeout = default_config.searxng_timeout
    if engine_deadline_s is None:
        engine_deadline_s = default_config.engine_deadline_s
    client = SearXNGClient(searxng_url, timeout=searxng_timeout)
    return await _fetch_and_merge(
        query,
        client=client,
        max_results=max_results,
        engine_deadline_s=engine_deadline_s,
        language=language,
        time_range=time_range,
        categories=categories,
    )


# Exported tools for external use
TOOLS: list[Callable[..., Any]] = [search]
