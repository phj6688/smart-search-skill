"""Search tools for web scraping and retrieving information from news sources."""

import asyncio
import os
import re
from collections.abc import Callable
from typing import Annotated, Any, Literal, cast

import aiohttp
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from .configuration import Configuration

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

    # Try SearXNG first
    try:
        if await searxng_client.health_check():
            language = _extract_language(region)
            results = await searxng_client.search(
                query=query,
                language=language,
                time_range=timelimit,
                max_results=configuration.max_search_results_tool
            )

            if results:
                print(f"✅ SearXNG: {len(results)} results")
                return results
            else:
                print("⚠️ SearXNG: No results, trying DuckDuckGo")
    except Exception as e:
        print(f"❌ SearXNG failed: {e}")

    # Fallback to DuckDuckGo
    try:
        print("🔄 Using DuckDuckGo fallback")
        search_engine = DuckDuckGoSearchResults(
            max_results=configuration.max_search_results_tool,
            backend='text'
        )

        result = await search_engine.ainvoke({
            "query": query,
            "region": region,
            "timelimit": timelimit
        })

        print("✅ DuckDuckGo: Results retrieved")
        return cast(list[dict[str, Any]], result)

    except Exception as e:
        print(f"❌ Both engines failed: {e}")
        return []

async def _ddg_search(query: str, max_results: int) -> list:
    """Run DDG search using DDGS().text() — returns up to max_results structured dicts."""
    try:
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
    except Exception as e:
        print(f"DuckDuckGo failed: {e}")
        return []


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

    searxng_task = client.search(query, language, time_range, max_results, categories=categories)
    ddg_task = _ddg_search(query, max_results)
    searxng_results, ddg_results = await asyncio.gather(
        searxng_task, ddg_task, return_exceptions=True
    )

    if isinstance(searxng_results, Exception):
        searxng_results = []
    if isinstance(ddg_results, Exception):
        ddg_results = []

    # Merge, deduplicate by link
    seen_links: set = set()
    merged = []
    for r in list(searxng_results) + list(ddg_results):
        link = r.get("link", "")
        if link and link in seen_links:
            continue
        seen_links.add(link)
        merged.append(r)

    if not merged:
        print(f"Warning: No results for '{query}' from either engine")
    elif not searxng_results:
        print(f"Warning: SearXNG returned 0 results for '{query}', used DDG only")

    return merged[:max_results]


# Exported tools for external use
TOOLS: list[Callable[..., Any]] = [search]
