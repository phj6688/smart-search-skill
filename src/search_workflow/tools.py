"""Search tools for web scraping and retrieving information from news sources."""

import asyncio
import aiohttp
from typing import Any, List, Optional, cast, Literal, Callable
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated
from .configuration import Configuration

# Define available region and timeframe options
Region = Literal[
    'us-en', 'uk-en', 'au-en', 'ca-en', 'in-en', 'de-de', 'at-de',
    'ch-de', 'ch-fr', 'ch-it', 'es-es', 'mx-es', 'ar-es', 'ue-es'
]
Timeframe = Optional[Literal['d', 'w', 'm', 'y']]

class SearXNGClient:
    """SearXNG API client with fallback capabilities"""
    
    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip('/')
        
    async def search(
        self, 
        query: str, 
        language: str = "en",
        time_range: Optional[str] = None,
        max_results: int = 10
    ) -> List[dict]:
        """Search using SearXNG API"""
        params = {
            "q": query,
            "categories": "news",
            "language": language,
            "format": "json",
            "safesearch": "0"
        }
        
        if time_range:
            # Map DuckDuckGo timeframes to SearXNG
            time_map = {'d': 'day', 'w': 'week', 'm': 'month', 'y': 'year'}
            params["time_range"] = time_map.get(time_range, time_range)
            
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
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
    
    def _normalize_results(self, raw_results: List[dict], max_results: int) -> List[dict]:
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
        """Check if SearXNG is available"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/healthz", timeout=5) as response:
                    return response.status == 200
        except:
            return False

# Initialize SearXNG client
searxng_client = SearXNGClient()

def _extract_language(region: Region) -> str:
    """Extract language from region code"""
    return region.split('-')[1] if '-' in region else 'en'

async def search(
    query: str,
    region: Region,
    timelimit: Timeframe,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[dict[str, Any]]]:
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
            backend='news'
        )
        
        result = await search_engine.ainvoke({
            "query": query,
            "region": region,
            "timelimit": timelimit
        })
        
        print(f"✅ DuckDuckGo: Results retrieved")
        return cast(List[dict[str, Any]], result)
        
    except Exception as e:
        print(f"❌ Both engines failed: {e}")
        return []

# Exported tools for external use
TOOLS: List[Callable[..., Any]] = [search]
