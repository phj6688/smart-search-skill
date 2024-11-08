"""Search tools for web scraping and retrieving information from news sources."""

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

async def search(
    query: str,
    region: Region,
    timelimit: Timeframe,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[List[dict[str, Any]]]:
    """Performs a web search for news-related content. Use DuckDucGo to retrieve news results based on query, region, and timeframe.
        Args:
            query (str): Search query string.
            region (Region): Geographic region for localized results.
            timelimit (Timeframe): Optional timeframe for filtering results.
            config (RunnableConfig): Configuration object with search settings.

        Returns:
            Optional[List[dict[str, Any]]]: List of search results, each as a dictionary.
    """
    configuration = Configuration.from_runnable_config(config)
    search_engine = DuckDuckGoSearchResults(
        max_results=configuration.max_search_results_tool,
        backend='news'
    )
    
    result = await search_engine.ainvoke({
        "query": query,
        "region": region,
        "timelimit": timelimit
    })
    
    return cast(List[dict[str, Any]], result)


# Exported tools for external use
TOOLS: List[Callable[..., Any]] = [search]
