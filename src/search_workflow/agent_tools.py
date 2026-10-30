"""LangChain tool definitions; loaded only for the graph path."""

from collections.abc import Callable
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg

from . import tools


async def search(
    query: str,
    region: tools.Region,
    timelimit: tools.Timeframe,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> list[dict[str, Any]] | None:
    """Performs a web search for news-related content using SearXNG with DuckDuckGo fallback.

    Args:
        query: Search query string
        region: Geographic region for localized results
        timelimit: Optional timeframe for filtering results
    """
    return await tools.search(query, region, timelimit, config=config)


TOOLS: list[Callable[..., Any]] = [search]
