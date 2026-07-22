"""SSE MCP server exposing search_workflow's hybrid web search as a tool.

Mounted under ``/mcp`` in a FastAPI app; uvicorn target is
``search_workflow.mcp_server:app``. Mirrors the SSE transport pattern used by
the medchat MCP sidecar. Exposes a single tool, ``web_search``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from langchain_core.runnables import RunnableConfig
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from .graph import run_workflow

# Docker DNS name the LibreChat container uses to reach this server. The MCP
# SDK's DNS-rebinding guard answers 421 for an unlisted Host, so name it here.
_SERVICE_HOST = os.getenv("MCP_SERVICE_HOST", "search-mcp")

# Model used only for ranking/evaluation inside the workflow. Defaults to the
# code's native OpenAI-style model; overridable per deployment. The OpenAI base
# URL and key are read from OPENAI_BASE_URL / OPENAI_API_KEY by langchain.
_RANKER_MODEL = os.getenv("SEARCH_RANKER_MODEL", "openai/gpt-4o-mini")

# Upper bound on results fetched per engine and returned after ranking.
_MAX_RESULTS_CAP = int(os.getenv("MAX_SEARCH_RESULTS_TOOL", "10"))

_TRANSPORT_SECURITY = TransportSecuritySettings(
    allowed_hosts=[f"{_SERVICE_HOST}:*", "127.0.0.1:*", "localhost:*"],
    allowed_origins=[
        f"http://{_SERVICE_HOST}:*",
        "http://127.0.0.1:*",
        "http://localhost:*",
    ],
)


def _coerce_results(raw: Any) -> list[dict]:
    """Normalise run_workflow's discriminated dict into a list of result dicts.

    run_workflow returns {"status": "ok", "results": [...]} on success and
    {"status": "error", "error": {"type", "message"}} on failure. Map the ok
    shape to its results list and the error shape to a single {"error": ...}
    entry, so the calling model sees the failure rather than an empty array.
    """
    if isinstance(raw, dict) and raw.get("status") == "ok":
        results = raw.get("results", [])
        return results if isinstance(results, list) else [results]
    if isinstance(raw, dict) and raw.get("status") == "error":
        return [{"error": raw.get("error", {})}]
    # Contract guarantees a discriminated dict; surface anything else as an error
    # instead of guessing its shape.
    return [{"error": {"type": "unexpected_shape", "message": str(raw)}}]


def create_mcp_server() -> FastMCP:
    server = FastMCP("search", transport_security=_TRANSPORT_SECURITY)

    @server.tool()
    async def web_search(query: str, max_results: int = 5) -> str:
        """Search the public web and return AI-ranked results.

        Runs SearXNG and DuckDuckGo in parallel, merges and deduplicates, then
        ranks the results with an LLM. Use for current events, live facts,
        recent documentation, or anything outside the model's training data.

        Args:
            query: The search query.
            max_results: Maximum ranked results to return (1-10).

        Returns:
            A JSON array of objects with title, link, and snippet fields.
        """
        k = max(1, min(int(max_results), _MAX_RESULTS_CAP))
        config: RunnableConfig = {
            "configurable": {
                "model": _RANKER_MODEL,
                "max_search_results_tool": _MAX_RESULTS_CAP,
                "max_search_results_evaluator": k,
            }
        }
        raw = await run_workflow(query, config)
        results = _coerce_results(raw)
        return json.dumps(results[:k], ensure_ascii=False)

    return server


def create_app() -> FastAPI:
    app = FastAPI(
        title="search-mcp",
        version="0.4.0",
        description="Hybrid web search (SearXNG + DuckDuckGo) exposed as an MCP SSE tool",
    )

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    mcp = create_mcp_server()
    app.mount("/mcp", mcp.sse_app())
    return app


app = create_app()
