"""MCP server consumes run_workflow's discriminated result (HLB-654).

_coerce_results maps the {"status": "ok"|"error", ...} envelope run_workflow now
returns into the JSON array the web_search tool hands to LibreChat, instead of
guessing shapes. The web_search tests drive the FastMCP tool in-process against
a stubbed run_workflow so both the ok and error envelopes are exercised end to
end (tool call -> JSON string content block).
"""

import json

import pytest

from search_workflow import mcp_server


def test_mcp_coerce_ok_shape_returns_results_list() -> None:
    raw = {"status": "ok", "results": [{"title": "T", "link": "l", "snippet": "s"}]}
    assert mcp_server._coerce_results(raw) == [
        {"title": "T", "link": "l", "snippet": "s"}
    ]


def test_mcp_coerce_error_shape_returns_error_entry() -> None:
    raw = {"status": "error", "error": {"type": "workflow_error", "message": "boom"}}
    assert mcp_server._coerce_results(raw) == [
        {"error": {"type": "workflow_error", "message": "boom"}}
    ]


async def test_mcp_coerce_web_search_ok_returns_results_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{"title": "T", "link": "https://example.test", "snippet": "s"}]

    async def fake_run_workflow(query: str, config: object = None) -> dict:
        return {"status": "ok", "results": payload}

    monkeypatch.setattr(mcp_server, "run_workflow", fake_run_workflow)
    server = mcp_server.create_mcp_server()

    content, _structured = await server.call_tool(
        "web_search", {"query": "q", "max_results": 5}
    )

    assert json.loads(content[0].text) == payload


async def test_mcp_coerce_web_search_error_returns_error_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_workflow(query: str, config: object = None) -> dict:
        return {
            "status": "error",
            "error": {"type": "workflow_error", "message": "boom"},
        }

    monkeypatch.setattr(mcp_server, "run_workflow", fake_run_workflow)
    server = mcp_server.create_mcp_server()

    content, _structured = await server.call_tool(
        "web_search", {"query": "q", "max_results": 5}
    )

    assert json.loads(content[0].text) == [
        {"error": {"type": "workflow_error", "message": "boom"}}
    ]
