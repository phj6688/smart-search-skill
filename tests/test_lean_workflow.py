"""Offline HTTP ranking contracts, including retry and raw-result fallback."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from search_workflow import tools

RAW = [
    {"title": "Mixed CASE α", "link": "https://A.test/Path?Key=VALUE", "snippet": "Long " * 80},
    {"title": "Second", "link": "https://b.test/Case", "snippet": "Second snippet"},
    {"title": "Third", "link": "https://c.test/Case", "snippet": "Third snippet"},
]


@pytest.fixture
def fetch(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls = []

    async def fake_search(query: str, **kwargs: Any) -> list[dict[str, str]]:
        calls.append((query, kwargs))
        tools.METRICS.record_provenance(
            {"engines_used": ["ddg", "searxng"], "n_searxng": 2, "n_ddg": 1}
        )
        return [dict(item) for item in RAW]

    monkeypatch.setattr(tools, "search_direct", fake_search)
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    return calls


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> Callable:
    real_client = httpx.AsyncClient

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> list[httpx.Request]:
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return handler(request)

        monkeypatch.setattr(
            httpx, "AsyncClient",
            lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
        )
        return requests

    return install


def _reply(content: Any) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.mark.parametrize("model", ["gpt-test", "openai/gpt-test"])
async def test_ranks_verbatim_with_one_minimized_request(fetch: list, http: Callable, model: str) -> None:
    from search_workflow.lean import run_workflow_lean

    requests = http(lambda request: _reply('{"selected": [1, 0]}'))
    out = await run_workflow_lean("Exact Query", {"configurable": {"model": model}})

    assert out == {
        "status": "ok", "results": [RAW[1], RAW[0]],
        "engines_used": ["duckduckgo", "searxng"],
        "degraded": False, "degraded_reason": None,
    }
    assert fetch == [("Exact Query", {
        "max_results": 10, "searxng_url": "http://localhost:9090",
        "searxng_timeout": 30, "engine_deadline_s": 4.5, "safesearch": 0,
    })]
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer offline-test-key"
    body = json.loads(request.content)
    assert body["model"] == "gpt-test"
    assert body["temperature"] == 0
    assert "response_format" not in body
    system, user = body["messages"]
    assert system["role"] == "system" and "Exact Query" in system["content"]
    assert '{"selected": [<int indices>]}' in system["content"]
    assert user == {"role": "user", "content": (
        "[0] Mixed CASE α\n" + "Long " * 40
        + "\n\n[1] Second\nSecond snippet\n\n[2] Third\nThird snippet"
    )}
    assert tools.METRICS.snapshot()["llm_calls"] == 1


@pytest.mark.parametrize("content", [
    '```json\n{"selected": [1]}\n```',
    'Here are the picks: {"selected": [1]} trailing text {"selected": [0]}',
])
async def test_fences_and_surrounding_text_are_accepted(fetch: list, http: Callable, content: str) -> None:
    from search_workflow.lean import run_workflow_lean

    http(lambda request: _reply(content))
    out = await run_workflow_lean("query")
    assert out["results"] == [RAW[1]]
    assert out["degraded"] is False


@pytest.mark.parametrize("content", [
    "```json\n{garbled}\n```", "not JSON", '{"wrong": []}',
    '{"selected": ["bad"]}', '{"selected": null}', None,
])
async def test_invalid_selection_falls_back_without_truncation(fetch: list, http: Callable, content: Any) -> None:
    from search_workflow.lean import run_workflow_lean

    requests = http(lambda request: _reply(content))
    out = await run_workflow_lean("query", {"configurable": {"max_search_results_evaluator": 1}})
    assert out["status"] == "ok"
    assert out["results"] == RAW
    assert out["degraded"] is True
    assert out["degraded_reason"] == "evaluator"
    assert len(requests) == 1
    assert tools.METRICS.snapshot()["llm_calls"] == 1


async def test_selection_drops_invalid_and_duplicate_indices_and_caps(fetch: list, http: Callable) -> None:
    from search_workflow.lean import run_workflow_lean

    http(lambda request: _reply('{"selected": [-1, 99, 2, 2, 0, 1]}'))
    out = await run_workflow_lean("query", {"configurable": {"max_search_results_evaluator": 2}})
    assert out["results"] == [RAW[2], RAW[0]]


async def test_empty_selection_is_healthy_and_clears_previous_degradation(fetch: list, http: Callable) -> None:
    from search_workflow.lean import run_workflow_lean

    tools.METRICS.record_evaluator_degraded()
    http(lambda request: _reply('{"selected": []}'))
    out = await run_workflow_lean("query")
    assert out["results"] == []
    assert out["degraded"] is False


async def test_empty_fetch_skips_http(monkeypatch: pytest.MonkeyPatch, http: Callable) -> None:
    from search_workflow.lean import run_workflow_lean

    async def empty(query: str, **kwargs: Any) -> list:
        return []

    monkeypatch.setattr(tools, "search_direct", empty)
    requests = http(lambda request: pytest.fail("Empty fetch must not rank"))
    tools.METRICS.record_evaluator_degraded()
    out = await run_workflow_lean("query")
    assert out == {"status": "ok", "results": [], "engines_used": [],
                   "degraded": False, "degraded_reason": None}
    assert not requests
    assert tools.METRICS.snapshot()["llm_calls"] == 0


@pytest.mark.parametrize("failure", ["rate_limit", "timeout"])
async def test_http_transient_failures_retry_one_logical_call(
    fetch: list, http: Callable, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    from search_workflow import retry
    from search_workflow.lean import run_workflow_lean

    waits = []

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)

    def respond(request: httpx.Request) -> httpx.Response:
        if len(requests) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(429, headers={"Retry-After": "2"})
        return _reply('{"selected": [0]}')

    requests = http(respond)
    out = await run_workflow_lean("query")
    assert out["status"] == "ok" and out["results"] == [RAW[0]]
    assert len(requests) == 2 and len(waits) == 1
    if failure == "rate_limit":
        assert waits == [2.0]
    assert tools.METRICS.snapshot()["llm_calls"] == 1


@pytest.mark.parametrize("status", [400, 401, 500])
async def test_non_429_http_errors_are_typed_and_not_retried(fetch: list, http: Callable, status: int) -> None:
    from search_workflow.lean import run_workflow_lean

    requests = http(lambda request: httpx.Response(status, headers={"Retry-After": "0"}))
    out = await run_workflow_lean("query")
    assert out["status"] == "error"
    assert out["error"]["type"] == "workflow_error"
    assert out["error"]["message"].startswith("Workflow execution error: ")
    assert str(status) in out["error"]["message"]
    assert len(requests) == 1


async def test_other_provider_returns_typed_error_without_http(fetch: list, http: Callable) -> None:
    from search_workflow.lean import run_workflow_lean

    requests = http(lambda request: pytest.fail("Unsupported provider must not reach HTTP"))
    out = await run_workflow_lean("query", {"configurable": {"model": "ollama/llama3.1"}})
    assert out["status"] == "error"
    assert out["error"]["type"] == "workflow_error"
    assert "ollama" in out["error"]["message"]
    assert not requests


@pytest.mark.parametrize("primary,legacy,expected", [
    ("https://primary.test/v1/", "https://legacy.test/v1", "https://primary.test/v1/chat/completions"),
    (None, "https://legacy.test/v1/", "https://legacy.test/v1/chat/completions"),
])
async def test_base_url_precedence(fetch: list, http: Callable, monkeypatch: pytest.MonkeyPatch,
                                   primary: str | None, legacy: str, expected: str) -> None:
    from search_workflow.lean import run_workflow_lean

    if primary is not None:
        monkeypatch.setenv("OPENAI_BASE_URL", primary)
    monkeypatch.setenv("OPENAI_API_BASE", legacy)
    requests = http(lambda request: _reply('{"selected": [0]}'))
    assert (await run_workflow_lean("query"))["status"] == "ok"
    assert str(requests[0].url) == expected


async def test_configured_fetch_options_reach_both_engines(
    http: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from search_workflow.lean import run_workflow_lean

    calls = {}

    async def searxng(self: tools.SearXNGClient, **kwargs: Any) -> list:
        calls["searxng"] = (self.base_url, self.timeout, kwargs)
        return [RAW[0]]

    async def ddg(query: str, max_results: int, **kwargs: Any) -> list:
        calls["ddg"] = (query, max_results, kwargs)
        return [RAW[1]]

    monkeypatch.setattr(tools.SearXNGClient, "search", searxng)
    monkeypatch.setattr(tools, "_ddg_search", ddg)
    http(lambda request: _reply('{"selected": [0]}'))
    out = await run_workflow_lean("User Query", {"configurable": {
        "region": "de-de", "timelimit": "w", "max_search_results_tool": 7,
        "searxng_url": "http://searxng.test:8080", "searxng_timeout": 9,
        "engine_deadline_s": 1.5,
    }})
    assert out["status"] == "ok"
    assert calls == {
        "searxng": ("http://searxng.test:8080", 9, {
            "query": "User Query", "language": "de", "time_range": "w",
            "max_results": 7, "categories": "general",
        }),
        "ddg": ("User Query", 7, {"region": "de-de", "timelimit": "w"}),
    }


async def test_engine_degradation_wins_over_malformed_evaluator(
    http: Callable, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from search_workflow.lean import run_workflow_lean

    async def fetch(query: str, **kwargs: Any) -> list:
        tools.METRICS.record_provenance(
            {"engines_used": ["ddg"], "n_searxng": 0, "n_ddg": 3}
        )
        return RAW

    monkeypatch.setattr(tools, "search_direct", fetch)
    http(lambda request: _reply("bad JSON"))
    out = await run_workflow_lean("query")
    assert out["results"] == RAW
    assert out["engines_used"] == ["duckduckgo"]
    assert out["degraded"] is True
    assert out["degraded_reason"] == "engine"


@pytest.mark.parametrize("typed", [False, True])
async def test_fetch_errors_keep_the_typed_contract(monkeypatch: pytest.MonkeyPatch, typed: bool) -> None:
    from search_workflow.errors import SearchError
    from search_workflow.lean import run_workflow_lean

    async def fetch(query: str, **kwargs: Any) -> list:
        if typed:
            raise SearchError("fetch failed", error_type="fetch_error")
        raise RuntimeError("fetch failed")

    monkeypatch.setattr(tools, "search_direct", fetch)
    out = await run_workflow_lean("query")
    assert out == {"status": "error", "error": {
        "type": "fetch_error" if typed else "workflow_error",
        "message": "fetch failed" if typed else "Workflow execution error: fetch failed",
    }}
    assert tools.METRICS.snapshot()["llm_calls"] == 0


async def test_mcp_other_provider_returns_error_without_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    from search_workflow import mcp_server

    async def empty(query: str, **kwargs: Any) -> list:
        return RAW

    monkeypatch.setattr(tools, "search_direct", empty)
    monkeypatch.setattr(mcp_server, "_RANKER_MODEL", "ollama/llama3.1")
    content, _ = await mcp_server.create_mcp_server().call_tool(
        "web_search", {"query": "User Query", "max_results": 2}
    )
    out = json.loads(content[0].text)
    assert len(out) == 1 and "ollama" in str(out[0]["error"])
