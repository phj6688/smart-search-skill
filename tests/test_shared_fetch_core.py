"""Shared fetch core + URL-normalized dedup (HLB-650).

Both `search` and `search_direct` route their fetch through _fetch_and_merge,
which dedups on normalize_url(link). Three groups of tests:

1. normalize_url unit tests (`pytest -k normalize_url`).
2. Dedup tests drawn from tests/fixtures/FIX-MERGE.json: COLLAPSE (a
   tracking-param twin and a trailing-slash twin merge to one) and SURVIVAL
   (a ?page=2 twin stays distinct).
3. An end-to-end test driving run_workflow through the shared core offline,
   asserting the {"status": "ok", "results": [...]} shape and that both
   engines were fetched in parallel on the tool path.
"""

import json
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools
from search_workflow.utils import SelectionResponse
from tests.fixtures_fallback import configure_fallback_state

FIX_MERGE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "FIX-MERGE.json").read_text()
)


# --- normalize_url unit tests --------------------------------------------


def test_normalize_url_lowercases_scheme_and_host() -> None:
    assert tools.normalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"


def test_normalize_url_strips_default_ports() -> None:
    assert tools.normalize_url("http://example.com:80/a") == "http://example.com/a"
    assert tools.normalize_url("https://example.com:443/a") == "https://example.com/a"


def test_normalize_url_keeps_non_default_port() -> None:
    assert (
        tools.normalize_url("http://example.com:8080/a") == "http://example.com:8080/a"
    )


def test_normalize_url_default_port_is_scheme_paired() -> None:
    # An explicit non-default port must be kept even when it equals the OTHER
    # scheme's default, so the two never collapse onto the wrong dedup key.
    assert tools.normalize_url("https://example.com:80/a") != tools.normalize_url(
        "https://example.com/a"
    )
    assert tools.normalize_url("http://example.com:443/a") != tools.normalize_url(
        "http://example.com/a"
    )


def test_normalize_url_malformed_port_falls_back_to_raw() -> None:
    # An out-of-range port makes urlsplit().port raise; the raw string is
    # returned so one bad result cannot abort the whole merge.
    bad = "http://example.com:99999/a"
    assert tools.normalize_url(bad) == bad


def test_normalize_url_strips_single_trailing_slash() -> None:
    assert tools.normalize_url("https://example.com/a/") == "https://example.com/a"
    # Only one slash comes off, never two.
    assert tools.normalize_url("https://example.com/a//") == "https://example.com/a/"


def test_normalize_url_strips_only_tracking_params() -> None:
    got = tools.normalize_url(
        "https://x.com/a?utm_source=nl&utm_medium=email&id=7&fbclid=z&gclid=w"
    )
    assert got == "https://x.com/a?id=7"


def test_normalize_url_keeps_non_tracking_params_and_query_case() -> None:
    assert tools.normalize_url("https://x.com/a?Page=2&Q=Ab") == "https://x.com/a?Page=2&Q=Ab"


def test_normalize_url_leaves_path_case_untouched() -> None:
    assert tools.normalize_url("https://x.com/Foo/Bar") == "https://x.com/Foo/Bar"


def test_normalize_url_page_param_keeps_urls_distinct() -> None:
    assert tools.normalize_url("https://x.com/a?page=2") != tools.normalize_url(
        "https://x.com/a"
    )


# --- dedup tests drawn from FIX-MERGE.json --------------------------------


def _record(variant: str) -> dict[str, Any]:
    for rec in FIX_MERGE:
        if rec["variant"] == variant:
            return rec
    raise AssertionError(f"no {variant!r} record in FIX-MERGE.json")


def _split_by_engine(
    rec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    searxng = [dict(r) for r in rec["results"] if r.get("engine") == "searxng"]
    ddg = [dict(r) for r in rec["results"] if r.get("engine") != "searxng"]
    return searxng, ddg


def _group_links(rec: dict[str, Any], group: str) -> list[str]:
    return [
        r["link"]
        for r, g in zip(rec["results"], rec["labels"]["duplicate_groups"])
        if g == group
    ]


async def _run_core(
    monkeypatch: pytest.MonkeyPatch,
    searxng_results: list[dict[str, Any]],
    ddg_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    async def fake_searxng(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, Any]]:
        return [dict(r) for r in searxng_results]

    async def fake_ddg(query: str, max_results: int) -> list[dict[str, Any]]:
        return [dict(r) for r in ddg_results]

    monkeypatch.setattr(tools.SearXNGClient, "search", fake_searxng)
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg)
    return await tools.search_direct("q", max_results=50)


@pytest.mark.parametrize("variant", ["trailing_slash", "utm_query"])
async def test_collapse_twin_merges_to_one(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    rec = _record(variant)
    twins = _group_links(rec, "dg-a")
    assert len(twins) == 2, "fixture record must carry a two-member duplicate group"
    # The pair differs only by a trailing slash or a utm_* param, so it must
    # share one normalized key.
    assert tools.normalize_url(twins[0]) == tools.normalize_url(twins[1])

    searxng, ddg = _split_by_engine(rec)
    result = await _run_core(monkeypatch, searxng, ddg)

    twin_key = tools.normalize_url(twins[0])
    survivors = [r for r in result if tools.normalize_url(r["link"]) == twin_key]
    assert len(survivors) == 1, "the twin pair did not collapse to one"
    # One of the four inputs was dropped; the two non-twin links survive.
    assert len(result) == len(rec["results"]) - 1


async def test_survival_page_param_keeps_two_urls_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ?page=2 is a real page selector, not a tracking tag; both URLs survive.
    base = _record("distinct_pages_one_domain")["results"][0]["link"]
    page1 = base
    page2 = f"{base}?page=2"
    assert tools.normalize_url(page1) != tools.normalize_url(page2)

    searxng = [{"title": "p1", "link": page1, "snippet": ""}]
    ddg = [{"title": "p2", "link": page2, "snippet": ""}]
    result = await _run_core(monkeypatch, searxng, ddg)

    assert len(result) == 2
    assert {r["link"] for r in result} == {page1, page2}


# --- end-to-end: run_workflow through the shared core ----------------------


class _AgentBound:
    """The agent step: emit a `search` tool call so the graph reaches tools."""

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "shared core probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_1",
                }
            ],
        )


class _EvaluatorBound:
    """The evaluator step: select the first merged result by index."""

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        return SelectionResponse(selected=[0])


class _ToolCallingModel:
    def bind_tools(self, tools_: object) -> _AgentBound:
        return _AgentBound()

    def with_structured_output(self, schema: object) -> _EvaluatorBound:
        return _EvaluatorBound()


async def test_run_workflow_drives_shared_core_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Override the default agent stub so the agent emits a tool call: without
    # it the graph ends at the agent and never touches the shared core.
    model = _ToolCallingModel()
    # load_chat_model now takes a temperature kwarg (temperature=0 for the
    # select-by-index evaluator call), so the stub must accept it.
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda name, **kwargs: model
    )

    # Both engines answer at the seam the core dispatches to. No health probe
    # to stub: the tool path no longer issues one.
    configure_fallback_state(monkeypatch, "searxng_ok")

    out = await graph.run_workflow("shared core probe")

    assert out["status"] == "ok"
    assert isinstance(out["results"], list) and out["results"]
    # Select-by-index returns a fetched result verbatim; assert the shape
    # rather than an invented link (the evaluator no longer regenerates URLs).
    assert out["results"][0]["link"].startswith("http")

    # The tool path fetched both engines in parallel through the shared core:
    # one outbound request each, no LLM call inside the core.
    snapshot = tools.METRICS.snapshot()
    assert snapshot["outbound_search_requests"] == 2
    assert snapshot["engines_used"] == {"searxng": 1, "ddg": 1}
    assert snapshot["llm_calls"] == 0


# --- probe removal: the SearXNG leg makes one request per tool query -------


class _FakeSearxngResponse:
    """Minimal stand-in for the aiohttp response context manager.

    SearXNGClient.search does `async with session.get(...) as response:` then
    reads `response.status` and `await response.json()`. Nothing else is
    touched, so only those three surfaces are implemented.
    """

    status = 200

    async def json(self) -> dict[str, Any]:
        return {
            "results": [
                {"url": "https://searxng.test/x", "title": "x", "content": "c"}
            ]
        }

    async def __aenter__(self) -> "_FakeSearxngResponse":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


async def test_tool_query_issues_one_searxng_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Count real outbound GETs to SearXNG. Before the probe was dropped a single
    # tool query hit SearXNG twice: once for health_check, once for the search.
    # The redundant probe is gone, so the SearXNG leg must now make exactly one
    # request. DDG rides DDGS().text(), not aiohttp, so patching ClientSession
    # here isolates the SearXNG leg; stubbing _ddg_search keeps it offline.
    searxng_gets: list[str] = []

    def fake_get(self: aiohttp.ClientSession, url: str, *args: Any, **kwargs: Any) -> _FakeSearxngResponse:
        searxng_gets.append(url)
        return _FakeSearxngResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "get", fake_get)

    async def no_ddg(query: str, max_results: int) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(tools, "_ddg_search", no_ddg)

    results = await tools.search("probe", "us-en", None, config={})

    # DDG is stubbed out, so every captured aiohttp GET is a SearXNG request:
    # assert the whole list is one, so a restored /health probe also fails here.
    assert len(searxng_gets) == 1, (
        "tool path must issue exactly one SearXNG request; a second was the "
        f"redundant health probe. got: {searxng_gets}"
    )
    assert searxng_gets[0].endswith("/search")
    # The single search leg still contributes its result to the merge.
    assert [r["link"] for r in results] == ["https://searxng.test/x"]
    assert tools.METRICS.snapshot()["engines_used"] == {"searxng": 1}
