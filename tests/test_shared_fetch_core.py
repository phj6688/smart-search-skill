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

import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools
from search_workflow.utils import ArticlesResponse, ArticleStrict
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
    """The evaluator step: rank the merged tool output into one article."""

    async def ainvoke(self, value: object, config: object = None) -> ArticlesResponse:
        return ArticlesResponse(
            articles=[
                ArticleStrict(
                    title="Merged article",
                    link="https://example.test/a",
                    snippet="ranked from the merged core output",
                    similarity=0.9,
                )
            ]
        )


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
    monkeypatch.setattr("search_workflow.graph.load_chat_model", lambda name: model)

    # Both engines answer at the seam the core dispatches to.
    configure_fallback_state(monkeypatch, "searxng_ok")

    async def healthy() -> bool:
        return True

    monkeypatch.setattr(tools.searxng_client, "health_check", healthy)

    out = await graph.run_workflow("shared core probe")

    assert out["status"] == "ok"
    assert isinstance(out["results"], list) and out["results"]
    assert out["results"][0]["link"] == "https://example.test/a"

    # The tool path fetched both engines in parallel through the shared core:
    # one outbound request each, no LLM call inside the core.
    snapshot = tools.METRICS.snapshot()
    assert snapshot["outbound_search_requests"] == 2
    assert snapshot["engines_used"] == {"searxng": 1, "ddg": 1}
    assert snapshot["llm_calls"] == 0
