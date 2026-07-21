"""Malformed evaluator selection falls back to raw results (HLB-658).

When the evaluator's structured output is UNPARSEABLE or SCHEMA-INVALID, the
node must NOT raise: the merged, deduplicated raw results the tool step already
fetched are still good, only the selection is unusable. run_workflow surfaces
those raw results with degraded=True and degraded_reason="evaluator", reusing
the metadata triple HLB-657 defined. The healthy selection path stays
degraded=False, and the malformed branch never yields a SearchError or a
status:error envelope: typed errors stay reserved for real failures with no
results.

The evaluator LLM is stubbed through the graph.load_chat_model seam. The engine
boundary is stubbed with a known corpus that carries a mixed-case URL, so the
byte-identity of the fallback links is meaningful (raw results are returned
verbatim, not lowercased or regenerated). Both engines serve rows, so the
engine-degradation path stays off and the evaluator reason is what surfaces.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools
from search_workflow.utils import SelectionResponse

# Known raw corpus. The first link is deliberately mixed-case so a fallback that
# lowercased or regenerated links would fail the byte-identity assertion. Links
# are all distinct, so nothing dedups away and the merged set is these three in
# fetch order (SearXNG rows first, then DDG).
_RAW: list[dict[str, str]] = [
    {"title": "Mixed case host result", "link": "https://Example.COM/Path/One", "snippet": "s1"},
    {"title": "Second result", "link": "https://beta.example/two", "snippet": "s2"},
    {"title": "Third from ddg", "link": "https://gamma.example/three", "snippet": "s3"},
]


class _AgentBound:
    """Agent step: emit a `search` tool call so the graph reaches the tools node."""

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "malformed selection probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_1",
                }
            ],
        )


class _RaisingValueErrorSelector:
    """with_structured_output(...).ainvoke raising: unparseable structured output."""

    async def ainvoke(self, value: object, config: object = None) -> Any:
        raise ValueError("could not parse structured output")


class _ObjectWithoutSelected:
    """Stand-in return value that lacks a usable .selected attribute."""


class _NoSelectedSelector:
    """Returns an object without a usable .selected (schema-invalid)."""

    async def ainvoke(self, value: object, config: object = None) -> Any:
        return _ObjectWithoutSelected()


class _ValidationErrorSelector:
    """Constructs an invalid SelectionResponse, raising pydantic ValidationError.

    Mirrors what with_structured_output raises when the model's JSON does not
    satisfy the schema (an object is not int-coercible for selected: list[int]).
    """

    async def ainvoke(self, value: object, config: object = None) -> Any:
        return SelectionResponse(selected=[object()])


class _HealthySelector:
    """Returns a valid single-index selection so the join path runs normally."""

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        return SelectionResponse(selected=[0])


class _StubModel:
    """Covers the ChatOpenAI surface the agent and evaluator nodes touch."""

    def __init__(self, selector: object) -> None:
        self._selector = selector

    def bind_tools(self, tools_: object) -> _AgentBound:
        return _AgentBound()

    def with_structured_output(self, schema: object) -> object:
        return self._selector


def _wire(monkeypatch: pytest.MonkeyPatch, selector: object) -> None:
    """Stub the LLM seam and both engines: healthy fetch, chosen selector."""
    model = _StubModel(selector)
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda name, **kwargs: model
    )

    async def fake_searxng_search(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, str]]:
        return [dict(_RAW[0]), dict(_RAW[1])]

    async def fake_ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
        return [dict(_RAW[2])]

    monkeypatch.setattr(tools.SearXNGClient, "search", fake_searxng_search)
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg_search)


_MALFORMED_SELECTORS = {
    "ainvoke_raises_value_error": _RaisingValueErrorSelector,
    "returns_object_without_selected": _NoSelectedSelector,
    "ainvoke_raises_validation_error": _ValidationErrorSelector,
}


@pytest.mark.parametrize("selector_name", sorted(_MALFORMED_SELECTORS))
async def test_malformed_selection_falls_back_to_raw_degraded_evaluator(
    monkeypatch: pytest.MonkeyPatch, selector_name: str
) -> None:
    _wire(monkeypatch, _MALFORMED_SELECTORS[selector_name]())

    out = await graph.run_workflow("malformed selection probe")

    assert out["status"] == "ok"
    assert out["degraded"] is True
    assert out["degraded_reason"] == "evaluator"
    # The fallback returns the merged/deduped raw fetched results verbatim.
    assert out["results"] == _RAW
    # Links are byte-identical, including the mixed-case host: not lowercased,
    # not regenerated.
    assert [r["link"] for r in out["results"]] == [r["link"] for r in _RAW]
    assert out["results"][0]["link"] == "https://Example.COM/Path/One"


async def test_healthy_selection_is_not_degraded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire(monkeypatch, _HealthySelector())

    out = await graph.run_workflow("healthy selection probe")

    assert out["status"] == "ok"
    assert out["degraded"] is False
    assert out["degraded_reason"] is None
    # Healthy path joins the selected index back to the fetched object.
    assert out["results"] == [_RAW[0]]


@pytest.mark.parametrize("selector_name", sorted(_MALFORMED_SELECTORS))
async def test_malformed_branch_never_raises_searcherror_or_error_status(
    monkeypatch: pytest.MonkeyPatch, selector_name: str
) -> None:
    _wire(monkeypatch, _MALFORMED_SELECTORS[selector_name]())

    # The call completing at all proves no SearchError propagated to the caller.
    out = await graph.run_workflow("malformed selection probe")

    assert out["status"] == "ok"
    assert out["status"] != "error"
    # Typed-error envelope is absent on the malformed branch; it stays reserved
    # for real failures with no results.
    assert "error" not in out
    assert out["results"] == _RAW
