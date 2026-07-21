"""Held-out behavioural probe for HLB-658.

Issue: "Return degraded raw results when evaluator output is malformed"
Anchor: src/search_workflow/graph.py:evaluator

Acceptance criteria checked (black-box, via run_workflow):
  1. The evaluator node catches UNPARSEABLE and SCHEMA-INVALID structured
     output and falls back to the merged/deduped raw results from the tool
     step instead of raising/erroring.
  2. The fallback is returned in the typed result shape with
     degraded=True and degraded_reason="evaluator"; the healthy path
     carries degraded=False and degraded_reason=None.
  3. The malformed-output branch NEVER raises SearchError and NEVER returns
     a status:"error" shape (status stays "ok").

The evaluator must actually run, so we do NOT patch the graph or the
evaluator/run_workflow. We patch only the two sanctioned boundaries:
  - search_workflow.graph.load_chat_model  -> one stub model that serves the
    agent node (bind_tools -> tool call) and the evaluator node
    (with_structured_output -> malformed output).
  - the search engines (SearXNGClient.search / _ddg_search) -> known results,
    so real merged raw results exist at the evaluator.

This file lives OUTSIDE tests/, so the repo's autouse conftest patch of
load_chat_model does not apply; the probe installs its own stub.
"""

import asyncio
import contextlib
import inspect
import json
import pathlib
from types import SimpleNamespace
from unittest import mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

import search_workflow.graph as graph
import search_workflow.tools as tools

# ASSUMPTION: raw result items expose their URL under one of these keys; we
# assert on the serialized blob so field naming does not matter.
KNOWN_LINKS = ("https://example.com/alpha", "https://example.com/beta")
FETCHED = [
    {"title": "Alpha", "link": KNOWN_LINKS[0], "url": KNOWN_LINKS[0],
     "href": KNOWN_LINKS[0], "snippet": "alpha", "content": "alpha",
     "engine": "searxng"},
    {"title": "Beta", "link": KNOWN_LINKS[1], "url": KNOWN_LINKS[1],
     "href": KNOWN_LINKS[1], "snippet": "beta", "content": "beta",
     "engine": "searxng"},
]

TOOL_CALL = {
    "name": "search",
    "args": {"query": "q", "region": "us-en", "timelimit": None},
    "id": "c1",
    "type": "tool_call",
}

try:  # prefer the real schema for the healthy contrast, if importable
    from search_workflow.graph import SelectionResponse
except Exception:  # pragma: no cover - fallback to a duck type
    SelectionResponse = None


class _Sel(BaseModel):
    selected: list


def _unparseable():
    raise ValueError("could not parse")  # variant A: unparseable LLM output


def _schema_invalid():
    _Sel()  # variant B: missing required field -> pydantic ValidationError
    return None


def _valid():
    if SelectionResponse is not None:
        try:
            return SelectionResponse(selected=[0])
        except Exception:
            pass
    return SimpleNamespace(selected=[0])


def _make_model(structured_behavior):
    class _Bad:
        async def ainvoke(self, *a, **k):
            return structured_behavior()

    class _Stub:
        def bind_tools(self, *a, **k):
            return self

        def with_structured_output(self, *a, **k):
            return _Bad()

        async def ainvoke(self, *a, **k):
            return AIMessage(content="", tool_calls=[dict(TOOL_CALL)])

    return _Stub()


def _engine_stub(original, value):
    if inspect.iscoroutinefunction(original):
        async def _repl(*a, **k):
            return list(value)
        return _repl

    def _repl(*a, **k):
        return list(value)
    return _repl


@contextlib.contextmanager
def _wired(structured_behavior):
    searx = _engine_stub(tools.SearXNGClient.search, FETCHED)
    ddg = _engine_stub(tools._ddg_search, FETCHED)
    with mock.patch.object(graph, "load_chat_model",
                           lambda *a, **k: _make_model(structured_behavior)), \
            mock.patch.object(tools.SearXNGClient, "search", searx), \
            mock.patch.object(tools, "_ddg_search", ddg):
        yield


def _run(structured_behavior):
    with _wired(structured_behavior):
        return asyncio.run(graph.run_workflow("q"))


@pytest.mark.parametrize(
    "behavior", [_unparseable, _schema_invalid],
    ids=["unparseable", "schema-invalid"])
def test_malformed_evaluator_degrades_to_raw_results(behavior):
    result = _run(behavior)

    # (3) never raised, never a status:error shape
    assert result["status"] == "ok"

    # (1) fell back to the merged raw fetched results (non-empty, known links)
    assert isinstance(result["results"], list) and result["results"]
    blob = json.dumps(result["results"], default=str)
    for link in KNOWN_LINKS:
        assert link in blob

    # (2) degraded flags mark the evaluator fallback
    assert result["degraded"] is True
    assert result["degraded_reason"] == "evaluator"


def test_healthy_evaluator_is_not_degraded():
    result = _run(_valid)
    assert result["status"] == "ok"
    assert result["degraded"] is False
    assert result.get("degraded_reason") is None


def test_graph_source_handles_evaluator_degraded_branch():
    src = pathlib.Path(graph.__file__).read_text()
    assert "degraded_reason" in src
    assert '"evaluator"' in src or "'evaluator'" in src
    assert "except" in src  # malformed output is caught, not propagated
