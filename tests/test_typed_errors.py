"""Typed discriminated results from run_workflow (HLB-654).

run_workflow returns a discriminated dict instead of a Union[list|dict|str]:
success = {"status": "ok", "results": [...]}, failure = {"status": "error",
"error": {"type", "message"}}. It never returns a bare string; the only thing
that may leave the function untyped is a genuinely unexpected exception, which
these tests do not permit for the known failure modes.

The ToolNode test proves the other half of the contract: a SearchError raised
inside the LangGraph tool path is serialized into a ToolMessage rather than
crashing the host process.
"""

import json
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from search_workflow import graph
from search_workflow.errors import SearchError


def _stub_ainvoke(monkeypatch: pytest.MonkeyPatch, behavior) -> None:
    """Replace the compiled graph's ainvoke with a canned final state or raise."""

    async def fake_ainvoke(state: object, config: object = None) -> object:
        return behavior()

    monkeypatch.setattr(graph.graph, "ainvoke", fake_ainvoke)


async def test_run_workflow_success_returns_ok_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{"title": "T", "link": "https://example.test/x", "snippet": "s"}]
    _stub_ainvoke(
        monkeypatch,
        lambda: {"messages": [AIMessage(content=json.dumps(payload))]},
    )

    out = await graph.run_workflow("q")

    # status/results stay exactly as HLB-654 defined; HLB-657 adds the
    # provenance metadata alongside. No search ran (ainvoke is stubbed) and the
    # metrics reset per test, so the attribution record is absent: engines_used
    # empty, not degraded.
    assert out["status"] == "ok"
    assert out["results"] == payload
    assert out["engines_used"] == []
    assert out["degraded"] is False
    assert out["degraded_reason"] is None


async def test_run_workflow_bad_json_returns_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ainvoke(
        monkeypatch,
        lambda: {"messages": [AIMessage(content="not-json-at-all")]},
    )

    out = await graph.run_workflow("q")

    assert out["status"] == "error"
    assert out["error"]["type"] == "json_parse_error"
    assert isinstance(out["error"]["message"], str)


async def test_run_workflow_no_messages_returns_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ainvoke(monkeypatch, lambda: {"messages": []})

    out = await graph.run_workflow("q")

    assert out["status"] == "error"
    assert out["error"]["type"] == "empty_result"


async def test_run_workflow_value_error_returns_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise ValueError("bad message format")

    _stub_ainvoke(monkeypatch, _raise)

    out = await graph.run_workflow("q")

    assert out["status"] == "error"
    assert out["error"]["type"] == "message_format_error"
    assert "bad message format" in out["error"]["message"]


async def test_run_workflow_unexpected_exception_returns_error_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise RuntimeError("kaboom")

    _stub_ainvoke(monkeypatch, _raise)

    out = await graph.run_workflow("q")

    assert out["status"] == "error"
    assert out["error"]["type"] == "workflow_error"
    assert "kaboom" in out["error"]["message"]


async def test_run_workflow_never_returns_bare_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_ainvoke(
        monkeypatch,
        lambda: {"messages": [AIMessage(content="still-not-json")]},
    )

    out = await graph.run_workflow("q")

    assert isinstance(out, dict)
    assert not isinstance(out, str)


class _ToolState(TypedDict):
    messages: Annotated[list, add_messages]


async def test_toolnode_error_serializes_searcherror() -> None:
    """A SearchError raised in the tool path becomes a ToolMessage, not a crash."""

    @tool
    def failing_search(query: str) -> str:
        """Search tool wired to fail so the tool path can be exercised."""
        raise SearchError("engine unreachable", error_type="workflow_error")

    builder = StateGraph(_ToolState)
    builder.add_node("tools", ToolNode([failing_search], handle_tool_errors=SearchError))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    compiled = builder.compile()

    ai = AIMessage(
        content="",
        tool_calls=[{"name": "failing_search", "args": {"query": "x"}, "id": "call_1"}],
    )
    result = compiled.invoke({"messages": [ai]})

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].status == "error"
    assert "engine unreachable" in tool_messages[0].content
