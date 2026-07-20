"""Evaluator selects fetched results by index (HLB-655).

The evaluator asks the model for indices into the fetched result set, then joins
those indices back to the fetched objects. It never lets the model re-emit
title/link/snippet, so URLs cannot be mutated or lowercased and no similarity
score is invented. These tests pin that join, the mixed-case URL byte identity,
the golden message the TOOLS path emits, out-of-range dropping, and the cap.
"""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from search_workflow import graph
from search_workflow.state import State
from search_workflow.utils import SelectionResponse

GOLDEN = Path(__file__).parent / "golden" / "HLB-655_evaluator_tools_message.json"

# Fetched corpus for the golden test. Every entry carries only the fields the
# evaluator returns, so the emitted message is these entries verbatim.
GOLDEN_FETCHED = [
    {
        "title": "Alpha reactor reaches first criticality",
        "link": "https://alpha.example/news/criticality",
        "snippet": "The Alpha reactor achieved sustained criticality during commissioning.",
    },
    {
        "title": "Bravo grid absorbs record solar output",
        "link": "https://Bravo.EXAMPLE/Grid/Record",
        "snippet": "Operators reported a record midday solar contribution on the Bravo grid.",
    },
    {
        "title": "Charlie storage plant clears final review",
        "link": "https://charlie.example/storage/review",
        "snippet": "The Charlie storage plant passed its final regulatory review this week.",
    },
]


class _StubSelector:
    """Returns a fixed SelectionResponse so the join path is deterministic."""

    def __init__(self, indices: list[int]) -> None:
        self._indices = indices

    def bind_tools(self, tools: object) -> "_StubSelector":
        return self

    def with_structured_output(self, schema: object) -> "_StubSelector":
        return self

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        return SelectionResponse(selected=self._indices)


def _install_selector(monkeypatch: pytest.MonkeyPatch, indices: list[int]) -> None:
    model = _StubSelector(indices)
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model",
        lambda model_name, temperature=0.1: model,
    )


def _tools_path_state(query: str, fetched: list[dict[str, object]]) -> State:
    """State as it stands right after the tools node, feeding the evaluator."""
    return State(
        messages=[
            HumanMessage(content=query),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": query}, "id": "call_1"}
                ],
            ),
            ToolMessage(content=json.dumps(fetched), tool_call_id="call_1"),
        ]
    )


async def _run_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    indices: list[int],
    fetched: list[dict[str, object]],
    max_results: int = 5,
    query: str = "evaluator selection probe",
) -> AIMessage:
    _install_selector(monkeypatch, indices)
    state = _tools_path_state(query, fetched)
    config = {"configurable": {"max_search_results_evaluator": max_results}}
    result = await graph.evaluator(state, config=config)
    return result["messages"][0]


async def test_golden_tools_message(monkeypatch: pytest.MonkeyPatch) -> None:
    message = await _run_evaluator(monkeypatch, [0, 2], GOLDEN_FETCHED)

    assert isinstance(message, AIMessage)
    assert message.content == GOLDEN.read_text(encoding="utf-8").rstrip("\n")

    # Every link in the emitted message must appear verbatim in the fetched set.
    fetched_links = {entry["link"] for entry in GOLDEN_FETCHED}
    for entry in json.loads(message.content):
        assert entry["link"] in fetched_links


async def test_byte_identity_preserves_mixed_case_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed_case = "https://Example.COM/CaseSensitive/Path"
    fetched = [
        {
            "title": "Case sensitive resource announcement",
            "link": mixed_case,
            "snippet": "This fetched entry carries a deliberately mixed-case URL path.",
        }
    ]

    message = await _run_evaluator(monkeypatch, [0], fetched)

    payload = json.loads(message.content)
    # Byte-identical to the fetched string: not lowercased, not regenerated.
    assert payload[0]["link"] == mixed_case
    assert mixed_case in message.content


async def test_out_of_range_indices_are_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = [
        {"title": "Kept alpha entry", "link": "https://a.example/x", "snippet": "kept"},
        {"title": "Unpicked bravo entry", "link": "https://b.example/y", "snippet": "n"},
    ]

    message = await _run_evaluator(monkeypatch, [5, 0, 99, -1], fetched)

    payload = json.loads(message.content)
    assert payload == [fetched[0]]


async def test_duplicate_indices_yield_each_result_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A model that repeats an index must not return the same result twice.
    fetched = [
        {"title": "Alpha", "link": "https://a.example/x", "snippet": "s"},
        {"title": "Bravo", "link": "https://b.example/y", "snippet": "s"},
    ]

    message = await _run_evaluator(monkeypatch, [0, 0, 1, 1, 0], fetched)

    payload = json.loads(message.content)
    assert payload == [fetched[0], fetched[1]]


async def test_selection_capped_at_max_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = [
        {"title": f"Entry number {i}", "link": f"https://e.example/{i}", "snippet": "s"}
        for i in range(5)
    ]

    message = await _run_evaluator(
        monkeypatch, [0, 1, 2, 3, 4], fetched, max_results=2
    )

    payload = json.loads(message.content)
    assert payload == [fetched[0], fetched[1]]


async def test_at_least_one_when_a_relevant_result_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = [
        {"title": "Sole relevant entry", "link": "https://only.example/z", "snippet": "s"},
    ]

    message = await _run_evaluator(monkeypatch, [0], fetched)

    payload = json.loads(message.content)
    assert len(payload) == 1
    assert payload[0] == fetched[0]
