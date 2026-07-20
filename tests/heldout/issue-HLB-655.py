"""Held-out behavioural probe for HLB-655: rework evaluator to select results by index.

Acceptance criteria checked:
1. utils.SelectionResponse exists with `selected: list[int]`; graph.py wires
   with_structured_output to SelectionResponse with temperature=0.
2. Evaluator joins selected indices back to fetched result objects: returned
   title/link/snippet are byte-identical to the fetched fixture; out-of-range
   indices are dropped (selected=[1, 99] yields exactly one result).
3. URL byte-identity with a mixed-case URL (no lowercasing).
4. Golden regression test for the TOOLS-path message shape exists under tests/.
5. README.md / SKILL.md describe index-based selection, old ranking claims gone.
"""
import asyncio
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args, get_origin
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import search_workflow.graph as graph_mod
from search_workflow.utils import SelectionResponse

FETCHED = [
    {"title": "Plain result", "link": "https://example.org/plain",
     "snippet": "An unrelated first result."},
    {"title": "Case Sensitive Title",
     "link": "https://example.com/CaseSensitive/Path",
     "snippet": "Snippet bytes that must survive Untouched."},
]


def _repo_root() -> Path:
    cand = Path(graph_mod.__file__).resolve().parents[2]
    return cand if (cand / "README.md").exists() else Path.cwd()


def test_selection_response_shape():
    fields = getattr(SelectionResponse, "model_fields", None) or SelectionResponse.__fields__
    assert "selected" in fields, "SelectionResponse must have a 'selected' field"
    ann = fields["selected"].annotation
    assert get_origin(ann) is list and get_args(ann) == (int,), (
        f"SelectionResponse.selected must be list[int], got {ann!r}"
    )


def test_graph_source_uses_selection_response_and_temperature_zero():
    src = Path(graph_mod.__file__).read_text(encoding="utf-8")
    assert "SelectionResponse" in src, "graph.py must target SelectionResponse"
    assert re.search(r"temperature\s*=\s*0(\.0+)?(?![\d.])", src), (
        "graph.py evaluator must set temperature=0"
    )
    assert "similarity" not in src.lower(), (
        "sort-by-invented-similarity padding path must be deleted from graph.py"
    )


class _Structured:
    def __init__(self, resp):
        self._resp = resp

    async def ainvoke(self, *a, **k):
        return self._resp

    def invoke(self, *a, **k):
        return self._resp


class _ModelStub:
    def __init__(self, resp):
        self._resp = resp

    def with_structured_output(self, *a, **k):
        return _Structured(self._resp)

    def bind_tools(self, *a, **k):
        return self

    async def ainvoke(self, *a, **k):
        return self._resp


def _extract_results(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "articles"):
            if isinstance(payload.get(key), list):
                return payload[key]
    return None


def test_evaluator_joins_by_index_drops_out_of_range_preserves_bytes():
    fetched_json = json.dumps(FETCHED)
    msgs = [
        HumanMessage(content="find case sensitive docs"),
        AIMessage(content=fetched_json),
        ToolMessage(content=fetched_json, tool_call_id="call_1", name="search"),
    ]
    try:
        from search_workflow.state import State
        state = State(messages=msgs)
    except Exception:
        state = SimpleNamespace(messages=msgs, is_last_step=False)

    config = {"configurable": {"max_search_results_evaluator": 5}}
    # Sanctioned seam: stub the model factory, never the evaluator itself.
    # selected=[1, 99] -> index 1 joined back, 99 out of range and dropped.
    stub = _ModelStub(SelectionResponse(selected=[1, 99]))
    with patch.object(graph_mod, "load_chat_model", return_value=stub):
        out = graph_mod.evaluator(state, config)
        if inspect.iscoroutine(out):
            out = asyncio.run(out)

    assert isinstance(out, dict) and out.get("messages"), f"unexpected node output: {out!r}"
    content = out["messages"][-1].content
    if isinstance(content, list):
        content = "".join(str(c) for c in content)
    results = _extract_results(json.loads(content))
    assert results is not None, f"could not find results list in: {content!r}"
    assert len(results) == 1, (
        f"expected exactly 1 result (index 1 kept, 99 dropped), got {len(results)}: {results!r}"
    )
    item = results[0]
    link = item.get("link") or item.get("url")
    assert link == "https://example.com/CaseSensitive/Path", (
        f"link must be byte-identical to fetched URL (no case folding), got {link!r}"
    )
    assert item.get("title") == FETCHED[1]["title"], f"title mutated: {item.get('title')!r}"
    assert item.get("snippet") == FETCHED[1]["snippet"], f"snippet mutated: {item.get('snippet')!r}"


def test_docs_describe_index_based_selection():
    root = _repo_root()
    readme = (root / "README.md").read_text(encoding="utf-8")
    skill = (root / "SKILL.md").read_text(encoding="utf-8")
    assert "The LLM evaluator ranks results" not in readme, (
        "README.md still carries the old ranking claim"
    )
    assert "ranked by an LLM evaluator" not in skill, (
        "SKILL.md still carries the old ranking claim"
    )
    assert re.search(r"by index", readme + "\n" + skill, re.IGNORECASE), (
        "neither README.md nor SKILL.md mentions index-based selection"
    )


def test_golden_tools_message_regression_exists():
    tests_dir = _repo_root() / "tests"
    assert tests_dir.is_dir(), "tests/ directory missing"
    names = [p for p in tests_dir.rglob("*") if "golden" in p.name.lower()]
    assert names, "no file or directory under tests/ with 'golden' in its name"
    hits = [
        p for p in tests_dir.rglob("*.py")
        if "golden_tools_message" in p.read_text(encoding="utf-8", errors="replace")
    ]
    assert hits, "no test under tests/ references golden_tools_message"
