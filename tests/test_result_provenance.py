"""Result-path provenance metadata on run_workflow (HLB-657).

run_workflow's SUCCESS envelope now carries engines_used/degraded/degraded_reason
alongside status/results. The triple is READ off the S03 attribution record
(tools.METRICS's per-query provenance from HLB-646/650), never re-derived from
result contents or control flow. These tests pin:

1. The triple per state over the five-state fallback fixture, driven end to end
   through run_workflow so the real _fetch_and_merge records the provenance.
2. A synthetic attribution record swapped in under run_workflow, proving the
   surfaced metadata FOLLOWS the record (read, not recompute).
3. The SearXNG-down case: engines_used == ["duckduckgo"], degraded, reason
   "engine".
4. The new keys ride alongside status/results and do not disturb the MCP path.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, mcp_server, tools
from search_workflow.utils import SelectionResponse
from tests.fixtures_fallback import FALLBACK_STATE_NAMES, configure_fallback_state

# state name -> (surfaced engines_used, degraded, degraded_reason).
# "ddg" is surfaced as "duckduckgo"; searxng_ok_ddg_unused stays NOT degraded
# because DDG did serve raw rows (they merely deduped away), while
# searxng_raises/empty are degraded because SearXNG served nothing.
_EXPECTED: dict[str, tuple[list[str], bool, str | None]] = {
    "searxng_ok": (["duckduckgo", "searxng"], False, None),
    "searxng_raises": (["duckduckgo"], True, "engine"),
    "searxng_empty": (["duckduckgo"], True, "engine"),
    "searxng_ok_ddg_unused": (["searxng"], False, None),
    "both_fail": ([], True, "engine"),
}


class _AgentBound:
    """Agent step: emit a `search` tool call so the graph reaches the tools node."""

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "provenance probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_1",
                }
            ],
        )


class _EvaluatorBound:
    """Evaluator step: select the first fetched result by index."""

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        return SelectionResponse(selected=[0])


class _ToolCallingModel:
    def bind_tools(self, tools_: object) -> _AgentBound:
        return _AgentBound()

    def with_structured_output(self, schema: object) -> _EvaluatorBound:
        return _EvaluatorBound()


def _wire_tool_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ToolCallingModel()
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda name, **kwargs: model
    )
    # HLB-652 removed the SearXNG health probe; the fallback fixture stubs the
    # engine boundary directly, so no health_check patch is needed.


@pytest.mark.parametrize("state", FALLBACK_STATE_NAMES)
async def test_run_workflow_surfaces_triple_per_state(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    _wire_tool_path(monkeypatch)
    configure_fallback_state(monkeypatch, state)

    out = await graph.run_workflow("provenance probe")

    expected_engines, expected_degraded, expected_reason = _EXPECTED[state]
    assert out["status"] == "ok"
    assert out["engines_used"] == expected_engines
    assert out["degraded"] is expected_degraded
    assert out["degraded_reason"] == expected_reason


async def test_searxng_down_surfaces_duckduckgo_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wire_tool_path(monkeypatch)
    configure_fallback_state(monkeypatch, "searxng_raises")

    out = await graph.run_workflow("provenance probe")

    assert out["engines_used"] == ["duckduckgo"]
    assert out["degraded"] is True
    assert out["degraded_reason"] == "engine"


async def test_surfaced_metadata_follows_synthetic_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The triple must follow the attribution record, not the merged results.

    The graph runs the searxng_ok state (both engines serve, not degraded), but
    the attribution record is swapped for a synthetic single-engine degraded
    record. run_workflow must surface the SYNTHETIC values, proving it reads the
    record rather than re-deriving engines from the result contents.
    """
    _wire_tool_path(monkeypatch)
    configure_fallback_state(monkeypatch, "searxng_ok")

    synthetic = {
        "n_searxng": 0,
        "n_ddg": 4,
        "n_after_dedup": 4,
        "elapsed_ms": 1.0,
        "ddg_ok": True,
        "fell_back": True,
        "engines_used": ["ddg"],
    }
    monkeypatch.setattr(tools.METRICS, "last_provenance", lambda: dict(synthetic))

    out = await graph.run_workflow("provenance probe")

    # Real run served both engines; the synthetic record says DDG-only degraded.
    assert out["engines_used"] == ["duckduckgo"]
    assert out["degraded"] is True
    assert out["degraded_reason"] == "engine"


def test_surface_provenance_maps_and_flags() -> None:
    """Direct unit cover: mapping reads engine names and served-count off record."""
    record = {"n_searxng": 3, "n_ddg": 0, "engines_used": ["searxng"]}
    out = graph._surface_provenance(record)
    assert out == {
        "engines_used": ["searxng"],
        "degraded": True,
        "degraded_reason": "engine",
    }


def test_surface_provenance_missing_record_is_not_degraded() -> None:
    out = graph._surface_provenance(None)
    assert out == {"engines_used": [], "degraded": False, "degraded_reason": None}


async def test_metadata_rides_through_coerce_results_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_coerce_results reads status/results only; the new keys do not disturb it."""
    _wire_tool_path(monkeypatch)
    configure_fallback_state(monkeypatch, "searxng_raises")

    out = await graph.run_workflow("provenance probe")
    # Sanity: the success envelope carries the new metadata keys.
    assert {"engines_used", "degraded", "degraded_reason"} <= set(out)

    coerced = mcp_server._coerce_results(out)
    assert coerced == out["results"]
