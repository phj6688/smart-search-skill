"""No-LLM deterministic mode and evaluator payload minimization (HLB-661).

run_workflow(query, use_llm=False) returns the shared fetch core output in the
SAME typed envelope the LLM path uses, without entering the agent/evaluator
nodes and without ever constructing a chat model, so tools.METRICS.llm_calls
stays 0. engines_used/degraded/degraded_reason come from the same S03
attribution record (_surface_provenance, HLB-657), so single-engine degradation
surfaces as degraded=True on both paths.

On the LLM path the evaluator payload is minimized: no raw URL (no "http"
substring) and no snippet over ~200 chars reaches the model, while the returned
URLs stay byte-identical to the fetched ones on both paths.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools
from search_workflow.utils import SelectionResponse
from tests.fixtures_fallback import configure_fallback_state

# A single-engine, distinct-domain corpus so reciprocal rank fusion preserves the
# fetch order (single engine) and the domain cap never trims (one URL per
# registrable domain). The Bravo link is mixed-case on purpose: a path that
# lowercased or regenerated URLs would fail the byte-identity assertions. The
# Alpha snippet is deliberately far longer than the cap so truncation is visible.
_LONG_TAIL = "TAILMARKER_BEYOND_THE_CAP"
_CORPUS: list[dict[str, str]] = [
    {
        "title": "Alpha headline about reactors",
        "link": "https://alpha.example/a",
        "snippet": "A" * 250 + _LONG_TAIL,
    },
    {
        "title": "Bravo headline about grids",
        "link": "https://Bravo.EXAMPLE/B",
        "snippet": "A short bravo snippet without any link in it.",
    },
    {
        "title": "Charlie headline about storage",
        "link": "https://charlie.example/c",
        "snippet": "A short charlie snippet without any link in it.",
    },
]


def _stub_engines(
    monkeypatch: pytest.MonkeyPatch,
    searxng_results: list[dict[str, str]],
    ddg_results: list[dict[str, str]],
) -> None:
    """Stub both engine boundaries the shared fetch core dispatches to."""

    async def fake_searxng_search(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, str]]:
        return [dict(r) for r in searxng_results]

    async def fake_ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
        return [dict(r) for r in ddg_results]

    monkeypatch.setattr(tools.SearXNGClient, "search", fake_searxng_search)
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg_search)


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the graph's load_chat_model seam to a sentinel that RAISES.

    A construction on the deterministic path would trip this, proving no chat
    model is built when use_llm=False.
    """

    def _raise(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "load_chat_model must not be called on the use_llm=False path"
        )

    monkeypatch.setattr("search_workflow.graph.load_chat_model", _raise)


class _AgentBound:
    """Agent step: emit a `search` tool call so the graph reaches the tools node."""

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "payload probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_1",
                }
            ],
        )


class _CapturingSelector:
    """with_structured_output proxy that records the payload it is handed."""

    def __init__(self, indices: list[int]) -> None:
        self._indices = indices
        self.captured: object | None = None

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        self.captured = value
        return SelectionResponse(selected=self._indices)


class _ToolCallingModel:
    """Covers both node seams: bind_tools for the agent, structured for the eval."""

    def __init__(self, selector: _CapturingSelector) -> None:
        self._selector = selector

    def bind_tools(self, tools_: object) -> _AgentBound:
        return _AgentBound()

    def with_structured_output(self, schema: object) -> _CapturingSelector:
        return self._selector


def _install_model(
    monkeypatch: pytest.MonkeyPatch, selector: _CapturingSelector
) -> None:
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model",
        lambda name, **kwargs: _ToolCallingModel(selector),
    )


def _payload_text(captured: object) -> str:
    to_string: Callable[[], str] | None = getattr(captured, "to_string", None)
    return to_string() if callable(to_string) else str(captured)


# --- 1. deterministic path: ok, non-empty, zero LLM calls --------------------


async def test_use_llm_false_returns_ok_without_touching_the_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fallback_state(monkeypatch, "searxng_ok")
    _forbid_llm(monkeypatch)

    out = await graph.run_workflow("deterministic probe", use_llm=False)

    assert out["status"] == "ok"
    assert out["results"], "shared fetch core should return non-empty results"
    # The sentinel never raised, and no node ran: the counter stays untouched.
    assert tools.METRICS.snapshot()["llm_calls"] == 0


# --- 2. five-state degraded flag on the deterministic path --------------------


@pytest.mark.parametrize(
    ("state", "expected_degraded"),
    [
        ("searxng_ok", False),  # both engines served raw rows
        ("searxng_raises", True),  # only DDG served
        ("searxng_empty", True),  # only DDG served
    ],
)
async def test_use_llm_false_degraded_follows_engines_served(
    monkeypatch: pytest.MonkeyPatch, state: str, expected_degraded: bool
) -> None:
    configure_fallback_state(monkeypatch, state)
    _forbid_llm(monkeypatch)

    out = await graph.run_workflow("deterministic probe", use_llm=False)

    assert out["status"] == "ok"
    assert out["degraded"] is expected_degraded


# --- 3. evaluator payload minimization on the LLM path -----------------------


async def test_evaluator_payload_drops_urls_and_caps_snippets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_engines(monkeypatch, _CORPUS, [])
    selector = _CapturingSelector(indices=[0, 1])
    _install_model(monkeypatch, selector)

    out = await graph.run_workflow("payload probe", use_llm=True)

    assert selector.captured is not None, "evaluator model was never invoked"
    payload = _payload_text(selector.captured)

    # No raw URL reaches the model: the link field is withheld, indices sent.
    assert "http" not in payload
    # The oversized snippet is truncated at the cap, so its tail is gone while
    # the 200-char prefix is still present.
    assert _LONG_TAIL not in payload
    assert "A" * 200 in payload

    # Selected indices resolve back to byte-identical fetched URLs (including the
    # mixed-case Bravo host): the returned results are never rewritten.
    assert out["status"] == "ok"
    result_links = [r["link"] for r in out["results"]]
    assert result_links == ["https://alpha.example/a", "https://Bravo.EXAMPLE/B"]


# --- 4. URL byte-identity across both paths ----------------------------------


async def test_urls_byte_identical_across_llm_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_engines(monkeypatch, _CORPUS, [])

    # LLM path first: select every fetched index so the sets are comparable.
    selector = _CapturingSelector(indices=[0, 1, 2])
    _install_model(monkeypatch, selector)
    llm_out = await graph.run_workflow("payload probe", use_llm=True)

    # Deterministic path over the same engine stubs, with the LLM seam forbidden.
    _forbid_llm(monkeypatch)
    det_out = await graph.run_workflow("payload probe", use_llm=False)

    fetched_links = sorted(r["link"] for r in _CORPUS)
    llm_links = sorted(r["link"] for r in llm_out["results"])
    det_links = sorted(r["link"] for r in det_out["results"])

    # Returned URLs are byte-identical to the fetched ones on BOTH paths, so the
    # two paths agree URL-for-URL.
    assert llm_links == fetched_links
    assert det_links == fetched_links
    # The mixed-case host survives verbatim, not lowercased, on either path.
    assert "https://Bravo.EXAMPLE/B" in llm_links
    assert "https://Bravo.EXAMPLE/B" in det_links
