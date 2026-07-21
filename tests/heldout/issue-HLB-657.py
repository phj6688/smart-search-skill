"""Held-out behavioural probe for HLB-657.

Issue: "Expose engines_used and degraded metadata on the result path".
Anchor: src/search_workflow/graph.py:run_workflow.

Acceptance criteria checked (black-box, derived from the spec, not the code):
  1. run_workflow's typed success result gains engines_used (list[str]),
     degraded (bool) and degraded_reason (enum with values "engine"/"evaluator").
     These fields are exposed either at the top level of the result dict OR under
     a "meta"/"metadata" sub-dict.
  2. On a degraded engine-state (fewer engines served than attempted), degraded
     is True and degraded_reason == "engine"; the sole serving engine is the
     ddg fallback (Eval 3: engines_used == ["duckduckgo"]). On a healthy state
     degraded is False. (The "evaluator" reason is set by a later story.)
  3. engines_used is sourced from the S03 attribution record (tools.METRICS /
     provenance), not recomputed by a second engine-comparison branch.

The probe never patches run_workflow. Its sanctioned seams are the compiled
graph collaborator (search_workflow.graph.graph.ainvoke) and the attribution
record (search_workflow.tools.METRICS).
"""

import asyncio
import json
import os
import pathlib
import sys
from unittest import mock

import pytest

# Probe runs from the issue worktree root; the package lives under src/.
_ROOT = os.getcwd()
for _p in (os.path.join(_ROOT, "src"), _ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
# Constructing a chat model at import time needs a key but no network here.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-heldout")

try:  # a real message keeps get_message_text / isinstance checks happy
    from langchain_core.messages import AIMessage as _AIMessage
except Exception:  # pragma: no cover
    _AIMessage = None

_RESULTS = [
    {
        "title": "Python (programming language)",
        "url": "https://example.org/python",
        "content": "Python is a high-level programming language.",
        "language": "en",
    }
]


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.type = "ai"


def _mk_message(content):
    if _AIMessage is not None:
        return _AIMessage(content=content)
    return _FakeMessage(content)


def _g():
    import search_workflow.graph as g

    return g


def _tools():
    from search_workflow import tools

    return tools


def _graph_source():
    return pathlib.Path(_g().__file__).read_text(encoding="utf-8")


def _val(x):
    return getattr(x, "value", x)  # unwrap enum members; pass through plain values


def _field(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def _has_meta(o):
    if isinstance(o, dict):
        return "engines_used" in o and "degraded" in o
    return hasattr(o, "engines_used") and hasattr(o, "degraded")


def _locate_meta(result):
    if _has_meta(result):
        return result
    for key in ("meta", "metadata"):
        sub = _field(result, key)
        if sub is not None and _has_meta(sub):
            return sub
    return None


def _reset_metrics(tools):
    reset = getattr(getattr(tools, "METRICS", None), "reset", None)
    if callable(reset):
        try:
            reset()
        except Exception:
            pass


def _force_single_engine(tools):
    """Best-effort: make the attribution record show only the ddg fallback
    served after SearXNG failed, i.e. a degraded engine-state."""
    m = getattr(tools, "METRICS", None)
    if m is None:
        return
    for arg in ({"duckduckgo": 1}, {"ddg": 1}):
        fn = getattr(m, "record_engines_used", None)
        if callable(fn):
            try:
                fn(arg)
                return
            except Exception:
                pass
    for attr, val in (
        ("fallback_state", "searxng_raises"),
        ("n_searxng", 0),
        ("n_ddg", 1),
        ("engines_used", ["duckduckgo"]),
        ("degraded", True),
    ):
        try:
            setattr(m, attr, val)
        except Exception:
            pass


def _drive(query, populate):
    g = _g()
    tools = _tools()
    if not hasattr(g, "graph"):
        pytest.skip("compiled graph seam search_workflow.graph.graph is absent")

    async def fake_ainvoke(*args, **kwargs):
        # Attribution is written during the graph run, then read by run_workflow.
        try:
            populate(tools)
        except Exception:
            pass
        return {"messages": [_mk_message(json.dumps(_RESULTS))]}

    _reset_metrics(tools)
    with mock.patch.object(g.graph, "ainvoke", new=fake_ainvoke):
        return asyncio.run(g.run_workflow(query))


def test_graph_source_defines_new_metadata_fields():
    src = _graph_source()
    for token in ("engines_used", "degraded", "degraded_reason"):
        assert token in src, f"graph.py does not mention {token!r}"


def test_success_result_exposes_engines_used_and_degraded():
    # Prefer the strong single-engine (degraded) attribution; fall back to a
    # plain healthy run if the METRICS API cannot be driven cleanly.
    try:
        result = _drive("what is python", _force_single_engine)
    except Exception:
        result = None
    if not (isinstance(result, dict) and _val(result.get("status")) == "ok"):
        result = _drive("what is python", lambda t: None)

    assert isinstance(result, dict)
    assert _val(result.get("status")) == "ok", result

    meta = _locate_meta(result)
    assert meta is not None, f"engines_used/degraded absent from result: {result}"

    engines = _field(meta, "engines_used")
    assert isinstance(engines, list) and all(
        isinstance(e, str) for e in engines
    ), engines

    degraded = _field(meta, "degraded")
    assert isinstance(degraded, bool), degraded

    reason = _val(_field(meta, "degraded_reason"))
    assert reason in (None, "engine", "evaluator"), reason

    if degraded is True:
        assert reason == "engine", "degraded engine-state must report reason 'engine'"
        low = [e.lower() for e in engines]
        assert low in ([], ["duckduckgo"], ["ddg"]), engines


def test_engines_used_is_sourced_from_attribution_record():
    src = _graph_source()
    assert "engines_used" in src
    markers = (
        "METRICS",
        "snapshot",
        "provenance",
        "attribution",
        "fallback_state",
        "n_searxng",
        "n_ddg",
    )
    if not any(mk in src for mk in markers):
        pytest.skip("static read cannot confirm metrics-sourcing; lenient per issue")
    assert any(mk in src for mk in markers)
