"""Held-out behavioural probe for issue HLB-661.

Title: Add no-LLM deterministic mode with snippet minimization.
Anchor: src/search_workflow/graph.py:run_workflow

Acceptance criteria checked here (black-box, from the issue text only):
  R1  run_workflow(query, use_llm=False) over stubbed engines returns a
      non-empty typed result list and performs ZERO LLM calls (llm_calls == 0).
  R2  With use_llm=False the `degraded` flag is true exactly when a single
      engine served the results (false when two engines served).
  R3  On the LLM path the payload sent to the selector caps each snippet at
      ~200 chars and contains NO "http" substring (URLs are sent as indices,
      not raw URLs).
  R4  Returned URLs are byte-identical to the fetched ones on BOTH the
      use_llm=False and use_llm=True paths (no rewriting / truncation).
  R6  .github/workflows/live-canary.yml exercises run_workflow(..., use_llm=...)
      (a default-config, DDG-only, no-LLM leg).

Interfaces exercised (named by the issue):
  - search_workflow.graph.run_workflow            (public entry point)
Sanctioned patch seams ONLY (never run_workflow/agent/evaluator internals):
  - search_workflow.graph.load_chat_model         (model seam)
  - search_workflow.tools.SearXNGClient.search    (engine seam)
  - search_workflow.tools._ddg_search             (engine seam)

Offline / deterministic: no network is touched; the model and both search
engines are replaced with in-process stubs.
"""

import contextlib
import importlib
import inspect
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


# --------------------------------------------------------------------------- #
# Path bootstrap: this probe lives OUTSIDE the repo. Walk up from the current
# working directory (the issue worktree root) to the dir holding
# src/search_workflow, and prepend that src/ to sys.path.
# --------------------------------------------------------------------------- #
def _repo_root():
    d = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(d, "src", "search_workflow")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent


_ROOT = _repo_root()
if _ROOT is not None:
    _SRC = os.path.join(_ROOT, "src")
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)


QUERY = "climate policy news"

# Exact tool-call shape supplied by the issue so the LLM path drives
# agent -> tools -> evaluator -> completion.
TOOL_CALL = {
    "name": "search",
    "args": {"query": "q", "region": "us-en", "timelimit": None},
    "id": "c1",
    "type": "tool_call",
}


# --------------------------------------------------------------------------- #
# Imports of the system under test (lazy, so collection errors are readable).
# --------------------------------------------------------------------------- #
def _import_sut():
    graph = importlib.import_module("search_workflow.graph")
    tools = importlib.import_module("search_workflow.tools")
    return graph, tools


def _get_ai_message_cls():
    try:
        from langchain_core.messages import AIMessage

        return AIMessage
    except Exception:
        try:
            from langchain.schema import AIMessage  # type: ignore

            return AIMessage
        except Exception:
            return None


def _make_selection():
    """Selection object (index 0) the evaluator would return.

    Defensively import SelectionResponse; fall back to a duck-typed stand-in.
    """
    for modname in ("search_workflow.graph", "search_workflow.utils"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        cls = getattr(mod, "SelectionResponse", None)
        if cls is not None:
            try:
                return cls(selected=[0])
            except Exception:
                pass
    return SimpleNamespace(selected=[0])


# --------------------------------------------------------------------------- #
# Model stubs (installed only via the sanctioned graph.load_chat_model seam).
# --------------------------------------------------------------------------- #
class _Tripwire:
    """Records every interaction; RAISES on any actual LLM invocation.

    bind_tools / with_structured_output are setup and merely recorded; ainvoke
    and invoke are the real 'LLM calls' and raise, so R1 can prove zero calls.
    """

    def __init__(self, trips):
        self._trips = trips

    def bind_tools(self, *a, **k):
        self._trips.append("bind_tools")
        return self

    def with_structured_output(self, *a, **k):
        self._trips.append("with_structured_output")
        return self

    async def ainvoke(self, *a, **k):
        self._trips.append("ainvoke")
        raise AssertionError("LLM must not be invoked when use_llm=False")

    def invoke(self, *a, **k):
        self._trips.append("invoke")
        raise AssertionError("LLM must not be invoked when use_llm=False")


class _StructuredProxy:
    def __init__(self, selection, captured, parent):
        self._selection = selection
        self._captured = captured
        self._parent = parent

    async def ainvoke(self, payload=None, *a, **k):
        self._parent.struct_calls += 1
        self._captured.append(payload)
        return self._selection

    def invoke(self, payload=None, *a, **k):
        self._parent.struct_calls += 1
        self._captured.append(payload)
        return self._selection


class _ToolCallModel:
    """Drives the real LLM path: first agent turn emits a `search` tool call,
    later turns emit a final answer, and the evaluator selection is captured.
    """

    def __init__(self, selection, captured, ai_cls):
        self._selection = selection
        self._captured = captured
        self._ai_cls = ai_cls
        self.agent_calls = 0
        self.struct_calls = 0

    def bind_tools(self, *a, **k):
        return self

    def with_structured_output(self, *a, **k):
        return _StructuredProxy(self._selection, self._captured, self)

    async def ainvoke(self, messages=None, *a, **k):
        return self._agent_response()

    def invoke(self, messages=None, *a, **k):
        return self._agent_response()

    def _agent_response(self):
        self.agent_calls += 1
        if self.agent_calls == 1:
            return self._ai_cls(content="", tool_calls=[dict(TOOL_CALL)])
        return self._ai_cls(content="done")


# --------------------------------------------------------------------------- #
# Engine stubs (installed only via the sanctioned tools seams).
# --------------------------------------------------------------------------- #
def _make_results(urls, snippet="short summary"):
    out = []
    for i, u in enumerate(urls):
        out.append(
            {
                "title": f"Title {i}",
                "link": u,
                "url": u,
                "href": u,
                "snippet": snippet,
                "content": snippet,
                "body": snippet,
                "description": snippet,
            }
        )
    return out


def _searx_stub(results):
    async def _fn(self, *a, **k):  # instance method: takes self
        return [dict(r) for r in results]

    return _fn


def _ddg_stub(results):
    async def _fn(*a, **k):  # module-level function
        return [dict(r) for r in results]

    return _fn


@contextlib.contextmanager
def _patch_engines(searx_results, ddg_results):
    with mock.patch(
        "search_workflow.tools.SearXNGClient.search", new=_searx_stub(searx_results)
    ), mock.patch(
        "search_workflow.tools._ddg_search", new=_ddg_stub(ddg_results)
    ):
        yield


@contextlib.contextmanager
def _patch_model(returned_model):
    with mock.patch(
        "search_workflow.graph.load_chat_model",
        new=lambda *a, **k: returned_model,
    ):
        yield


# --------------------------------------------------------------------------- #
# run_workflow driver + result helpers.
# --------------------------------------------------------------------------- #
def _has_use_llm(graph):
    return "use_llm" in inspect.signature(graph.run_workflow).parameters


def _run(graph, query, use_llm=None):
    import asyncio

    kwargs = {}
    if use_llm is not None and _has_use_llm(graph):
        kwargs["use_llm"] = use_llm
    result = graph.run_workflow(query, **kwargs)
    if inspect.iscoroutine(result):
        result = asyncio.run(result)
    return result


def _ok_items(result):
    assert isinstance(result, dict), f"expected dict result, got {type(result)!r}"
    assert result.get("status") == "ok", f"expected status 'ok', got {result!r}"
    items = result.get("results")
    assert isinstance(items, list) and items, f"expected non-empty results, got {result!r}"
    return items


def _item_url(item):
    d = None
    if isinstance(item, dict):
        d = item
    elif hasattr(item, "model_dump"):
        try:
            d = item.model_dump()
        except Exception:
            d = None
    if d is None and hasattr(item, "__dict__"):
        d = vars(item)
    if isinstance(d, dict):
        for k in ("link", "url", "href", "source_url", "source"):
            v = d.get(k)
            if isinstance(v, str) and v:
                return v
        for v in d.values():
            if isinstance(v, str) and "://" in v:
                return v
    for k in ("link", "url", "href"):
        v = getattr(item, k, None)
        if isinstance(v, str) and v:
            return v
    return None


def _read_llm_calls(tools):
    """Best-effort read of the S03 llm_calls counter; None when unavailable."""
    metrics = getattr(tools, "METRICS", None)
    if metrics is not None:
        for name in ("llm_calls", "S03", "s03", "llm"):
            v = getattr(metrics, name, None)
            if isinstance(v, int):
                return v
        get = getattr(metrics, "get", None)
        if callable(get):
            try:
                v = get("llm_calls")
                if isinstance(v, int):
                    return v
            except Exception:
                pass
    v = getattr(tools, "llm_calls", None)
    if isinstance(v, int):
        return v
    return None


# --------------------------------------------------------------------------- #
# R1: no-LLM mode returns non-empty results and never calls the LLM.
# --------------------------------------------------------------------------- #
def test_r1_no_llm_mode_returns_results_and_zero_llm_calls():
    graph, tools = _import_sut()
    assert _has_use_llm(graph), "run_workflow must expose a `use_llm` parameter"

    trips = []
    tripwire = _Tripwire(trips)
    searx = _make_results(["https://searx.test/a", "https://searx.test/b"])
    ddg = _make_results(["https://ddg.test/a", "https://ddg.test/b"])

    with _patch_model(tripwire), _patch_engines(searx, ddg):
        before = _read_llm_calls(tools)
        result = _run(graph, QUERY, use_llm=False)
        after = _read_llm_calls(tools)

    # The LLM was never actually invoked (agent/evaluator nodes skipped).
    assert "ainvoke" not in trips, "no-LLM mode invoked the model (async)"
    assert "invoke" not in trips, "no-LLM mode invoked the model (sync)"

    _ok_items(result)  # status ok + non-empty typed result list

    # Tolerant llm_calls counter check: assert no increment when the field exists.
    if before is not None and after is not None:
        assert after == before, "no-LLM mode must not increment llm_calls (S03)"


def test_llm_path_still_completes():
    """Contrast probe: the default (use_llm=True) path still returns results."""
    graph, _ = _import_sut()
    ai_cls = _get_ai_message_cls()
    if ai_cls is None:
        pytest.skip("langchain_core AIMessage unavailable; cannot drive LLM path")

    model = _ToolCallModel(_make_selection(), [], ai_cls)
    searx = _make_results(["https://searx.test/a", "https://searx.test/b"])
    ddg = _make_results(["https://ddg.test/a", "https://ddg.test/b"])

    with _patch_model(model), _patch_engines(searx, ddg):
        result = _run(graph, QUERY, use_llm=True)

    _ok_items(result)


# --------------------------------------------------------------------------- #
# R2: degraded is true exactly when a single engine served the results.
# --------------------------------------------------------------------------- #
def test_r2_degraded_true_when_single_engine_serves():
    graph, _ = _import_sut()
    # Only DDG returns results; SearXNG serves nothing => one engine served.
    with _patch_engines([], _make_results(["https://ddg.test/x", "https://ddg.test/y"])):
        result = _run(graph, QUERY, use_llm=False)

    _ok_items(result)
    assert bool(result.get("degraded")) is True, (
        "degraded must be True when exactly one engine served the results: "
        f"{result!r}"
    )


def test_r2_degraded_false_when_two_engines_serve():
    graph, _ = _import_sut()
    # Distinct URLs per engine so both clearly contribute after dedup.
    searx = _make_results(["https://searx.test/1", "https://searx.test/2"])
    ddg = _make_results(["https://ddg.test/1", "https://ddg.test/2"])
    with _patch_engines(searx, ddg):
        result = _run(graph, QUERY, use_llm=False)

    _ok_items(result)
    assert "engines_used" in result, "HLB-657 engines_used metadata missing"
    eu = result["engines_used"]
    assert isinstance(eu, list), f"engines_used must be a list, got {eu!r}"
    assert "degraded" in result, "HLB-657 degraded metadata missing"

    if len(eu) >= 2:
        # Two engines served => not degraded.
        assert bool(result["degraded"]) is False, (
            f"two engines served ({eu}) must not be degraded: {result!r}"
        )
    else:
        # Environment consulted a single engine: invariant still holds
        # (degraded true exactly when one engine served).
        assert bool(result["degraded"]) is True, (
            f"single engine served ({eu}) must be degraded: {result!r}"
        )


# --------------------------------------------------------------------------- #
# R3: LLM payload caps snippets at ~200 chars and carries no raw URLs.
# --------------------------------------------------------------------------- #
def test_r3_llm_payload_caps_snippets_and_has_no_raw_urls():
    graph, _ = _import_sut()
    ai_cls = _get_ai_message_cls()
    if ai_cls is None:
        pytest.skip("langchain_core AIMessage unavailable; cannot drive LLM path")

    # Long snippet (no http) + a distinctive raw URL in link/url/href.
    long_snip = "X" * 500
    leak_urls = [
        "http://leak.example/secret-0",
        "http://leak.example/secret-1",
    ]
    results = _make_results(leak_urls, snippet=long_snip)

    captured = []
    model = _ToolCallModel(_make_selection(), captured, ai_cls)

    with _patch_model(model), _patch_engines(results, results):
        _run(graph, QUERY, use_llm=True)

    if not captured:
        pytest.skip("could not capture the selector payload (no structured call observed)")

    combined = "\n".join(str(p) for p in captured)

    # Guard against a false pass: require evidence the payload actually carried
    # the candidate data before asserting on its shape.
    informative = ("Title" in combined) or ("X" * 20 in combined)
    if not informative:
        pytest.skip("captured payload was not informative enough to assert on")

    # R3a: no "http" substring — URLs are sent as INDICES, not raw URLs.
    assert "http" not in combined, (
        "LLM payload must not contain raw URLs (send indices instead)"
    )
    for u in leak_urls:
        assert u not in combined, f"raw fetched URL leaked into LLM payload: {u}"

    # R3b: snippets capped at ~200 chars (no long uncapped snippet run).
    assert long_snip not in combined, "uncapped 500-char snippet present in payload"
    assert ("X" * 300) not in combined, "snippet exceeds the ~200 char cap in payload"


# --------------------------------------------------------------------------- #
# R4: returned URLs byte-identical to fetched URLs on BOTH paths.
# --------------------------------------------------------------------------- #
def test_r4_returned_urls_byte_identical_both_paths():
    graph, _ = _import_sut()
    ai_cls = _get_ai_message_cls()
    if ai_cls is None:
        pytest.skip("langchain_core AIMessage unavailable; cannot drive LLM path")

    searx_urls = ["https://searx.test/one", "https://searx.test/two"]
    ddg_urls = ["https://ddg.test/one", "https://ddg.test/two"]
    fetched = set(searx_urls) | set(ddg_urls)
    searx = _make_results(searx_urls)
    ddg = _make_results(ddg_urls)

    # use_llm=False path.
    with _patch_model(_Tripwire([])), _patch_engines(searx, ddg):
        no_llm = _run(graph, QUERY, use_llm=False)
    for item in _ok_items(no_llm):
        u = _item_url(item)
        assert u is not None, f"result item without a URL: {item!r}"
        assert u in fetched, f"no-LLM path rewrote a URL: {u!r} not in {fetched}"

    # use_llm=True path.
    model = _ToolCallModel(_make_selection(), [], ai_cls)
    with _patch_model(model), _patch_engines(searx, ddg):
        llm = _run(graph, QUERY, use_llm=True)
    for item in _ok_items(llm):
        u = _item_url(item)
        assert u is not None, f"result item without a URL: {item!r}"
        assert u in fetched, f"LLM path rewrote a URL: {u!r} not in {fetched}"


# --------------------------------------------------------------------------- #
# R6: live-canary workflow exercises the no-LLM leg.
# --------------------------------------------------------------------------- #
def test_r6_live_canary_has_use_llm_leg():
    root = _ROOT if _ROOT is not None else _repo_root()
    if root is None:
        pytest.skip("could not locate repo root from cwd")
    wf_dir = os.path.join(root, ".github", "workflows")
    if not os.path.isdir(wf_dir):
        pytest.skip(".github/workflows not present in this checkout")
    path = os.path.join(wf_dir, "live-canary.yml")
    assert os.path.exists(path), "live-canary.yml must exist (R6 default-config leg)"
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert "use_llm" in text, (
        "live-canary.yml must exercise run_workflow(query, use_llm=...) (R6)"
    )
