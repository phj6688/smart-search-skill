"""Held-out behavioural probe for Linear issue HLB-660.

HLB-660: "Add retry with backoff to both LLM call sites"
Anchor: src/search_workflow/graph.py:agent

Policy under test (2-3 attempts total on rate-limit/timeout, exponential backoff WITH
jitter, honor Retry-After, raise immediately on auth/validation, one shared helper routed
through BOTH the agent node and the evaluator node).

Acceptance criteria exercised here, black-box, through the public run_workflow interface,
patching ONLY the sanctioned model seam (search_workflow.graph.load_chat_model) and the
engine seams (search_workflow.tools.SearXNGClient.search, _ddg_search). The retry helper,
agent, evaluator and run_workflow internals are NEVER patched.

  R2 (retry only 429; 2-3 attempts total, backoff+jitter between attempts):
    - A model stub raising an OpenAI 429 (RateLimitError) on EVERY attempt causes exactly
      2 or 3 total ainvoke attempts at the call site, then the query does NOT succeed, and
      at least one positive backoff wait occurred between attempts.
      -> test_rate_limit_every_attempt_exhausts_two_to_three
    - A model stub raising an OpenAI timeout on every attempt is likewise retried 2-3 times.
      -> test_timeout_every_attempt_is_retried
  R4 (raise IMMEDIATELY, no retry, on authentication errors):
    - A model stub raising an OpenAI AuthenticationError is invoked EXACTLY once (no retry).
      -> test_auth_error_no_retry_single_attempt
  R3 (honor the Retry-After header when present):
    - A model stub whose 429 carries "Retry-After: 1" makes the helper wait >= 1s (recorded
      via a patched sleep) before the retry.
      -> test_retry_after_header_honored
  All / end-to-end (a 429 once then success = exactly one retry and a completed query):
    - A model stub raising a 429 once then succeeding results in exactly one retry (2
      attempts at the agent call site, exactly one 429 raised) and a completed run_workflow.
      -> test_rate_limit_once_then_success_end_to_end
  R1 (a single retry helper lives in the graph module) is checked with a tolerant static
      smoke assertion; the behavioural tests carry the weight.
      -> test_retry_logic_present_in_graph_module_source
"""

import asyncio
import contextlib
import datetime
import enum
import importlib
import inspect
import os
import sys
import types
import typing
from unittest import mock

import pytest


# --------------------------------------------------------------------------------------
# Bootstrap: locate the repo `src/` that contains search_workflow and put it on sys.path.
# Runs from the issue worktree root, so walk up from cwd (and this file) to find it.
# --------------------------------------------------------------------------------------
def _bootstrap_pkg_dir():
    starts = []
    try:
        starts.append(os.getcwd())
    except Exception:
        pass
    try:
        starts.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    for start in starts:
        d = os.path.abspath(start)
        while True:
            cand = os.path.join(d, "src")
            if os.path.isdir(os.path.join(cand, "search_workflow")):
                if cand not in sys.path:
                    sys.path.insert(0, cand)
                return os.path.join(cand, "search_workflow")
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return None


_PKG_DIR = _bootstrap_pkg_dir()

# Import the system under test by its public interface.
AIMessage = pytest.importorskip("langchain_core.messages").AIMessage
sw_graph = pytest.importorskip("search_workflow.graph")
sw_tools = pytest.importorskip("search_workflow.tools")


# --------------------------------------------------------------------------------------
# OpenAI error factories (the provider-native path the classifier most likely keys on).
# Guarded per-test with importorskip("openai"); openai is a declared dependency.
# --------------------------------------------------------------------------------------
def _rate_limit_error(retry_after="1"):
    import httpx
    from openai import RateLimitError

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(429, headers={"retry-after": str(retry_after)}, request=req)
    return RateLimitError("rate limit exceeded", response=resp, body=None)


def _timeout_error():
    import httpx
    from openai import APITimeoutError

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return APITimeoutError(request=req)


def _auth_error():
    import httpx
    from openai import AuthenticationError

    req = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    resp = httpx.Response(401, headers={}, request=req)
    return AuthenticationError("invalid api key", response=resp, body=None)


# --------------------------------------------------------------------------------------
# Stub model satisfying BOTH call sites: bind_tools -> self, with_structured_output ->
# proxy with async ainvoke, and an async ainvoke used by the agent node. A single
# load_chat_model patch returns this for both the agent and evaluator nodes.
# --------------------------------------------------------------------------------------
class _StructuredProxy:
    def __init__(self, model):
        self._model = model

    async def ainvoke(self, *args, **kwargs):
        self._model.struct_calls += 1
        beh = self._model.struct_behavior
        if beh is not None:
            return beh(self._model)
        return _build_structured_result(self._model.captured_schema)


class StubModel:
    def __init__(self, agent_behavior, struct_behavior=None):
        self.agent_behavior = agent_behavior
        self.struct_behavior = struct_behavior
        self.agent_calls = 0
        self.struct_calls = 0
        self.raised_429 = 0
        self.raised_timeout = 0
        self.raised_auth = 0
        self.captured_schema = None

    def bind_tools(self, *args, **kwargs):
        return self

    def with_structured_output(self, schema=None, *args, **kwargs):
        self.captured_schema = schema
        return _StructuredProxy(self)

    async def ainvoke(self, *args, **kwargs):
        self.agent_calls += 1
        return self.agent_behavior(self)


def _always_rate_limit(stub):
    stub.raised_429 += 1
    raise _rate_limit_error()


def _always_timeout(stub):
    stub.raised_timeout += 1
    raise _timeout_error()


def _always_auth(stub):
    stub.raised_auth += 1
    raise _auth_error()


def _rate_then_success(stub):
    # First attempt raises a 429 (with Retry-After: 1); the retry attempt succeeds with an
    # AIMessage carrying no tool calls. Used only by tests that assert retry MECHANICS
    # (attempt count, honored Retry-After) and never inspect run_workflow's status.
    if stub.agent_calls == 1:
        stub.raised_429 += 1
        raise _rate_limit_error(retry_after="1")
    return AIMessage(content="final answer")


# A bare-text agent answer routes straight to __end__, so run_workflow then json.loads
# that text and returns a json_parse_error, never "ok". To exercise the end-to-end
# retry-then-complete path we must drive a real completion: the agent emits a `search`
# tool call -> the tools node fetches -> the evaluator returns a JSON selection.
_SEARCH_TOOL_CALL = {
    "name": "search",
    "args": {"query": "test query about retries", "region": "us-en", "timelimit": None},
    "id": "call_retry_1",
    "type": "tool_call",
}

_ENGINE_RESULTS = [
    {"title": "Alpha", "link": "https://example.com/alpha", "url": "https://example.com/alpha",
     "href": "https://example.com/alpha", "snippet": "alpha", "content": "alpha"},
    {"title": "Beta", "link": "https://example.com/beta", "url": "https://example.com/beta",
     "href": "https://example.com/beta", "snippet": "beta", "content": "beta"},
]


def _rate_then_tool_call(stub):
    # First attempt raises a 429 once; the retry returns an agent message carrying a
    # `search` tool call so the graph runs agent -> tools -> evaluator -> completion.
    if stub.agent_calls == 1:
        stub.raised_429 += 1
        raise _rate_limit_error(retry_after="1")
    return AIMessage(content="", tool_calls=[dict(_SEARCH_TOOL_CALL)])


def _selection_result():
    # A valid evaluator selection: pick the first fetched result by index. Prefer the
    # real SelectionResponse schema if importable; fall back to a duck type.
    for modname in ("search_workflow.graph", "search_workflow.utils"):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        sel = getattr(mod, "SelectionResponse", None)
        if sel is not None:
            try:
                return sel(selected=[0])
            except Exception:
                pass
    return types.SimpleNamespace(selected=[0])


def _selection_behavior(model):
    return _selection_result()


# --------------------------------------------------------------------------------------
# Dynamic structured-output builder: constructs a valid instance of whatever schema the
# evaluator passes to with_structured_output, without reading the evaluator source.
# --------------------------------------------------------------------------------------
def _build_structured_result(schema):
    try:
        return _build_pydantic(schema)
    except Exception:
        return mock.MagicMock()


def _build_pydantic(schema):
    if schema is None:
        return mock.MagicMock()
    mf = getattr(schema, "model_fields", None)  # pydantic v2
    if isinstance(mf, dict):
        kwargs = {
            fname: _value_for_field(fname, getattr(finfo, "annotation", None))
            for fname, finfo in mf.items()
        }
        return schema(**kwargs)
    of = getattr(schema, "__fields__", None)  # pydantic v1
    if isinstance(of, dict):
        kwargs = {}
        for fname, finfo in of.items():
            ann = getattr(finfo, "outer_type_", None) or getattr(finfo, "type_", None)
            kwargs[fname] = _value_for_field(fname, ann)
        return schema(**kwargs)
    if callable(schema):
        return schema()
    return mock.MagicMock()


def _value_for_field(name, ann):
    lname = (name or "").lower()
    if any(tok in lname for tok in ("url", "link", "href", "uri")):
        return "https://example.com"
    return _value_for_annotation(ann)


def _value_for_annotation(ann):
    if ann is None:
        return "test"
    tname = str(getattr(ann, "__name__", "")).lower()
    if isinstance(ann, type):
        if issubclass(ann, bool):
            return True
        if issubclass(ann, int):
            return 1
        if issubclass(ann, float):
            return 1.0
        if issubclass(ann, str):
            return "https://example.com" if ("url" in tname or "uri" in tname) else "test"
        if issubclass(ann, datetime.datetime):
            return datetime.datetime.now()
        if issubclass(ann, datetime.date):
            return datetime.date.today()
        if issubclass(ann, enum.Enum):
            members = list(ann)
            return members[0] if members else "test"
        if issubclass(ann, dict):
            return {}
        if issubclass(ann, (list, tuple, set, frozenset)):
            return []
        if hasattr(ann, "model_fields") or hasattr(ann, "__fields__"):
            return _build_pydantic(ann)
    origin = typing.get_origin(ann)
    args = typing.get_args(ann)
    if origin is typing.Literal:
        return args[0] if args else "test"
    if origin in (list, set, frozenset):
        inner = args[0] if args else str
        return [_value_for_annotation(inner)]
    if origin is tuple:
        if args and Ellipsis not in args:
            return tuple(_value_for_annotation(a) for a in args)
        inner = args[0] if args else str
        return (_value_for_annotation(inner),)
    if origin is dict:
        return {}
    union_type = getattr(types, "UnionType", None)
    if origin is typing.Union or (union_type is not None and origin is union_type):
        non_none = [a for a in args if a is not type(None)]
        return _value_for_annotation(non_none[0]) if non_none else None
    if hasattr(ann, "model_fields") or hasattr(ann, "__fields__"):
        return _build_pydantic(ann)
    return "test"


# --------------------------------------------------------------------------------------
# Sleep recorder: records requested delays without ever waiting, so backoff assertions
# are deterministic and fast. Covers `await asyncio.sleep(...)` and `time.sleep(...)`.
# --------------------------------------------------------------------------------------
class _SleepRecorder:
    def __init__(self):
        self.delays = []

    async def async_sleep(self, delay=0, *args, **kwargs):
        try:
            self.delays.append(float(delay))
        except Exception:
            self.delays.append(0.0)
        return None

    def sync_sleep(self, delay=0, *args, **kwargs):
        try:
            self.delays.append(float(delay))
        except Exception:
            self.delays.append(0.0)
        return None


def _no_network(*args, **kwargs):
    return []


def _engine_stub(original, value):
    # Match the original's sync/async nature so the tool node can call it either way.
    if inspect.iscoroutinefunction(original):
        async def _repl(*args, **kwargs):
            return [dict(r) for r in value]
        return _repl

    def _repl(*args, **kwargs):
        return [dict(r) for r in value]
    return _repl


@contextlib.contextmanager
def _patched(stub, recorder, engine_results=None):
    if engine_results is None:
        searx_repl = _no_network
        ddg_repl = _no_network
    else:
        searx_repl = _engine_stub(sw_tools.SearXNGClient.search, engine_results)
        ddg_repl = _engine_stub(sw_tools._ddg_search, engine_results)
    patches = [
        mock.patch("search_workflow.graph.load_chat_model", return_value=stub),
        mock.patch("asyncio.sleep", recorder.async_sleep),
        mock.patch("time.sleep", recorder.sync_sleep),
        mock.patch.object(sw_tools.SearXNGClient, "search", searx_repl),
        mock.patch("search_workflow.tools._ddg_search", ddg_repl),
    ]
    # If the retry helper imported `sleep` into the graph namespace, patch that binding too.
    graph_sleep = getattr(sw_graph, "sleep", None)
    if graph_sleep is not None:
        repl = recorder.async_sleep if asyncio.iscoroutinefunction(graph_sleep) else recorder.sync_sleep
        patches.append(mock.patch.object(sw_graph, "sleep", repl))

    started = []
    try:
        for p in patches:
            p.start()
            started.append(p)
        yield
    finally:
        for p in reversed(started):
            p.stop()


# --------------------------------------------------------------------------------------
# Drive the public run_workflow. Tolerant to whether it re-raises the exhausted error or
# converts it to a typed error status (sibling HLB-654 work), and to sync vs async.
# --------------------------------------------------------------------------------------
def _invoke_run_workflow(query="test query about retries"):
    run_workflow = sw_graph.run_workflow
    sig = inspect.signature(run_workflow)
    required_pos = [
        p
        for p in sig.parameters.values()
        if p.default is p.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    res = run_workflow(query) if required_pos else run_workflow()
    if inspect.isawaitable(res):
        return asyncio.run(res)
    return res


def _drive(query="test query about retries"):
    result = None
    raised = None
    try:
        result = _invoke_run_workflow(query)
    except BaseException as exc:  # noqa: BLE001 - we classify below
        raised = exc
    return result, raised


def _extract_status(result):
    if isinstance(result, dict):
        return result.get("status")
    return getattr(result, "status", None)


def _nonempty_results(result):
    res = result.get("results") if isinstance(result, dict) else getattr(result, "results", None)
    return bool(res)


def _run_ok(result, raised):
    if raised is not None:
        return False
    status = _extract_status(result)
    if status is not None:
        return str(status).lower() == "ok"
    if isinstance(result, dict):
        if result.get("error"):
            return False
        return len(result) > 0
    return result is not None


# ======================================================================================
# Behavioural tests
# ======================================================================================
def test_rate_limit_every_attempt_exhausts_two_to_three():
    """R2: a 429 on every attempt is retried 2-3 times, then the query does not succeed."""
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    stub = StubModel(_always_rate_limit, struct_behavior=_always_rate_limit)
    rec = _SleepRecorder()
    with _patched(stub, rec):
        result, raised = _drive()
    assert 2 <= stub.agent_calls <= 3, (
        f"a persistent 429 must exhaust 2-3 total attempts; got {stub.agent_calls}"
    )
    assert not _run_ok(result, raised), (
        "an always-429 model must not yield a successful run"
    )
    assert any(d > 0 for d in rec.delays), (
        f"expected at least one positive backoff wait between attempts; got {rec.delays}"
    )


def test_timeout_every_attempt_is_retried():
    """R2: a timeout on every attempt is retried 2-3 times, then the query does not succeed."""
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    stub = StubModel(_always_timeout, struct_behavior=_always_timeout)
    rec = _SleepRecorder()
    with _patched(stub, rec):
        result, raised = _drive()
    assert 2 <= stub.agent_calls <= 3, (
        f"a persistent timeout must exhaust 2-3 total attempts; got {stub.agent_calls}"
    )
    assert not _run_ok(result, raised), (
        "an always-timeout model must not yield a successful run"
    )


def test_auth_error_no_retry_single_attempt():
    """R4: an authentication error is raised immediately, invoked exactly once (no retry)."""
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    stub = StubModel(_always_auth, struct_behavior=_always_auth)
    rec = _SleepRecorder()
    with _patched(stub, rec):
        result, raised = _drive()
    assert stub.agent_calls == 1, (
        f"an authentication error must NOT be retried; got {stub.agent_calls} attempts"
    )
    assert not _run_ok(result, raised), (
        "an authentication error must not yield a successful run"
    )


def test_retry_after_header_honored():
    """R3: a 429 carrying Retry-After: 1 makes the helper wait >= 1s before the retry."""
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    stub = StubModel(_rate_then_success)
    rec = _SleepRecorder()
    with _patched(stub, rec):
        _drive()
    assert stub.raised_429 == 1, "stub should raise exactly one 429 before succeeding"
    assert stub.agent_calls == 2, (
        f"expected exactly one retry (2 attempts) at the agent call site; got {stub.agent_calls}"
    )
    assert any(d >= 1.0 for d in rec.delays), (
        f"helper must honor Retry-After: 1 (wait >= 1s before retry); recorded waits: {rec.delays}"
    )


def test_rate_limit_once_then_success_end_to_end():
    """All: a 429 once then success = exactly one retry and a completed run_workflow.

    The retry succeeds with a `search` tool call (not a bare-text answer), so the graph
    runs agent -> tools -> evaluator and run_workflow reaches an "ok" completion.
    """
    pytest.importorskip("openai")
    pytest.importorskip("httpx")
    stub = StubModel(_rate_then_tool_call, struct_behavior=_selection_behavior)
    rec = _SleepRecorder()
    with _patched(stub, rec, engine_results=_ENGINE_RESULTS):
        result, raised = _drive()
    assert stub.raised_429 == 1, "exactly one 429 should have been raised"
    assert stub.agent_calls == 2, (
        f"expected exactly one retry (fail + success) at the agent call site; got {stub.agent_calls}"
    )
    status = _extract_status(result)
    assert status is not None and str(status).lower() == "ok", (
        f"one 429 then success must complete the query with status ok; "
        f"raised={raised!r} result={result!r}"
    )
    assert _nonempty_results(result), (
        f"a completed run must carry a non-empty results list; result={result!r}"
    )


def test_retry_logic_present_in_graph_module_source():
    """R1 (tolerant static smoke): the graph module carries retry logic per HLB-660."""
    if _PKG_DIR is None:
        pytest.skip("search_workflow package directory could not be located")
    graph_path = os.path.join(_PKG_DIR, "graph.py")
    if not os.path.isfile(graph_path):
        pytest.skip(f"graph.py not found at {graph_path}")
    with open(graph_path, encoding="utf-8") as fh:
        src = fh.read().lower()
    assert "retry" in src, (
        "graph.py should reference retry logic (the shared retry helper) per HLB-660"
    )
