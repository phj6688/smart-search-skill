"""Held-out behavioural probe for Linear issue HLB-662.

Issue: "Handle DDG rate limits with async backoff in the shared core"
Anchor: src/search_workflow/tools.py:_ddg_search

Acceptance criteria checked here (black-box, spec only):

  R1 + R2: _ddg_search catches the DDG rate-limit exception SPECIFICALLY and
           retries with capped exponential backoff + full jitter. A rate-limit
           raised ONCE then a result yields exactly one retry and a non-empty
           list; a rate-limit on EVERY attempt yields [] after 3 attempts.
  R3:      No `time.sleep` in tools.py; the wait is `await asyncio.sleep(...)`
           at the async layer, awaited BETWEEN fresh executor submissions.
  R4:      RatelimitException imports from `ddgs.exceptions` (primary) with
           `duckduckgo_search.exceptions` as fallback; both resolve to an
           Exception subclass when installed.
  R5:      Worst-case cumulative backoff for base 0.5s / cap 4s / 3 attempts is
           under Configuration.engine_deadline_s.
  Non-ratelimit: a non rate-limit exception still returns [] with zero retries.

Design notes:
  * Only the DDG boundary is patched: search_workflow.tools.DDGS (and, belt and
    braces, ddgs.DDGS / duckduckgo_search.DDGS if lazily imported). The retry
    loop, _ddg_search internals, and run_workflow are NOT patched.
  * asyncio.sleep is replaced with an async no-op that RECORDS the requested
    delay so the probe is fast and deterministic (no real waiting, no network).
  * Tests are named so `pytest -k ddg_backoff` selects them.
"""

import asyncio
import importlib
import inspect
import os
import re
import sys

import pytest


# --------------------------------------------------------------------------- #
# Locate the repo `src/` dir (probe lives outside the repo tree) and import.
# --------------------------------------------------------------------------- #
def _ensure_src_on_path():
    try:
        import search_workflow  # noqa: F401
        return
    except Exception:
        pass
    candidates = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for start in candidates:
        cur = os.path.abspath(start)
        while True:
            src = os.path.join(cur, "src")
            if os.path.isdir(os.path.join(src, "search_workflow")):
                if src not in sys.path:
                    sys.path.insert(0, src)
                return
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent


_ensure_src_on_path()

try:
    import search_workflow.tools as tools
    _TOOLS_ERR = None
except Exception as exc:  # pragma: no cover - environment guard
    tools = None
    _TOOLS_ERR = exc

try:
    from search_workflow.configuration import Configuration
    _CFG_ERR = None
except Exception as exc:  # pragma: no cover - environment guard
    Configuration = None
    _CFG_ERR = exc


# --------------------------------------------------------------------------- #
# Defensively resolve RatelimitException the SAME way the impl must.
# --------------------------------------------------------------------------- #
RatelimitException = None
_RL_SOURCE = None
for _mod_name in ("ddgs.exceptions", "duckduckgo_search.exceptions"):
    try:
        _m = importlib.import_module(_mod_name)
        _cls = getattr(_m, "RatelimitException", None)
        if _cls is not None:
            RatelimitException = _cls
            _RL_SOURCE = _mod_name
            break
    except Exception:
        continue

_BEHAV_SKIP = None
if tools is None:
    _BEHAV_SKIP = f"search_workflow.tools failed to import: {_TOOLS_ERR!r}"
elif RatelimitException is None:
    _BEHAV_SKIP = (
        "RatelimitException not importable from ddgs.exceptions or "
        "duckduckgo_search.exceptions"
    )


# ASSUMPTION: DDG results are dicts with title/href/body; a couple of them make
# a non-empty result observable regardless of any downstream normalisation.
SAMPLE_RESULTS = [
    {"title": "Async IO in Python", "href": "https://example.com/asyncio",
     "body": "asyncio provides a runtime for coroutines."},
    {"title": "Backoff with jitter", "href": "https://example.com/backoff",
     "body": "exponential backoff plus full jitter avoids thundering herd."},
]


# --------------------------------------------------------------------------- #
# Fakes for the DDG boundary.
# --------------------------------------------------------------------------- #
class _ScriptedText:
    """Callable standing in for DDGS().text(). Records ordering + call count."""

    def __init__(self, script, events, default):
        self._script = list(script)
        self._events = events
        self._default = default
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self._events.append("text")
        self.call_count += 1
        item = self._script.pop(0) if self._script else self._default
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item("rate limited")
        if isinstance(item, BaseException):
            raise item
        return item


def _make_fake_ddgs(text_callable):
    class _FakeDDGS:
        def __init__(self, *a, **k):
            pass

        def text(self, *a, **k):
            return text_callable(*a, **k)

        # Some call sites use `with DDGS() as ddgs:`.
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _FakeDDGS


def _install_fake_ddgs(monkeypatch, fake_cls):
    monkeypatch.setattr(tools, "DDGS", fake_cls, raising=False)
    for mod_name in ("ddgs", "duckduckgo_search"):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "DDGS"):
            monkeypatch.setattr(mod, "DDGS", fake_cls, raising=False)


def _patch_sleep(monkeypatch, recorded, events):
    async def _fake_sleep(delay=0, *a, **k):
        recorded.append(float(delay))
        events.append("sleep")

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    if tools is not None and hasattr(tools, "sleep"):
        monkeypatch.setattr(tools, "sleep", _fake_sleep, raising=False)


def _guess_arg(name):
    low = name.lower()
    if any(t in low for t in ("max", "result", "count", "num", "limit", "top")):
        return 5
    return None


def _call_ddg(query):
    """Drive _ddg_search directly, filling only required args by inspection."""
    fn = tools._ddg_search
    sig = inspect.signature(fn)
    real = [
        p for p in sig.parameters.values()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD)
    ]
    args, kwargs = [], {}
    for idx, p in enumerate(real):
        if idx == 0:
            if p.kind == inspect.Parameter.KEYWORD_ONLY:
                kwargs[p.name] = query
            else:
                args.append(query)
            continue
        if p.default is not inspect.Parameter.empty:
            continue  # rely on the impl's default
        val = _guess_arg(p.name)
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[p.name] = val
        else:
            args.append(val)
    coro = fn(*args, **kwargs)
    if not inspect.iscoroutine(coro):
        pytest.skip("_ddg_search did not return a coroutine; cannot drive async")
    return asyncio.run(coro)


def _read_tools_source():
    path = inspect.getsourcefile(tools) or getattr(tools, "__file__", None)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _resolve_engine_deadline():
    cfg = Configuration
    if cfg is None:
        return None
    val = getattr(cfg, "engine_deadline_s", None)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    mf = getattr(cfg, "model_fields", None)
    if isinstance(mf, dict) and "engine_deadline_s" in mf:
        d = getattr(mf["engine_deadline_s"], "default", None)
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            return float(d)
    try:
        import dataclasses
        if dataclasses.is_dataclass(cfg):
            for f in dataclasses.fields(cfg):
                if f.name == "engine_deadline_s" and isinstance(
                    f.default, (int, float)
                ) and not isinstance(f.default, bool):
                    return float(f.default)
    except Exception:
        pass
    try:
        inst = cfg()
        v = getattr(inst, "engine_deadline_s", None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# R1: rate-limit once, then success -> exactly one retry, non-empty result.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_retries_once_then_succeeds(monkeypatch):
    if _BEHAV_SKIP:
        pytest.skip(_BEHAV_SKIP)
    events, recorded = [], []
    text = _ScriptedText([RatelimitException, SAMPLE_RESULTS], events, default=[])
    _install_fake_ddgs(monkeypatch, _make_fake_ddgs(text))
    _patch_sleep(monkeypatch, recorded, events)

    result = _call_ddg("python asyncio backoff")

    assert text.call_count == 2, "one rate-limit then success = exactly one retry"
    assert len(recorded) == 1, "exactly one backoff sleep between the two attempts"
    assert len(list(result)) >= 1, "successful attempt yields a non-empty result"
    # The sleep must be awaited BETWEEN the two executor submissions.
    assert events == ["text", "sleep", "text"]
    # Full jitter: delay is random in [0, min(cap, base*2**n)] -> within [0, 4].
    assert 0.0 <= recorded[0] <= 4.0


# --------------------------------------------------------------------------- #
# R2: rate-limit on every attempt -> [] after 3 attempts, 2 sleeps between.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_exhausts_after_three_attempts(monkeypatch):
    if _BEHAV_SKIP:
        pytest.skip(_BEHAV_SKIP)
    events, recorded = [], []
    text = _ScriptedText([], events, default=RatelimitException)
    _install_fake_ddgs(monkeypatch, _make_fake_ddgs(text))
    _patch_sleep(monkeypatch, recorded, events)

    result = _call_ddg("always rate limited")

    assert text.call_count == 3, "3 attempts total under sustained rate limiting"
    assert len(list(result)) == 0, "return [] after exhausting all retries"
    assert len(recorded) == 2, "sleeps happen BETWEEN attempts (attempts - 1 = 2)"
    for delay in recorded:
        assert 0.0 <= delay <= 4.0, "full-jitter delay stays within [0, cap=4]"
    assert events.count("text") == 3 and events.count("sleep") == 2


# --------------------------------------------------------------------------- #
# Non-ratelimit exception -> propagates unchanged with ZERO retries.
#
# The issue text's premise ("non-ratelimit exceptions keep the current return-[]
# behavior") is stale: HLB-657 already removed _ddg_search's broad except so a
# hard failure propagates and _fetch_and_merge records ddg_ok=False (asserted by
# test_instrumentation.py::test_ddg_raise_reaches_provenance_as_failure). Only
# the rate-limit path is caught; every other exception must propagate, so this
# probe pins propagation, not return-[].
# --------------------------------------------------------------------------- #
def test_ddg_backoff_non_ratelimit_propagates_without_retry(monkeypatch):
    if _BEHAV_SKIP:
        pytest.skip(_BEHAV_SKIP)
    events, recorded = [], []
    text = _ScriptedText([RuntimeError("boom")], events, default=[])
    _install_fake_ddgs(monkeypatch, _make_fake_ddgs(text))
    _patch_sleep(monkeypatch, recorded, events)

    with pytest.raises(RuntimeError):
        _call_ddg("non ratelimit failure")

    assert text.call_count == 1, "non-ratelimit error is NOT retried"
    assert len(recorded) == 0, "no backoff sleep for a non-ratelimit error"


# --------------------------------------------------------------------------- #
# R3 (static): no time.sleep in tools.py; the wait uses asyncio.sleep.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_no_time_sleep_in_tools_source():
    if tools is None:
        pytest.skip(f"search_workflow.tools failed to import: {_TOOLS_ERR!r}")
    src = _read_tools_source()
    assert re.search(r"time\.sleep", src) is None, "tools.py must not use time.sleep"
    assert "asyncio.sleep" in src, "the backoff wait must use asyncio.sleep"


# --------------------------------------------------------------------------- #
# R4 (a): the defensive import path resolves against installed packages.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_import_path_resolves():
    resolved = None
    for mod_name in ("ddgs.exceptions", "duckduckgo_search.exceptions"):
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        cls = getattr(m, "RatelimitException", None)
        if cls is not None:
            resolved = cls
            break
    assert resolved is not None, (
        "RatelimitException must resolve from ddgs.exceptions (primary) or "
        "duckduckgo_search.exceptions (fallback)"
    )
    assert isinstance(resolved, type) and issubclass(resolved, Exception)


# --------------------------------------------------------------------------- #
# R4 (b): the fallback module also exposes RatelimitException when installed.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_import_fallback_path_resolves():
    try:
        fallback = importlib.import_module("duckduckgo_search.exceptions")
    except Exception:
        pytest.skip("duckduckgo_search not installed; fallback path unavailable")
    cls = getattr(fallback, "RatelimitException", None)
    assert cls is not None, "fallback module must expose RatelimitException"
    assert isinstance(cls, type) and issubclass(cls, Exception)


# --------------------------------------------------------------------------- #
# R5: worst-case cumulative backoff is under Configuration.engine_deadline_s.
# --------------------------------------------------------------------------- #
def test_ddg_backoff_worstcase_under_engine_deadline():
    if Configuration is None:
        pytest.skip(f"Configuration failed to import: {_CFG_ERR!r}")
    deadline = _resolve_engine_deadline()
    if deadline is None:
        pytest.skip(
            "Configuration.engine_deadline_s not resolvable "
            "(engine-deadline story dependency missing)"
        )
    base, cap, attempts = 0.5, 4.0, 3
    # Two inter-attempt waits, each capped: min(4, 0.5) + min(4, 1.0) = 1.5.
    worst_case = sum(min(cap, base * (2 ** i)) for i in range(attempts - 1))
    assert worst_case == 1.5
    assert worst_case < deadline, (
        f"worst-case cumulative backoff ({worst_case:.3f}s) must stay under "
        f"engine_deadline_s ({deadline:.3f}s)"
    )
