"""Held-out behavioural probe for HLB-653.

Issue: "Wire SearXNG timeout through Configuration and add per-engine deadline."
Anchor: src/search_workflow/configuration.py:Configuration ; package `search_workflow`.

Acceptance criteria checked black-box (no impl reading, engine-boundary mocks only):
  R1  Tool-path SearXNG client built from Configuration.searxng_url / .searxng_timeout;
      effective default timeout is 30 (replacing the hardcoded 12s class default).
  R2  Configuration exposes engine_deadline_s, a constant in [4.0, 4.5] seconds.
  R3  Each per-engine fetch wrapped in asyncio.wait_for(deadline): a hung engine
      yields fallback results instead of stalling the whole call.
  R5  Explicit-default test for every new config knob, incl. SEARXNG_URL env fallback.

Probe runs from the issue worktree root, outside tests/. pytest + stdlib + monkeypatch.
"""

import asyncio
import importlib
import os
import time
from pathlib import Path

import pytest

SENTINEL = "HELDOUT_DDG_FALLBACK_TOKEN"


def _tools_source() -> str:
    import search_workflow.tools as tools
    return Path(tools.__file__).read_text(encoding="utf-8")


def test_engine_deadline_constant():
    """R2: Configuration().engine_deadline_s exists and is within [4.0, 4.5]."""
    import search_workflow.configuration as cfg
    c = cfg.Configuration()
    assert hasattr(c, "engine_deadline_s"), "Configuration missing engine_deadline_s"
    val = float(c.engine_deadline_s)
    assert 4.0 <= val <= 4.5, f"engine_deadline_s={val} not in [4.0, 4.5]"


def test_default_searxng_timeout_is_30():
    """R1: with a clean env the effective default searxng_timeout is 30 (not 12)."""
    import search_workflow.configuration as cfg
    prev = os.environ.pop("SEARXNG_TIMEOUT", None)
    try:
        importlib.reload(cfg)
        c = cfg.Configuration()
        assert hasattr(c, "searxng_timeout"), "Configuration missing searxng_timeout"
        assert int(c.searxng_timeout) == 30, f"default searxng_timeout={c.searxng_timeout}, want 30"
    finally:
        if prev is not None:
            os.environ["SEARXNG_TIMEOUT"] = prev
        importlib.reload(cfg)


def test_searxng_url_env_fallback():
    """R5: SEARXNG_URL env value flows through Configuration.searxng_url."""
    import search_workflow.configuration as cfg
    prev = os.environ.get("SEARXNG_URL")
    os.environ["SEARXNG_URL"] = "http://alt:9090"
    try:
        importlib.reload(cfg)
        c = cfg.Configuration()
        assert hasattr(c, "searxng_url"), "Configuration missing searxng_url"
        assert c.searxng_url == "http://alt:9090", (
            f"searxng_url did not honour SEARXNG_URL env: got {c.searxng_url!r}"
        )
    finally:
        if prev is None:
            os.environ.pop("SEARXNG_URL", None)
        else:
            os.environ["SEARXNG_URL"] = prev
        importlib.reload(cfg)


def test_wait_for_wraps_engine_fetches():
    """R3: tools.py guards per-engine fetches with asyncio.wait_for on the deadline."""
    src = _tools_source()
    assert "engine_deadline_s" in src, "tools.py does not read engine_deadline_s"
    assert src.count("wait_for") >= 2 or (
        "wait_for" in src and "engine_deadline_s" in src
    ), "tools.py does not wrap engine fetches in wait_for with the deadline"


def test_tools_reads_config_knobs():
    """R1: tool path references the Configuration timeout + deadline knobs."""
    src = _tools_source()
    assert "searxng_timeout" in src, "tools.py does not reference searxng_timeout"
    assert "engine_deadline_s" in src, "tools.py does not reference engine_deadline_s"


def test_hung_engine_falls_back_within_deadline(monkeypatch):
    """R3 behavioural: a hung SearXNG engine hits the deadline and DDG fallback wins."""
    import search_workflow.configuration as cfg
    import search_workflow.tools as tools

    deadline = float(getattr(cfg.Configuration(), "engine_deadline_s", 4.25))

    async def hung_search(self, *a, **k):
        await asyncio.sleep(10)
        return []

    monkeypatch.setattr(tools.SearXNGClient, "search", hung_search, raising=False)

    class _AwaitableList(list):
        # Behaves as a list when called sync/in-thread, and resolves when awaited,
        # so the fake works whether _ddg_search is invoked sync or async.
        def __await__(self):
            async def _c():
                return list(self)
            return _c().__await__()

    ddg_item = {
        "title": SENTINEL, "url": "http://ddg.example", "href": "http://ddg.example",
        "content": SENTINEL, "snippet": SENTINEL, "body": SENTINEL,
    }

    def fake_ddg(*a, **k):
        return _AwaitableList([dict(ddg_item)])

    assert hasattr(tools, "_ddg_search"), "tools._ddg_search boundary not present"
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg, raising=False)

    async def _outer():
        try:
            call = tools.search_direct("q", max_results=5)
        except TypeError:
            call = tools.search_direct("q")
        if not (asyncio.iscoroutine(call) or hasattr(call, "__await__")):
            return call
        return await asyncio.wait_for(call, timeout=deadline + 3.0)

    start = time.monotonic()
    try:
        result = asyncio.run(_outer())
    except TimeoutError:
        pytest.fail(
            "search_direct did not honour the per-engine deadline: a hung SearXNG "
            "engine stalled the whole call past engine_deadline_s"
        )
    elapsed = time.monotonic() - start

    assert elapsed <= deadline + 2.5, (
        f"call took {elapsed:.1f}s; per-engine deadline (~{deadline}s) not enforced"
    )
    assert SENTINEL in str(result), (
        "DDG fallback result missing when the SearXNG engine hung; deadline did not "
        "yield fallback"
    )
