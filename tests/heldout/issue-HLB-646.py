"""Held-out behavioural probe for issue HLB-646.

Acceptance criteria checked:
1. tests/ contains a five-state ``fallback_state`` fixture covering
   searxng_ok, searxng_raises, searxng_empty, searxng_ok_ddg_unused, both_fail.
2. src/search_workflow/tools.py declares the per-query counters
   (outbound_search_requests, llm_calls, cache_hit, engines_used) and the
   provenance record fields (n_searxng, n_ddg, n_after_dedup, elapsed_ms,
   ddg_ok, fell_back).
3. search_direct merges SearXNG and DDG results (2 + 1 distinct -> 3) and a
   machine-readable provenance record is observable, with elapsed_ms >= 0.
"""
import asyncio
import importlib
import inspect
import logging
import pathlib
import re
import sys
from unittest import mock

ROOT = pathlib.Path.cwd()  # probe runs from the issue's worktree root
sys.path.insert(0, str(ROOT / "src"))

TOOLS_PATH = ROOT / "src" / "search_workflow" / "tools.py"
STATES = ("searxng_ok", "searxng_raises", "searxng_empty", "searxng_ok_ddg_unused", "both_fail")
COUNTERS = ("outbound_search_requests", "llm_calls", "cache_hit", "engines_used")
FIELDS = ("n_searxng", "n_ddg", "n_after_dedup", "elapsed_ms", "ddg_ok", "fell_back")
METRIC_NAMES = ("METRICS", "metrics", "COUNTERS", "counters", "get_metrics", "last_query_record", "LAST_RECORD")

FAKE_SEARXNG = [
    {"title": "sx one", "link": "https://sx.example/1", "url": "https://sx.example/1", "snippet": "a", "content": "a"},
    {"title": "sx two", "link": "https://sx.example/2", "url": "https://sx.example/2", "snippet": "b", "content": "b"},
]
FAKE_DDG = [
    {"title": "ddg one", "link": "https://ddg.example/3", "url": "https://ddg.example/3", "snippet": "c", "content": "c"},
]


def test_fallback_state_fixture_covers_five_states():
    tests_dir = ROOT / "tests"
    assert tests_dir.is_dir(), "tests/ directory missing"
    blob = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in sorted(tests_dir.rglob("*.py"))
        if "heldout" not in p.name  # never self-match this probe
    )
    assert "def fallback_state" in blob or all(s in blob for s in STATES), \
        "no fallback_state fixture (or five-state parametrize) found under tests/"
    missing = [s for s in STATES if s not in blob]
    assert not missing, f"fallback_state state names missing from tests/: {missing}"


def test_tools_declares_counters_and_record_fields():
    src = TOOLS_PATH.read_text(encoding="utf-8", errors="replace")
    missing = [n for n in COUNTERS + FIELDS if n not in src]
    assert not missing, f"names missing from src/search_workflow/tools.py: {missing}"


def _stub(original, payload):
    fresh = lambda: [dict(item) for item in payload]  # noqa: E731
    if inspect.iscoroutinefunction(original):
        async def repl(*a, **k):
            return fresh()
    else:
        def repl(*a, **k):
            return fresh()
    return repl


def _record_from(obj):
    if isinstance(obj, dict):
        if "n_searxng" in obj:
            return obj
        for v in obj.values():  # e.g. metrics keyed by query
            if isinstance(v, dict) and "n_searxng" in v:
                return v
    if hasattr(obj, "n_searxng"):
        return {f: getattr(obj, f, None) for f in FIELDS}
    return None


def _find_record(tools_mod, caplog):
    # ASSUMPTION: issue names the record fields but not the exact exposure
    # surface, so accept a module-level metrics object or a structured log.
    for name in METRIC_NAMES:
        obj = getattr(tools_mod, name, None)
        if obj is None:
            continue
        if callable(obj) and not isinstance(obj, type):
            try:
                obj = obj()
            except Exception:
                continue
        rec = _record_from(obj)
        if rec is not None:
            return rec
    for lr in caplog.records:
        if not lr.name.startswith("search_workflow"):
            continue
        rec = _record_from(lr.__dict__)  # extra= lands on record attributes
        if rec is not None:
            return rec
        msg = lr.getMessage()
        if "n_searxng" in msg:
            return msg
    return None


def _elapsed_ms(record):
    if isinstance(record, dict) and record.get("elapsed_ms") is not None:
        return float(record["elapsed_ms"])
    text = record if isinstance(record, str) else str(record)
    m = re.search(r"elapsed_ms\W{0,4}(-?[0-9]+(?:\.[0-9]+)?)", text)
    assert m, f"elapsed_ms not present in observed provenance record: {text!r}"
    return float(m.group(1))


def test_search_direct_merges_and_emits_provenance(caplog):
    tools = importlib.import_module("search_workflow.tools")
    ddg_name = next((n for n in ("_ddg_search",) if hasattr(tools, n)), None)
    assert ddg_name, "DDG fetch helper _ddg_search not found in search_workflow.tools"
    caplog.set_level(logging.DEBUG)
    # Patch only the engine boundaries; search_direct (the anchor) stays real.
    with mock.patch.object(tools.SearXNGClient, "search",
                           _stub(tools.SearXNGClient.search, FAKE_SEARXNG)), \
         mock.patch.object(tools, ddg_name,
                           _stub(getattr(tools, ddg_name), FAKE_DDG)):
        if inspect.iscoroutinefunction(tools.search_direct):
            result = asyncio.run(tools.search_direct("q", max_results=5))
        else:
            result = tools.search_direct("q", max_results=5)
    if isinstance(result, dict) and "results" in result:
        result = result["results"]
    assert len(result) == 3, f"expected 2 SearXNG + 1 distinct DDG merged results, got {len(result)}"
    record = _find_record(tools, caplog)
    assert record is not None, \
        "no provenance record observable via metrics object or search_workflow.* log containing n_searxng"
    assert _elapsed_ms(record) >= 0
