"""Held-out probe for HLB-652 - Remove the pre-search SearXNG health probe from tools.py.

Acceptance criteria checked (black-box, engine-boundary mocks only):
  R1/R2: `health_check` appears nowhere under src/search_workflow/ (call removed + method deleted).
  Core intact: tools.py still defines the `search` tool and the shared fetch core (asyncio.gather).
  R3: an empty SearXNG 200 is a zero-result SUCCESS - DDG results still merge in, no crash.
  R4: the SearXNG leg errors ONLY when the request raises - DDG results still returned.
  R5: instrumentation shows n_searxng == 1 per query (no second health-probe round trip).

Rules honoured: never patch search_direct / the search tool; only the engine boundaries
(SearXNGClient.search, _ddg_search) are mocked. Deterministic, offline, single file.
"""
import asyncio
import inspect
import json
import logging
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# --- bootstrap the src/ layout when the package is not installed ----------------
_SRC = Path(os.getcwd()) / "src"
if (_SRC / "search_workflow").is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import search_workflow.tools as tools  # noqa: E402


def _pkg_sources():
    pkg = Path(tools.__file__).resolve().parent
    return {p: p.read_text(encoding="utf-8", errors="ignore") for p in pkg.rglob("*.py")}


def _mock_like(original, *, returns=None, raises=None):
    """Mock matching the original's sync/async signature; ignores all call args."""
    if inspect.iscoroutinefunction(original):
        async def _a(*a, **k):
            if raises is not None:
                raise raises
            return returns
        return _a

    def _s(*a, **k):
        if raises is not None:
            raise raises
        return returns
    return _s


def _patch_engines(*, searxng_returns=None, searxng_raises=None, ddg_returns=None):
    sx = _mock_like(tools.SearXNGClient.search, returns=searxng_returns, raises=searxng_raises)
    ddg = _mock_like(tools._ddg_search, returns=ddg_returns)
    return patch.object(tools.SearXNGClient, "search", sx), patch.object(tools, "_ddg_search", ddg)


# --- static: health_check fully removed (R1 call removed + R2 method deleted) ---
def test_health_check_absent_from_package():
    offenders = [str(p) for p, txt in _pkg_sources().items() if "health_check" in txt]
    assert not offenders, f"health_check still present in: {offenders}"


# --- static: search tool + shared fetch core still present ----------------------
def test_tools_keeps_search_and_gather_core():
    txt = Path(tools.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "asyncio.gather" in txt, "shared parallel fetch core (asyncio.gather) missing"
    assert hasattr(tools, "search"), "search tool missing from tools module"
    assert hasattr(tools, "search_direct"), "search_direct entrypoint missing"


# --- behavioural (R3): empty SearXNG 200 is success, DDG merges in --------------
def test_empty_searxng_is_success_ddg_merges():
    ddg = [{"title": "d", "link": "https://d.example/1", "snippet": "s"}]
    p1, p2 = _patch_engines(searxng_returns=[], ddg_returns=ddg)
    with p1, p2:
        result = asyncio.run(tools.search_direct("q", max_results=10))
    assert isinstance(result, list)
    assert len(result) == 1
    assert "d.example/1" in json.dumps(result, default=str)


# --- behavioural (R4): SearXNG raises -> DDG still returned, no crash -----------
def test_searxng_error_absorbed_ddg_returned():
    ddg = [{"title": "d", "link": "https://d.example/2", "snippet": "s"}]
    p1, p2 = _patch_engines(searxng_raises=RuntimeError("down"), ddg_returns=ddg)
    with p1, p2:
        result = asyncio.run(tools.search_direct("q", max_results=10))
    assert isinstance(result, list)
    assert len(result) == 1
    assert "d.example/2" in json.dumps(result, default=str)


# --- instrumentation (R5): one SearXNG request per query (was 2 with the probe) -
def _extract_n_searxng(metrics, caplog):
    for rec in caplog.records:
        if not rec.name.startswith("search_workflow"):
            continue
        if "n_searxng" in rec.__dict__:
            return int(rec.__dict__["n_searxng"])
        m = re.search(r'["\']?n_searxng["\']?\s*[:=]\s*(\d+)', rec.getMessage())
        if m:
            return int(m.group(1))
    if metrics is not None and hasattr(metrics, "snapshot"):
        snap = metrics.snapshot()
        if isinstance(snap, dict):
            if isinstance(snap.get("n_searxng"), int):
                return snap["n_searxng"]
            osr = snap.get("outbound_search_requests")
            if isinstance(osr, dict):
                for k, v in osr.items():
                    if "searx" in k.lower() and isinstance(v, int):
                        return v
    return None


def test_n_searxng_is_one_per_query(caplog):
    metrics = getattr(tools, "METRICS", None)
    if metrics is not None and hasattr(metrics, "reset"):
        metrics.reset()
    one = [{"title": "x", "link": "https://x.example/1", "snippet": "s"}]
    p1, p2 = _patch_engines(searxng_returns=one, ddg_returns=one)
    with caplog.at_level(logging.DEBUG):
        with p1, p2:
            asyncio.run(tools.search_direct("q", max_results=10))
    n = _extract_n_searxng(metrics, caplog)
    if n is None:
        pytest.skip(
            "no observable n_searxng metric/provenance surface; "
            "health_check-absence test covers R1/R2"
        )
    assert n == 1, f"expected exactly 1 SearXNG request per query, got {n}"
