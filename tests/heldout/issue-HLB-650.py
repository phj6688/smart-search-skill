"""Held-out behavioural probe for HLB-650.

"Unify both search paths onto one shared fetch core with URL-normalized dedup"

Acceptance criteria checked, black-box:
1. tools.normalize_url exists: lowercases scheme+host, strips default ports and
   trailing slash, strips only utm_*/fbclid/gclid params, leaves path and query
   case untouched.
2. search_direct dedups the SearXNG + DDG merge by normalized URL: the
   tracking-param and trailing-slash twins collapse to one, ?page=2 survives.
3. search_direct keeps its signature and runs with no OPENAI_API_KEY; no LLM
   call in the core (load_chat_model absent from tools.py).
4. Both the LangGraph `search` tool and search_direct call one shared
   module-level async core; the fan-out uses asyncio.gather.
5. No shared/pooled aiohttp ClientSession: every instantiation is per-call
   inside an `async with`.

Engine-boundary mocks only (SearXNGClient.search, _ddg_search); search_direct
and the `search` tool themselves are never patched.
"""

import ast
import asyncio
import inspect
import sys
from pathlib import Path

# ASSUMPTION: probe is executed from the issue's worktree root, per task spec.
ROOT = Path.cwd()
SRC = ROOT / "src"
TOOLS_PATH = SRC / "search_workflow" / "tools.py"
sys.path.insert(0, str(SRC))

import search_workflow.tools as tools  # noqa: E402


def test_normalize_url_exists_and_behavior():
    normalize_url = getattr(tools, "normalize_url", None)
    assert callable(normalize_url), "tools.normalize_url must exist and be callable"
    # scheme/host case, default port, trailing slash all canonicalized
    assert normalize_url("HTTP://Example.COM:80/a/") == normalize_url("http://example.com/a")
    # only tracking params stripped; real query params survive
    kept = normalize_url("https://example.com/a?utm_source=x&page=2")
    assert "page=2" in kept
    assert "utm_source" not in kept
    tracked = normalize_url("https://example.com/a?fbclid=1&gclid=2")
    assert "fbclid" not in tracked
    assert "gclid" not in tracked
    # path case untouched
    assert "CaseSensitive/Path" in normalize_url("https://example.com/CaseSensitive/Path")


SEARX_RESULTS = [
    {"title": "a", "link": "https://example.com/a/", "snippet": "s"},
]
DDG_RESULTS = [
    {"title": "a2", "link": "https://example.com/a?utm_source=x", "snippet": "s"},
    {"title": "b", "link": "https://example.com/a?page=2", "snippet": "s"},
]


def _fake_like(template, payload):
    """Return a stub matching the sync/async nature of the real callable."""
    if inspect.iscoroutinefunction(template):
        async def fake(*args, **kwargs):
            return payload
    else:
        def fake(*args, **kwargs):
            return payload
    return fake


def test_search_direct_dedup_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert hasattr(tools, "SearXNGClient"), "engine boundary SearXNGClient missing"
    assert hasattr(tools, "_ddg_search"), "engine boundary _ddg_search missing"
    monkeypatch.setattr(
        tools.SearXNGClient, "search",
        _fake_like(tools.SearXNGClient.search, SEARX_RESULTS),
    )
    monkeypatch.setattr(
        tools, "_ddg_search",
        _fake_like(tools._ddg_search, DDG_RESULTS),
    )
    results = asyncio.run(tools.search_direct("q", max_results=10))
    assert isinstance(results, list)
    assert len(results) == 2, (
        "trailing-slash/utm twins must collapse to one and page=2 must "
        f"survive; got {results!r}"
    )
    assert any("page=2" in str(r) for r in results), "?page=2 variant must survive dedup"


def _called_names(node):
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_both_paths_call_one_shared_async_core():
    text = TOOLS_PATH.read_text()
    assert "asyncio.gather" in text, "shared core must fan out via asyncio.gather"
    tree = ast.parse(text)
    top = {
        n.name: n for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "search" in top, "LangGraph `search` tool missing from tools.py"
    assert "search_direct" in top, "search_direct missing from tools.py"
    async_defs = {
        name for name, node in top.items() if isinstance(node, ast.AsyncFunctionDef)
    }
    shared = (
        _called_names(top["search"])
        & _called_names(top["search_direct"])
        & async_defs
    ) - {"search", "search_direct"}
    assert shared, (
        "search and search_direct must both call the same module-level "
        "async core function"
    )


def test_no_pooled_client_session():
    for lineno, line in enumerate(TOOLS_PATH.read_text().splitlines(), 1):
        if "ClientSession(" in line:
            assert "async with" in line, (
                f"tools.py:{lineno} instantiates ClientSession outside "
                f"'async with' (pooled/shared session forbidden): {line.strip()}"
            )


def test_no_llm_call_in_core():
    assert "load_chat_model" not in TOOLS_PATH.read_text(), (
        "tools.py core must not call the LLM (load_chat_model found)"
    )
