"""Held-out behavioural probe for issue HLB-654.

Acceptance criteria checked:
1. run_workflow returns no bare error strings: its source contains no
   'return f"' / 'return "Error' patterns, and forcing the compiled graph's
   ainvoke to fail yields either a raised SearchError/WorkflowError-typed
   exception or a discriminated {"status": "error", ...} dict, never a str
   (or str subclass).
2. The success path still returns a usable non-error shape (legacy list or
   {"status": "ok", ...} dict).
3. pyproject.toml carries version = "0.4.0".
4. CHANGELOG.md documents the migration, naming the old string shape and the
   new typed/discriminated shape.
5. mcp_server._coerce_results no longer wraps bare strings as
   [{"error": str(raw)}] and handles the discriminated "status" shape.
"""

import asyncio
import inspect
import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path.cwd()  # probe is executed from the issue worktree root
GRAPH_PY = ROOT / "src" / "search_workflow" / "graph.py"
MCP_PY = ROOT / "src" / "search_workflow" / "mcp_server.py"

sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("OPENAI_API_KEY", "probe-key")  # keep import/runtime deterministic


def _import_graph_module():
    import importlib

    return importlib.import_module("search_workflow.graph")


def _call(func, *args):
    if inspect.iscoroutinefunction(func):
        return asyncio.run(func(*args))
    result = func(*args)
    if inspect.iscoroutine(result):
        return asyncio.run(result)
    return result


def _run_workflow_body(src: str) -> str:
    m = re.search(r"^(?:async\s+)?def\s+run_workflow\b", src, re.M)
    assert m, "run_workflow not found in src/search_workflow/graph.py"
    rest = src[m.end():]
    nxt = re.search(r"^(?:async def |def |class |@)", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _patch_graph_failure(monkeypatch, mod, async_impl, sync_impl):
    compiled = getattr(mod, "graph", None)
    if compiled is None:
        pytest.fail("search_workflow.graph exposes no module-level compiled `graph`")
    monkeypatch.setattr(compiled, "ainvoke", async_impl, raising=False)
    monkeypatch.setattr(compiled, "invoke", sync_impl, raising=False)


def test_run_workflow_source_has_no_string_error_returns():
    body = _run_workflow_body(GRAPH_PY.read_text())
    assert 'return f"' not in body, "run_workflow still returns f-string errors"
    assert re.search(r'return\s+f?"Error', body) is None, (
        "run_workflow still returns bare 'Error...' strings"
    )


def test_error_path_yields_typed_error_or_status_dict(monkeypatch):
    mod = _import_graph_module()

    async def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    def _boom_sync(*args, **kwargs):
        raise RuntimeError("boom")

    _patch_graph_failure(monkeypatch, mod, _boom, _boom_sync)
    try:
        result = _call(mod.run_workflow, "q")
    except Exception as exc:  # a raised typed error is an accepted outcome
        names = {cls.__name__ for cls in type(exc).__mro__}
        assert any("SearchError" in n or "WorkflowError" in n for n in names), (
            f"error path raised untyped {type(exc).__name__}: {exc!r}"
        )
        return
    assert not isinstance(result, str), (
        f"error path returned a bare string (str or str subclass): {result!r}"
    )
    assert isinstance(result, dict), (
        f"error path must return a discriminated dict, got {type(result).__name__}: {result!r}"
    )
    assert result.get("status") == "error", (
        f"error dict missing status == 'error': {result!r}"
    )


def test_success_path_returns_non_error_shape(monkeypatch):
    from types import SimpleNamespace

    mod = _import_graph_module()
    final_state = {
        "messages": [
            SimpleNamespace(
                content='[{"title": "t", "link": "https://x.example/a", "snippet": "s"}]'
            )
        ]
    }

    async def _ok(*args, **kwargs):
        return final_state

    def _ok_sync(*args, **kwargs):
        return final_state

    _patch_graph_failure(monkeypatch, mod, _ok, _ok_sync)
    result = _call(mod.run_workflow, "q")
    assert not (isinstance(result, str) and "error" in result.lower()), (
        f"success path returned an error string: {result!r}"
    )
    if isinstance(result, dict) and "status" in result:
        assert result["status"] == "ok", f"success dict has status != 'ok': {result!r}"


def test_pyproject_version_is_0_4_0():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version\s*=\s*"0\.4\.0"', text, re.M), (
        "pyproject.toml does not pin version = \"0.4.0\""
    )


def test_changelog_names_old_and_new_shapes():
    text = (ROOT / "CHANGELOG.md").read_text().lower()
    assert "string" in text, "CHANGELOG.md never mentions the old string error shape"
    assert "status" in text or "searcherror" in text, (
        "CHANGELOG.md never names the new typed/discriminated shape"
    )


def test_mcp_server_coerce_results_handles_discriminated_shape():
    text = MCP_PY.read_text()
    assert 'return [{"error": str(raw)}]' not in text, (
        "_coerce_results still wraps bare strings as [{'error': str(raw)}]"
    )
    assert "status" in text, (
        "mcp_server.py shows no handling of the discriminated 'status' shape"
    )
