"""Held-out behavioural probe for issue HLB-664.

Chore: "drop duckduckgo-search, standardize on ddgs."

Acceptance criteria exercised (verbatim from the issue Verification list):
  1. rg "duckduckgo-search" pyproject.toml returns no hits in dependencies.
  2. rg "DuckDuckGoSearchResults" src/ returns no hits.
  3. A runtime test exercising DDGS().text() through the shared core passes
     (cassette-backed); an import-only check is insufficient (runtime-call gate).
  4. The RatelimitException import-path test from the backoff story still passes
     with the legacy package absent (RatelimitException resolves from ddgs).
  5. pyproject pins a tested ddgs range with floor AND upper bound
     (e.g. ddgs>=9.14,<10) and uv sync resolves.

Frozen black-box probe: derived from the issue text only, never from the
implementation. Offline, deterministic, no network. Runs from the issue
worktree root; it locates the repo by walking up cwd to the directory that
contains src/search_workflow and prepends that repo's src/ to sys.path.
"""

import asyncio
import inspect
import re
import sys
import tomllib
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Repo discovery + sys.path wiring (probe lives OUTSIDE the repo tests/ tree)
# ---------------------------------------------------------------------------
def _find_repo_root():
    """Walk up from cwd (issue worktree root) to the dir holding src/search_workflow."""
    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    # Also consider walking up from this file, in case cwd is unexpected.
    candidates += [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    seen = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        if (base / "src" / "search_workflow").is_dir():
            return base
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is not None:
    _src = str(_REPO_ROOT / "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)


def _require_repo():
    if _REPO_ROOT is None:
        pytest.fail("could not locate repo root containing src/search_workflow")
    return _REPO_ROOT


# ---------------------------------------------------------------------------
# Dependency parsing helpers
# ---------------------------------------------------------------------------
def _load_pyproject():
    root = _require_repo()
    pp = root / "pyproject.toml"
    assert pp.is_file(), f"pyproject.toml not found at {pp}"
    with pp.open("rb") as fh:
        return tomllib.load(fh)


def _all_declared_deps(pyproject):
    """Only inspect the actual dependency lists, never the whole-file prose."""
    deps = []
    proj = pyproject.get("project", {}) or {}
    deps.extend(proj.get("dependencies", []) or [])
    opt = proj.get("optional-dependencies", {}) or {}
    for group in opt.values():
        deps.extend(group or [])
    return [d for d in deps if isinstance(d, str)]


def _dep_name(dep_str):
    m = re.match(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dep_str)
    if not m:
        return ""
    return m.group(1).lower().replace("_", "-")


def _version_tuple(v):
    parts = re.split(r"[.\-+]", v)
    out = []
    for p in parts:
        m = re.match(r"^(\d+)", p)
        out.append(int(m.group(1)) if m else 0)
    return tuple(out)


# ---------------------------------------------------------------------------
# Test 1 - duckduckgo-search dropped from dependencies
# ---------------------------------------------------------------------------
def test_pyproject_drops_duckduckgo_search():
    pyproject = _load_pyproject()
    deps = _all_declared_deps(pyproject)
    offenders = [d for d in deps if _dep_name(d) == "duckduckgo-search"]
    assert not offenders, (
        "duckduckgo-search must not appear in [project.dependencies] or any "
        f"optional-dependencies group; found: {offenders}"
    )


# ---------------------------------------------------------------------------
# Test 2 - ddgs pinned with BOTH a floor (>= 9.14) AND an upper bound (<)
# ---------------------------------------------------------------------------
def test_ddgs_pinned_with_floor_and_upper_bound():
    pyproject = _load_pyproject()
    deps = _all_declared_deps(pyproject)
    ddgs_deps = [d for d in deps if _dep_name(d) == "ddgs"]
    assert ddgs_deps, f"ddgs dependency not declared; deps={deps}"
    dep_str = ddgs_deps[0]

    has_lower = has_upper = False
    lower_ok = False

    parsed = False
    try:
        from packaging.requirements import Requirement
        from packaging.version import Version

        req = Requirement(dep_str)
        for spec in req.specifier:
            if spec.operator in (">=", ">"):
                has_lower = True
                if Version(spec.version) >= Version("9.14"):
                    lower_ok = True
            if spec.operator in ("<", "<="):
                has_upper = True
        parsed = True
    except Exception:
        parsed = False

    if not parsed:
        # Tolerant regex fallback: extract every (operator, version) pair.
        pairs = re.findall(r"(>=|<=|==|~=|>|<)\s*([0-9][0-9A-Za-z.\-]*)", dep_str)
        for op, ver in pairs:
            if op in (">=", ">"):
                has_lower = True
                if _version_tuple(ver) >= _version_tuple("9.14"):
                    lower_ok = True
            if op in ("<", "<="):
                has_upper = True

    assert has_lower, f"ddgs pin missing a lower bound (>=): {dep_str!r}"
    assert lower_ok, f"ddgs floor must be >= 9.14: {dep_str!r}"
    assert has_upper, f"ddgs pin missing an upper bound (< / <=): {dep_str!r}"


# ---------------------------------------------------------------------------
# Test 3 - no legacy DuckDuckGoSearchResults reference in the package source
# ---------------------------------------------------------------------------
def test_no_duckduckgo_search_results_in_src():
    root = _require_repo()
    pkg = root / "src" / "search_workflow"
    py_files = sorted(pkg.glob("*.py"))
    assert py_files, f"no python source files found under {pkg}"
    offenders = []
    for f in py_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        if "DuckDuckGoSearchResults" in text:
            offenders.append(f.name)
    assert not offenders, (
        "legacy DuckDuckGoSearchResults reference must be gone from src/; "
        f"found in: {offenders}"
    )


# ---------------------------------------------------------------------------
# Test 4 - runtime-call gate: a real DDGS().text() call flows through the core
# ---------------------------------------------------------------------------
class _FakeDDGS:
    text_called = False
    last_args = None

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, *args, **kwargs):
        _FakeDDGS.text_called = True
        _FakeDDGS.last_args = (args, kwargs)
        return [
            {"title": "Result One", "href": "https://example.com/1", "body": "body one"},
            {"title": "Result Two", "href": "https://example.com/2", "body": "body two"},
        ]


def _linklike(item):
    if isinstance(item, dict):
        keys = {str(k).lower() for k in item.keys()}
        return bool(keys & {"link", "url", "href", "title", "name", "body", "snippet", "content"})
    return any(hasattr(item, attr) for attr in ("link", "url", "href", "title", "name"))


def _maybe_run(result):
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _drive_callable(func):
    """Try a few call shapes tolerant to the exact signature; return the result."""
    query = "python typed errors"
    attempts = []
    try:
        sig = inspect.signature(func)
        params = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        names = {p.name for p in params}
        kw = {}
        if names & {"max_results", "num_results", "count", "k", "n"}:
            for candidate in ("max_results", "num_results", "count", "k", "n"):
                if candidate in names:
                    kw[candidate] = 3
                    break
        attempts.append(((query,), kw))
    except (TypeError, ValueError):
        pass
    attempts.append(((query,), {"max_results": 3}))
    attempts.append(((query, 3), {}))
    attempts.append(((query,), {}))

    last_exc = None
    for args, kwargs in attempts:
        try:
            return _maybe_run(func(*args, **kwargs))
        except TypeError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no call attempt executed")


def test_ddgs_text_runtime_through_core():
    _require_repo()
    _FakeDDGS.text_called = False
    _FakeDDGS.last_args = None

    from search_workflow import tools

    patches = ExitStack()
    # Patch every plausible binding of the DDGS client so a local import inside
    # the function is also covered. create=True keeps this tolerant.
    patches.enter_context(mock.patch.object(tools, "DDGS", _FakeDDGS, create=True))
    try:
        import ddgs as _ddgs_mod
        patches.enter_context(mock.patch.object(_ddgs_mod, "DDGS", _FakeDDGS, create=True))
    except Exception:
        pass

    result = None
    with patches:
        ddg_fn = getattr(tools, "_ddg_search", None)
        if ddg_fn is not None:
            result = _drive_callable(ddg_fn)
        else:
            # Fallback: drive the higher-level entry point (still a real .text call).
            search_direct = getattr(tools, "search_direct", None)
            assert search_direct is not None, (
                "neither tools._ddg_search nor tools.search_direct is available"
            )
            result = _drive_callable(search_direct)

    # Runtime-call gate: the point is a REAL DDGS().text() invocation, not an import.
    assert _FakeDDGS.text_called, "DDGS().text() was never called - import-only path is insufficient"

    assert result is not None, "shared core returned None"
    items = result
    if isinstance(result, dict):
        # Some cores wrap results; dig out the first list value.
        for v in result.values():
            if isinstance(v, (list, tuple)):
                items = v
                break
    assert isinstance(items, (list, tuple)), f"expected a list of results, got {type(items)!r}"
    assert len(items) > 0, "shared core returned an empty result list"
    assert any(_linklike(it) for it in items), (
        "returned results are not normalized (no link/title-like field found)"
    )


# ---------------------------------------------------------------------------
# Test 5 - RatelimitException resolves from ddgs with the legacy package absent
# ---------------------------------------------------------------------------
def test_ratelimit_exception_resolves_from_ddgs():
    # Primary path that MUST work once duckduckgo-search is dropped.
    from ddgs.exceptions import RatelimitException

    assert isinstance(RatelimitException, type)
    assert issubclass(RatelimitException, Exception), (
        "RatelimitException must be an Exception subclass"
    )
    # Do NOT require duckduckgo_search to be importable - the legacy package
    # may be absent after this chore.
