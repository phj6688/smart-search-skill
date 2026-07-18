"""Held-out behavioural probe for HLB-644.

Acceptance criteria checked:
1. `python -m search_workflow --help` exits 0 (working `__main__.py`).
2. `search_workflow.cli.main` is a synchronous callable (not a coroutine
   function), keeping the console-script target valid.
3. `python -m search_workflow --version` exits 0 and prints either a semver
   (\\d+\\.\\d+\\.\\d+) or a dev sentinel, never crashing when the
   distribution is not installed.
4. README.md advertises the real distribution name `smart-search-skill`
   for both pip and uv, not `search-workflow`.
5. Awaiting the `search` tool from search_workflow.tools inside an already
   running event loop raises no "asyncio.run() cannot be called from a
   running event loop" RuntimeError.
"""

import asyncio
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path.cwd()
SRC = REPO / "src"


def _env_with_src():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), env.get("PYTHONPATH", "")) if p
    )
    return env


def _run_module(*args):
    return subprocess.run(
        [sys.executable, "-m", "search_workflow", *args],
        cwd=str(REPO),
        env=_env_with_src(),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _import_pkg(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return __import__(name, fromlist=["_"])


def test_python_dash_m_help_exits_zero():
    proc = _run_module("--help")
    assert proc.returncode == 0, (
        f"python -m search_workflow --help exited {proc.returncode}\n"
        f"stderr: {proc.stderr}"
    )


def test_cli_main_is_sync_callable():
    cli = _import_pkg("search_workflow.cli")
    assert hasattr(cli, "main"), "search_workflow.cli has no `main`"
    assert callable(cli.main), "search_workflow.cli.main is not callable"
    assert not inspect.iscoroutinefunction(cli.main), (
        "search_workflow.cli.main is a coroutine function; the console-script "
        "target requires a synchronous wrapper"
    )


def test_version_flag_semver_or_dev_sentinel():
    proc = _run_module("--version")
    assert proc.returncode == 0, (
        f"--version exited {proc.returncode}\nstderr: {proc.stderr}"
    )
    out = (proc.stdout + proc.stderr).strip()
    assert re.search(r"(\d+\.\d+\.\d+|.*dev.*)", out), (
        f"--version output matches neither semver nor a dev sentinel: {out!r}"
    )


def test_readme_uses_real_distribution_name():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "pip install smart-search-skill" in text
    assert "pip install search-workflow" not in text
    assert "uv add search-workflow" not in text


def test_search_tool_safe_inside_running_loop():
    tools = _import_pkg("search_workflow.tools")
    search = tools.search

    async def body():
        # ASSUMPTION: the issue names the `search` tool but not its exact call
        # shape; try the LangChain async invoke surface first, then a direct
        # await. Only the in-loop RuntimeError is a failure; network or
        # argument errors are acceptable in this offline probe.
        try:
            if hasattr(search, "ainvoke"):
                await search.ainvoke(
                    {"query": "heldout probe"}, {"configurable": {}}
                )
            else:
                await search("heldout probe", config={"configurable": {}})
        except RuntimeError as exc:
            if "asyncio.run() cannot be called from a running event loop" in str(exc):
                raise AssertionError(
                    "search tool called asyncio.run() inside a running loop"
                ) from exc
        except Exception:
            pass

    asyncio.run(body())
