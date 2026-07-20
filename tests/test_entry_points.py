"""Entry-point contract tests for the CLI (HLB-644).

Covers: sync console-script wrapper, `python -m search_workflow`, `--version`,
README honesty for the documented non-network commands, awaiting the `search`
tool inside a running event loop, and a wheel-install subprocess gate.
"""

import asyncio
import json
import re
import shlex
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
README = PROJECT_ROOT / "README.md"

VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")
DEV_SENTINEL = "0.0.0.dev0"

# Non-network commands documented in the README, executed verbatim below.
README_HELP_VERSION_COMMANDS = [
    "python -m search_workflow --help",
    "python -m search_workflow --version",
    "search-workflow --help",
    "search-workflow --version",
]
README_QUERY_COMMAND = 'python -m search_workflow "neo4j python driver documentation"'


def _run(cmd: list[str], timeout: int = 120, **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kwargs)


def _console_script() -> str:
    path = shutil.which("search-workflow")
    assert path is not None, "console script 'search-workflow' not on PATH (run uv sync)"
    return path


def _assert_version_output(output: str) -> None:
    # Never assert a literal version: accept any x.y.z or the dev sentinel.
    assert VERSION_PATTERN.search(output) or DEV_SENTINEL in output, (
        f"version output {output!r} matches neither \\d+.\\d+.\\d+ nor {DEV_SENTINEL}"
    )


def test_main_is_sync_and_async_impl_exists() -> None:
    from search_workflow import cli

    assert callable(cli.main)
    assert not asyncio.iscoroutinefunction(cli.main), (
        "cli.main must be sync: the console script never awaits a coroutine"
    )
    assert asyncio.iscoroutinefunction(cli._async_main)


def test_python_dash_m_help_exits_zero() -> None:
    result = _run([sys.executable, "-m", "search_workflow", "--help"])
    assert result.returncode == 0, result.stderr
    assert "Search Workflow CLI" in result.stdout


def test_python_dash_m_version() -> None:
    result = _run([sys.executable, "-m", "search_workflow", "--version"])
    assert result.returncode == 0, result.stderr
    _assert_version_output(result.stdout.strip())


def test_console_script_help_and_version() -> None:
    script = _console_script()
    help_result = _run([script, "--help"])
    assert help_result.returncode == 0, help_result.stderr
    assert "Search Workflow CLI" in help_result.stdout

    version_result = _run([script, "--version"])
    assert version_result.returncode == 0, version_result.stderr
    _assert_version_output(version_result.stdout.strip())


def test_readme_names_real_distribution() -> None:
    text = README.read_text()
    assert "pip install smart-search-skill" in text
    assert "uv add smart-search-skill" in text
    assert "pip install search-workflow" not in text
    assert "uv add search-workflow" not in text


def test_readme_documents_nonnetwork_commands() -> None:
    text = README.read_text()
    for command in [*README_HELP_VERSION_COMMANDS, README_QUERY_COMMAND]:
        assert command in text, f"README missing documented command: {command}"


@pytest.mark.parametrize("command", README_HELP_VERSION_COMMANDS)
def test_readme_nonnetwork_commands_run_verbatim(command: str) -> None:
    assert command in README.read_text()
    argv = shlex.split(command)
    # Resolve the interpreter/script but keep the arguments verbatim.
    argv[0] = sys.executable if argv[0] == "python" else _console_script()
    result = _run(argv)
    assert result.returncode == 0, f"{command!r} failed: {result.stderr}"


def test_readme_query_command_with_stubbed_workflow(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from search_workflow import cli

    assert README_QUERY_COMMAND in README.read_text()

    async def fake_run_workflow(query: str, config: Any = None) -> dict[str, Any]:
        assert query == "neo4j python driver documentation"
        # run_workflow returns the discriminated success shape.
        return {
            "status": "ok",
            "results": [{"title": "t", "link": "https://example.com", "snippet": "s"}],
        }

    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    cli_args = shlex.split(README_QUERY_COMMAND)[3:]  # drop "python -m search_workflow"
    monkeypatch.setattr(sys, "argv", ["search-workflow", *cli_args])

    exit_code = cli.main()
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["link"] == "https://example.com"


async def test_search_tool_awaitable_inside_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Embedded consumers (LangGraph, MCP server) await `search` inside their own
    # loop; the sync CLI wrapper must not push asyncio.run into the tool path.
    from search_workflow import tools

    stub_results = [{"title": "t", "link": "https://example.com", "snippet": "s"}]

    async def fake_health_check() -> bool:
        return True

    async def fake_searxng_search(**kwargs: Any) -> list[dict[str, str]]:
        return stub_results

    monkeypatch.setattr(tools.searxng_client, "health_check", fake_health_check)
    monkeypatch.setattr(tools.searxng_client, "search", fake_searxng_search)

    assert asyncio.get_running_loop() is not None
    try:
        results = await tools.search("query", "us-en", None, config={})
    except RuntimeError as exc:
        pytest.fail(f"search raised RuntimeError inside a running loop: {exc}")
    assert results == stub_results


@pytest.mark.wheel_install
def test_wheel_install_subprocess_gates(tmp_path: Path) -> None:
    if shutil.which("uv") is None:
        pytest.skip("uv not available to build the wheel")

    dist_dir = tmp_path / "dist"
    build = _run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=str(PROJECT_ROOT),
        timeout=300,
    )
    assert build.returncode == 0, build.stderr
    wheels = list(dist_dir.glob("smart_search_skill-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=False)
    venv_python = venv_dir / "bin" / "python"

    # uv resolves from its local cache first, so this passes without network
    # after a prior sync; a genuinely offline cold cache is a skip, not a fail.
    install = _run(
        ["uv", "pip", "install", "--python", str(venv_python), str(wheels[0])],
        timeout=600,
    )
    if install.returncode != 0:
        pytest.skip(f"could not install wheel into clean venv: {install.stderr[-500:]}")

    help_result = _run([str(venv_python), "-m", "search_workflow", "--help"])
    assert help_result.returncode == 0, help_result.stderr

    version_result = _run([str(venv_dir / "bin" / "search-workflow"), "--version"])
    assert version_result.returncode == 0, version_result.stderr
    _assert_version_output(version_result.stdout.strip())
