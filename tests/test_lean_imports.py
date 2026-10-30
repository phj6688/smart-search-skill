"""Exercise the lean import and call paths in an uncontaminated process."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def _run_python(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env={**os.environ, "PYTHON_DOTENV_DISABLED": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lean_imports_and_calls_without_agent_dependencies() -> None:
    _run_python('''
        import asyncio
        import importlib.abc
        import importlib.util
        import json
        import sys

        class BlockAgentImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("langchain", "langgraph", "openai")):
                    raise ImportError(f"Forbidden import: {fullname}")

        sys.meta_path.insert(0, BlockAgentImports())
        import search_workflow
        assert "search_workflow.graph" not in sys.modules
        from search_workflow import Configuration
        import search_workflow.configuration
        from search_workflow import tools, lean
        from search_workflow.utils import SelectionResponse
        assert Configuration.from_runnable_config({}).model == "gpt-4o-mini"
        assert SelectionResponse(selected=[0]).selected == [0]
        mcp_server = None
        if (importlib.util.find_spec("fastapi") is not None
                and importlib.util.find_spec("mcp") is not None):
            from search_workflow import mcp_server

        import httpx
        real_client = httpx.AsyncClient
        requests = []
        fetched = []
        async def fetch(query, **kwargs):
            return fetched
        tools.search_direct = fetch
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": '{"selected": [0]}'}}]
            })
        httpx.AsyncClient = lambda **kwargs: real_client(
            transport=httpx.MockTransport(respond), **kwargs
        )

        async def check():
            empty = await lean.run_workflow_lean("query")
            assert empty["status"] == "ok" and empty["results"] == []
            assert not requests
            fetched.append({"title": "Title", "link": "https://a.test/Mixed", "snippet": "Text"})
            ranked = await lean.run_workflow_lean("query")
            assert ranked["status"] == "ok" and ranked["results"] == fetched
            if mcp_server is not None:
                mcp_server._RANKER_MODEL = "openai/test-model"
                content, _ = await mcp_server.create_mcp_server().call_tool(
                    "web_search", {"query": "query"}
                )
                assert json.loads(content[0].text) == fetched
            assert not [name for name in sys.modules
                        if name.startswith(("langchain", "langgraph", "openai"))]
        asyncio.run(check())
    ''')


def test_package_graph_exports_resolve_lazily() -> None:
    if importlib.util.find_spec("langchain_core") is None:
        pytest.skip("agent extra is not installed")
    _run_python('''
        import sys
        import search_workflow
        assert "search_workflow.graph" not in sys.modules
        from search_workflow import run_workflow, Configuration, TOOLS
        from search_workflow.graph import run_workflow as graph_workflow
        from search_workflow.agent_tools import search
        from search_workflow.tools import TOOLS as legacy_tools
        assert run_workflow is graph_workflow
        assert TOOLS is legacy_tools
        assert TOOLS == [search]
        for name in search_workflow.__all__:
            assert getattr(search_workflow, name) is not None
    ''')


@pytest.mark.parametrize("flag", ["--help", "--version"])
def test_cli_information_does_not_require_agent_dependencies(flag: str) -> None:
    _run_python(f'''
        import importlib.abc
        import sys
        class BlockAgentImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith(("langchain", "langgraph", "openai")):
                    raise ImportError(fullname)
        sys.meta_path.insert(0, BlockAgentImports())
        sys.argv = ["search-workflow", {flag!r}]
        from search_workflow.cli import main
        main()
    ''')
