"""Zero-OpenAI-egress proof for the local provider (HLB-659).

Drives ``run_workflow`` end to end with ``model="ollama/llama3.1"`` under the
egress socket-guard fixture, exercising BOTH LLM call sites (the agent node and
the evaluator node). The local provider is pointed at an allowlisted loopback
address standing in for the Ollama daemon. langchain-ollama is not part of the
base install, so a real ChatOllama cannot be constructed here; the sanctioned
fallback applies: stub the local model's ainvoke / with_structured_output so no
real network is needed, and let the egress guard prove no non-loopback connect
occurs. The load-bearing assertion is zero OpenAI egress across both sites.
"""

import socket

import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools, utils
from search_workflow.utils import SelectionResponse
from search_workflow.utils import load_chat_model as real_load_chat_model
from tests.fixtures_fallback import configure_fallback_state

# Any of these appearing as a socket peer means a query or snippet left for
# OpenAI, which is exactly the privacy regression this story removes.
_OPENAI_HOSTS = frozenset({"api.openai.com", "openai.com"})

# Loopback endpoint the local provider is pointed at, standing in for a local
# Ollama daemon on its default port. It is allowlisted by egress_guard. The fake
# never opens a real socket (the suite runs under --block-network), but carrying
# the URL documents the local-first configuration under test.
_OLLAMA_STUB_URL = "http://127.0.0.1:11434"


class _OllamaEvaluatorBound:
    """Stands in for ChatOllama(...).with_structured_output(SelectionResponse)."""

    def __init__(self, recorder: list[str]) -> None:
        self._recorder = recorder

    async def ainvoke(self, value: object, config: object = None) -> SelectionResponse:
        self._recorder.append("evaluator")
        return SelectionResponse(selected=[0])


class _OllamaAgentBound:
    """Stands in for ChatOllama(...).bind_tools(TOOLS); emits a search call."""

    def __init__(self, recorder: list[str]) -> None:
        self._recorder = recorder

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        self._recorder.append("agent")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "local probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_local_1",
                }
            ],
        )


class _FakeLocalChatModel:
    """ChatOllama stand-in pointed at a loopback URL; never reaches OpenAI."""

    def __init__(self, base_url: str, recorder: list[str]) -> None:
        self.base_url = base_url
        self._recorder = recorder

    def bind_tools(self, tools_: object) -> _OllamaAgentBound:
        return _OllamaAgentBound(self._recorder)

    def with_structured_output(self, schema: object) -> _OllamaEvaluatorBound:
        return _OllamaEvaluatorBound(self._recorder)


async def test_run_workflow_local_provider_has_zero_openai_egress(
    monkeypatch: pytest.MonkeyPatch, egress_guard: None
) -> None:
    call_sites: list[str] = []
    openai_constructions: list[str] = []
    fake_local = _FakeLocalChatModel(_OLLAMA_STUB_URL, call_sites)

    # Route the ollama provider to the loopback-pointed fake. Record any attempt
    # to construct the OpenAI integration instead of raising, so a stray openai
    # path surfaces as an assertion here rather than an obscured crash.
    def _record_openai(model_name: str, temperature: float) -> _FakeLocalChatModel:
        openai_constructions.append(model_name)
        return fake_local

    monkeypatch.setitem(
        utils._PROVIDERS, "ollama", lambda name, temperature: fake_local
    )
    monkeypatch.setitem(utils._PROVIDERS, "openai", _record_openai)

    # Undo conftest's autouse stub for the graph so run_workflow goes through
    # the real provider dispatch; the real loader reads utils._PROVIDERS (patched
    # above) at call time.
    monkeypatch.setattr(graph, "load_chat_model", real_load_chat_model)

    # Keep the tools node off the real network: stub both search engines at the
    # seam the shared core dispatches to, plus the SearXNG health probe.
    configure_fallback_state(monkeypatch, "searxng_ok")

    async def healthy() -> bool:
        return True

    monkeypatch.setattr(tools.searxng_client, "health_check", healthy)

    # Record every socket peer on top of the active egress guard, so the
    # zero-OpenAI-egress claim is asserted at the socket level too. The guard is
    # the current connect; wrap it so a non-loopback host still raises.
    connected_hosts: list[object] = []
    guarded_connect = socket.socket.connect

    def recording_connect(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, (tuple, list)) else address
        connected_hosts.append(host)
        return guarded_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", recording_connect)

    config = {"configurable": {"model": "ollama/llama3.1"}}
    out = await graph.run_workflow("local probe", config)

    assert out["status"] == "ok", out
    assert isinstance(out["results"], list) and out["results"]

    # Both LLM call sites ran through the local provider, not OpenAI.
    assert "agent" in call_sites, "agent node did not use the local provider"
    assert "evaluator" in call_sites, "evaluator node did not use the local provider"

    # Zero OpenAI egress: the OpenAI integration was never constructed, and no
    # socket peer was an OpenAI host (egress_guard would also have raised on any
    # non-loopback connect).
    assert openai_constructions == [], (
        f"OpenAI provider was constructed during a local run: {openai_constructions!r}"
    )
    assert not (_OPENAI_HOSTS & set(connected_hosts)), (
        f"a socket reached an OpenAI host: {connected_hosts!r}"
    )
