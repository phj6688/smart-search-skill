"""Shared fixtures.

`fallback_state` lives in tests/fixtures_fallback.py and is re-exported here
so every test module gets it without importing the mocks directly. Later
stories use this fixture; they never re-create engine stubs.

HLB-647 adds the VCR wiring (cassette dir, header scrubbing) and an autouse
OpenAI stub so the suite runs offline with OPENAI_API_KEY unset.
"""

import os
import socket
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage

from search_workflow.utils import ArticlesResponse, ArticleStrict
from tests.fixtures_fallback import (  # noqa: F401
    FALLBACK_STATE_NAMES,
    FallbackExpectation,
    FallbackScenario,
    configure_fallback_state,
    fallback_state,
)

# Written into cassettes in place of Authorization / X-Api-Key values, so the
# hygiene test can assert the scrub ran instead of proving a negative.
HEADER_PLACEHOLDER = "SCRUBBED"

# Lets the delete-a-cassette negative test point a pytest subprocess at a
# pruned copy of tests/cassettes/ without touching the checked-in cassettes.
CASSETTE_ROOT_ENV = "SEARCH_WORKFLOW_CASSETTE_ROOT"

# Hosts a guarded test is allowed to reach. Everything else is treated as
# outbound egress and blocked.
EGRESS_ALLOWLIST = frozenset({"localhost", "127.0.0.1", "::1"})


class EgressBlockedError(RuntimeError):
    """Raised when a guarded test attempts a non-allowlisted outbound connect."""


@pytest.fixture
def egress_guard(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail a test that opens a socket to a non-allowlisted host.

    Wraps socket.socket.connect so a connect to anything outside
    EGRESS_ALLOWLIST raises EgressBlockedError instead of touching the network.
    Allowlisted local addresses fall through to the real connect.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: object) -> object:
        host = address[0] if isinstance(address, (tuple, list)) else address
        if host not in EGRESS_ALLOWLIST:
            raise EgressBlockedError(f"blocked outbound connect to {host!r}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    yield


@pytest.fixture(autouse=True)
def _reset_search_metrics() -> Iterator[None]:
    # Counters are cumulative across queries; without a reset per test the
    # known-count assertions would depend on execution order.
    from search_workflow import tools

    tools.METRICS.reset()
    yield


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    """Mirror pytest-recording's default layout under an overridable root."""
    module = request.node.fspath
    root = os.environ.get(CASSETTE_ROOT_ENV) or os.path.join(
        module.dirname, "cassettes"
    )
    return os.path.join(root, module.purebasename)


@pytest.fixture
def vcr_config() -> dict[str, object]:
    """Scrub credential headers before any cassette hits disk."""
    return {
        "filter_headers": [
            ("authorization", HEADER_PLACEHOLDER),
            ("x-api-key", HEADER_PLACEHOLDER),
        ],
    }


class CannedStructuredModel:
    """Stand-in for `chat_model.with_structured_output(ArticlesResponse)`."""

    async def ainvoke(self, value: object, config: object = None) -> ArticlesResponse:
        return ArticlesResponse(
            articles=[
                ArticleStrict(
                    title="Canned offline article",
                    link="https://example.test/canned-article",
                    snippet="Stub snippet returned by the offline OpenAI stand-in.",
                    similarity=0.9,
                )
            ]
        )


class CannedChatModel:
    """Covers the ChatOpenAI surface the agent and evaluator nodes touch."""

    def bind_tools(self, tools: object) -> "CannedChatModel":
        return self

    def with_structured_output(self, schema: object) -> CannedStructuredModel:
        return CannedStructuredModel()

    async def ainvoke(self, value: object, config: object = None) -> AIMessage:
        return AIMessage(content="canned agent reply")


@pytest.fixture(autouse=True)
def stub_openai(monkeypatch: pytest.MonkeyPatch) -> CannedChatModel:
    """Keep every test away from the OpenAI API.

    load_chat_model is patched at its definition and at graph's imported
    reference; the key is deleted so anything that slips past the stub fails
    loudly instead of spending tokens.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = CannedChatModel()
    monkeypatch.setattr(
        "search_workflow.utils.load_chat_model", lambda model_name: model
    )
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda model_name: model
    )
    return model
