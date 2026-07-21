"""Retry with backoff on both LLM call sites (HLB-660).

One helper, ainvoke_with_retry, wraps model.ainvoke at the agent and evaluator
nodes. It retries ONLY rate-limit (429) and timeout errors, 2-3 attempts total,
with exponential backoff plus jitter, and honors a Retry-After header. Auth and
request-validation errors raise immediately. A ValueError/ValidationError from
the structured-output call is NOT retried, so HLB-658's degraded fallback still
fires.

Every model is patched through the graph.load_chat_model seam (the conftest
autouse stub patches it; these tests re-patch it in the test body, which wins).
asyncio.sleep is patched so backoff waits are recorded, not slept.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from search_workflow import graph, tools
from search_workflow.retry import DEFAULT_MAX_ATTEMPTS, ainvoke_with_retry
from search_workflow.utils import SelectionResponse


def _rate_limit_error(retry_after: str | None = None) -> openai.RateLimitError:
    """A real openai.RateLimitError (status 429), optionally with Retry-After."""
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(429, headers=headers, request=request)
    return openai.RateLimitError("rate limited", response=response, body=None)


def _auth_error() -> openai.AuthenticationError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(401, request=request)
    return openai.AuthenticationError("bad key", response=response, body=None)


class _CountingModel:
    """Records ainvoke calls, then raises `exc` or returns `result`."""

    def __init__(self, exc: BaseException | None = None, result: Any = None) -> None:
        self.calls = 0
        self._exc = exc
        self._result = result

    async def ainvoke(self, value: Any, config: Any = None) -> Any:
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._result


class _FailThenSucceedModel:
    """Raises `exc` for the first `failures` calls, then returns `result`."""

    def __init__(self, exc: BaseException, failures: int, result: Any) -> None:
        self.calls = 0
        self._exc = exc
        self._failures = failures
        self._result = result

    async def ainvoke(self, value: Any, config: Any = None) -> Any:
        self.calls += 1
        if self.calls <= self._failures:
            raise self._exc
        return self._result


@pytest.fixture
def recorded_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Patch asyncio.sleep to record waits instead of sleeping."""
    waits: list[float] = []

    async def fake_sleep(delay: float) -> None:
        waits.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return waits


async def test_rate_limit_every_attempt_exhausts_then_raises(
    recorded_sleeps: list[float],
) -> None:
    model = _CountingModel(exc=_rate_limit_error())

    with pytest.raises(openai.RateLimitError):
        await ainvoke_with_retry(model, "prompt")

    # 2-3 attempts total, then the last error surfaces.
    assert model.calls == DEFAULT_MAX_ATTEMPTS
    assert 2 <= model.calls <= 3
    # One backoff wait between each pair of attempts.
    assert len(recorded_sleeps) == DEFAULT_MAX_ATTEMPTS - 1


async def test_authentication_error_is_not_retried(
    recorded_sleeps: list[float],
) -> None:
    model = _CountingModel(exc=_auth_error())

    with pytest.raises(openai.AuthenticationError):
        await ainvoke_with_retry(model, "prompt")

    # Auth errors raise immediately: invoked exactly once, no backoff wait.
    assert model.calls == 1
    assert recorded_sleeps == []


async def test_retry_after_header_sets_the_wait(
    recorded_sleeps: list[float],
) -> None:
    # Raise a 429 carrying Retry-After: 1 once, then succeed.
    model = _FailThenSucceedModel(
        exc=_rate_limit_error(retry_after="1"), failures=1, result="ok"
    )

    result = await ainvoke_with_retry(model, "prompt")

    assert result == "ok"
    assert model.calls == 2
    # The wait honored the header rather than the shorter default backoff base.
    assert recorded_sleeps
    assert recorded_sleeps[0] >= 1


async def test_timeout_error_is_retried(recorded_sleeps: list[float]) -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    timeout = openai.APITimeoutError(request=request)
    model = _FailThenSucceedModel(exc=timeout, failures=1, result="ok")

    result = await ainvoke_with_retry(model, "prompt")

    assert result == "ok"
    assert model.calls == 2


# --- End-to-end through run_workflow -----------------------------------------

# A small raw corpus so the tool step yields results and the evaluator selects.
_RAW: list[dict[str, str]] = [
    {"title": "Alpha result", "link": "https://alpha.example/one", "snippet": "s1"},
    {"title": "Bravo result", "link": "https://bravo.example/two", "snippet": "s2"},
]


class _AgentBound:
    """Agent step: emit a `search` tool call so the graph reaches the tools node."""

    async def ainvoke(self, value: Any, config: Any = None) -> AIMessage:
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {
                        "query": "retry probe",
                        "region": "us-en",
                        "timelimit": None,
                    },
                    "id": "call_1",
                }
            ],
        )


class _FlakyAgentBound:
    """Agent step that raises a 429 once, then emits the tool call."""

    def __init__(self, parent: _FlakyAgentModel) -> None:
        self._parent = parent
        self._delegate = _AgentBound()

    async def ainvoke(self, value: Any, config: Any = None) -> AIMessage:
        self._parent.agent_calls += 1
        if self._parent.agent_calls == 1:
            raise _rate_limit_error()
        return await self._delegate.ainvoke(value, config)


class _HealthySelector:
    async def ainvoke(self, value: Any, config: Any = None) -> SelectionResponse:
        return SelectionResponse(selected=[0])


class _RaisingSelector:
    """with_structured_output(...).ainvoke raising a malformed-output ValueError."""

    async def ainvoke(self, value: Any, config: Any = None) -> Any:
        raise ValueError("could not parse structured output")


class _FlakyAgentModel:
    """Agent LLM raises 429 once then succeeds; evaluator selects healthily."""

    def __init__(self) -> None:
        self.agent_calls = 0

    def bind_tools(self, tools_: object) -> _FlakyAgentBound:
        return _FlakyAgentBound(self)

    def with_structured_output(self, schema: object) -> _HealthySelector:
        return _HealthySelector()


class _MalformedEvaluatorModel:
    """Agent succeeds; evaluator's structured output raises ValueError (658)."""

    def bind_tools(self, tools_: object) -> _AgentBound:
        return _AgentBound()

    def with_structured_output(self, schema: object) -> _RaisingSelector:
        return _RaisingSelector()


def _wire_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub both engines so the tool step yields results without the network."""

    async def fake_searxng_search(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, str]]:
        return [dict(_RAW[0])]

    async def fake_ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
        return [dict(_RAW[1])]

    monkeypatch.setattr(tools.SearXNGClient, "search", fake_searxng_search)
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg_search)


async def test_end_to_end_one_retry_then_completes(
    monkeypatch: pytest.MonkeyPatch, recorded_sleeps: list[float]
) -> None:
    model = _FlakyAgentModel()
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda name, **kwargs: model
    )
    _wire_engines(monkeypatch)

    out = await graph.run_workflow("retry probe")

    # Exactly one retry: the agent LLM was invoked twice (fail once, succeed).
    assert model.agent_calls == 2
    assert out["status"] == "ok"
    assert out["results"] == [_RAW[0]]


async def test_malformed_structured_output_not_retried_still_degrades(
    monkeypatch: pytest.MonkeyPatch, recorded_sleeps: list[float]
) -> None:
    """A ValueError from the selection is not retried and reaches 658's fallback."""
    model = _MalformedEvaluatorModel()
    monkeypatch.setattr(
        "search_workflow.graph.load_chat_model", lambda name, **kwargs: model
    )
    _wire_engines(monkeypatch)

    out = await graph.run_workflow("retry probe")

    # No backoff wait was ever scheduled: the malformed-output error is not
    # retried (attempt count 1 at the evaluator call site).
    assert recorded_sleeps == []
    # 658's degraded fallback still fires with the raw fetched results.
    assert out["status"] == "ok"
    assert out["degraded"] is True
    assert out["degraded_reason"] == "evaluator"
    assert out["results"] == _RAW
