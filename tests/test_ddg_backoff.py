"""DuckDuckGo rate-limit backoff in the shared core (HLB-662).

`_ddg_search` retries a DDG RatelimitException with capped exponential backoff
plus full jitter, at the async layer (await asyncio.sleep between a FRESH
executor submission per attempt), and gives up with an empty list after the
attempt budget. Non-ratelimit exceptions are left to propagate unchanged.

Reconciliation note: the issue text assumed the pre-existing behavior was a
broad `except Exception` that printed and returned []. dev-v04 (HLB-657) had
already replaced that with propagation so `_fetch_and_merge` records
ddg_ok=False for a hard outage (see test_instrumentation.py, which is in the
do-not-regress set). "Preserve current behavior for non-ratelimit exceptions"
therefore means propagate, and the non-ratelimit test below pins propagation
rather than a swallowed []. Only the ratelimit path is retried and only it
returns [] on exhaustion.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from search_workflow import tools

try:
    from ddgs.exceptions import RatelimitException
except ImportError:  # older package name
    from duckduckgo_search.exceptions import RatelimitException


def _install_fake_ddgs(
    monkeypatch: pytest.MonkeyPatch,
    actions: list[object],
    events: list[tuple[str, object]] | None = None,
) -> list[str]:
    """Install a fake ddgs.DDGS whose .text() consumes `actions` in order.

    Each action is "ratelimit" (raise RatelimitException), "error" (raise a
    plain RuntimeError), or a list of raw {title,href,body} dicts to return.
    Once the actions run out, the last action repeats. Returns the list of
    per-call query strings so a caller can assert the exact .text() call count;
    when `events` is given, every call also appends ("text", query) so ordering
    against asyncio.sleep can be checked.
    """
    calls: list[str] = []
    seq = list(actions)

    import ddgs

    class _FakeDDGS:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def text(self, query: str, max_results: int = 10) -> list[dict[str, str]]:
            calls.append(query)
            if events is not None:
                events.append(("text", query))
            action = seq[len(calls) - 1] if len(calls) <= len(seq) else seq[-1]
            if action == "ratelimit":
                raise RatelimitException("throttled")
            if action == "error":
                raise RuntimeError("hard failure")
            return [dict(r) for r in action]  # type: ignore[union-attr]

    monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)
    return calls


_RAW = [
    {"title": "One", "href": "https://example.test/1", "body": "first"},
    {"title": "Two", "href": "https://example.test/2", "body": "second"},
]


async def test_ddg_backoff_ratelimit_once_then_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single RatelimitException triggers exactly one retry then succeeds."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep())
    calls = _install_fake_ddgs(monkeypatch, ["ratelimit", _RAW])

    result = await tools._ddg_search("q", max_results=10)

    assert len(calls) == 2, "one ratelimit should cause exactly one retry"
    assert result, "second attempt returned results, so result must be non-empty"
    assert set(result[0]) == {"title", "link", "snippet"}


async def test_ddg_backoff_ratelimit_every_attempt_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ratelimit on every attempt exhausts the budget and returns []."""
    monkeypatch.setattr(asyncio, "sleep", _noop_sleep())
    calls = _install_fake_ddgs(
        monkeypatch, ["ratelimit", "ratelimit", "ratelimit", "ratelimit"]
    )

    result = await tools._ddg_search("q", max_results=10)

    assert len(calls) == tools._DDG_RETRY_ATTEMPTS == 3
    assert result == []


async def test_ddg_backoff_non_ratelimit_exception_propagates_no_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-ratelimit error propagates on the first call with no retry.

    Propagation (not a swallowed []) is what keeps _fetch_and_merge recording
    ddg_ok=False for a hard DDG outage; see the module docstring.
    """
    slept: list[float] = []
    monkeypatch.setattr(asyncio, "sleep", _recording_sleep(slept))
    calls = _install_fake_ddgs(monkeypatch, ["error"])

    with pytest.raises(RuntimeError):
        await tools._ddg_search("q", max_results=10)

    assert len(calls) == 1, "non-ratelimit error must not retry"
    assert slept == [], "no backoff sleep on a non-ratelimit error"


async def test_ddg_backoff_sleep_is_awaited_between_executor_submissions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """asyncio.sleep is awaited BETWEEN executor submissions, each delay in [0,4]."""
    events: list[tuple[str, object]] = []
    delays: list[float] = []

    async def recording_sleep(delay: float) -> None:
        events.append(("sleep", delay))
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", recording_sleep)
    _install_fake_ddgs(
        monkeypatch, ["ratelimit", "ratelimit", _RAW], events=events
    )

    result = await tools._ddg_search("q", max_results=10)

    kinds = [kind for kind, _ in events]
    # Two ratelimited attempts then success: text, sleep, text, sleep, text.
    assert kinds == ["text", "sleep", "text", "sleep", "text"]
    assert len(delays) == 2
    for delay in delays:
        assert 0.0 <= delay <= 4.0
    assert result, "third attempt succeeded"


def test_ddg_backoff_no_time_sleep_in_tools_source() -> None:
    """The retry loop must never block the executor thread with time.sleep."""
    source = Path(tools.__file__).read_text()
    assert re.search(r"time\.sleep", source) is None


def test_ddg_backoff_import_path_resolves() -> None:
    """RatelimitException resolves from one of the two installed packages."""
    try:
        from ddgs.exceptions import RatelimitException as Exc
    except ImportError:
        from duckduckgo_search.exceptions import RatelimitException as Exc
    assert issubclass(Exc, Exception)


def test_ddg_backoff_worst_case_under_engine_deadline() -> None:
    """Worst-case cumulative backoff (derived from the constants) < deadline.

    A 3-attempt run has two inter-attempt waits; each is at most its jitter
    ceiling min(cap, base*2**i). Summing those ceilings is the worst case a
    full-jitter run can reach, and it must stay under engine_deadline_s.
    """
    base = tools._DDG_BACKOFF_BASE_S
    cap = tools._DDG_BACKOFF_CAP_S
    attempts = tools._DDG_RETRY_ATTEMPTS

    worst_case = sum(min(cap, base * 2**i) for i in range(attempts - 1))

    assert worst_case == 1.5
    assert worst_case < tools.default_config.engine_deadline_s


def _noop_sleep():
    async def _sleep(delay: float) -> None:
        return None

    return _sleep


def _recording_sleep(sink: list[float]):
    async def _sleep(delay: float) -> None:
        sink.append(delay)

    return _sleep
