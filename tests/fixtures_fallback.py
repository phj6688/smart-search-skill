"""Five-state fallback fixture for search instrumentation tests (HLB-646).

Later stories import `fallback_state` (re-exported from tests/conftest.py)
instead of re-creating engine mocks. Each state stubs the exact seam
search_direct dispatches to (SearXNGClient.search and _ddg_search), so
assertions exercise result-source attribution rather than control flow:
the fallback can fire on a raise OR an empty response, and the record must
describe whose results were actually returned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from search_workflow import tools


class EngineDownError(RuntimeError):
    """Raised by mock engines to simulate an unreachable backend."""


@dataclass(frozen=True)
class FallbackExpectation:
    """Expected provenance-record values for one fixture state."""

    n_searxng: int
    n_ddg: int
    n_after_dedup: int
    ddg_ok: bool
    fell_back: bool
    engines_used: frozenset[str]


@dataclass(frozen=True)
class FallbackScenario:
    """What a test receives: the state name plus its expected provenance."""

    name: str
    expected: FallbackExpectation


_SEARXNG_A = {"title": "SearXNG A", "link": "https://example.test/a", "snippet": "a"}
_SEARXNG_B = {"title": "SearXNG B", "link": "https://example.test/b", "snippet": "b"}
_DDG_A = {"title": "DDG A", "link": "https://example.test/a", "snippet": "a via ddg"}
_DDG_B = {"title": "DDG B", "link": "https://example.test/b", "snippet": "b via ddg"}
_DDG_C = {"title": "DDG C", "link": "https://example.test/c", "snippet": "c via ddg"}


def _raise_engine_down() -> list[dict[str, str]]:
    raise EngineDownError("engine unreachable")


def _copies(*results: dict[str, str]) -> Callable[[], list[dict[str, str]]]:
    # Fresh dicts per call: search_direct callers may mutate results, and a
    # shared literal would leak state between parametrized runs.
    return lambda: [dict(r) for r in results]


# state name -> (searxng behavior, ddg behavior); each callable may raise.
_STATES: dict[
    str,
    tuple[Callable[[], list[dict[str, str]]], Callable[[], list[dict[str, str]]]],
] = {
    # Both engines answer; DDG's b duplicates SearXNG's b, DDG's c survives.
    "searxng_ok": (_copies(_SEARXNG_A, _SEARXNG_B), _copies(_DDG_B, _DDG_C)),
    # SearXNG raises; every returned result is DDG's.
    "searxng_raises": (_raise_engine_down, _copies(_DDG_B, _DDG_C)),
    # SearXNG answers with zero results; same attribution as a raise.
    "searxng_empty": (_copies(), _copies(_DDG_B, _DDG_C)),
    # DDG answers fine but every DDG result is deduplicated away, so a record
    # keyed to the executed branch would claim DDG was used; attribution says no.
    "searxng_ok_ddg_unused": (
        _copies(_SEARXNG_A, _SEARXNG_B),
        _copies(_DDG_A, _DDG_B),
    ),
    # Both engines raise; nothing returned, so no fallback actually happened.
    "both_fail": (_raise_engine_down, _raise_engine_down),
}

EXPECTATIONS: dict[str, FallbackExpectation] = {
    "searxng_ok": FallbackExpectation(
        n_searxng=2,
        n_ddg=2,
        n_after_dedup=3,
        ddg_ok=True,
        fell_back=False,
        engines_used=frozenset({"searxng", "ddg"}),
    ),
    "searxng_raises": FallbackExpectation(
        n_searxng=0,
        n_ddg=2,
        n_after_dedup=2,
        ddg_ok=True,
        fell_back=True,
        engines_used=frozenset({"ddg"}),
    ),
    "searxng_empty": FallbackExpectation(
        n_searxng=0,
        n_ddg=2,
        n_after_dedup=2,
        ddg_ok=True,
        fell_back=True,
        engines_used=frozenset({"ddg"}),
    ),
    "searxng_ok_ddg_unused": FallbackExpectation(
        n_searxng=2,
        n_ddg=2,
        n_after_dedup=2,
        ddg_ok=True,
        fell_back=False,
        engines_used=frozenset({"searxng"}),
    ),
    "both_fail": FallbackExpectation(
        n_searxng=0,
        n_ddg=0,
        n_after_dedup=0,
        ddg_ok=False,
        fell_back=False,
        engines_used=frozenset(),
    ),
}

FALLBACK_STATE_NAMES: tuple[str, ...] = tuple(_STATES)


def configure_fallback_state(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> FallbackScenario:
    """Stub both engines at the SearXNGClient/_ddg_search boundary for `state`."""
    searxng_behavior, ddg_behavior = _STATES[state]

    async def fake_searxng_search(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, str]]:
        return searxng_behavior()

    async def fake_ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
        return ddg_behavior()

    monkeypatch.setattr(tools.SearXNGClient, "search", fake_searxng_search)
    monkeypatch.setattr(tools, "_ddg_search", fake_ddg_search)
    return FallbackScenario(name=state, expected=EXPECTATIONS[state])


@pytest.fixture(params=FALLBACK_STATE_NAMES)
def fallback_state(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> FallbackScenario:
    """Parametrized five-state engine fixture; yields one scenario per state."""
    return configure_fallback_state(monkeypatch, request.param)
