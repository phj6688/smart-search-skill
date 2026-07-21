"""Explicit-default tests for the SearXNG timeout + per-engine deadline knobs.

HLB-653 routes Configuration.searxng_url and Configuration.searxng_timeout into
the tool-path SearXNG client (effective default 30s) and bounds each engine leg
with Configuration.engine_deadline_s. These tests pin every new default and
prove a hung SearXNG leg yields the DDG fallback within the deadline rather than
stalling for the full request timeout.
"""

import asyncio
import time

import pytest

from search_workflow import tools
from search_workflow.configuration import Configuration


def test_engine_deadline_default_between_4_and_4_5() -> None:
    # A per-engine deadline shorter than the 30s request timeout is the whole
    # point: it caps p99 when one engine hangs. Bounds are the issue contract.
    deadline = Configuration().engine_deadline_s
    assert isinstance(deadline, float)
    assert 4.0 <= deadline <= 4.5


def test_searxng_timeout_default_is_30() -> None:
    # The routed request timeout default; the old class default was 12.
    assert Configuration().searxng_timeout == 30


def test_searxng_url_env_flows_through_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # SEARXNG_URL is honored via Configuration's default_factory, so a fresh
    # Configuration built after the env is set picks it up.
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.internal:8888")
    assert Configuration().searxng_url == "http://searxng.internal:8888"


async def test_search_direct_routes_configuration_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # With no explicit timeout/url, search_direct must build the client from
    # Configuration's defaults (timeout 30), not the old hardcoded 12.
    seen: dict[str, object] = {}

    class SpyClient(tools.SearXNGClient):
        def __init__(
            self, base_url: str = "http://localhost:9090", timeout: int = 12
        ) -> None:
            super().__init__(base_url, timeout)
            seen["timeout"] = self.timeout
            seen["base_url"] = self.base_url

        async def search(self, *args: object, **kwargs: object) -> list[dict]:
            return []

    monkeypatch.setattr(tools, "SearXNGClient", SpyClient)

    async def no_ddg(query: str, max_results: int) -> list[dict]:
        return []

    monkeypatch.setattr(tools, "_ddg_search", no_ddg)

    await tools.search_direct("q")

    assert seen["timeout"] == 30
    assert seen["base_url"] == Configuration().searxng_url


async def test_hung_searxng_leg_falls_back_to_ddg_within_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A SearXNG leg that sleeps far past the deadline must not stall the query.
    # The wait_for wrapper cancels it, gather absorbs the timeout, and the DDG
    # results are returned within deadline + a small margin.
    deadline = 0.2
    sleep_s = 5.0

    async def hung_searxng(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict]:
        await asyncio.sleep(sleep_s)
        return [{"title": "late", "link": "https://searxng.test/late", "snippet": ""}]

    async def fast_ddg(query: str, max_results: int) -> list[dict]:
        return [{"title": "ddg", "link": "https://ddg.test/x", "snippet": ""}]

    monkeypatch.setattr(tools.SearXNGClient, "search", hung_searxng)
    monkeypatch.setattr(tools, "_ddg_search", fast_ddg)

    started = time.monotonic()
    results = await tools.search_direct("q", engine_deadline_s=deadline)
    elapsed = time.monotonic() - started

    assert elapsed < deadline + 0.5, f"hung leg stalled the query: {elapsed:.3f}s"
    assert [r["link"] for r in results] == ["https://ddg.test/x"]
