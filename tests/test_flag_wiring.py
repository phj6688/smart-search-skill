"""--region / --timelimit / safesearch reach the engine request params (HLB-663).

The CLI parsed --region and --timelimit but wired only max_results into the run
config, so region/timelimit were dropped and SearXNG's safesearch was hardcoded
to "0". These tests pin the wired path end to end at the engine seam, offline:

  * the emitted SearXNG request carries the region-derived language, the mapped
    time_range (d/w/m/y -> day/week/month/year), and the configured safesearch;
  * the DDG call (DDGS().text) receives region and timelimit;
  * with no flags set the SearXNG request carries safesearch=0 and NO time_range;
  * Configuration.safesearch accepts 0/1/2 and rejects anything else;
  * `search-workflow --help` lists --safesearch.

Everything is patched at the HTTP / DDGS boundary, so no network is touched.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import pytest

from search_workflow import cli, tools
from search_workflow.configuration import Configuration


class _FakeSearxngResponse:
    """Async-context response stand-in for aiohttp's session.get(...)."""

    status = 200

    async def json(self) -> dict[str, Any]:
        return {"results": [{"url": "https://searxng.test/x", "title": "x", "content": "c"}]}

    async def __aenter__(self) -> _FakeSearxngResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _patch_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Capture the SearXNG request params and the DDGS().text kwargs.

    Returns (searxng_params, ddg_calls); both are appended to as the tool runs.
    """
    searxng_params: list[dict[str, Any]] = []
    ddg_calls: list[dict[str, Any]] = []

    def fake_get(
        self: aiohttp.ClientSession, url: str, *args: Any, **kwargs: Any
    ) -> _FakeSearxngResponse:
        searxng_params.append(dict(kwargs.get("params") or {}))
        return _FakeSearxngResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "get", fake_get)

    class _FakeDDGS:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        def text(self, query: str, **kwargs: Any) -> list[dict[str, str]]:
            ddg_calls.append({"query": query, **kwargs})
            return []

    # _ddg_search does `from ddgs import DDGS` at call time, so patch the source
    # module, not a tools-level alias.
    monkeypatch.setattr("ddgs.DDGS", _FakeDDGS)
    return searxng_params, ddg_calls


async def test_region_timelimit_safesearch_reach_both_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searxng_params, ddg_calls = _patch_engines(monkeypatch)

    # region/timelimit/safesearch carried on the run config exactly as the CLI
    # main wires args.region / args.timelimit / args.safesearch.
    config = {
        "configurable": {
            "region": "de-de",
            "timelimit": "w",
            "safesearch": 1,
        }
    }
    await tools.search("berlin news", "us-en", None, config=config)

    assert len(searxng_params) == 1, searxng_params
    params = searxng_params[0]
    # region -> language via the existing region-to-language mapping (de-de -> de).
    assert params["language"] == "de"
    # timelimit 'w' -> SearXNG time_range 'week'.
    assert params["time_range"] == "week"
    # configured safesearch replaces the old hardcoded "0".
    assert params["safesearch"] == "1"

    assert len(ddg_calls) == 1, ddg_calls
    ddg = ddg_calls[0]
    # The DDG call receives the ddgs-native region/timelimit codes verbatim.
    assert ddg["region"] == "de-de"
    assert ddg["timelimit"] == "w"


async def test_default_no_flags_safesearch_zero_and_no_time_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    searxng_params, _ddg_calls = _patch_engines(monkeypatch)

    # No configurable region/timelimit/safesearch: behavior must match the old
    # hardcoded default (safesearch off, no time filter).
    await tools.search("berlin news", "us-en", None, config={})

    assert len(searxng_params) == 1, searxng_params
    params = searxng_params[0]
    assert params["safesearch"] == "0"
    assert "time_range" not in params


def test_configuration_safesearch_accepts_0_1_2() -> None:
    for level in (0, 1, 2):
        assert Configuration(safesearch=level).safesearch == level


def test_configuration_safesearch_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        Configuration(safesearch=3)


def test_help_lists_safesearch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["search-workflow", "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    assert "--safesearch" in capsys.readouterr().out
