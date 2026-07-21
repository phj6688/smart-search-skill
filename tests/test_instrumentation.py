"""Per-query search instrumentation tests (HLB-646).

Every assertion pins the provenance record and counters to whose results were
actually returned (result-source attribution), never to which branch executed:
the searxng_ok_ddg_unused state exists precisely because a branch-keyed record
would claim DDG was used there.
"""

import logging

import pytest

from search_workflow import tools
from tests.fixtures_fallback import FallbackScenario, configure_fallback_state

PROVENANCE_LOGGER = "search_workflow.tools"
PROVENANCE_FIELDS = (
    "n_searxng",
    "n_ddg",
    "n_after_dedup",
    "elapsed_ms",
    "ddg_ok",
    "fell_back",
)


def _provenance_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if hasattr(r, "provenance")]


async def test_fallback_state_provenance(
    fallback_state: FallbackScenario, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger=PROVENANCE_LOGGER):
        results = await tools.search_direct("instrumentation probe")

    records = _provenance_records(caplog)
    assert len(records) == 1, (
        f"expected exactly one provenance record per query, got {len(records)}"
    )
    record = records[0]
    assert record.name.startswith("search_workflow")

    expected = fallback_state.expected
    provenance = record.provenance
    # Raw queries can carry PII; they must never reach the provenance record
    # or the log message.
    assert "query" not in provenance
    assert "instrumentation probe" not in record.getMessage()
    assert provenance["n_searxng"] == expected.n_searxng
    assert provenance["n_ddg"] == expected.n_ddg
    assert provenance["n_after_dedup"] == expected.n_after_dedup
    assert provenance["ddg_ok"] is expected.ddg_ok
    assert provenance["fell_back"] is expected.fell_back
    assert set(provenance["engines_used"]) == set(expected.engines_used)
    assert isinstance(provenance["elapsed_ms"], float)
    assert provenance["elapsed_ms"] >= 0.0

    # The provenance record still counts distinct URLs after dedup; the returned
    # list is that set after the HLB-651 domain cap (at most RRF_DOMAIN_CAP per
    # registrable domain) and truncation. Every fallback URL sits on one
    # registrable domain (example.test), so a post-dedup set above the cap is
    # trimmed to it.
    assert len(results) == min(expected.n_after_dedup, tools.RRF_DOMAIN_CAP)

    message = record.getMessage()
    for field in PROVENANCE_FIELDS:
        assert field in message, f"{field} missing from provenance message"


async def test_ddg_raise_reaches_provenance_as_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A raising DDG backend must surface as ddg_ok=False, not an empty list.

    Patches the DDGS class itself so the real _ddg_search runs: the old broad
    except swallowed the raise and provenance recorded ddg_ok=True.
    """

    async def searxng_ok(
        self: tools.SearXNGClient,
        query: str,
        language: str = "en",
        time_range: str | None = None,
        max_results: int = 10,
        categories: str = "general",
    ) -> list[dict[str, str]]:
        return [{"title": "A", "link": "https://example.test/a", "snippet": "a"}]

    def ddg_down(*args: object, **kwargs: object) -> None:
        raise RuntimeError("ddg unreachable")

    monkeypatch.setattr(tools.SearXNGClient, "search", searxng_ok)
    monkeypatch.setattr("ddgs.DDGS", ddg_down)

    with caplog.at_level(logging.INFO, logger=PROVENANCE_LOGGER):
        results = await tools.search_direct("ddg outage probe")

    provenance = _provenance_records(caplog)[0].provenance
    assert provenance["ddg_ok"] is False
    assert provenance["n_ddg"] == 0
    assert "ddg" not in provenance["engines_used"]
    # The failure never crashes search_direct; SearXNG results still return.
    assert len(results) == 1


async def test_counters_known_count_both_engines_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fallback_state(monkeypatch, "searxng_ok")

    await tools.search_direct("known count probe")

    snapshot = tools.METRICS.snapshot()
    # One query dispatches exactly one outbound request per engine.
    assert snapshot["outbound_search_requests"] == 2
    assert snapshot["llm_calls"] == 0
    assert snapshot["cache_hit"] == 0
    assert snapshot["engines_used"] == {"searxng": 1, "ddg": 1}


async def test_counters_reset_zeroes_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_fallback_state(monkeypatch, "searxng_ok")
    await tools.search_direct("reset probe")

    tools.METRICS.reset()

    snapshot = tools.METRICS.snapshot()
    assert snapshot == {
        "outbound_search_requests": 0,
        "llm_calls": 0,
        "cache_hit": 0,
        "engines_used": {},
    }
