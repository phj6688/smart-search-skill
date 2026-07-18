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
    assert provenance["n_searxng"] == expected.n_searxng
    assert provenance["n_ddg"] == expected.n_ddg
    assert provenance["n_after_dedup"] == expected.n_after_dedup
    assert provenance["ddg_ok"] is expected.ddg_ok
    assert provenance["fell_back"] is expected.fell_back
    assert set(provenance["engines_used"]) == set(expected.engines_used)
    assert isinstance(provenance["elapsed_ms"], float)
    assert provenance["elapsed_ms"] >= 0.0

    assert len(results) == expected.n_after_dedup

    message = record.getMessage()
    for field in PROVENANCE_FIELDS:
        assert field in message, f"{field} missing from provenance message"


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
