"""Shared fixtures.

`fallback_state` lives in tests/fixtures_fallback.py and is re-exported here
so every test module gets it without importing the mocks directly. Later
stories use this fixture; they never re-create engine stubs.
"""

from collections.abc import Iterator

import pytest

from tests.fixtures_fallback import (  # noqa: F401
    FALLBACK_STATE_NAMES,
    FallbackExpectation,
    FallbackScenario,
    configure_fallback_state,
    fallback_state,
)


@pytest.fixture(autouse=True)
def _reset_search_metrics() -> Iterator[None]:
    # Counters are cumulative across queries; without a reset per test the
    # known-count assertions would depend on execution order.
    from search_workflow import tools

    tools.METRICS.reset()
    yield
