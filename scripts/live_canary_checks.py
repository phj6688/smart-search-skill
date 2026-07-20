"""Cassette-staleness canary for .github/workflows/live-canary.yml (HLB-649).

Fetches a live SearXNG /search response and compares its shape, the keys of
the entries under "results", against the recorded replay cassette. The
offline suite replays that cassette forever, so shape drift shows up here
first instead of as a silently stale fixture. Canary only, never a merge
gate.

Exit codes: 0 = shape matches, or SearXNG unreachable (skipped with a
notice), or live results empty (engine flakiness, warned). 1 = shape drift.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASSETTE = (
    REPO_ROOT
    / "tests"
    / "cassettes"
    / "test_vcr_replay"
    / "test_searxng_search_replay.yaml"
)
# Same query the cassette was recorded with, so the comparison is like for like.
QUERY = "python programming"


def entry_key_floor(entries: list[dict]) -> set[str]:
    """Keys present on every entry: the shape consumers can rely on."""
    common = set(entries[0])
    for entry in entries[1:]:
        common &= set(entry)
    return common


def cassette_shape(cassette: Path) -> set[str]:
    doc = yaml.safe_load(cassette.read_text())
    body = doc["interactions"][0]["response"]["body"]["string"]
    results = json.loads(body)["results"]
    if not results:
        raise SystemExit(f"cassette {cassette} holds no results entries")
    return entry_key_floor(results)


def live_search(base_url: str, timeout: float) -> tuple[list[dict], float]:
    params = urllib.parse.urlencode(
        {
            "q": QUERY,
            "categories": "general",
            "language": "en",
            "format": "json",
            "safesearch": "0",
        }
    )
    url = f"{base_url.rstrip('/')}/search?{params}"
    start = time.monotonic()
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed = time.monotonic() - start
    return payload.get("results", []), elapsed


def record_duration(seconds: float) -> None:
    """Expose the live-leg duration so a later workflow step can assert the
    wall-clock ceiling without re-running the search."""
    line = f"live_leg_seconds={seconds:.2f}"
    print(line)
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("SEARXNG_URL", "http://127.0.0.1:9090"),
        help="SearXNG instance to probe (default: SEARXNG_URL or 127.0.0.1:9090)",
    )
    parser.add_argument("--cassette", type=Path, default=DEFAULT_CASSETTE)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    expected = cassette_shape(args.cassette)

    try:
        results, elapsed = live_search(args.base_url, args.timeout)
    except (urllib.error.URLError, OSError) as exc:
        print(
            f"::notice::SearXNG unreachable at {args.base_url} ({exc}); "
            "shape check skipped"
        )
        return 0

    record_duration(elapsed)

    if not results:
        print(
            "::warning::live SearXNG returned zero results "
            "(engine flakiness, not shape drift); shape check skipped"
        )
        return 0

    live = entry_key_floor(results)
    added = sorted(live - expected)
    missing = sorted(expected - live)
    if added:
        print(f"::notice::live entries carry keys the cassette floor lacks: {added}")
    if missing:
        print(
            "::error::cassette shape drift: keys on every recorded entry "
            f"but no longer on every live entry: {missing}"
        )
        return 1

    print(
        f"shape check passed: {len(expected)} cassette floor keys present "
        f"on every live entry ({len(results)} results, {elapsed:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
