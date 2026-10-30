"""Offline VCR replay tests (HLB-647).

SearXNG replays through SearXNGClient's aiohttp session. All ddgs backends,
including "duckduckgo", use primp, a Rust HTTP client vcrpy cannot intercept.
The DDG test replaces HttpClient.request with an httpx shim so vcrpy can
replay the existing cassette while keeping the real DuckDuckGo parser.
The recorded body uses gzip so replay needs no optional Brotli decoder.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from ddgs import DDGS
from ddgs.http_client import HttpClient, Response
from vcr.cassette import Cassette

from search_workflow.tools import SearXNGClient
from tests.conftest import CASSETTE_ROOT_ENV

TESTS_DIR = Path(__file__).resolve().parent
SEARXNG_CASSETTE = Path("test_vcr_replay") / "test_searxng_search_replay.yaml"


@pytest.mark.vcr
async def test_searxng_search_replay() -> None:
    client = SearXNGClient(base_url="http://localhost:9090")
    results = await client.search("python programming", max_results=5)

    assert results, "SearXNG replay returned no results"
    for result in results:
        assert set(result) >= {"title", "link", "snippet"}
    assert all(result["link"].startswith("http") for result in results)


@pytest.mark.vcr(match_on=["method", "uri", "body"])
def test_ddg_text_replay(monkeypatch: pytest.MonkeyPatch, vcr: Cassette) -> None:
    def replay_request(self: HttpClient, *args: Any, **kwargs: Any) -> Response:
        # DuckDuckGo only uses status_code and text, not primp's text helpers.
        with httpx.Client(timeout=20, trust_env=False) as client:
            return Response(client.request(*args, **kwargs))

    monkeypatch.setattr(HttpClient, "request", replay_request)
    results = DDGS(timeout=20).text(
        "python programming", max_results=5, backend="duckduckgo"
    )

    assert results, "DDG replay returned no results"
    for result in results:
        assert result["title"]
        assert result["href"].startswith("http")
    assert vcr.play_count == 1, "DDG did not replay the recorded request"


def _run_searxng_replay(cassette_root: Path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    env[CASSETTE_ROOT_ENV] = str(cassette_root)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{TESTS_DIR / 'test_vcr_replay.py'}::test_searxng_search_replay",
            "-q",
            "--block-network",
        ],
        env=env,
        cwd=str(TESTS_DIR.parent),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_searxng_replay_without_cassette_fails_negative(tmp_path: Path) -> None:
    """Deleting the cassette must break the replay test it feeds.

    The control run against an intact copy proves the subprocess setup is
    sound, so the pruned run's failure can only mean the response came from
    the cassette through aiohttp interception, not from a live server.
    """
    cassette_root = tmp_path / "cassettes"
    shutil.copytree(TESTS_DIR / "cassettes", cassette_root)

    control = _run_searxng_replay(cassette_root)
    assert control.returncode == 0, (
        f"control run failed:\n{control.stdout}\n{control.stderr}"
    )

    (cassette_root / SEARXNG_CASSETTE).unlink()
    pruned = _run_searxng_replay(cassette_root)
    assert pruned.returncode != 0, (
        "replay test passed without its cassette; aiohttp interception "
        f"is not real:\n{pruned.stdout}\n{pruned.stderr}"
    )
