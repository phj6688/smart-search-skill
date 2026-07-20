"""Cassette hygiene scans (HLB-647).

Both tests read interaction fields from the parsed YAML rather than
text-matching whole files: a placeholder appearing in a response body must
not mask a leaked header, and "api.openai.com" inside recorded HTML must not
false-positive the host check.
"""

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from tests.conftest import HEADER_PLACEHOLDER

CASSETTE_ROOT = Path(__file__).resolve().parent / "cassettes"
SCRUBBED_HEADERS = {"authorization", "x-api-key"}
SECRET_KEY_RE = re.compile(r"sk-[A-Za-z0-9_-]{8,}")


def _cassette_paths() -> list[Path]:
    paths = sorted(CASSETTE_ROOT.rglob("*.yaml"))
    assert paths, f"no cassettes found under {CASSETTE_ROOT}"
    return paths


def _interactions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, dict), f"{path} is not a vcrpy cassette"
    interactions = data.get("interactions") or []
    assert interactions, f"{path} contains no interactions"
    return interactions


def _header_values(headers: dict | None) -> Iterator[tuple[str, str]]:
    for name, values in (headers or {}).items():
        if isinstance(values, (str, bytes)):
            values = [values]
        for value in values:
            yield name, value.decode() if isinstance(value, bytes) else str(value)


def test_cassettes_contain_no_real_credential_headers() -> None:
    for path in _cassette_paths():
        for interaction in _interactions(path):
            for side in ("request", "response"):
                headers = interaction.get(side, {}).get("headers", {})
                for name, value in _header_values(headers):
                    if name.lower() in SCRUBBED_HEADERS:
                        assert value == HEADER_PLACEHOLDER, (
                            f"{path}: {side} header {name} holds a real value"
                        )
                    assert not value.lower().startswith("bearer "), (
                        f"{path}: bearer token in {side} header {name}"
                    )
                    assert not SECRET_KEY_RE.search(value), (
                        f"{path}: sk- key in {side} header {name}"
                    )


def test_no_cassette_interaction_targets_openai() -> None:
    for path in _cassette_paths():
        for interaction in _interactions(path):
            uri = interaction["request"]["uri"]
            host = urlsplit(uri).hostname or ""
            assert host != "api.openai.com", (
                f"{path} recorded OpenAI traffic: {uri}"
            )
