"""Static checks on the live canary workflow definition (HLB-649).

The workflow's push trigger fires on dev-v04, not on feature branches, so a
real run cannot gate this change. These tests pin the definition instead:
triggers, canary steps, hardening, and the promise that ci.yml stays free of
wall-clock assertions.
"""

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "live-canary.yml"
CI_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
COMPOSE_PATH = REPO_ROOT / ".github" / "searxng-compose.yml"
SCRIPT_PATH = REPO_ROOT / "scripts" / "live_canary_checks.py"
CASSETTE_PATH = (
    REPO_ROOT
    / "tests"
    / "cassettes"
    / "test_vcr_replay"
    / "test_searxng_search_replay.yaml"
)

PINNED_USES_RE = re.compile(r"uses:\s*(\S+?)@([0-9a-f]{40})\b")
ANY_USES_RE = re.compile(r"uses:\s*(\S+)")


def _load(path: Path) -> dict:
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict), f"{path} did not parse to a mapping"
    return doc


def _triggers(doc: dict) -> dict:
    # PyYAML resolves a bare `on:` key to boolean True.
    return doc.get("on") or doc[True]


def _steps() -> list[dict]:
    jobs = _load(WORKFLOW_PATH)["jobs"]
    assert len(jobs) == 1, "live canary is a single-job workflow"
    return next(iter(jobs.values()))["steps"]


def test_workflow_parses_and_declares_all_three_triggers() -> None:
    triggers = _triggers(_load(WORKFLOW_PATH))

    assert "workflow_dispatch" in triggers
    assert triggers["push"]["branches"] == ["dev-v04"]

    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert crons, "schedule trigger missing"
    for cron in crons:
        minute = cron.split()[0]
        assert minute.isdigit() and minute != "0", (
            f"cron minute must be off-minute, got {cron!r}"
        )


def test_cassette_staleness_step_runs_the_canary_script() -> None:
    steps = _steps()
    canary = [
        step
        for step in steps
        if "scripts/live_canary_checks.py" in (step.get("run") or "")
    ]
    assert len(canary) == 1, "expected one step running scripts/live_canary_checks.py"
    assert SCRIPT_PATH.is_file()
    assert CASSETTE_PATH.is_file()
    script_text = SCRIPT_PATH.read_text()
    for part in CASSETTE_PATH.relative_to(REPO_ROOT).parts:
        assert part in script_text, f"canary script does not reference {part!r}"

    skipped = [
        step for step in steps if "skipped" in (step.get("name") or "").lower()
    ]
    assert skipped, "expected a skip-notice step for when SearXNG is down"


def test_ddgs_import_path_canary_step() -> None:
    steps = {step.get("name"): step for step in _steps()}
    step = steps.get("ddgs import-path canary")
    assert step is not None, "step named 'ddgs import-path canary' missing"
    run = step["run"]
    assert "from ddgs.exceptions import RatelimitException" in run
    assert "DDGS(" in run


def test_searxng_startup_failure_does_not_fail_the_job() -> None:
    start = [
        step for step in _steps() if (step.get("name") or "").startswith("Start")
    ]
    assert len(start) == 1
    assert "DDG-only" in start[0]["run"]
    assert "GITHUB_OUTPUT" in start[0]["run"]


def test_no_reference_to_the_runtime_compose_tree() -> None:
    assert "docker/" not in WORKFLOW_PATH.read_text()
    assert "docker/" not in COMPOSE_PATH.read_text()


def test_wall_clock_ceiling_lives_in_the_canary_only() -> None:
    jobs = _load(WORKFLOW_PATH)["jobs"]
    job = next(iter(jobs.values()))
    assert isinstance(job.get("timeout-minutes"), int)
    assert job["timeout-minutes"] > 0

    ceiling_steps = [
        step for step in _steps() if "ceiling" in (step.get("name") or "").lower()
    ]
    assert len(ceiling_steps) == 1, "expected one ceiling assert step"
    assert "60" in ceiling_steps[0]["run"]

    ci_text = CI_PATH.read_text()
    assert "timeout-minutes" not in ci_text
    assert "ceiling" not in ci_text
    assert "live_leg_seconds" not in ci_text


def test_hardening_permissions_credentials_and_pins() -> None:
    doc = _load(WORKFLOW_PATH)
    assert doc["permissions"] == {"contents": "read"}

    for step in _steps():
        uses = step.get("uses", "")
        if uses.startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] is False

    workflow_text = WORKFLOW_PATH.read_text()
    all_uses = ANY_USES_RE.findall(workflow_text)
    pinned = dict(PINNED_USES_RE.findall(workflow_text))
    assert len(all_uses) == len(pinned), "every uses: must be pinned to a full SHA"

    ci_pins = dict(PINNED_USES_RE.findall(CI_PATH.read_text()))
    for action, sha in pinned.items():
        assert action in ci_pins, f"{action} is not an action ci.yml uses"
        assert sha == ci_pins[action], f"{action} pin diverges from ci.yml"


def test_throwaway_compose_binds_searxng_to_localhost_9090() -> None:
    compose = _load(COMPOSE_PATH)
    searxng = compose["services"]["searxng"]
    assert searxng["ports"] == ["127.0.0.1:9090:8080"]
    env = searxng["environment"]
    assert "SEARXNG_SECRET" in env
