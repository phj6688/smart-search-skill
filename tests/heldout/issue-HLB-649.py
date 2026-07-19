"""Held-out behavioural probe for HLB-649: scheduled live integration canary workflow.

Acceptance criteria checked (statically, against the workflow definition):
1. .github/workflows/live-canary.yml exists, parses as YAML, and its `on`
   triggers include schedule (with a cron entry), workflow_dispatch, and
   push scoped to the dev-v04 branch.
2. The workflow carries a cassette-staleness canary (raw text mentions
   "cassette") and a named ddgs import-path canary (raw text mentions "ddgs").
3. A wall-clock ceiling is expressed in live-canary.yml only; ci.yml carries
   no wall-clock assertion ("wall" absent from ci.yml).
4. The workflow never touches docker/: the string "docker/" is absent.
5. Any compose file the workflow references lives under .github/.
6. Hardening: workflow or job permissions grant contents: read, or at
   minimum the workflow does not use pull_request_target.

The probe never executes the workflow; the live run is validated by the
pipeline, not here.
"""
import pathlib
import re

import pytest
import yaml

WORKFLOW_PATH = pathlib.Path(".github/workflows/live-canary.yml")
CI_PATH = pathlib.Path(".github/workflows/ci.yml")


@pytest.fixture(scope="module")
def canary_text():
    assert WORKFLOW_PATH.is_file(), f"{WORKFLOW_PATH} must exist"
    return WORKFLOW_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def canary_doc(canary_text):
    doc = yaml.safe_load(canary_text)
    assert isinstance(doc, dict), "live-canary.yml must parse to a YAML mapping"
    return doc


def _on_mapping(doc):
    # PyYAML parses the bare key `on` as boolean True; accept both spellings.
    on = doc.get(True, doc.get("on"))
    assert isinstance(on, dict), "workflow `on` must be a mapping of triggers"
    return on


def test_triggers_schedule_dispatch_and_push_dev_v04(canary_doc):
    on = _on_mapping(canary_doc)
    assert "schedule" in on, "missing schedule trigger"
    schedule = on["schedule"]
    assert isinstance(schedule, list) and any(
        isinstance(entry, dict) and "cron" in entry for entry in schedule
    ), "schedule trigger must contain at least one cron entry"
    assert "workflow_dispatch" in on, "missing workflow_dispatch trigger"
    assert "push" in on, "missing push trigger"
    push = on["push"] or {}
    branches = push.get("branches", []) if isinstance(push, dict) else []
    if isinstance(branches, str):
        branches = [branches]
    assert "dev-v04" in branches, "push trigger must be scoped to dev-v04"


def test_cassette_staleness_and_ddgs_canary_steps_present(canary_text):
    lowered = canary_text.lower()
    assert "cassette" in lowered, "cassette-staleness canary step missing"
    assert "ddgs" in lowered, "ddgs import-path canary step missing"


def test_wall_clock_ceiling_only_in_canary_workflow(canary_text):
    has_timeout = "timeout-minutes" in canary_text
    has_marker = re.search(r"wall|ceiling|canary", canary_text, re.IGNORECASE)
    has_duration_assert = re.search(r"elapsed|duration|SECONDS", canary_text)
    assert (has_timeout and has_marker) or has_duration_assert, (
        "live-canary.yml must enforce a wall-clock ceiling "
        "(timeout-minutes with a wall/ceiling/canary marker, or an explicit "
        "duration assertion step)"
    )
    if CI_PATH.is_file():
        ci_text = CI_PATH.read_text(encoding="utf-8").lower()
        assert "wall" not in ci_text, "ci.yml must carry no wall-clock assertion"


def test_never_touches_docker_directory(canary_text):
    assert "docker/" not in canary_text, "live-canary.yml must not reference docker/"


def test_referenced_compose_files_live_under_dot_github(canary_text):
    compose_refs = re.findall(
        r"[A-Za-z0-9_./-]*compose[A-Za-z0-9_./-]*\.ya?ml", canary_text
    )
    for ref in compose_refs:
        assert ref.startswith(".github/"), (
            f"compose file {ref!r} must live under .github/"
        )


def test_hardened_permissions_or_no_pull_request_target(canary_doc, canary_text):
    perms = canary_doc.get("permissions") or {}
    job_perms = [
        job.get("permissions") or {}
        for job in (canary_doc.get("jobs") or {}).values()
        if isinstance(job, dict)
    ]
    contents_read = (
        isinstance(perms, dict) and perms.get("contents") == "read"
    ) or any(
        isinstance(p, dict) and p.get("contents") == "read" for p in job_perms
    )
    assert contents_read or "pull_request_target" not in canary_text, (
        "workflow must grant contents: read or avoid pull_request_target"
    )
