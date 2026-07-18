"""Held-out behavioural probe for HLB-645: tests/ scaffold and GitHub Actions CI matrix.

Acceptance criteria checked, black-box:
1. .github/workflows/ci.yml exists and parses as YAML with jobs.
2. Exactly one job carries job-level continue-on-error: true, and that job's
   serialized text mentions mypy (allow-fail mypy job).
3. The pytest job's strategy matrix includes both Python "3.11" and "3.12".
4. Some step runs python -c "import search_workflow" (fresh-runner probe);
   raw workflow text contains `import search_workflow`.
5. Raw workflow text references follow-up issue HLB-665 and never touches docker/.
6. tests/test_smoke.py exists and passes when run via pytest as a subprocess
   from the worktree root.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

# ASSUMPTION: probe is invoked from the issue's worktree root (per task spec).
ROOT = Path.cwd()
CI_PATH = ROOT / ".github" / "workflows" / "ci.yml"


def _raw_text() -> str:
    assert CI_PATH.is_file(), f"missing workflow file: {CI_PATH}"
    return CI_PATH.read_text(encoding="utf-8")


def _jobs() -> dict:
    workflow = yaml.safe_load(_raw_text())
    assert isinstance(workflow, dict), "ci.yml did not parse to a mapping"
    jobs = workflow.get("jobs")
    assert isinstance(jobs, dict) and jobs, "ci.yml has no jobs"
    return jobs


def test_ci_workflow_exists_and_parses():
    jobs = _jobs()
    assert len(jobs) >= 1


def test_exactly_one_allow_fail_job_and_it_is_mypy():
    jobs = _jobs()
    allow_fail = {
        name: job
        for name, job in jobs.items()
        if isinstance(job, dict) and job.get("continue-on-error") is True
    }
    assert len(allow_fail) == 1, (
        f"expected exactly one job with continue-on-error: true, "
        f"found {sorted(allow_fail)}"
    )
    name, job = next(iter(allow_fail.items()))
    serialized = (name + "\n" + yaml.safe_dump(job)).lower()
    assert "mypy" in serialized, f"allow-fail job {name!r} does not mention mypy"


def test_pytest_job_matrix_covers_311_and_312():
    jobs = _jobs()
    candidates = [
        (name, job)
        for name, job in jobs.items()
        if isinstance(job, dict) and "pytest" in yaml.safe_dump(job).lower()
    ]
    assert candidates, "no job mentions pytest"
    for _name, job in candidates:
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        matrix_text = " ".join(
            str(v) for values in matrix.values() if isinstance(values, list)
            for v in values
        )
        if "3.11" in matrix_text and "3.12" in matrix_text:
            return
    pytest.fail("no pytest job has a strategy matrix with both 3.11 and 3.12")


def test_fresh_runner_probe_imports_package():
    assert "import search_workflow" in _raw_text(), (
        "no step runs the fresh-runner probe (import search_workflow)"
    )


def test_followup_reference_and_no_docker_paths():
    raw = _raw_text()
    assert "HLB-665" in raw, "missing follow-up reference HLB-665 next to mypy job"
    assert "docker/" not in raw, "workflow must not read, mount, or modify docker/"


def test_smoke_tests_exist_and_pass():
    smoke = ROOT / "tests" / "test_smoke.py"
    assert smoke.is_file(), f"missing {smoke}"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_smoke.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"tests/test_smoke.py failed:\n{result.stdout}\n{result.stderr}"
    )
