"""Held-out behavioural probe for HLB-648.

Acceptance criteria checked, black-box:
1. tests/fixtures/FIX-MERGE.json parses as JSON with exactly 60 top-level entries.
2. tests/fixtures/FIX-NEWS.json has 30 entries; tests/fixtures/FIX-INJECT.json has
   20 injection cases plus clean twins (40 flat entries, or 20 entries each
   carrying a paired clean-variant field).
3. tests/fixtures/RUBRIC.md exists and is non-trivial (>300 chars).
4. An egress socket-guard lives under tests/ (mentions "socket" plus an
   allowlist/guard term) and a canary test named like egress...canary exists.
5. `python scripts/eval_gates.py --help` exits 0.
6. `python -m pytest tests/ -q -k egress_canary` exits 0 from the worktree root.

Probe is read-only and runs from the issue worktree root.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()  # probe is executed from the worktree root
FIXTURES = ROOT / "tests" / "fixtures"


def _entries(path):
    """Top-level entries: the list itself, or the largest list inside a dict."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        assert lists, f"{path.name}: dict form must hold its corpus in a list value"
        return max(lists, key=len)
    raise AssertionError(f"{path.name}: unexpected top-level JSON type {type(data)}")


def _test_sources():
    for p in sorted((ROOT / "tests").rglob("*.py")):
        if "heldout" in p.name.lower() or "heldout" in str(p.parent).lower():
            continue
        yield p, p.read_text(encoding="utf-8", errors="replace")


def test_fix_merge_has_exactly_60_entries():
    path = FIXTURES / "FIX-MERGE.json"
    assert path.is_file(), f"missing {path}"
    assert len(_entries(path)) == 60


def test_fix_news_has_exactly_30_entries():
    path = FIXTURES / "FIX-NEWS.json"
    assert path.is_file(), f"missing {path}"
    assert len(_entries(path)) == 30


def test_fix_inject_has_20_cases_with_clean_twins():
    path = FIXTURES / "FIX-INJECT.json"
    assert path.is_file(), f"missing {path}"
    entries = _entries(path)
    if len(entries) == 20:
        # ASSUMPTION: paired shape carries the clean twin in a field whose key
        # mentions "clean" (issue does not name the field).
        for e in entries:
            assert isinstance(e, dict) and any(
                "clean" in k.lower() for k in e
            ), "each of the 20 cases must carry a paired clean-variant field"
    else:
        assert len(entries) == 40, (
            f"expected 40 flat entries (20 injection + 20 clean twins) or 20 "
            f"paired entries, got {len(entries)}"
        )
        blob = json.dumps(entries).lower()
        assert "inject" in blob and "clean" in blob, (
            "flat corpus must distinguish injection cases from clean twins"
        )


def test_rubric_exists_and_is_nontrivial():
    path = FIXTURES / "RUBRIC.md"
    assert path.is_file(), f"missing {path}"
    assert len(path.read_text(encoding="utf-8")) > 300


def test_egress_socket_guard_present_under_tests():
    terms = ("allowlist", "allowed_hosts", "guard")
    hit = any(
        "socket" in src and any(t in src for t in terms)
        for _, src in _test_sources()
    )
    assert hit, "no tests/ file mentions socket plus allowlist/allowed_hosts/guard"


def test_egress_canary_test_exists():
    pat = re.compile(r"def\s+test_\w*(?:egress\w*canary|canary\w*egress)\w*\s*\(")
    hit = any(pat.search(src) for _, src in _test_sources())
    assert hit, "no canary test named like test_*egress*canary* found under tests/"


def test_eval_gates_help_exits_zero():
    script = ROOT / "scripts" / "eval_gates.py"
    assert script.is_file(), f"missing {script}"
    proc = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, f"--help exited {proc.returncode}: {proc.stderr}"


def test_egress_canary_passes_under_pytest():
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "-k", "egress_canary"],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"pytest -k egress_canary exited {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
