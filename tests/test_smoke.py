"""Smoke tests for CI (HLB-645): package import and entry-point execution."""

import importlib
import subprocess
import sys


def test_import_package() -> None:
    module = importlib.import_module("search_workflow")
    assert module is not None


def test_module_entry_point_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "search_workflow", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
