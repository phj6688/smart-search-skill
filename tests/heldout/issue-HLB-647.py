"""Held-out behavioural probe for HLB-647: offline VCR test harness.

Acceptance criteria checked, black-box:
1. tests/conftest.py exists, names the OpenAI stub seam (load_chat_model) and
   scrubs Authorization headers (filter_headers / scrub / before_record).
2. tests/cassettes/ exists with at least one YAML cassette.
3. Every cassette parses as YAML; no recorded URI/host targets api.openai.com;
   no header value is a real bearer token or sk- key (placeholders only).
4. .github/workflows/ci.yml runs the suite with --block-network.
5. pyproject.toml declares pytest-recording.
6. A negative test exists that deletes/copies a cassette to prove aiohttp
   interception under --block-network.
7. The full suite passes offline: pytest --block-network exits 0 with
   OPENAI_API_KEY removed from the environment.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

# ASSUMPTION: probe is invoked from the issue's worktree root.
ROOT = Path.cwd()
SECRET_RE = re.compile(r"(?i)^(Bearer\s+\S{10,}|sk-[A-Za-z0-9]{10,})")


def _cassettes():
    d = ROOT / "tests" / "cassettes"
    return sorted(p for p in d.rglob("*") if p.suffix in (".yaml", ".yml"))


def _walk(node, key=None):
    """Yield (key, value) for every mapping entry, recursively."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk(v, k)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item, key)


def test_conftest_names_stub_seam_and_scrubs_authorization():
    conftest = ROOT / "tests" / "conftest.py"
    assert conftest.is_file(), "tests/conftest.py missing"
    text = conftest.read_text(encoding="utf-8")
    low = text.lower()
    assert "load_chat_model" in text, "OpenAI stub seam load_chat_model not referenced"
    assert "authorization" in low, "no Authorization header scrubbing mentioned"
    assert any(m in low for m in ("filter_headers", "scrub", "before_record")), \
        "no scrub mechanism (filter_headers/scrub/before_record) in conftest"


def test_cassettes_directory_has_recordings():
    assert (ROOT / "tests" / "cassettes").is_dir(), "tests/cassettes/ missing"
    assert _cassettes(), "no *.yaml/*.yml cassettes under tests/cassettes/"


def test_cassettes_parse_and_contain_no_openai_traffic_or_real_secrets():
    for path in _cassettes():
        docs = list(yaml.safe_load_all(path.read_text(encoding="utf-8")))
        assert docs, f"{path} parsed to nothing"
        for doc in docs:
            for key, value in _walk(doc):
                k = str(key).lower()
                if k in ("uri", "url", "host") and isinstance(value, str):
                    assert "api.openai.com" not in value, \
                        f"{path}: recorded OpenAI traffic in {key}={value!r}"
                if k == "headers" and isinstance(value, dict):
                    for hname, hval in value.items():
                        vals = hval if isinstance(hval, list) else [hval]
                        for v in vals:
                            if isinstance(v, str):
                                assert not SECRET_RE.match(v), \
                                    f"{path}: real credential in header {hname!r}"


def test_ci_workflow_blocks_network():
    ci = ROOT / ".github" / "workflows" / "ci.yml"
    assert ci.is_file(), ".github/workflows/ci.yml missing"
    assert "--block-network" in ci.read_text(encoding="utf-8"), \
        "CI does not run pytest with --block-network"


def test_pyproject_declares_pytest_recording():
    pyproject = ROOT / "pyproject.toml"
    assert pyproject.is_file(), "pyproject.toml missing"
    assert "pytest-recording" in pyproject.read_text(encoding="utf-8"), \
        "pytest-recording not declared in pyproject.toml"


def test_negative_cassette_deletion_test_exists():
    hits = []
    for path in (ROOT / "tests").rglob("*.py"):
        if "heldout" in path.name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"def\s+test_\w*negative\w*", text, re.IGNORECASE):
            continue
        low = text.lower()
        if "cassette" in low and re.search(r"unlink|remove|rmtree|copy", low):
            hits.append(path)
    assert hits, ("no negative test found that removes/copies a cassette "
                  "to prove aiohttp interception")


def test_suite_passes_offline_with_block_network_and_no_api_key():
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q",
         "-p", "no:cacheprovider", "--block-network"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, (
        f"offline suite failed (exit {proc.returncode})\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
    )
