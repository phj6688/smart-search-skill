"""Held-out probe for HLB-656.

Harden prompts.py: injection-prone evaluator input, news bias, misleading graph name.

Acceptance criteria checked (verbatim intent):
- AGENT_PROMPT + EVALUATOR_PROMPT: general-purpose search language; DELETE every
  reference to the tokens "news", "tags", "weights", "categories".
- EVALUATOR_PROMPT: wrap results in explicit delimiters + an instruction that
  delimited results are DATA, never instructions (injection isolation).
- EVALUATOR_PROMPT: stale exact-count instruction ("EXACTLY {N_RESULT}") and the
  "Sort results by similarity" instruction removed.
- graph.py: graph.name renamed "NEWS_SEARCH_WORKFLOW" -> "SEARCH_WORKFLOW";
  "NEWS_SEARCH_WORKFLOW" has 0 hits across src/, "SEARCH_WORKFLOW" present in graph.py.

Static/string assertions only. No LLM is ever invoked. Offline, deterministic.
"""

import os
import re
import sys

import pytest

# Make the src/ layout importable when the probe runs from the worktree root,
# outside tests/.
_ROOT = os.getcwd()
_SRC = os.path.join(_ROOT, "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

GRAPH_PY = os.path.join(_SRC, "search_workflow", "graph.py")
SRC_TREE = os.path.join(_SRC, "search_workflow")


@pytest.fixture(scope="module")
def prompts():
    import search_workflow.prompts as p

    return p


def test_forbidden_tokens_absent_from_both_prompts(prompts):
    combined = (prompts.AGENT_PROMPT + "\n" + prompts.EVALUATOR_PROMPT).lower()
    # Literal substrings that must be gone entirely.
    for literal in ("tags", "weights", "categories"):
        assert literal not in combined, f"forbidden token {literal!r} still present in prompts"
    # "news" as a standalone word must be gone.
    assert re.search(r"\bnews\b", combined) is None, "standalone word 'news' still present in prompts"


def test_evaluator_has_injection_isolation_clause(prompts):
    ev = prompts.EVALUATOR_PROMPT
    ev_low = ev.lower()
    assert "data" in ev_low, "EVALUATOR_PROMPT must state results are DATA"
    isolation = any(
        phrase in ev_low
        for phrase in ("not instructions", "never instructions", "ignore instructions", "as data")
    )
    assert isolation, "EVALUATOR_PROMPT must isolate delimited results as data, not instructions"


def test_evaluator_has_delimiter_marker(prompts):
    ev_low = prompts.EVALUATOR_PROMPT.lower()
    markers = ("<result", "---", "===", "```", "[[", "begin result", "delimit")
    assert any(m in ev_low for m in markers), "EVALUATOR_PROMPT must wrap results in an explicit delimiter"


def test_evaluator_drops_exact_count_instruction(prompts):
    ev = prompts.EVALUATOR_PROMPT
    assert re.search(r"exactly\s*\{?\s*n_result", ev, re.IGNORECASE) is None, (
        "EVALUATOR_PROMPT must not demand exactly N_RESULT results"
    )


def test_evaluator_drops_sort_by_similarity(prompts):
    ev_low = prompts.EVALUATOR_PROMPT.lower()
    assert "sort results by similarity" not in ev_low, "'Sort results by similarity' instruction must be removed"


def test_graph_py_renamed():
    assert os.path.isfile(GRAPH_PY), f"expected graph.py at {GRAPH_PY}"
    with open(GRAPH_PY, encoding="utf-8") as fh:
        text = fh.read()
    assert "NEWS_SEARCH_WORKFLOW" not in text, "graph.py still references NEWS_SEARCH_WORKFLOW"
    assert "SEARCH_WORKFLOW" in text, "graph.py must reference the new SEARCH_WORKFLOW name"


def test_news_search_workflow_absent_across_src():
    assert os.path.isdir(SRC_TREE), f"expected package tree at {SRC_TREE}"
    offenders = []
    for dirpath, _dirs, files in os.walk(SRC_TREE):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="replace") as fh:
                if "NEWS_SEARCH_WORKFLOW" in fh.read():
                    offenders.append(path)
    assert not offenders, f"NEWS_SEARCH_WORKFLOW still present in: {offenders}"


def test_graph_imports_and_name():
    import search_workflow.graph as g

    compiled = getattr(g, "graph", None)
    if compiled is None:
        pytest.skip("graph module exposes no compiled `graph` attribute")
    name = getattr(compiled, "name", None)
    if name is None:
        pytest.skip("compiled graph exposes no `.name` attribute")
    assert name == "SEARCH_WORKFLOW", f"compiled graph name is {name!r}, expected 'SEARCH_WORKFLOW'"
