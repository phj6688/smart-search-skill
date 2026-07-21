"""Static guards on the agent and evaluator prompts (HLB-656).

After the news-bias removal and injection hardening the prompts must carry no
news/tags/weights/categories vocabulary (those fields left the input schema and
skewed non-news queries), no stale exact-count or sort-by-similarity
instruction (the evaluator returns indices, not scored articles), and the
evaluator prompt must state that delimited results are DATA, never
instructions. The compiled graph carries the general-purpose name.
"""

import re

from search_workflow.graph import graph
from search_workflow.prompts import AGENT_PROMPT, EVALUATOR_PROMPT

# Vocabulary tied to the retired news-tag input schema. `news` is matched on a
# word boundary; the rest mirror the plain-substring repo grep.
_BANNED = ("news", "tags", "weights", "categories")


def _banned_present(text: str) -> list[str]:
    lowered = text.lower()
    hits = []
    for word in _BANNED:
        pattern = rf"\b{word}\b" if word == "news" else word
        if re.search(pattern, lowered):
            hits.append(word)
    return hits


def test_agent_prompt_has_no_news_schema_vocabulary() -> None:
    assert _banned_present(AGENT_PROMPT) == []


def test_evaluator_prompt_has_no_news_schema_vocabulary() -> None:
    assert _banned_present(EVALUATOR_PROMPT) == []


def test_prompts_drop_exact_count_and_similarity_sort() -> None:
    # The evaluator selects by index; an exact-count demand and a
    # sort-by-similarity instruction both contradict that contract.
    for prompt in (AGENT_PROMPT, EVALUATOR_PROMPT):
        lowered = prompt.lower()
        assert re.search(r"exactly\s*\{?n_result", lowered) is None
        assert "similarity" not in lowered


def test_evaluator_prompt_has_data_isolation_clause() -> None:
    # Delimited results are DATA and must never be obeyed as instructions.
    lowered = EVALUATOR_PROMPT.lower()
    assert "delimit" in lowered
    assert "data" in lowered
    assert "instruction" in lowered
    assert "never" in lowered


def test_evaluator_prompt_keeps_select_by_index_contract() -> None:
    # Still asks for `selected` indices and still exposes the placeholders the
    # evaluator node fills.
    assert "selected" in EVALUATOR_PROMPT
    assert "{N_RESULT}" in EVALUATOR_PROMPT
    assert "{SEARCH_QUERY}" in EVALUATOR_PROMPT


def test_graph_carries_general_purpose_name() -> None:
    assert graph.name == "SEARCH_WORKFLOW"
