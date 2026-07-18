"""OpenAI stub coverage (HLB-647): the evaluator path runs without a key.

The autouse `stub_openai` fixture deletes OPENAI_API_KEY, so a pass here
means the canned with_structured_output path was used, not the real API.
"""

import json
import os

from langchain_core.messages import HumanMessage

from search_workflow import graph
from search_workflow.state import State


async def test_evaluator_returns_canned_structured_output() -> None:
    assert "OPENAI_API_KEY" not in os.environ

    state = State(messages=[HumanMessage(content="canned query")])
    result = await graph.evaluator(state, config={})

    payload = json.loads(result["messages"][0].content)
    assert payload == [
        {
            "title": "Canned offline article",
            "link": "https://example.test/canned-article",
            "snippet": "Stub snippet returned by the offline OpenAI stand-in.",
            "similarity": 0.9,
        }
    ]


async def test_agent_node_uses_canned_model() -> None:
    state = State(messages=[HumanMessage(content="canned query")])
    result = await graph.agent(state, config={})

    assert result["messages"][0].content == "canned agent reply"
