"""OpenAI stub coverage (HLB-647): the evaluator path runs without a key.

The autouse `stub_openai` fixture deletes OPENAI_API_KEY, so a pass here
means the canned with_structured_output path was used, not the real API.
"""

import json
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from search_workflow import graph
from search_workflow.state import State


async def test_evaluator_selects_fetched_object_offline() -> None:
    assert "OPENAI_API_KEY" not in os.environ

    # The canned structured stub selects index 0; the evaluator must join that
    # index back to the fetched object it received, not fabricate an article.
    fetched = [
        {
            "title": "Offline fetched article",
            "link": "https://Example.test/Fetched",
            "snippet": "Fetched snippet returned by the offline tool output.",
        },
        {
            "title": "Second offline article",
            "link": "https://example.test/second",
            "snippet": "A second fetched snippet the stub does not select.",
        },
    ]
    state = State(
        messages=[
            HumanMessage(content="canned query"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "search", "args": {"query": "canned query"}, "id": "c1"}
                ],
            ),
            ToolMessage(content=json.dumps(fetched), tool_call_id="c1"),
        ]
    )
    result = await graph.evaluator(state, config={})

    payload = json.loads(result["messages"][0].content)
    assert payload == [fetched[0]]


async def test_agent_node_uses_canned_model() -> None:
    state = State(messages=[HumanMessage(content="canned query")])
    result = await graph.agent(state, config={})

    assert result["messages"][0].content == "canned agent reply"
