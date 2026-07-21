"""Define a custom Reasoning and Action agent with tool support."""

import json
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from .configuration import Configuration
from .errors import SearchError
from .prompts import AGENT_PROMPT, EVALUATOR_PROMPT
from .state import InputState, State
from .tools import METRICS, TOOLS
from .utils import SelectionResponse, load_chat_model

# S03 engine short names mapped to the documented result-path names. The
# attribution record uses "ddg"; the result path surfaces "duckduckgo".
_ENGINE_DISPLAY_NAMES = {"searxng": "searxng", "ddg": "duckduckgo"}


async def agent(
    state: State, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    """Initialize the prompt, run the model, and handle tool calls as the main agent."""
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_PROMPT),
        ("placeholder", "{messages}")
    ])
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=UTC).isoformat(),
        },
        config,
    )

    response = cast(AIMessage, await model.ainvoke(message_value, config))

    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }
    return {"messages": [response]}

def _fetched_results(messages: list[Any]) -> list[dict[str, Any]]:
    """Return the results the tool path fetched, from the most recent ToolMessage.

    ToolNode serializes the search tool's list return as a JSON string on the
    ToolMessage. Parsing it here lets the evaluator select by position instead
    of asking the model to re-emit (and possibly mutate) each field.
    """
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                return []
        else:
            parsed = content
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        return []
    return []


async def evaluator(
    state: State, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    """Select the most relevant fetched results by index, without regenerating them."""

    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_PROMPT),
        ("placeholder", "{messages}")
    ])
    search_query = state.messages[0].content
    max_results = configuration.max_search_results_evaluator

    fetched = _fetched_results(list(state.messages))

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=UTC).isoformat(),
            "N_RESULT": max_results,
            "SEARCH_QUERY": search_query,
        },
        config,
    )

    # temperature=0 keeps the same fetched set mapping to the same picks; the
    # model returns indices into `fetched`, never regenerated article fields.
    model = load_chat_model(
        configuration.model, temperature=0
    ).with_structured_output(SelectionResponse)
    structured_response: Any = await model.ainvoke(message_value, config)
    selected_indices = structured_response.selected

    # Join the picked indices back to the fetched objects. Values are copied
    # verbatim (no lowercasing, no regeneration); out-of-range indices drop, and
    # the cap keeps at most max_results without padding the shortfall.
    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for index in selected_indices:
        # Drop out-of-range and repeated indices; a model that returns the same
        # index twice must not yield the same result twice.
        if index in seen_indices or not 0 <= index < len(fetched):
            continue
        seen_indices.add(index)
        source = fetched[index]
        selected.append(
            {
                "title": source.get("title", ""),
                "link": source.get("link", ""),
                "snippet": source.get("snippet", ""),
            }
        )
        if len(selected) >= max_results:
            break

    response_content = json.dumps(selected, ensure_ascii=False)
    return {"messages": [AIMessage(content=response_content)]}

# Define workflow and its nodes
workflow = StateGraph(State, input=InputState, config_schema=Configuration)
workflow.add_node(agent)
workflow.add_node("tools", ToolNode(TOOLS))
workflow.add_node(evaluator)

# Set entry point and transitions
workflow.add_edge("__start__", "agent")

def route_agent_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output."""
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(last_message).__name__}")
    return "__end__" if not last_message.tool_calls else "tools"

workflow.add_conditional_edges("agent", route_agent_output)
workflow.add_edge("tools", "evaluator")
workflow.add_edge("evaluator", "__end__")

# Compile and name the workflow
graph = workflow.compile(interrupt_before=[], interrupt_after=[])
graph.name = "NEWS_SEARCH_WORKFLOW"

def _surface_provenance(provenance: dict[str, Any] | None) -> dict[str, Any]:
    """Map the S03 attribution record to the result-path metadata triple.

    Reads engines_used/n_searxng/n_ddg straight off the provenance record so
    the metadata FOLLOWS what the tool path attributed, instead of re-deriving
    which engine answered from result contents. An engine "served results" when
    it returned any raw rows (n>0), even if dedup later dropped them; both are
    always attempted, so fewer served than attempted is the degraded signal.
    """
    if not provenance:
        return {"engines_used": [], "degraded": False, "degraded_reason": None}
    engines_used = [
        _ENGINE_DISPLAY_NAMES.get(engine, engine)
        for engine in provenance.get("engines_used", [])
    ]
    served = int(provenance.get("n_searxng", 0) > 0) + int(
        provenance.get("n_ddg", 0) > 0
    )
    degraded = served < 2
    return {
        "engines_used": engines_used,
        "degraded": degraded,
        # "evaluator" is reserved for the malformed-evaluator consumer story;
        # engine-degradation surfaces "engine" and nothing else sets it here.
        "degraded_reason": "engine" if degraded else None,
    }


async def run_workflow(input_data: str, config: RunnableConfig = None) -> dict[str, Any]:
    """
    Execute the news search workflow with given input data.

    Args:
        input_data (str): The user input or query for initiating the workflow.
        config (RunnableConfig, optional): Configuration for the workflow.

    Returns:
        dict[str, Any]: A discriminated result. Success is
        {"status": "ok", "results": [...], "engines_used": [...],
        "degraded": <bool>, "degraded_reason": "engine" | None}; failure is
        {"status": "error", "error": {"type": <str>, "message": <str>}}. A bare
        string is never returned; handled failures return the typed error
        envelope rather than propagating. engines_used/degraded/degraded_reason
        are read from the S03 attribution record (tools.METRICS), not
        re-derived from result contents.
    """
    try:
        # Prepare initial state with user input as a HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=input_data)]
        }

        # Run the workflow with optional config
        if config:
            final_state = await graph.ainvoke(initial_state, config)
        else:
            final_state = await graph.ainvoke(initial_state)

        # Ensure `final_state` is a dictionary and contains 'messages'
        messages = final_state.get('messages')
        if not messages:
            raise SearchError(
                "No messages returned by the workflow.", error_type="empty_result"
            )

        # Retrieve the last message from the 'messages' list
        final_message = messages[-1]

        # Parse the final message content as JSON. JSONDecodeError subclasses
        # ValueError, so convert it before the ValueError handler can claim it.
        try:
            output_content = json.loads(final_message.content)
        except json.JSONDecodeError as json_error:
            raise SearchError(
                f"Error parsing JSON response: {json_error}",
                error_type="json_parse_error",
            ) from json_error

        results = output_content if isinstance(output_content, list) else [output_content]
        # Read the S03 attribution record the tool path recorded this run and
        # surface engines_used/degraded/degraded_reason alongside results. The
        # record is the only source; nothing here re-derives engines from the
        # result contents.
        metadata = _surface_provenance(METRICS.last_provenance())
        return {"status": "ok", "results": results, **metadata}

    except SearchError as search_error:
        return search_error.to_dict()
    except ValueError as ve:
        return SearchError(
            f"Error with message format: {str(ve)}",
            error_type="message_format_error",
        ).to_dict()
    except Exception as e:
        return SearchError(
            f"Workflow execution error: {str(e)}",
            error_type="workflow_error",
        ).to_dict()
