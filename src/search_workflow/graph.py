"""Define a custom Reasoning and Action agent with tool support."""

import json
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from .configuration import Configuration
from .errors import SearchError
from .prompts import AGENT_PROMPT, EVALUATOR_PROMPT
from .retry import ainvoke_with_retry
from .state import InputState, State
from .tools import METRICS, TOOLS, search_direct
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

    # One real LLM invocation. Counted here, not inside the retry wrapper, so a
    # retried 429 stays a single logical call; the use_llm=False path never
    # reaches this node, so its llm_calls stays 0.
    METRICS.record_llm_call()
    # Retry only rate-limit/timeout so a single 429 does not fail the query.
    response = cast(AIMessage, await ainvoke_with_retry(model, message_value, config))

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


# Cap on the snippet text handed to the evaluator LLM. The model selects by
# index and never re-emits result fields, so it needs only enough snippet to
# judge relevance; ~200 chars bounds the per-result token cost while the
# returned results (joined from `fetched`) stay the verbatim fetched bytes.
_EVALUATOR_SNIPPET_CAP = 200


def _minimized_evaluator_payload(fetched: list[dict[str, Any]]) -> str:
    """Render the fetched results as indexed, URL-free, snippet-capped records.

    The evaluator selects by position into `fetched` (the join back to the
    fetched objects happens in the node), so the model never needs the raw URLs
    or the full snippet. Withholding the `link` field keeps every URL out of the
    LLM payload, and truncating the snippet bounds the token cost. This minimizes
    only what the model SEES; the returned results are still the fetched objects,
    untouched.
    """
    records = []
    for index, item in enumerate(fetched):
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", ""))[:_EVALUATOR_SNIPPET_CAP]
        records.append(f"[{index}] {title}\n{snippet}")
    return "\n\n".join(records)


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

    # Hand the model an indexed, URL-free, snippet-capped view of the fetched
    # results instead of the raw tool messages. The selection contract is
    # unchanged (indices into `fetched`), so this only shrinks and de-URLs what
    # the model sees: no raw URL and no oversized snippet reaches the LLM, while
    # the join below still returns the fetched bytes verbatim.
    payload_messages = [HumanMessage(content=_minimized_evaluator_payload(fetched))]
    message_value = await prompt.ainvoke(
        {
            "messages": payload_messages,
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
    # One real LLM invocation; the use_llm=False path never runs this node, so
    # its llm_calls stays 0. Counted even if the selection comes back malformed,
    # because the model WAS invoked.
    METRICS.record_llm_call()
    try:
        # Same retry wrapper as the agent node. It retries only rate-limit and
        # timeout; a ValueError/ValidationError from a malformed selection is
        # non-retryable and propagates out unchanged, so this except still fires
        # and the degraded fallback below runs (HLB-658).
        structured_response: Any = await ainvoke_with_retry(
            model, message_value, config
        )
        selected_indices = getattr(structured_response, "selected", None)
    except (ValueError, ValidationError):
        # Unparseable or schema-invalid structured output. with_structured_output
        # raises ValueError/ValidationError on bad JSON or a schema mismatch.
        selected_indices = None

    if selected_indices is None:
        # The selection is unusable, but the merged, deduplicated raw results the
        # tool step fetched are still good. Fall back to them verbatim instead of
        # raising: a malformed selection is a degraded result, not a failure, and
        # discarding the fetched set would waste it. No retry. The flag rides up
        # to run_workflow, which surfaces degraded_reason="evaluator". An empty
        # but valid selection (selected=[]) is NOT this branch; it stays healthy.
        METRICS.record_evaluator_degraded()
        response_content = json.dumps(fetched, ensure_ascii=False)
        return {"messages": [AIMessage(content=response_content)]}

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
graph.name = "SEARCH_WORKFLOW"

def _surface_provenance(
    provenance: dict[str, Any] | None, evaluator_degraded: bool = False
) -> dict[str, Any]:
    """Map the S03 attribution record to the result-path metadata triple.

    Reads engines_used/n_searxng/n_ddg straight off the provenance record so
    the metadata FOLLOWS what the tool path attributed, instead of re-deriving
    which engine answered from result contents. An engine "served results" when
    it returned any raw rows (n>0), even if dedup later dropped them; both are
    always attempted, so fewer served than attempted is the degraded signal.

    evaluator_degraded is HLB-658's malformed-selection signal (the evaluator
    node fell back to raw results). It surfaces degraded_reason="evaluator", but
    engine degradation still WINS: a single-engine fetch is a more fundamental
    data-quality loss than an unusable selection over an otherwise-complete
    fetch, so the evaluator reason only shows when the engines were healthy.
    """
    if not provenance:
        engines_used: list[str] = []
        engine_degraded = False
    else:
        engines_used = [
            _ENGINE_DISPLAY_NAMES.get(engine, engine)
            for engine in provenance.get("engines_used", [])
        ]
        served = int(provenance.get("n_searxng", 0) > 0) + int(
            provenance.get("n_ddg", 0) > 0
        )
        engine_degraded = served < 2
    if engine_degraded:
        degraded_reason: str | None = "engine"
    elif evaluator_degraded:
        degraded_reason = "evaluator"
    else:
        degraded_reason = None
    return {
        "engines_used": engines_used,
        "degraded": engine_degraded or evaluator_degraded,
        "degraded_reason": degraded_reason,
    }


async def _run_deterministic(
    input_data: str, config: RunnableConfig | None
) -> dict[str, Any]:
    """LLM-free path: return the shared fetch core output in the typed envelope.

    Reuses search_direct (the shared _fetch_and_merge core in tools.py), which
    records the SAME S03 attribution record the graph path does, so
    engines_used/degraded/degraded_reason come from _surface_provenance exactly
    as on the LLM path (no second attribution computation, no new field). The
    agent and evaluator nodes are never entered and load_chat_model is never
    called, so METRICS.llm_calls stays 0 and no OPENAI_API_KEY is read. A hard
    fetch failure with no results surfaces the same typed error envelope the LLM
    path returns; both engines returning empty is not a failure (status "ok",
    empty results, degraded via the attribution record), matching the LLM path.
    """
    try:
        configuration = Configuration.from_runnable_config(config or {})
        # Same knobs the agentic `search` tool feeds the shared core, so the two
        # paths fetch identically; the returned dicts are surfaced verbatim.
        results = await search_direct(
            input_data,
            max_results=configuration.max_search_results_tool,
            searxng_url=configuration.searxng_url,
            searxng_timeout=configuration.searxng_timeout,
            engine_deadline_s=configuration.engine_deadline_s,
        )
        # Read the attribution record search_direct just recorded. The evaluator
        # flag is not consulted: there is no evaluator on this path.
        metadata = _surface_provenance(METRICS.last_provenance())
        return {"status": "ok", "results": results, **metadata}

    except SearchError as search_error:
        return search_error.to_dict()
    except Exception as e:
        return SearchError(
            f"Workflow execution error: {str(e)}",
            error_type="workflow_error",
        ).to_dict()


async def run_workflow(
    input_data: str,
    config: RunnableConfig = None,
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """
    Execute the search workflow with given input data.

    Args:
        input_data (str): The user input or query for initiating the workflow.
        config (RunnableConfig, optional): Configuration for the workflow.
        use_llm (bool): When False, run the deterministic LLM-free path: return
            the shared fetch core's merged, deduplicated results directly,
            without entering the agent or evaluator nodes and without ever
            constructing a chat model. The success envelope is identical to the
            LLM path's, built from the same attribution record.

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
    if not use_llm:
        return await _run_deterministic(input_data, config)

    try:
        # Clear last query's evaluator-degradation flag so this query starts
        # clean. The engine attribution record is overwritten every query by the
        # tool path, but this flag is only ever set (never cleared) by the
        # evaluator, so it must be reset here per query.
        METRICS.clear_evaluator_degraded()

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
        # record is the only source for engine attribution; the evaluator flag
        # adds the malformed-selection degradation (HLB-658). Nothing here
        # re-derives engines from the result contents.
        metadata = _surface_provenance(
            METRICS.last_provenance(),
            evaluator_degraded=METRICS.evaluator_degraded(),
        )
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
