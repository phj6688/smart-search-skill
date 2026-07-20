"""Define a custom Reasoning and Action agent with tool support."""

import json
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from .configuration import Configuration
from .errors import SearchError
from .prompts import AGENT_PROMPT, EVALUATOR_PROMPT
from .state import InputState, State
from .tools import TOOLS
from .utils import ArticlesResponse, load_chat_model


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

async def evaluator(
    state: State, config: RunnableConfig
) -> dict[str, list[AIMessage]]:
    """Evaluate tool results and return the most relevant entries based on the search query."""


    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVALUATOR_PROMPT),
        ("placeholder", "{messages}")
    ])
    search_query = state.messages[0].content
    max_results = configuration.max_search_results_evaluator


    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=UTC).isoformat(),
            "N_RESULT": max_results,
            "SEARCH_QUERY": search_query,
        },
        config,
    )

    model = load_chat_model(configuration.model).with_structured_output(ArticlesResponse)
    structured_response: Any = await model.ainvoke(message_value, config)

    # Process and serialize response
    articles_data = (
        structured_response.articles if isinstance(structured_response, ArticlesResponse)
        else [item[0] if isinstance(item, tuple) else item for item in structured_response]
    )

    articles_data.sort(key=lambda x: x.similarity, reverse=True)
    if len(articles_data) > max_results:
        articles_data = articles_data[:max_results]


    response_content = json.dumps([article.dict() for article in articles_data], ensure_ascii=False)
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

async def run_workflow(input_data: str, config: RunnableConfig = None) -> dict[str, Any]:
    """
    Execute the news search workflow with given input data.

    Args:
        input_data (str): The user input or query for initiating the workflow.
        config (RunnableConfig, optional): Configuration for the workflow.

    Returns:
        dict[str, Any]: A discriminated result. Success is
        {"status": "ok", "results": [...]}; failure is
        {"status": "error", "error": {"type": <str>, "message": <str>}}. A bare
        string is never returned; handled failures return the typed error
        envelope rather than propagating.
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
        return {"status": "ok", "results": results}

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
