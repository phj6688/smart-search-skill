"""Define a custom Reasoning and Action agent with tool support."""

from datetime import datetime, timezone
from typing import Dict, List, Literal, cast, Any, Union

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from .configuration import Configuration
from .state import InputState, State
from .tools import TOOLS
from .utils import load_chat_model, ArticlesResponse


import json


async def agent(
    state: State, config: RunnableConfig
) -> Dict[str, List[AIMessage]]:
    """Initialize the prompt, run the model, and handle tool calls as the main agent.

    Args:
        state (State): The conversation's current state.
        config (RunnableConfig): Model run configuration.

    Returns:
        dict: Dictionary with the model's response message.
    """
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", configuration.AGENT_PROMPT),
        ("placeholder", "{messages}")
    ])
    model = load_chat_model(configuration.model).bind_tools(TOOLS)

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
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
) -> Dict[str, List[AIMessage]]:
    """Evaluate tool results and return the most relevant entries based on the search query.

    Args:
        state (State): The conversation's current state.
        config (RunnableConfig): Model run configuration.

    Returns:
        dict: Dictionary with evaluated, filtered results.
    """
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages([
        ("system", configuration.EVALUATOR_PROMPT),
        ("placeholder", "{messages}")
    ])
    search_query = state.messages[0].content  # Assumes the initial message is the user query.

    message_value = await prompt.ainvoke(
        {
            "messages": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
            "N_RESULT": configuration.max_search_results_evaluator,
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

async def run_workflow(input_data: str) -> Union[Dict[str, Any], str]:
    """
    Execute the news search workflow with given input data.

    Args:
        input_data (str): The user input or query for initiating the workflow.

    Returns:
        Union[Dict[str, Any], str]: The final message content in JSON format, 
        or an error message if the workflow fails.
    """
    try:
        # Prepare initial state with user input as a HumanMessage
        initial_state = {
            "messages": [HumanMessage(content=input_data)]
        }
        
        # Run the workflow
        final_state = await graph.ainvoke(initial_state)

        # Ensure `final_state` is a dictionary and contains 'messages'
        messages = final_state.get('messages')
        if not messages:
            return "Error: No messages returned by the workflow."

        # Retrieve the last message from the 'messages' list
        final_message = messages[-1]

        # Try to parse the final message content as JSON
        try:
            output_content = json.loads(final_message.content)
            if isinstance(output_content, list):
                return output_content  # Return as list of dicts if JSON-parsed correctly
            else:
                return {"result": output_content}  # Wrap if single dict or other structure

        except json.JSONDecodeError as json_error:
            return f"Error parsing JSON response: {json_error}"

    except ValueError as ve:
        return f"Error with message format: {str(ve)}"
    except Exception as e:
        return f"Workflow execution error: {str(e)}"

