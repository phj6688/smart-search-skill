"""State structures for managing agent interactions and lifecycle."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep
from typing_extensions import Annotated
from .utils import ArticleStrict, ArticlesResponse


@dataclass
class InputState:
    """Initial input state for the agent, capturing the conversation history.

    Attributes:
        messages (Sequence[AnyMessage]): Tracks the execution state across various interaction stages.
            Follows a pattern of HumanMessage -> AIMessage (with tool calls) -> ToolMessage(s) -> AIMessage response.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(default_factory=list)

@dataclass
class State(InputState):
    """Extended state including lifecycle and output tracking for the agent.

    Attributes:
        is_last_step (IsLastStep): Indicates if the current step is the last before reaching the limit.
        evaluator_output (List[ArticleStrict]): Holds the evaluated, filtered results from the search tools.
    """
    
    is_last_step: IsLastStep = field(default=False)
    evaluator_output: ArticlesResponse = field(default_factory=list) #List[ArticleStrict] = field(default_factory=list)
