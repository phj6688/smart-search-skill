"""Custom Search and Evaluation Agent.

This module defines a custom agent with the following capabilities:
1. Dynamically searches for relevant content based on specific tags and keywords, adjusting `timelimit` based on context.
2. Utilizes tools for retrieving relevant articles.
3. Evaluates and filters search results to provide only the most relevant content based on similarity scores.

Components:
- `graph`: The main graph structure that defines the flow between agent, tools, evaluator, and final output.
- `TOOLS`: A collection of tools available for the agent to use in retrieving and processing information.
- `Configuration`: Handles dynamic configurations such as `N_RESULT`, `timelimit` options, and other customizable parameters.
- `State` and `InputState`: Track conversation states and maintain context throughout the agent’s interactions.
- Utility functions for loading models and handling structured output.
"""

from .graph import run_workflow
from .configuration import Configuration
from .state import State, InputState
from .tools import TOOLS
from .utils import ArticleStrict, ArticlesResponse, get_message_text

__all__ = [
    "run_workflow",
    "Configuration",
    "State",
    "InputState",
    "TOOLS",
    "ArticleStrict",
    "ArticlesResponse",
    "get_message_text"
]
