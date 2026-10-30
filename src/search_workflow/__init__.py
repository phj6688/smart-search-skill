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

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "run_workflow": ".graph",
    "Configuration": ".configuration",
    "State": ".state",
    "InputState": ".state",
    "TOOLS": ".tools",
    "ArticleStrict": ".utils",
    "ArticlesResponse": ".utils",
    "get_message_text": ".utils",
}


def __getattr__(name: str) -> Any:
    """Resolve public exports only when requested by a caller."""
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)

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
