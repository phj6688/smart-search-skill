"""Configurations for the custom agent, defining prompts, model, and limits."""

from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from . import prompts

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
MAX_SEARCH_RESULTS_TOOL = 10
MAX_SEARCH_RESULTS_EVALUATOR = 5

@dataclass(kw_only=True)
class Configuration:
    """Configurable parameters for the agent's operation."""
    
    AGENT_PROMPT: str = field(
        default=prompts.AGENT_PROMPT,
        metadata={"description": "System prompt guiding the agent's interactions."}
    )
    
    EVALUATOR_PROMPT: str = field(
        default=prompts.EVALUATOR_PROMPT,
        metadata={"description": "System prompt guiding the evaluator's interactions."}
    )
    
    model: str = field(
        default=DEFAULT_MODEL,
        metadata={"description": "Identifier for the language model."}
    )
    
    max_search_results_tool: int = field(
        default=MAX_SEARCH_RESULTS_TOOL,
        metadata={"description": "Max search results to retrieve per query."}
    )
    
    max_search_results_evaluator: int = field(
        default=MAX_SEARCH_RESULTS_EVALUATOR,
        metadata={"description": "Max results to return after evaluation."}
    )
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object.

        Args:
            config (Optional[RunnableConfig]): Configuration from an external source.

        Returns:
            Configuration: Initialized configuration object with defaults or overrides.
        """
        config = ensure_config(config)
        configurable_data = config.get("configurable", {})
        valid_fields = {field.name for field in fields(cls) if field.init}
        
        # Filter out any extraneous keys and initialize the Configuration
        filtered_config = {key: value for key, value in configurable_data.items() if key in valid_fields}
        
        try:
            return cls(**filtered_config)
        except TypeError as e:
            raise ValueError(f"Invalid configuration data provided: {e}")
