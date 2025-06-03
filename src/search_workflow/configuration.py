
"""Configuration management with environment variable support."""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass(kw_only=True)
class Configuration:
    """Configuration for the search workflow."""
    
    # Core LLM settings
    model: str = field(
        default="gpt-4o-mini",
        metadata={"description": "The LLM model to use for evaluation"}
    )
    
    # OpenAI API configuration
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        metadata={"description": "OpenAI API key"}
    )
    
    # Search configuration
    max_search_results_tool: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS_TOOL", "10")),
        metadata={"description": "Maximum results to fetch from search engines"}
    )
    
    max_search_results_evaluator: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS_EVALUATOR", "5")),
        metadata={"description": "Maximum results to return after AI evaluation"}
    )
    
    # SearXNG configuration
    searxng_enabled: bool = field(
        default=True,
        metadata={"description": "Enable SearXNG search engine"}
    )
    
    searxng_url: str = field(
        default_factory=lambda: os.getenv("SEARXNG_URL", "http://localhost:9090"),
        metadata={"description": "SearXNG instance URL"}
    )
    
    searxng_timeout: int = field(
        default=30,
        metadata={"description": "SearXNG request timeout in seconds"}
    )
    
    # Search strategy
    search_strategy: str = field(
        default="hybrid",  # "searxng", "duckduckgo", "hybrid"
        metadata={"description": "Search strategy: hybrid, searxng-only, or duckduckgo-only"}
    )
    
    # Debugging and logging
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true",
        metadata={"description": "Enable debug mode"}
    )
    
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        metadata={"description": "Logging level"}
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Changed from raising error to just warning
        if not self.openai_api_key:
            print("⚠️  Warning: OPENAI_API_KEY not set. Some features may not work.")
        
        if self.max_search_results_evaluator > self.max_search_results_tool:
            raise ValueError(
                "max_search_results_evaluator cannot be greater than max_search_results_tool"
            )


    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """Create configuration from LangChain runnable config."""
        # Get configurable values from the input config
        configurable_values = config.get("configurable", {})
        
        # Get default values from environment or dataclass defaults
        default_kwargs = {
            f.name: f.default_factory() if callable(f.default_factory) else f.default
            for f in cls.__dataclass_fields__.values()
            if hasattr(f, 'default_factory') or hasattr(f, 'default')
        }
        
        # Start with default values
        final_kwargs = default_kwargs.copy()
        
        # Override with values from the configurable dictionary
        for key, value in configurable_values.items():
            if key in final_kwargs:
                # Ensure correct type conversion for int fields
                if key in ["max_search_results_tool", "max_search_results_evaluator", "searxng_timeout"]:
                    try:
                        final_kwargs[key] = int(value)
                    except (ValueError, TypeError):
                        print(f"⚠️  Warning: Could not convert '{key}' value '{value}' to int. Using default.")
                elif key == "debug":
                    final_kwargs[key] = str(value).lower() == "true"
                else:
                    final_kwargs[key] = value
                    
        # Create the Configuration object with final arguments
        return cls(**final_kwargs)

    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def validate_environment(self) -> List[str]:
        """Validate environment and return list of issues."""
        issues = []
        
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY is not set")
        
        if self.searxng_enabled:
            # Could add SearXNG connectivity check here
            pass
            
        return issues

# Global configuration instance
default_config = Configuration()
