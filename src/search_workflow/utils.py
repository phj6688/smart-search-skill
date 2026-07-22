"""Utility functions and data models for agent interaction."""

from collections.abc import Callable

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, confloat, validator

load_dotenv()

def get_message_text(msg: BaseMessage) -> str:
    """Extracts and returns the text content from a message object.

    Args:
        msg (BaseMessage): Message object containing content as a string, dict, or list.

    Returns:
        str: Extracted text content, or an empty string if unavailable.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    elif isinstance(content, list):
        return "".join(c if isinstance(c, str) else c.get("text", "") for c in content).strip()
    return ""


# Points the missing-dependency error at the exact install incantation, so a
# user who asks for a local model without the extra learns how to get it.
_LOCAL_EXTRA_HINT = (
    "langchain-ollama is not installed; install the 'local' extra: "
    "pip install smart-search-skill[local]"
)


def _load_openai(model_name: str, temperature: float) -> BaseChatModel:
    # langchain-openai is a base dependency, but the import lives here so the
    # provider dispatch is the single place the concrete integration is named.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)


def _load_ollama(model_name: str, temperature: float) -> BaseChatModel:
    # langchain-ollama ships only in the `local` extra, so a base install must
    # not import it; defer the import until an ollama model is actually asked
    # for and translate the miss into a hint naming the extra.
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(_LOCAL_EXTRA_HINT) from exc

    return ChatOllama(model=model_name, temperature=temperature)


# provider -> constructor. A plain dict keeps the base install to langchain-core
# plus langchain-openai: no `langchain` meta-package and no init_chat_model.
_PROVIDERS: dict[str, Callable[[str, float], BaseChatModel]] = {
    "openai": _load_openai,
    "ollama": _load_ollama,
}


def load_chat_model(model_name: str, temperature: float = 0.1) -> BaseChatModel:
    """Load a chat model from a ``provider/model`` string.

    A bare name with no ``/`` (e.g. "gpt-4o-mini") resolves to the openai
    provider, so the historical config strings and the deployed
    "openai/gpt-4o-mini" default both keep working.

    Args:
        model_name (str): Either "provider/model" (e.g. "ollama/llama3.1") or a
            bare model name, which defaults to the openai provider.
        temperature (float): Sampling temperature forwarded to the integration.
            The evaluator's index selection passes 0 so the same fetched set
            maps to the same picks.

    Returns:
        BaseChatModel: An initialized chat model for the requested provider.

    Raises:
        ValueError: The provider prefix is not a known provider.
        ImportError: An ollama model was requested without the ``local`` extra.
    """
    provider, separator, name = model_name.partition("/")
    if not separator:
        # No slash means the whole string is an openai model name.
        provider, name = "openai", model_name
    constructor = _PROVIDERS.get(provider)
    if constructor is None:
        raise ValueError(
            f"Unknown model provider {provider!r} in {model_name!r}; "
            f"known providers: {sorted(_PROVIDERS)}"
        )
    return constructor(name, temperature)


class ArticleStrict(BaseModel):
    """Schema for validating an article entry with title, link, snippet, and similarity score."""

    title: str = Field(
        ...,
        # min_length=5,
        # max_length=500,
        description="Descriptive title of the news article, 5-500 characters."
    )
    link: str = Field(
        ...,
        description="URL to the article, automatically converted to lowercase."
    )
    snippet: str = Field(
        ...,
        # min_length=20,
        # max_length=1000,
        description="Brief summary of the article, 20-1000 characters."
    )
    #similarity: confloat(ge=0.0, le=1.0) = Field(
    similarity: float = Field(
        ...,
    description="A float between 0.0 and 1.0 indicating the relevance score, where 0.0 means completely unrelated and 1.0 means identical."
    )

    # pydantic v1 validators are classmethods; `cls` is correct, so N805 is a
    # false positive here.
    @validator("title")
    def title_cannot_be_generic(cls, v: str) -> str:  # noqa: N805
        """Ensures title is specific and not a generic placeholder."""
        generic_titles = {"untitled", "article", "news"}
        if v.lower() in generic_titles:
            raise ValueError("Title must be specific and not generic.")
        return v

    @validator("title")
    def validate_title_length(cls, v: str) -> str:  # noqa: N805
        if len(v) < 5 or len(v) > 500:
            raise ValueError("Title must be between 5 and 500 characters.")
        return v

    @validator("snippet")
    def validate_snippet_length(cls, v: str) -> str:  # noqa: N805
        if len(v) < 20 or len(v) > 1000:
            raise ValueError("Snippet must be between 20 and 1000 characters.")
        return v

    @validator("link", pre=True)
    def link_to_lowercase(cls, v: str) -> str:  # noqa: N805
        """Converts the article link to lowercase for consistency."""
        return v.lower()

    @validator("similarity")
    def validate_similarity(cls, v: float) -> float:  # noqa: N805
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity must be between 0.0 and 1.0.")
        return v

class ArticlesResponse(BaseModel):
    """Container for a list of validated articles in the response."""
    articles: list[ArticleStrict]


class SelectionResponse(BaseModel):
    """Indices the evaluator picks from the fetched search results.

    The model returns positions into the fetched list instead of re-emitting
    title/link/snippet, so it cannot mutate a URL or invent a similarity score;
    the join back to the fetched objects happens in the evaluator node.
    """

    selected: list[int]
