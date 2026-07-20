"""Utility functions and data models for agent interaction."""

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
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


def load_chat_model(model_name: str, temperature: float = 0.1) -> ChatOpenAI:
    """Initialize and return the chat model based on the model name.

    Args:
        model_name (str): The name of the model, such as "gpt-4".
        temperature (float): Sampling temperature. The evaluator's index
            selection passes 0 so the same fetched set maps to the same picks.

    Returns:
        ChatOpenAI: An initialized ChatOpenAI instance configured with standard parameters.
    """
    return ChatOpenAI(model=model_name, temperature=temperature)


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
    def title_cannot_be_generic(cls, v):  # noqa: N805
        """Ensures title is specific and not a generic placeholder."""
        generic_titles = {"untitled", "article", "news"}
        if v.lower() in generic_titles:
            raise ValueError("Title must be specific and not generic.")
        return v

    @validator("title")
    def validate_title_length(cls, v):  # noqa: N805
        if len(v) < 5 or len(v) > 500:
            raise ValueError("Title must be between 5 and 500 characters.")
        return v

    @validator("snippet")
    def validate_snippet_length(cls, v):  # noqa: N805
        if len(v) < 20 or len(v) > 1000:
            raise ValueError("Snippet must be between 20 and 1000 characters.")
        return v

    @validator("link", pre=True)
    def link_to_lowercase(cls, v: str) -> str:  # noqa: N805
        """Converts the article link to lowercase for consistency."""
        return v.lower()

    @validator("similarity")
    def validate_similarity(cls, v):  # noqa: N805
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
