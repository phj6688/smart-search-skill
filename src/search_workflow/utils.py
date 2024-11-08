"""Utility functions and data models for agent interaction."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, validator, confloat
from typing import List


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


def load_chat_model(model_name: str) -> ChatOpenAI:
    """Initialize and return the chat model based on the model name.

    Args:
        model_name (str): The name of the model, such as "gpt-4".

    Returns:
        ChatOpenAI: An initialized ChatOpenAI instance configured with standard parameters.
    """
    return ChatOpenAI(model=model_name, temperature=0.1)


class ArticleStrict(BaseModel):
    """Schema for validating an article entry with title, link, snippet, and similarity score."""

    title: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Descriptive title of the news article, 5-500 characters."
    )
    link: str = Field(
        ...,
        description="URL to the article, automatically converted to lowercase."
    )
    snippet: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description="Brief summary of the article, 20-1000 characters."
    )
    similarity: confloat(ge=0.0, le=1.0) = Field(
        ...,
        description="Relevance score, from 0.0 (unrelated) to 1.0 (identical)."
    )

    @validator("title")
    def title_cannot_be_generic(cls, v):
        """Ensures title is specific and not a generic placeholder."""
        generic_titles = {"untitled", "article", "news"}
        if v.lower() in generic_titles:
            raise ValueError("Title must be specific and not generic.")
        return v

    @validator("link", pre=True)
    def link_to_lowercase(cls, v: str) -> str:
        """Converts the article link to lowercase for consistency."""
        return v.lower()


class ArticlesResponse(BaseModel):
    """Container for a list of validated articles in the response."""
    articles: List[ArticleStrict]
