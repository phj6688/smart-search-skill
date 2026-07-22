"""Retry wrapper shared by the two LLM call sites (agent + evaluator nodes).

One helper, two callers: graph.agent and graph.evaluator both route their
model.ainvoke through ainvoke_with_retry. Only rate-limit (429) and timeout
errors retry, with exponential backoff plus jitter, honoring a Retry-After
header when the provider sends one.

Everything else re-raises unchanged after the first attempt. That is load
bearing for HLB-658: a malformed structured-output selection surfaces as a
ValueError/ValidationError, which the evaluator node catches to fall back to raw
results (degraded_reason="evaluator"). This wrapper must never retry or swallow
those, so they propagate out untouched and 658's except block still fires.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any, Protocol

# openai ships transitively with langchain-openai, so the isinstance checks are
# the preferred classifier. The import stays guarded so a provider swap that
# drops openai degrades to attribute-based detection (status_code / Retry-After)
# instead of an ImportError at import time.
try:
    import openai

    _RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
        openai.RateLimitError,
        openai.APITimeoutError,
    )
    _NON_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
        openai.AuthenticationError,
        openai.BadRequestError,
    )
except ImportError:  # pragma: no cover - openai is a base dep in this repo
    _RETRYABLE_TYPES = ()
    _NON_RETRYABLE_TYPES = ()


# Total attempts across the initial call and its retries. Three keeps us inside
# the 2-3 bound while giving a transient rate limit two chances to clear.
DEFAULT_MAX_ATTEMPTS = 3
# Exponential backoff base; attempt N waits BASE * 2**(N-1) plus jitter.
_BACKOFF_BASE_S = 0.5
# Additive jitter cap. Kept small so it spreads concurrent retriers without ever
# dominating the wait. It perturbs the DELAY only, never the attempt count.
_JITTER_MAX_S = 0.25


class SupportsAInvoke(Protocol):
    """The awaitable surface both call sites share.

    load_chat_model returns a langchain BaseChatModel; the agent binds tools and
    the evaluator wraps it in with_structured_output, and both of those return a
    Runnable. All three expose async ainvoke, so typing against this Protocol
    (not ChatOpenAI) keeps the retry behavior provider-agnostic.
    """

    # Positional-only: langchain's Runnable.ainvoke names its first parameter
    # `input`, so matching by position (not name) keeps the structural check
    # provider-agnostic.
    async def ainvoke(self, value: Any, config: Any = None, /) -> Any: ...


async def ainvoke_with_retry(
    model: SupportsAInvoke,
    value: Any,
    config: Any = None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Any:
    """Await model.ainvoke, retrying only rate-limit and timeout errors.

    On success returns the model result unchanged. On a non-retryable error
    (auth, bad request, or a ValueError/ValidationError from structured output)
    it re-raises on the first attempt with no wrapping. On exhausted retries it
    raises the last rate-limit/timeout error, which flows to run_workflow's
    typed-error path.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return await model.ainvoke(value, config)
        except Exception as exc:
            if attempt == max_attempts or not _is_retryable(exc):
                raise
            await asyncio.sleep(_retry_wait_seconds(exc, attempt))


def _is_retryable(exc: BaseException) -> bool:
    """Only rate limits and timeouts retry; everything else raises immediately."""
    # ValueError/ValidationError are HLB-658's malformed-structured-output
    # signal (ValidationError subclasses ValueError here). They must reach the
    # evaluator's except block unretried and unwrapped, so refuse them first.
    if isinstance(exc, ValueError):
        return False
    if _NON_RETRYABLE_TYPES and isinstance(exc, _NON_RETRYABLE_TYPES):
        return False
    if _RETRYABLE_TYPES and isinstance(exc, _RETRYABLE_TYPES):
        return True
    # Provider-agnostic fallbacks for a client whose error classes we do not
    # recognize: an HTTP 429 status or a Retry-After header is a throttle the
    # caller is meant to wait out and retry.
    if getattr(exc, "status_code", None) == 429:
        return True
    return _retry_after_seconds(exc) is not None


def _retry_wait_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before the next attempt."""
    # A server-sent Retry-After dictates the wait exactly; honor it rather than
    # the computed backoff so we do not retry ahead of when the provider allows.
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return retry_after
    backoff = _BACKOFF_BASE_S * (2.0 ** (attempt - 1))
    return backoff + random.uniform(0, _JITTER_MAX_S)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """Parse a numeric Retry-After (seconds) off the error, or None if absent."""
    headers = _error_headers(exc)
    if headers is None:
        return None
    raw = _header_lookup(headers, "retry-after")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        # A non-numeric Retry-After (e.g. an HTTP-date) is not honored here; the
        # backoff schedule takes over instead of guessing a wait.
        return None
    return seconds if seconds >= 0 else None


def _error_headers(exc: BaseException) -> Any:
    """The response headers carried by a client error, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        return headers
    # Some client errors carry headers directly rather than under .response.
    return getattr(exc, "headers", None)


def _header_lookup(headers: Any, name: str) -> Any:
    get = getattr(headers, "get", None)
    if get is None:
        return None
    # httpx.Headers is case-insensitive so lowercase suffices there; a plain dict
    # is not, so also try the title-cased header a caller might have stored.
    value = get(name)
    if value is None:
        value = get(name.title())
    return value
