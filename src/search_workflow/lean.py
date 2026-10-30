"""Fetch and rank web results without loading the agent or a model SDK."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import httpx

from . import tools
from .configuration import Configuration
from .errors import SearchError
from .prompts import EVALUATOR_PROMPT
from .results import _minimized_evaluator_payload, _surface_provenance
from .retry import ainvoke_with_retry
from .tools import METRICS
from .utils import SelectionResponse


class _EvaluatorHTTPError(RuntimeError):
    """Expose status and response headers to the shared retry classifier."""

    def __init__(self, response: httpx.Response) -> None:
        self.status_code = response.status_code
        self.response = response
        super().__init__(f"Evaluator HTTP {response.status_code}")


class _EvaluatorRequest:
    def __init__(self, client: httpx.AsyncClient, url: str) -> None:
        self.client = client
        self.url = url

    async def ainvoke(self, value: Any, config: Any = None) -> httpx.Response:
        response = await self.client.post(self.url, json=value)
        if response.is_error:
            raise _EvaluatorHTTPError(response)
        return response


def _parse_selection(response: httpx.Response) -> SelectionResponse:
    content = response.json()["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Evaluator content must be text")
    # Starting at the first object also discards markdown fences and prose;
    # raw_decode ignores any closing fence or commentary after that object.
    start = content.find("{")
    if start < 0:
        raise ValueError("Evaluator did not return a JSON object")
    value, _ = json.JSONDecoder().raw_decode(content[start:])
    return SelectionResponse.model_validate(value)


async def _evaluate(
    fetched: list[dict[str, Any]],
    query: str,
    configuration: Configuration,
    config: dict | None,
) -> list[dict[str, Any]]:
    provider, separator, model = configuration.model.partition("/")
    if not separator:
        provider, model = "openai", configuration.model
    if provider != "openai":
        raise ValueError(f"Unsupported model provider {provider!r} in {configuration.model!r}")

    max_results = configuration.max_search_results_evaluator
    system_message = EVALUATOR_PROMPT.format(
        system_time=datetime.now(tz=UTC).isoformat(),
        N_RESULT=max_results,
        SEARCH_QUERY=query,
    ) + '\nAnswer with only a JSON object {"selected": [<int indices>]}.'
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": _minimized_evaluator_payload(fetched)},
        ],
    }
    base = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "https://api.openai.com/v1"
    )
    api_key = os.getenv("OPENAI_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        METRICS.record_llm_call()
        response = await ainvoke_with_retry(
            _EvaluatorRequest(client, f"{base.rstrip('/')}/chat/completions"),
            payload,
            config,
        )

    try:
        selection = _parse_selection(response)
    except (ValueError, KeyError, IndexError, TypeError):
        METRICS.record_evaluator_degraded()
        return fetched

    selected: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for index in selection.selected:
        if index in seen_indices or not 0 <= index < len(fetched):
            continue
        seen_indices.add(index)
        source = fetched[index]
        selected.append({
            "title": source.get("title", ""),
            "link": source.get("link", ""),
            "snippet": source.get("snippet", ""),
        })
        if len(selected) >= max_results:
            break
    return selected


async def run_workflow_lean(input_data: str, config: dict | None = None) -> dict:
    """Return the workflow envelope using direct search and one HTTP evaluator."""
    try:
        METRICS.clear_evaluator_degraded()
        configuration = Configuration.from_runnable_config(config or {})
        search_options: dict[str, Any] = {}
        if configuration.region is not None:
            search_options["language"] = tools._extract_language(configuration.region)
            search_options["region"] = configuration.region
        if configuration.timelimit is not None:
            search_options["time_range"] = configuration.timelimit
            search_options["timelimit"] = configuration.timelimit
        fetched = await tools.search_direct(
            input_data,
            max_results=configuration.max_search_results_tool,
            searxng_url=configuration.searxng_url,
            searxng_timeout=configuration.searxng_timeout,
            engine_deadline_s=configuration.engine_deadline_s,
            safesearch=configuration.safesearch,
            **search_options,
        )
        results = (
            await _evaluate(fetched, input_data, configuration, config)
            if fetched else fetched
        )
        metadata = _surface_provenance(
            METRICS.last_provenance(), evaluator_degraded=METRICS.evaluator_degraded()
        )
        return {"status": "ok", "results": results, **metadata}
    except SearchError as search_error:
        return search_error.to_dict()
    except Exception as exc:
        return SearchError(
            f"Workflow execution error: {exc}", error_type="workflow_error"
        ).to_dict()
