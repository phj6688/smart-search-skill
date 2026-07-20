"""Provider-string model loading (HLB-659).

load_chat_model parses a ``provider/model`` string and dispatches to the
matching integration: a bare name is openai, ``ollama/...`` is the local
integration. These tests exercise the real loader. The autouse stub in
conftest patches the module attribute ``search_workflow.utils.load_chat_model``,
but this module captured the real function object at import time, so the
reference bound here is unaffected by that patch.
"""

import importlib.util

import pytest

from search_workflow.utils import load_chat_model

_HAS_OLLAMA = importlib.util.find_spec("langchain_ollama") is not None


def test_bare_name_resolves_to_openai_targeting_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The autouse conftest stub deletes OPENAI_API_KEY; ChatOpenAI validates a
    # key at construction (no network call), so set a dummy for the build only.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = load_chat_model("gpt-4o-mini")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-4o-mini"


def test_openai_prefix_resolves_to_openai_targeting_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = load_chat_model("openai/gpt-4o-mini")
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-4o-mini"


def test_deployed_ranker_default_resolves_through_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # R4: the deployed mcp_server default string must keep resolving to a
    # working openai model through the new loader. mcp_server needs the `mcp`
    # extra (fastapi); skip if it is not synced.
    pytest.importorskip("fastapi")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    from search_workflow import mcp_server

    assert mcp_server._RANKER_MODEL == "openai/gpt-4o-mini"
    model = load_chat_model(mcp_server._RANKER_MODEL)
    assert type(model).__name__ == "ChatOpenAI"
    assert model.model_name == "gpt-4o-mini"


def test_evaluator_temperature_is_forwarded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # HLB-655 relies on temperature=0 for the evaluator's index selection, so
    # the loader must forward the kwarg to the integration unchanged.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = load_chat_model("openai/gpt-4o-mini", temperature=0)
    assert model.temperature == 0


def test_unknown_provider_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown model provider"):
        load_chat_model("anthropic/claude-3")


@pytest.mark.skipif(_HAS_OLLAMA, reason="langchain-ollama is installed")
def test_ollama_without_local_extra_raises_naming_the_extra() -> None:
    # Requesting a local model without the extra must fail loudly and name the
    # `local` extra so the user knows the exact install to run.
    with pytest.raises(ImportError, match="local"):
        load_chat_model("ollama/llama3.1")


@pytest.mark.skipif(not _HAS_OLLAMA, reason="langchain-ollama not installed")
def test_ollama_with_extra_returns_chatollama_targeting_the_model() -> None:
    model = load_chat_model("ollama/llama3.1")
    assert type(model).__name__ == "ChatOllama"
    target = getattr(model, "model", None) or getattr(model, "model_name", None)
    assert target == "llama3.1"
