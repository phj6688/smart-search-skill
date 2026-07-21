"""Held-out behavioural probe for HLB-659.

Issue: "Replace hardcoded ChatOpenAI with provider-string model loading".
Anchor: src/search_workflow/utils.py:load_chat_model

Acceptance criteria checked (black-box, no network, no real Ollama server):
  1. Importing `search_workflow` succeeds without the optional `local` extra.
  2. load_chat_model("openai/gpt-4o-mini") resolves to the openai provider
     targeting gpt-4o-mini (class name ~ ChatOpenAI, or .model/.model_name).
  3. load_chat_model("gpt-4o-mini") (bare, no slash) also = openai provider.
  4. load_chat_model("ollama/llama3.1") either returns a ChatOllama object, or
     raises with a message naming the `local` extra / ollama / langchain-ollama.
  5. graph.py contains no hardcoded ChatOpenAI (call sites go via load_chat_model).
  6. pyproject.toml declares a `local` optional-dependency extra with langchain-ollama.

Constraints honoured: load_chat_model is NEVER patched. temperature kwarg has a
default, so calls pass only the model string.
"""

import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    # Probe runs from the issue worktree root; locate src/search_workflow.
    start = Path.cwd()
    for base in (start, *start.parents):
        if (base / "src" / "search_workflow").is_dir():
            return base
    return start


REPO_ROOT = _repo_root()
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_chat_model():
    from search_workflow.utils import load_chat_model  # public interface

    return load_chat_model


def _is_openai_gpt4o_mini(model) -> bool:
    if "ChatOpenAI" in type(model).__name__:
        return True
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    return name == "gpt-4o-mini"


def test_import_search_workflow_is_safe_without_local_extra():
    import importlib

    mod = importlib.import_module("search_workflow")
    assert mod is not None


def test_openai_provider_prefix_resolves_to_gpt4o_mini(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-key-for-construction-only")
    model = _load_chat_model()("openai/gpt-4o-mini")
    assert _is_openai_gpt4o_mini(model), (
        f"expected openai gpt-4o-mini, got {type(model).__name__} "
        f"model={getattr(model, 'model_name', getattr(model, 'model', None))!r}"
    )


def test_bare_model_name_defaults_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-key-for-construction-only")
    model = _load_chat_model()("gpt-4o-mini")
    assert _is_openai_gpt4o_mini(model), (
        f"bare name should default to openai; got {type(model).__name__}"
    )


def test_ollama_path_names_local_extra_or_returns_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-key-for-construction-only")
    load_chat_model = _load_chat_model()
    try:
        model = load_chat_model("ollama/llama3.1")
    except Exception as exc:  # langchain-ollama not in base install
        msg = str(exc).lower()
        assert ("local" in msg) or ("ollama" in msg) or ("langchain-ollama" in msg), (
            f"ollama failure should reference the local extra / ollama; got: {exc!r}"
        )
    else:
        assert "Ollama" in type(model).__name__, (
            f"ollama provider should build a ChatOllama-like object; got {type(model).__name__}"
        )


def test_graph_has_no_hardcoded_chatopenai():
    graph = REPO_ROOT / "src" / "search_workflow" / "graph.py"
    assert graph.is_file(), f"missing {graph}"
    assert "ChatOpenAI" not in graph.read_text(encoding="utf-8"), (
        "graph.py must call load_chat_model, not construct ChatOpenAI directly"
    )


def test_pyproject_declares_local_extra_with_langchain_ollama():
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file(), f"missing {pyproject}"
    text = pyproject.read_text(encoding="utf-8")

    try:
        import tomllib

        data = tomllib.loads(text)
        extras = data.get("project", {}).get("optional-dependencies", {})
        local = extras.get("local")
        assert local is not None, "no [project.optional-dependencies].local extra"
        assert any("langchain-ollama" in str(dep) for dep in local), (
            f"local extra must contain langchain-ollama; got {local!r}"
        )
    except ImportError:
        # ASSUMPTION: fall back to text scan if tomllib is unavailable.
        assert "local" in text and "langchain-ollama" in text
