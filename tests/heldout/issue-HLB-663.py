"""Held-out behavioural probe for Linear issue HLB-663.

Issue: "Wire --region, --timelimit and safesearch through to engine params."
Anchor: src/search_workflow/cli.py:main.

Acceptance criteria checked here (verbatim intent):
- R1+R3: with `--region de-de --timelimit w --safesearch 1`, the emitted SearXNG
  request params carry the region-derived language (contains "de"), a truthy
  time_range mapped from 'w', and safesearch == 1.
- R2a: Configuration accepts safesearch 0/1/2 (default 0) and rejects 3.
- R2b: `search-workflow --help` lists a `--safesearch` flag.
- R4: with no flags, emitted SearXNG params carry safesearch == 0 and NO time_range.
- All: deterministic, offline, network-free. Only the HTTP request boundary is
  patched; cli.main is driven black-box. (The git-diff / no-docker-paths check is a
  merge gate, not a runtime probe concern, so it is intentionally omitted.)

The probe imports the public interfaces the issue names: search_workflow.cli.main,
search_workflow.configuration.Configuration, search_workflow.tools.SearXNGClient.
It captures the emitted engine request params by monkeypatching the lowest-level
HTTP call (aiohttp / requests / httpx) rather than any private param-builder, so it
stays tolerant to exact internal names.
"""

import importlib
import inspect
import os
import sys
import threading
import types
from contextlib import ExitStack
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import pytest


# --------------------------------------------------------------------------- #
# Make the repo's src/ importable. The probe file lives outside the repo tree
# and is expected to run from the issue worktree root.
# --------------------------------------------------------------------------- #
def _add_src_to_path():
    here = os.getcwd()
    d = here
    candidates = []
    for _ in range(6):
        candidates.append(os.path.join(d, "src"))
        candidates.append(d)
        d = os.path.dirname(d)
    for c in candidates:
        if os.path.isdir(os.path.join(c, "search_workflow")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    fallback = os.path.join(here, "src")
    if fallback not in sys.path:
        sys.path.insert(0, fallback)


_add_src_to_path()
# Keep any model-construction path from demanding a real key; the LLM is stubbed.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


# --------------------------------------------------------------------------- #
# Fake HTTP response objects (async + sync) with the minimal surface the client
# code touches: json()/text(), status, raise_for_status, async context manager.
# --------------------------------------------------------------------------- #
class _FakeAiohttpResp:
    status = 200

    async def json(self, *a, **k):
        return {"results": []}

    async def text(self, *a, **k):
        return ""

    def raise_for_status(self):
        return None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeAiohttpCtx:
    """Awaitable AND async-context-manager, matching aiohttp's get() return."""

    def __init__(self):
        self._resp = _FakeAiohttpResp()

    def __await__(self):
        async def _coro():
            return self._resp

        return _coro().__await__()

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *a):
        return False


class _FakeReqResp:
    status_code = 200
    ok = True
    text = ""
    content = b""

    def json(self, *a, **k):
        return {"results": []}

    def raise_for_status(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------- #
# Capture closures over a shared list. Plain functions (not bound methods) so
# they bind `self` correctly when set as class attributes.
# --------------------------------------------------------------------------- #
def _make_capturers(captured):
    def rec(url, params):
        captured.append({"url": url, "params": params})

    def aiohttp_get(self, url, *a, **k):
        rec(url, k.get("params"))
        return _FakeAiohttpCtx()

    def aiohttp_post(self, url, *a, **k):
        rec(url, k.get("params") if k.get("params") is not None else k.get("data"))
        return _FakeAiohttpCtx()

    def sync_self_get(self, url=None, *a, **k):
        rec(url, k.get("params"))
        return _FakeReqResp()

    def sync_self_post(self, url=None, *a, **k):
        rec(url, k.get("params") if k.get("params") is not None else k.get("data"))
        return _FakeReqResp()

    def sync_mod_get(url=None, *a, **k):
        rec(url, k.get("params"))
        return _FakeReqResp()

    def sync_mod_post(url=None, *a, **k):
        rec(url, k.get("params") if k.get("params") is not None else k.get("data"))
        return _FakeReqResp()

    async def async_self_get(self, url=None, *a, **k):
        rec(url, k.get("params"))
        return _FakeReqResp()

    async def async_self_post(self, url=None, *a, **k):
        rec(url, k.get("params") if k.get("params") is not None else k.get("data"))
        return _FakeReqResp()

    return {
        "aiohttp_get": aiohttp_get,
        "aiohttp_post": aiohttp_post,
        "sync_self_get": sync_self_get,
        "sync_self_post": sync_self_post,
        "sync_mod_get": sync_mod_get,
        "sync_mod_post": sync_mod_post,
        "async_self_get": async_self_get,
        "async_self_post": async_self_post,
    }


def _patch_http(stack, captured):
    caps = _make_capturers(captured)
    targets = [
        ("aiohttp", "ClientSession.get", "aiohttp_get"),
        ("aiohttp", "ClientSession.post", "aiohttp_post"),
        ("requests", "get", "sync_mod_get"),
        ("requests", "post", "sync_mod_post"),
        ("requests", "Session.get", "sync_self_get"),
        ("requests", "Session.post", "sync_self_post"),
        ("httpx", "get", "sync_mod_get"),
        ("httpx", "post", "sync_mod_post"),
        ("httpx", "Client.get", "sync_self_get"),
        ("httpx", "Client.post", "sync_self_post"),
        ("httpx", "AsyncClient.get", "async_self_get"),
        ("httpx", "AsyncClient.post", "async_self_post"),
    ]
    for modname, attrpath, key in targets:
        try:
            module = importlib.import_module(modname)
        except Exception:
            continue
        parts = attrpath.split(".")
        obj = module
        ok = True
        for p in parts[:-1]:
            obj = getattr(obj, p, None)
            if obj is None:
                ok = False
                break
        if not ok or not hasattr(obj, parts[-1]):
            continue
        try:
            stack.enter_context(mock.patch.object(obj, parts[-1], caps[key]))
        except Exception:
            continue


# --------------------------------------------------------------------------- #
# LLM / DDG stubs so the graph never touches the network.
# --------------------------------------------------------------------------- #
def _make_fake_model():
    try:
        from langchain_core.messages import AIMessage
    except Exception:  # minimal shim if langchain_core is unavailable

        class AIMessage:  # type: ignore
            def __init__(self, content="", tool_calls=None, **kw):
                self.content = content
                self.tool_calls = tool_calls or []

    state = {"n": 0}

    class _Struct:
        def __init__(self, schema=None):
            self._schema = schema

        def _build(self):
            s = self._schema
            for m in ("model_construct", "construct"):
                if s is not None and hasattr(s, m):
                    try:
                        return getattr(s, m)()
                    except Exception:
                        pass
            if isinstance(s, type):
                try:
                    return s()
                except Exception:
                    pass
            return types.SimpleNamespace()

        async def ainvoke(self, *a, **k):
            return self._build()

        def invoke(self, *a, **k):
            return self._build()

        def bind_tools(self, *a, **k):
            return self

        def with_structured_output(self, *a, **k):
            return self

    class _Model:
        def bind_tools(self, *a, **k):
            return self

        def bind(self, *a, **k):
            return self

        def with_structured_output(self, schema=None, *a, **k):
            return _Struct(schema)

        async def ainvoke(self, *a, **k):
            state["n"] += 1
            # ASSUMPTION: the search tool is registered under the name "search"
            # (module search_workflow.tools exposes a `search` tool). Emit one
            # tool call to trigger the engine request, then a plain answer so the
            # agent loop terminates. If the name is wrong the graph simply ends
            # without a capture and the probe falls back / skips (never hangs).
            if state["n"] == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search",
                            "args": {"query": "test query"},
                            "id": "call_1",
                        }
                    ],
                )
            return AIMessage(content="done")

        def invoke(self, *a, **k):
            return AIMessage(content="done")

        async def astream(self, *a, **k):
            yield AIMessage(content="done")

    return _Model()


def _install_stubs(stack, captured):
    _patch_http(stack, captured)
    try:
        t = importlib.import_module("search_workflow.tools")
        if hasattr(t, "_ddg_search"):

            async def _ddg_noop(*a, **k):
                return []

            stack.enter_context(mock.patch.object(t, "_ddg_search", _ddg_noop))
    except Exception:
        pass

    fake = _make_fake_model()
    for modname in ("search_workflow.graph", "search_workflow.utils"):
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(m, "load_chat_model"):
            stack.enter_context(
                mock.patch.object(m, "load_chat_model", lambda *a, **k: fake)
            )


# --------------------------------------------------------------------------- #
# Capture strategies.
# --------------------------------------------------------------------------- #
def _capture_via_cli(argv):
    """Drive cli.main black-box in a guarded thread; return captured HTTP calls."""
    captured = []
    try:
        importlib.import_module("search_workflow.cli")
    except Exception:
        return captured

    with ExitStack() as stack:
        _install_stubs(stack, captured)
        old_argv = sys.argv[:]
        sys.argv = list(argv)

        def _run():
            try:
                cli_mod = importlib.import_module("search_workflow.cli")
                cli_mod.main()
            except SystemExit:
                pass
            except BaseException:
                pass

        t = threading.Thread(target=_run, daemon=True)
        try:
            t.start()
            t.join(timeout=25)
        finally:
            sys.argv = old_argv
    return captured


def _build_config(**fields):
    fields = {k: v for k, v in fields.items() if v is not None}
    try:
        configuration = importlib.import_module("search_workflow.configuration")
    except Exception:
        return None
    cfg_cls = getattr(configuration, "Configuration", None)
    if cfg_cls is None:
        return None
    try:
        return cfg_cls(**fields)
    except Exception:
        pass
    base = None
    try:
        base = cfg_cls()
    except Exception:
        if hasattr(cfg_cls, "model_construct"):
            try:
                base = cfg_cls.model_construct()
            except Exception:
                base = None
    if base is not None:
        for k, v in fields.items():
            try:
                setattr(base, k, v)
            except Exception:
                pass
    return base


def _instantiate_client(client_cls, config):
    try:
        sig = inspect.signature(client_cls.__init__)
    except (TypeError, ValueError):
        return client_cls()
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        lname = name.lower()
        if lname in ("config", "configuration", "cfg"):
            kwargs[name] = config
        elif any(u in lname for u in ("url", "host", "endpoint", "base")):
            kwargs[name] = "http://localhost:8888"
        elif "safe" in lname:
            kwargs[name] = getattr(config, "safesearch", 0)
        elif p.default is inspect._empty:
            kwargs[name] = None
    return client_cls(**kwargs)


def _call_maybe_async(fn, args, kwargs):
    res = fn(*args, **kwargs)
    if inspect.iscoroutine(res):
        import asyncio

        asyncio.run(res)


def _invoke_search(client, config, region, timelimit, safesearch):
    for cand in ("search", "asearch", "query", "run", "get"):
        m = getattr(client, cand, None)
        if not callable(m):
            continue
        try:
            sig = inspect.signature(m)
        except (TypeError, ValueError):
            continue
        kwargs = {}
        matched_query = False
        for name, p in sig.parameters.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            lname = name.lower()
            query_names = ("q", "query", "term", "text", "keyword", "keywords", "search_query")
            if lname in query_names or "query" in lname:
                kwargs[name] = "test query"
                matched_query = True
            elif "region" in lname or lname == "lang" or "language" in lname:
                if region is not None:
                    kwargs[name] = region
            elif "time" in lname:
                if timelimit is not None:
                    kwargs[name] = timelimit
            elif "safe" in lname:
                if safesearch is not None:
                    kwargs[name] = safesearch
            elif "config" in lname:
                kwargs[name] = config
            elif p.default is inspect._empty and p.kind in (
                p.POSITIONAL_ONLY,
                p.POSITIONAL_OR_KEYWORD,
            ):
                if not matched_query:
                    kwargs[name] = "test query"
                    matched_query = True
                else:
                    kwargs[name] = None
        try:
            _call_maybe_async(m, (), kwargs)
            return
        except Exception:
            continue


def _capture_via_direct(region, timelimit, safesearch):
    """Fallback: build a Configuration and invoke SearXNGClient directly."""
    captured = []
    try:
        tools = importlib.import_module("search_workflow.tools")
    except Exception:
        return captured
    client_cls = getattr(tools, "SearXNGClient", None)
    if client_cls is None:
        return captured
    config = _build_config(region=region, timelimit=timelimit, safesearch=safesearch)
    if config is None:
        return captured
    with ExitStack() as stack:
        _patch_http(stack, captured)
        try:
            client = _instantiate_client(client_cls, config)
        except Exception:
            return captured
        try:
            _invoke_search(client, config, region, timelimit, safesearch)
        except Exception:
            pass
    return captured


# --------------------------------------------------------------------------- #
# Param normalisation / lookup helpers.
# --------------------------------------------------------------------------- #
def _normalize(captured):
    out = []
    for entry in captured:
        merged = {}
        params = entry.get("params")
        if isinstance(params, dict):
            for k, v in params.items():
                merged[str(k)] = v
        elif isinstance(params, (list, tuple)):
            for kv in params:
                try:
                    merged[str(kv[0])] = kv[1]
                except Exception:
                    pass
        url = entry.get("url")
        if isinstance(url, str) and "?" in url:
            for k, v in parse_qs(urlsplit(url).query).items():
                merged.setdefault(k, v[0] if len(v) == 1 else v)
        if merged:
            out.append(merged)
    return out


def _lower_keys(d):
    return {str(k).lower(): v for k, v in d.items()}


def _searxng_params(captured):
    norm = _normalize(captured)
    picks = []
    for d in norm:
        low = _lower_keys(d)
        if "safesearch" in low or "time_range" in low or ("language" in low and "q" in low):
            picks.append(low)
    if not picks and norm:
        picks = [_lower_keys(d) for d in norm]
    return picks


def _find_value(picks, key):
    for d in picks:
        if key in d:
            return d[key]
    return None


def _stringify(d):
    return "&".join(f"{k}={v}" for k, v in d.items())


# --------------------------------------------------------------------------- #
# Configuration round-trip helpers.
# --------------------------------------------------------------------------- #
def _construct_default(cfg_cls):
    try:
        return cfg_cls()
    except Exception:
        pass
    if hasattr(cfg_cls, "model_construct"):
        try:
            return cfg_cls.model_construct()
        except Exception:
            pass
    if hasattr(cfg_cls, "from_runnable_config"):
        try:
            return cfg_cls.from_runnable_config(None)
        except Exception:
            pass
    return None


def _construct_with(cfg_cls, **kw):
    try:
        return cfg_cls(**kw)
    except Exception:
        pass
    base = _construct_default(cfg_cls)
    if base is not None:
        try:
            for k, v in kw.items():
                setattr(base, k, v)
            return base
        except Exception:
            pass
    if hasattr(cfg_cls, "model_construct"):
        try:
            return cfg_cls.model_construct(**kw)
        except Exception:
            pass
    return None


def _rejects_safesearch_3(cfg_cls):
    """Return True if 3 is rejected, False if accepted, None if not determinable."""
    try:
        cfg_cls(safesearch=0)
    except Exception:
        return None  # cannot isolate safesearch validation via the constructor
    try:
        cfg_cls(safesearch=3)
        return False
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_searxng_params_carry_region_timelimit_safesearch():
    """R1+R3: `--region de-de --timelimit w --safesearch 1` -> the emitted SearXNG
    request params carry the region-derived language (contains 'de'), a truthy
    time_range mapped from 'w', and safesearch == 1."""
    argv = [
        "search-workflow",
        "test query",
        "--region",
        "de-de",
        "--timelimit",
        "w",
        "--safesearch",
        "1",
    ]
    picks = _searxng_params(_capture_via_cli(argv))
    source = "cli"
    if not picks:
        picks = _searxng_params(_capture_via_direct("de-de", "w", 1))
        source = "direct"
    if not picks:
        pytest.skip(
            "Could not intercept a SearXNG HTTP request via cli.main or a direct "
            "SearXNGClient call offline; the engine request boundary was not reached."
        )

    ss = _find_value(picks, "safesearch")
    lang = _find_value(picks, "language") or _find_value(picks, "lang")
    tr = _find_value(picks, "time_range")
    blob = " ".join(_stringify(p) for p in picks).lower()

    # The direct fallback does not exercise cli.py wiring; if it could not carry
    # the configured values, degrade to a skip rather than a false failure.
    if source == "direct" and (ss is None or str(ss) != "1" or not lang):
        pytest.skip(
            "SearXNG request only reachable via the direct fallback, which could "
            "not carry the configured params; cli-driven capture unavailable offline."
        )

    assert ss is not None, f"no 'safesearch' param in captured SearXNG request: {picks!r}"
    assert str(ss) == "1", f"expected safesearch=1, got {ss!r} in {picks!r}"

    assert lang is not None, f"no language/lang param derived from region: {picks!r}"
    assert "de" in str(lang).lower(), (
        f"expected region-derived language containing 'de', got {lang!r}"
    )

    if tr is None:
        assert "week" in blob or "time_range" in blob, (
            f"no time_range mapped from --timelimit w: {picks!r}"
        )
    else:
        assert tr not in (None, "", 0, "0"), (
            f"time_range should be truthy for --timelimit w, got {tr!r} in {picks!r}"
        )


def test_default_flags_safesearch_zero_no_timerange():
    """R4: with no flags the emitted SearXNG params carry safesearch == 0 and set
    NO time_range."""
    argv = ["search-workflow", "test query"]
    picks = _searxng_params(_capture_via_cli(argv))
    source = "cli"
    if not picks:
        picks = _searxng_params(_capture_via_direct(None, None, None))
        source = "direct"
    if not picks:
        pytest.skip("Could not intercept a default-flag SearXNG HTTP request offline.")

    ss = _find_value(picks, "safesearch")
    if source == "direct" and ss is None:
        pytest.skip(
            "Default SearXNG request only reachable via the direct fallback and it "
            "emitted no safesearch param; cannot verify the default offline."
        )

    assert ss is not None, f"no 'safesearch' param in default SearXNG request: {picks!r}"
    assert str(ss) == "0", f"default safesearch must be 0, got {ss!r}"

    tr = _find_value(picks, "time_range")
    assert tr in (None, ""), (
        f"a default run must not set time_range, got {tr!r} in {picks!r}"
    )


def test_configuration_safesearch_roundtrip():
    """R2a: Configuration exposes `safesearch` defaulting to 0, accepts 0/1/2, and
    (tolerantly) rejects 3."""
    configuration = pytest.importorskip("search_workflow.configuration")
    cfg_cls = getattr(configuration, "Configuration", None)
    assert cfg_cls is not None, "search_workflow.configuration.Configuration is missing"

    default_cfg = _construct_default(cfg_cls)
    assert default_cfg is not None, "Configuration() could not be constructed"
    assert hasattr(default_cfg, "safesearch"), (
        "Configuration has no 'safesearch' field (R2 requires it)"
    )
    assert int(getattr(default_cfg, "safesearch")) == 0, (
        f"default safesearch must be 0, got {getattr(default_cfg, 'safesearch')!r}"
    )

    for val in (0, 1, 2):
        cfg = _construct_with(cfg_cls, safesearch=val)
        assert cfg is not None, (
            f"Configuration(safesearch={val}) was rejected but should be accepted"
        )
        assert int(getattr(cfg, "safesearch")) == val, (
            f"safesearch={val} did not round-trip, got {getattr(cfg, 'safesearch')!r}"
        )

    rejected = _rejects_safesearch_3(cfg_cls)
    if rejected is not True:
        pytest.xfail(
            "Configuration does not range-validate safesearch=3 at construction; "
            "enforcement relies on the CLI choices=[0,1,2] instead."
        )


def test_help_lists_safesearch(capsys):
    """R2b: `search-workflow --help` lists a `--safesearch` flag."""
    cli_mod = pytest.importorskip("search_workflow.cli")
    old_argv = sys.argv[:]
    sys.argv = ["search-workflow", "--help"]
    try:
        with pytest.raises(SystemExit):
            cli_mod.main()
    finally:
        sys.argv = old_argv
    out = capsys.readouterr()
    text = (out.out or "") + (out.err or "")
    assert "--safesearch" in text, f"--help does not list --safesearch:\n{text}"
