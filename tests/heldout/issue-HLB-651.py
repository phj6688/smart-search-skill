"""Held-out behavioural probe for HLB-651.

"Add reciprocal-rank-fusion merge with domain cap to the shared core"
Anchor: src/search_workflow/tools.py:search_direct

Acceptance criteria checked (black-box, engine-boundary mocks only):
  1. tools.py exposes a documented RRF k as a module-level int constant.
  2. Dedup runs BEFORE RRF: a URL returned by both engines appears once in the
     merged output, and its combined RRF score (both ranks fused) lifts a
     dual-engine consensus result above a single-engine rank-1 result.
  3. Domain cap: at most 2 results per registrable domain survive the merge.
  4. Both-engines representation: when both engines return results, the merged
     top slice contains entries from each engine.

Rules obeyed here: patch ONLY the engine boundaries (SearXNGClient.search and
tools._ddg_search); never patch search_direct or the merge function. Offline,
deterministic, no subprocess/network.
"""
import asyncio  # noqa: I001  (repo isort config lives outside this file's tree)
import importlib
import inspect
import pathlib
import re
import unittest.mock
import urllib.parse


# --- helpers ---------------------------------------------------------------

def _tools():
    return importlib.import_module("search_workflow.tools")  # SUT named by issue


def _r(link, i):
    # ASSUMPTION: engine results are dicts with at least {title, link, snippet}.
    return {"title": f"t{i}", "link": link, "snippet": f"s{i}"}


def _returner(value, is_async):
    seq = list(value)
    if is_async:
        async def _f(*a, **k):
            return list(seq)
        return _f

    def _f(*a, **k):
        return list(seq)
    return _f


def _link(item):
    if isinstance(item, dict):
        return item.get("link") or item.get("url") or item.get("href") or ""
    return getattr(item, "link", "") or getattr(item, "url", "") or ""


def _as_list(out):
    if isinstance(out, list):
        return out
    if isinstance(out, dict):
        for key in ("results", "articles", "items"):
            if isinstance(out.get(key), list):
                return out[key]
    for attr in ("results", "articles", "items"):
        v = getattr(out, attr, None)
        if isinstance(v, list):
            return v
    raise AssertionError(f"unexpected search_direct output type: {type(out)!r}")


def _run(sx_results, ddg_results, max_results):
    """Patch both engine boundaries and invoke search_direct end-to-end."""
    mod = _tools()
    sx_async = inspect.iscoroutinefunction(mod.SearXNGClient.search)
    ddg_async = inspect.iscoroutinefunction(mod._ddg_search)
    with unittest.mock.patch.object(mod.SearXNGClient, "search",
                                    _returner(sx_results, sx_async)), \
         unittest.mock.patch.object(mod, "_ddg_search",
                                    _returner(ddg_results, ddg_async)):
        # ASSUMPTION: search_direct(query, max_results=...) as named by issue.
        res = mod.search_direct("q", max_results=max_results)
        if inspect.isawaitable(res):
            res = asyncio.run(res)
    return [_link(x) for x in _as_list(res)]


def _host(link):
    return urllib.parse.urlparse(link).netloc.lower()


def _count(links, needle):
    return sum(1 for link in links if needle in link)


def _idx(links, needle):
    for i, link in enumerate(links):
        if needle in link:
            return i
    return -1


# --- tests -----------------------------------------------------------------

def test_rrf_k_is_documented_module_constant():
    tools = _tools()
    src = pathlib.Path(tools.__file__).read_text()
    assert re.search(r"rrf", src, re.IGNORECASE), "no reciprocal-rank-fusion marker in tools.py"
    const_re = re.compile(r"^[A-Z0-9_]*K[A-Z0-9_]*\s*(?::[^=\n]+)?=\s*\d+", re.M)
    rrfk_re = re.compile(r"^\s*rrf_?k\s*(?::[^=\n]+)?=\s*\d+", re.I | re.M)
    has_attr = isinstance(getattr(tools, "RRF_K", None), int)
    assert has_attr or const_re.search(src) or rrfk_re.search(src), \
        "no module-level integer RRF k constant found in tools.py"


def test_dedup_before_rrf_dual_engine_beats_single_engine_rank1():
    url_a = "https://alpha.test/page-a"    # single-engine (searxng) rank0
    url_b = "https://bravo.test/page-b"    # dual-engine: sx rank1 + ddg rank0
    url_c = "https://charlie.test/page-c"
    url_d = "https://delta.test/page-d"
    sx = [_r(url_a, 0), _r(url_b, 1), _r(url_c, 2)]
    ddg = [_r(url_b, 0), _r(url_d, 1)]     # same normalized link B in both engines
    links = _run(sx, ddg, 10)

    assert _count(links, "bravo.test/page-b") == 1, "B not deduped to a single merged entry"
    bi, ai = _idx(links, "bravo.test/page-b"), _idx(links, "alpha.test/page-a")
    assert bi != -1 and ai != -1, "expected both A and B in merged output"
    # B carries A's rank-0-equivalent term PLUS an extra fused term -> outranks A.
    assert bi < ai, "dual-engine consensus (B) should outrank single-engine rank-1 (A)"


def test_domain_cap_two_per_registrable_domain():
    sx = [_r("https://www.shop.com/1", 1), _r("https://www.shop.com/2", 2),
          _r("https://www.shop.com/3", 3)]
    ddg = [_r("https://www.shop.com/4", 4), _r("https://www.shop.com/5", 5),
           _r("https://other.test/x", 6)]
    links = _run(sx, ddg, 10)

    shop = [link for link in links if _host(link).endswith("shop.com")]
    assert len(shop) <= 2, f"domain cap breached: {len(shop)} shop.com results survived"
    assert len(links) >= 1, "merge dropped everything"


def test_both_engines_represented_in_merged_top():
    sx = [_r("https://a.sx.example/1", 1), _r("https://b.sx.example/2", 2),
          _r("https://c.sx.example/3", 3)]
    ddg = [_r("https://a.ddg.example/1", 4), _r("https://b.ddg.example/2", 5),
           _r("https://c.ddg.example/3", 6)]
    links = _run(sx, ddg, 6)

    assert any("sx.example" in link for link in links), "searxng engine unrepresented in merged top"
    assert any("ddg.example" in link for link in links), "ddg engine unrepresented in merged top"
