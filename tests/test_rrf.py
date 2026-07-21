"""Reciprocal-rank-fusion merge with a domain cap (HLB-651).

The shared core's merge is no longer plain concatenation. _rrf_merge applies a
fixed order: normalize_url dedup FIRST, then reciprocal rank fusion (RRF), then
a cap of RRF_DOMAIN_CAP per registrable domain, then truncation. These tests
pin, in that order:

* RRF scoring reads k from the RRF_K module constant, and a document both
  engines rank mid-list outranks a document one engine ranks first (reorder).
* dedup before RRF: a URL from both engines becomes one entry whose fused score
  adds both ranks and keeps both engines in the attribution set.
* the domain cap keeps at most two per REGISTRABLE domain (co.uk handled), and
  both engines appear at the top when both return results.
* an offline, deterministic nDCG@10 comparison over the FIX-MERGE labels: the
  RRF ordering scores at or above the concat-ordered baseline through the same
  dedup + cap + truncate pipeline.
"""

import json
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from search_workflow import tools

FIX_MERGE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "FIX-MERGE.json").read_text()
)


def _result(link: str, title: str = "t", snippet: str = "s") -> dict[str, Any]:
    return {"title": title, "link": link, "snippet": snippet}


# --- registrable-domain heuristic -----------------------------------------


def test_rrf_registrable_domain_simple_and_multipart() -> None:
    # last two labels for an ordinary host...
    assert tools._registrable_domain("docs.example.dev") == "example.dev"
    assert tools._registrable_domain("a.b.example.com") == "example.com"
    assert tools._registrable_domain("example.com") == "example.com"
    # ...but last three when the final two are a known multi-part suffix, so a
    # bare split does not collapse every co.uk site onto one key.
    assert tools._registrable_domain("a.example.co.uk") == "example.co.uk"
    assert tools._registrable_domain("example.co.uk") == "example.co.uk"
    assert tools._registrable_domain("shop.brand.com.au") == "brand.com.au"


# --- RRF scoring and the reorder assertion --------------------------------


def test_rrf_consensus_midlist_outranks_single_engine_top() -> None:
    """A URL both engines rank mid-list outranks a URL one engine ranks first.

    Reads k from the RRF_K module constant, not a local literal. Two engines'
    contributions overtake one engine's rank-0 hit only when k exceeds the
    consensus rank (2/(k+r) > 1/k  <=>  k > r); with the classic k=60 a
    ten-result set is nearly flat and the single-engine number one would win.
    """
    mid = 4  # 0-based middle of a ten-result list
    assert tools.RRF_K > mid, (
        "RRF_K must exceed the consensus rank for the reorder this fixture proves"
    )

    # Ten results per engine, each on its own registrable domain so the domain
    # cap never intervenes in this ranking-only assertion.
    searxng = [_result(f"https://sx-{i}.example/p") for i in range(10)]
    ddg = [_result(f"https://dg-{i}.example/p") for i in range(10)]

    consensus = "https://consensus.example/x"
    searxng[mid]["link"] = consensus  # mid-list in engine A
    ddg[mid]["link"] = consensus  # mid-list in engine B
    single_top = searxng[0]["link"]  # rank one, engine A only

    merged, _engines, _n = tools._rrf_merge(searxng, ddg, max_results=10)
    links = [r["link"] for r in merged]

    assert links.index(consensus) < links.index(single_top)


# --- dedup before RRF fuses both ranks ------------------------------------


def test_rrf_dedup_collapses_same_url_from_both_engines() -> None:
    url = "https://both.example/x"
    merged, engines_used, n_after_dedup = tools._rrf_merge(
        [_result(url, "sx")], [_result(url, "dg")], max_results=10
    )
    # One URL returned by both engines becomes a single entry.
    assert n_after_dedup == 1
    key = tools.normalize_url(url)
    survivors = [r for r in merged if tools.normalize_url(r["link"]) == key]
    assert len(survivors) == 1
    # Result-source attribution (HLB-646) is first-seen: the survivor is tagged
    # with the engine that returned it first, so DDG enters engines_used only
    # through a DDG-UNIQUE survivor, never through a URL both engines returned.
    # This is the same rule the searxng_ok_ddg_unused fallback state pins.
    assert engines_used == {"searxng"}


def test_rrf_dedup_fuses_both_ranks_and_beats_single_engine_top() -> None:
    """The deduped entry's fused score adds both engine ranks.

    A URL both engines rank SECOND (rank 1) must beat a URL one engine ranks
    first (rank 0): 2/(k+1) > 1/k holds for k > 1. Without fusing both ranks the
    single-engine rank-0 hit (1/k) would win, so this proves dedup ran before
    RRF and combined the ranks rather than letting the copies compete.
    """
    assert tools.RRF_K > 1
    dup = "https://dup.example/x"  # rank 1 in BOTH engines
    solo = "https://solo.example/y"  # rank 0 in searxng only

    searxng = [_result(solo, "solo"), _result(dup, "dup-sx")]
    ddg = [_result("https://other.example/z", "other"), _result(dup, "dup-dg")]

    merged, engines_used, _n = tools._rrf_merge(searxng, ddg, max_results=10)
    key = tools.normalize_url(dup)
    assert len([r for r in merged if tools.normalize_url(r["link"]) == key]) == 1
    assert engines_used == {"searxng", "ddg"}

    links = [r["link"] for r in merged]
    assert links.index(dup) < links.index(solo)


# --- domain cap ------------------------------------------------------------


def _cluster_record() -> dict[str, Any]:
    for rec in FIX_MERGE:
        if rec["category"] == "same_domain_cluster":
            return rec
    raise AssertionError("no same_domain_cluster record in FIX-MERGE.json")


def _split_by_engine(
    rec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    searxng = sorted(
        (dict(r) for r in rec["results"] if r["engine"] == "searxng"),
        key=lambda r: r["rank"],
    )
    ddg = sorted(
        (dict(r) for r in rec["results"] if r["engine"] != "searxng"),
        key=lambda r: r["rank"],
    )
    return searxng, ddg


def test_rrf_domain_cap_keeps_at_most_two_per_registrable_domain() -> None:
    rec = _cluster_record()
    # Five distinct pages, all on one registrable domain.
    domains = {
        tools._registrable_domain(urlsplit(r["link"]).hostname or "")
        for r in rec["results"]
    }
    assert len(rec["results"]) == 5 and len(domains) == 1

    searxng, ddg = _split_by_engine(rec)
    merged, _engines, n_after_dedup = tools._rrf_merge(searxng, ddg, max_results=10)

    assert n_after_dedup == 5, "the five pages are distinct URLs, none dedup away"
    assert len(merged) == tools.RRF_DOMAIN_CAP == 2, "domain cap keeps at most two"
    kept_domains = [
        tools._registrable_domain(urlsplit(r["link"]).hostname or "") for r in merged
    ]
    assert all(d == kept_domains[0] for d in kept_domains)


def test_rrf_domain_cap_uses_registrable_domain_not_bare_host() -> None:
    # Three subdomains of one registrable domain under a multi-part suffix; a
    # bare host split would treat them as distinct and never cap.
    searxng = [
        _result("https://a.example.co.uk/1"),
        _result("https://b.example.co.uk/2"),
        _result("https://c.example.co.uk/3"),
    ]
    merged, _engines, _n = tools._rrf_merge(searxng, [], max_results=10)
    assert len(merged) == 2


# --- both engines represented at the top ----------------------------------


def test_rrf_both_engines_represented_at_top() -> None:
    searxng = [_result("https://sx-a.example/1"), _result("https://sx-b.example/2")]
    ddg = [_result("https://dg-a.example/1"), _result("https://dg-b.example/2")]

    merged, engines_used, _n = tools._rrf_merge(searxng, ddg, max_results=10)
    assert engines_used == {"searxng", "ddg"}

    top_hosts = {urlsplit(r["link"]).hostname or "" for r in merged[:2]}
    assert any(h.startswith("sx-") for h in top_hosts), top_hosts
    assert any(h.startswith("dg-") for h in top_hosts), top_hosts


# --- nDCG@10: RRF ordering at or above the concat baseline -----------------


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def _ndcg_at_10(returned_relevances: list[int], all_relevances: list[int]) -> float:
    ideal = _dcg(sorted(all_relevances, reverse=True)[:10])
    return _dcg(returned_relevances[:10]) / ideal if ideal > 0 else 0.0


def _concat_cap_truncate(
    searxng: list[dict[str, Any]],
    ddg: list[dict[str, Any]],
    max_results: int,
) -> list[dict[str, Any]]:
    """The OLD ordering (SearXNG then dedup-surviving DDG) through the SAME
    dedup + domain cap + truncate trim, so nDCG isolates the ordering change."""
    seen: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for r in [*searxng, *ddg]:
        key = tools.normalize_url(r["link"])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(r)

    kept: list[dict[str, Any]] = []
    per_domain: dict[str, int] = {}
    for r in ordered:
        domain = tools._registrable_domain(urlsplit(r["link"]).hostname or "")
        if domain and per_domain.get(domain, 0) >= tools.RRF_DOMAIN_CAP:
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        kept.append(r)
        if len(kept) >= max_results:
            break
    return kept


def _relevances(
    returned: list[dict[str, Any]], rel_by_key: dict[str, int]
) -> list[int]:
    return [rel_by_key[tools.normalize_url(r["link"])] for r in returned]


def test_rrf_ndcg_at_or_above_concat_baseline() -> None:
    """RRF merge scores at or above the concat baseline on nDCG@10, offline.

    Both orderings run through the identical dedup + domain cap + truncate
    pipeline, differing only in the ranking step (RRF vs concat), so the metric
    measures exactly the reordering this story introduces. nDCG@10 uses the
    ideal ordering of each query's full labelled relevance list as the ideal DCG.
    """
    rrf_scores: list[float] = []
    concat_scores: list[float] = []
    cluster_rrf: list[float] = []
    cluster_concat: list[float] = []

    for rec in FIX_MERGE:
        searxng, ddg = _split_by_engine(rec)
        rel_by_key = {
            tools.normalize_url(r["link"]): rel
            for r, rel in zip(rec["results"], rec["labels"]["relevance"])
        }
        all_relevances = rec["labels"]["relevance"]

        rrf_out, _e, _n = tools._rrf_merge(searxng, ddg, max_results=10)
        concat_out = _concat_cap_truncate(searxng, ddg, max_results=10)

        rrf_score = _ndcg_at_10(_relevances(rrf_out, rel_by_key), all_relevances)
        concat_score = _ndcg_at_10(_relevances(concat_out, rel_by_key), all_relevances)
        rrf_scores.append(rrf_score)
        concat_scores.append(concat_score)
        if rec["category"] == "same_domain_cluster":
            cluster_rrf.append(rrf_score)
            cluster_concat.append(concat_score)

    mean_rrf = sum(rrf_scores) / len(rrf_scores)
    mean_concat = sum(concat_scores) / len(concat_scores)
    # Recorded for the PR body; kept in the assertion message so a regression
    # prints both numbers.
    assert mean_rrf >= mean_concat, (
        f"RRF nDCG@10 {mean_rrf:.4f} fell below concat baseline {mean_concat:.4f}"
    )
    # The win must be real, not a wash: on the same-domain clusters the cap
    # budget spent on the cross-engine consensus pair beats concat spending it
    # on one engine's top two.
    assert sum(cluster_rrf) / len(cluster_rrf) > sum(cluster_concat) / len(
        cluster_concat
    )
