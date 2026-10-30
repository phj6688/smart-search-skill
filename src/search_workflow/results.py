"""Shared result metadata and evaluator payloads, independent of model clients."""

from typing import Any

# S03 engine short names mapped to the documented result-path names. The
# attribution record uses "ddg"; the result path surfaces "duckduckgo".
_ENGINE_DISPLAY_NAMES = {"searxng": "searxng", "ddg": "duckduckgo"}



# Cap on the snippet text handed to the evaluator LLM. The model selects by
# index and never re-emits result fields, so it needs only enough snippet to
# judge relevance; ~200 chars bounds the per-result token cost while the
# returned results (joined from `fetched`) stay the verbatim fetched bytes.
_EVALUATOR_SNIPPET_CAP = 200


def _minimized_evaluator_payload(fetched: list[dict[str, Any]]) -> str:
    """Render the fetched results as indexed, URL-free, snippet-capped records.

    The evaluator selects by position into `fetched` (the join back to the
    fetched objects happens in the node), so the model never needs the raw URLs
    or the full snippet. Withholding the `link` field keeps every URL out of the
    LLM payload, and truncating the snippet bounds the token cost. This minimizes
    only what the model SEES; the returned results are still the fetched objects,
    untouched.
    """
    records = []
    for index, item in enumerate(fetched):
        title = str(item.get("title", ""))
        snippet = str(item.get("snippet", ""))[:_EVALUATOR_SNIPPET_CAP]
        records.append(f"[{index}] {title}\n{snippet}")
    return "\n\n".join(records)



def _surface_provenance(
    provenance: dict[str, Any] | None, evaluator_degraded: bool = False
) -> dict[str, Any]:
    """Map the S03 attribution record to the result-path metadata triple.

    Reads engines_used/n_searxng/n_ddg straight off the provenance record so
    the metadata FOLLOWS what the tool path attributed, instead of re-deriving
    which engine answered from result contents. An engine "served results" when
    it returned any raw rows (n>0), even if dedup later dropped them; both are
    always attempted, so fewer served than attempted is the degraded signal.

    evaluator_degraded is HLB-658's malformed-selection signal (the evaluator
    node fell back to raw results). It surfaces degraded_reason="evaluator", but
    engine degradation still WINS: a single-engine fetch is a more fundamental
    data-quality loss than an unusable selection over an otherwise-complete
    fetch, so the evaluator reason only shows when the engines were healthy.
    """
    if not provenance:
        engines_used: list[str] = []
        engine_degraded = False
    else:
        engines_used = [
            _ENGINE_DISPLAY_NAMES.get(engine, engine)
            for engine in provenance.get("engines_used", [])
        ]
        served = int(provenance.get("n_searxng", 0) > 0) + int(
            provenance.get("n_ddg", 0) > 0
        )
        engine_degraded = served < 2
    if engine_degraded:
        degraded_reason: str | None = "engine"
    elif evaluator_degraded:
        degraded_reason = "evaluator"
    else:
        degraded_reason = None
    return {
        "engines_used": engines_used,
        "degraded": engine_degraded or evaluator_degraded,
        "degraded_reason": degraded_reason,
    }

