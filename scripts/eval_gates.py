#!/usr/bin/env python3
"""Load and shape-validate the HLB-648 evaluation corpora.

Skeleton CLI. Each subcommand loads one corpus, checks its JSON shape, and
prints counts. No quality thresholds are enforced: a corpus that loads and
validates always exits 0. Wiring thresholds into CI or merge gates is out of
scope.

    eval_gates.py merge   [--fixture PATH]
    eval_gates.py news    [--fixture PATH]
    eval_gates.py inject  [--fixture PATH]
    eval_gates.py all
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

MERGE_FIXTURE = FIXTURES_DIR / "FIX-MERGE.json"
NEWS_FIXTURE = FIXTURES_DIR / "FIX-NEWS.json"
INJECT_FIXTURE = FIXTURES_DIR / "FIX-INJECT.json"

RESULT_FIELDS = {"title", "link", "snippet", "engine", "rank"}
MERGE_CATEGORIES = {"dup_pair", "case_sensitive_path", "same_domain_cluster"}


class CorpusError(ValueError):
    """Raised when a corpus does not match its expected shape."""


def _load_array(path: Path) -> list:
    if not path.exists():
        raise CorpusError(f"fixture not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise CorpusError(f"{path.name}: top level must be a JSON array")
    return data


def _require_results(item: dict, results: object, where: str) -> list:
    if not isinstance(results, list) or not results:
        raise CorpusError(f"{item.get('id', '?')}: {where} must be a non-empty list")
    for result in results:
        missing = RESULT_FIELDS - set(result)
        if missing:
            raise CorpusError(
                f"{item.get('id', '?')}: {where} result missing {sorted(missing)}"
            )
    return results


def _require_aligned(item: dict, values: object, length: int, label: str) -> list:
    if not isinstance(values, list) or len(values) != length:
        raise CorpusError(
            f"{item.get('id', '?')}: labels.{label} must align with {length} results"
        )
    return values


def validate_merge(path: Path) -> dict:
    data = _load_array(path)
    categories: Counter = Counter()
    duplicate_groups = 0
    for item in data:
        results = _require_results(item, item.get("results"), "results")
        category = item.get("category")
        if category not in MERGE_CATEGORIES:
            raise CorpusError(f"{item.get('id', '?')}: unknown category {category!r}")
        categories[category] += 1
        labels = item.get("labels", {})
        _require_aligned(item, labels.get("relevance"), len(results), "relevance")
        groups = _require_aligned(
            item, labels.get("duplicate_groups"), len(results), "duplicate_groups"
        )
        duplicate_groups += len({g for g in groups if g is not None})
    return {
        "items": len(data),
        "categories": dict(categories),
        "duplicate_groups": duplicate_groups,
    }


def validate_news(path: Path) -> dict:
    data = _load_array(path)
    kinds: Counter = Counter()
    for item in data:
        results = _require_results(item, item.get("results"), "results")
        for result in results:
            kinds[result.get("kind")] += 1
        labels = item.get("labels", {})
        _require_aligned(item, labels.get("relevance"), len(results), "relevance")
    return {"items": len(data), "result_kinds": dict(kinds)}


def validate_inject(path: Path) -> dict:
    data = _load_array(path)
    payload_fields: Counter = Counter()
    for item in data:
        clean = _require_results(item, item.get("clean", {}).get("results"), "clean.results")
        dirty = _require_results(item, item.get("dirty", {}).get("results"), "dirty.results")
        if len(clean) != len(dirty):
            raise CorpusError(f"{item.get('id', '?')}: clean and dirty differ in length")
        labels = item.get("labels", {})
        payload_index = labels.get("payload_index")
        expected = labels.get("expected_selection")
        payload_field = labels.get("payload_field")
        for name, value in (("payload_index", payload_index), ("expected_selection", expected)):
            if not isinstance(value, int) or not 0 <= value < len(dirty):
                raise CorpusError(f"{item.get('id', '?')}: labels.{name} out of range")
        if payload_field not in {"title", "snippet"}:
            raise CorpusError(f"{item.get('id', '?')}: labels.payload_field invalid")
        if expected == payload_index:
            raise CorpusError(
                f"{item.get('id', '?')}: expected_selection must not be the payload result"
            )
        payload_fields[payload_field] += 1
    return {"items": len(data), "payload_fields": dict(payload_fields)}


CORPORA = {
    "merge": (MERGE_FIXTURE, validate_merge),
    "news": (NEWS_FIXTURE, validate_news),
    "inject": (INJECT_FIXTURE, validate_inject),
}


def _run_one(name: str, fixture: Path) -> None:
    default_path, validator = CORPORA[name]
    path = fixture or default_path
    summary = validator(path)
    print(f"{name}: {path.name} OK")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in CORPORA:
        corpus_parser = sub.add_parser(name, help=f"load and validate the {name} corpus")
        corpus_parser.add_argument(
            "--fixture", type=Path, default=None, help="override the fixture path"
        )
    sub.add_parser("all", help="load and validate every corpus")
    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "all":
            for name in CORPORA:
                _run_one(name, CORPORA[name][0])
        else:
            _run_one(args.command, args.fixture)
    except CorpusError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
