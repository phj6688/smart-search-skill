"""Command line interface for search workflow."""
import argparse
import asyncio
import json
from importlib.metadata import PackageNotFoundError, version

from . import run_workflow

_DIST_NAME = "smart-search-skill"
# Sentinel for source checkouts where the distribution is not installed.
_DEV_VERSION = "0.0.0.dev0"


def _package_version() -> str:
    """Resolve the installed distribution version, with a dev sentinel fallback."""
    try:
        return version(_DIST_NAME)
    except PackageNotFoundError:
        return _DEV_VERSION


async def _async_main() -> int:
    """Async CLI implementation."""
    parser = argparse.ArgumentParser(description="Search Workflow CLI")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--version", action="version", version=_package_version())
    parser.add_argument("--region", default="us-en", help="Search region")
    parser.add_argument("--timelimit", choices=['d', 'w', 'm', 'y'], help="Time limit")
    parser.add_argument("--format", choices=['json', 'text'], default='json', help="Output format")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum results")

    args = parser.parse_args()

    # Configure search
    config = {
        "configurable": {
            "max_search_results_evaluator": args.max_results
        }
    }

    try:
        results = await run_workflow(args.query, config=config)

        if args.format == 'json':
            print(json.dumps(results, indent=2))
        else:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   {result['link']}")
                print(f"   {result['snippet'][:100]}...")
                print()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def main() -> int:
    """Sync entry point: console scripts call this directly, so it must not
    return a coroutine (the previous async main was never awaited)."""
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
