"""Command line interface for search workflow."""
import asyncio
import argparse
import json
from . import run_workflow

async def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Search Workflow CLI")
    parser.add_argument("query", help="Search query")
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

if __name__ == "__main__":
    exit(asyncio.run(main()))
