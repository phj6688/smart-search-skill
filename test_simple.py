# test_simple.py - Quick functionality test
import asyncio
from src.search_workflow import run_workflow

async def quick_test():
    print("🔍 Quick Search Workflow Test")
    try:
        results = await run_workflow("AI developments", config={
            "configurable": {"max_search_results_evaluator": 3}
        })
        
        print(f"✅ Success! Retrieved {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title'][:50]}...")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(quick_test())
