# test_docker.py - Test Docker SearXNG integration
import asyncio
import aiohttp

async def test_docker_searxng():
    print("🐳 Testing SearXNG Docker Integration")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            async with session.get("http://localhost:9090/healthz", timeout=5) as response:
                if response.status == 200:
                    print("✅ SearXNG is healthy")
                    
                    # API test
                    async with session.get(
                        "http://localhost:9090/search",
                        params={"q": "AI", "format": "json", "categories": "news"},
                        timeout=10
                    ) as search_response:
                        data = await search_response.json()
                        results = data.get("results", [])
                        print(f"✅ SearXNG API working: {len(results)} results")
                        
                        if results:
                            print(f"Sample: {results[0].get('title', 'No title')}")
                else:
                    print(f"❌ SearXNG health check failed: {response.status}")
                    
    except Exception as e:
        print(f"❌ SearXNG test failed: {e}")
        print("💡 Make sure SearXNG is running: cd docker && ./scripts/start.sh")

if __name__ == "__main__":
    asyncio.run(test_docker_searxng())
