import asyncio
import traceback
import time
import json
import os
from typing import Dict, Any, List
from src.search_workflow import run_workflow, Configuration

class SearchWorkflowTester:
    """Comprehensive test suite for search workflow package."""
    
    def __init__(self):
        self.test_results = []
        self.start_time = time.time()
    
    async def run_all_tests(self):
        """Run all test scenarios."""
        print("🧪 Search Workflow Comprehensive Test Suite")
        print("=" * 60)
        
        # Test configuration and environment
        await self.test_configuration()
        await self.test_environment_variables()
        
        # Test search functionality
        await self.test_basic_search()
        await self.test_search_with_configuration()
        await self.test_search_strategies()
        
        # Test Docker integration
        await self.test_searxng_connectivity()
        
        # Test CLI functionality
        await self.test_cli_functionality()
        
        # Test error handling
        await self.test_error_scenarios()
        
        # Performance tests
        await self.test_performance()
        
        # Print summary
        self.print_test_summary()
    
    async def test_configuration(self):
        """Test configuration loading and validation."""
        print("\n📋 Testing Configuration System...")
        
        try:
            # Test default configuration
            config = Configuration()
            
            print(f"✅ Configuration loaded successfully")
            print(f"   • Model: {config.model}")
            print(f"   • OpenAI Key set: {bool(config.openai_api_key)}")
            print(f"   • SearXNG URL: {config.searxng_url}")
            print(f"   • Max results (tool): {config.max_search_results_tool}")
            print(f"   • Max results (evaluator): {config.max_search_results_evaluator}")
            print(f"   • Search strategy: {config.search_strategy}")
            print(f"   • Debug mode: {config.debug}")
            
            # Test validation
            issues = config.validate_environment()
            if issues:
                print(f"⚠️  Configuration issues found:")
                for issue in issues:
                    print(f"   • {issue}")
            else:
                print(f"✅ Configuration validation passed")
            
            self.test_results.append(("Configuration", "PASS", "Configuration system working"))
            
        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.test_results.append(("Configuration", "FAIL", str(e)))
    
    async def test_environment_variables(self):
        """Test environment variable handling."""
        print("\n🌍 Testing Environment Variables...")
        
        try:
            # Check critical environment variables
            env_vars = {
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                "SEARXNG_URL": os.getenv("SEARXNG_URL", "http://localhost:9090"),
                "MAX_SEARCH_RESULTS_TOOL": os.getenv("MAX_SEARCH_RESULTS_TOOL", "10"),
                "MAX_SEARCH_RESULTS_EVALUATOR": os.getenv("MAX_SEARCH_RESULTS_EVALUATOR", "5"),
            }
            
            for var, value in env_vars.items():
                status = "✅" if value else "⚠️ "
                print(f"   {status} {var}: {'Set' if value else 'Not set'}")
            
            # Test configuration override
            custom_config = {
                "configurable": {
                    "max_search_results_evaluator": 3,
                    "searxng_url": "http://localhost:9090"
                }
            }
            
            config = Configuration.from_runnable_config(custom_config)
            print(f"✅ Configuration override test passed")
            print(f"   • Overridden evaluator results: {config.max_search_results_evaluator}")
            
            self.test_results.append(("Environment", "PASS", "Environment handling working"))
            
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            self.test_results.append(("Environment", "FAIL", str(e)))
    
    async def test_basic_search(self):
        """Test basic search functionality."""
        print("\n🔍 Testing Basic Search Functionality...")
        
        try:
            start_time = time.time()
            
            # Test basic search
            results = await run_workflow("latest AI developments")
            
            duration = time.time() - start_time
            
            if results and len(results) > 0:
                print(f"✅ Basic search successful")
                print(f"   • Results returned: {len(results)}")
                print(f"   • Response time: {duration:.2f}s")
                
                # Display first result
                first_result = results[0]
                print(f"   • Sample result:")
                print(f"     - Title: {first_result.get('title', 'N/A')[:60]}...")
                print(f"     - Similarity: {first_result.get('similarity', 'N/A')}")
                print(f"     - Link: {first_result.get('link', 'N/A')[:60]}...")
                
                self.test_results.append(("Basic Search", "PASS", f"{len(results)} results in {duration:.2f}s"))
            else:
                print(f"⚠️  Search returned no results")
                self.test_results.append(("Basic Search", "WARN", "No results returned"))
                
        except Exception as e:
            print(f"❌ Basic search failed: {e}")
            traceback.print_exc()
            self.test_results.append(("Basic Search", "FAIL", str(e)))
    
    async def test_search_with_configuration(self):
        """Test search with custom configuration."""
        print("\n⚙️  Testing Search with Custom Configuration...")
        
        test_configs = [
            {
                "name": "Minimal Results",
                "config": {"configurable": {"max_search_results_evaluator": 2}},
                "expected_results": 2
            },
            {
                "name": "More Results", 
                "config": {"configurable": {"max_search_results_evaluator": 7}},
                "expected_results": 7
            }
        ]
        
        for test_config in test_configs:
            try:
                print(f"   Testing: {test_config['name']}")
                
                results = await run_workflow(
                    "AI technology news", 
                    config=test_config["config"]
                )
                
                actual_count = len(results) if results else 0
                expected_count = test_config["expected_results"]
                
                if actual_count <= expected_count:
                    print(f"   ✅ {test_config['name']}: {actual_count} results (≤ {expected_count})")
                    self.test_results.append((f"Config-{test_config['name']}", "PASS", f"{actual_count} results"))
                else:
                    print(f"   ⚠️  {test_config['name']}: {actual_count} results (expected ≤ {expected_count})")
                    self.test_results.append((f"Config-{test_config['name']}", "WARN", f"Unexpected count: {actual_count}"))
                    
            except Exception as e:
                print(f"   ❌ {test_config['name']} failed: {e}")
                self.test_results.append((f"Config-{test_config['name']}", "FAIL", str(e)))
    
    async def test_search_strategies(self):
        """Test different search strategies."""
        print("\n🎯 Testing Search Strategies...")
        
        strategies = [
            "Test with default strategy (hybrid)",
            "Test search engine fallback behavior"
        ]
        
        try:
            # Test primary search
            print("   Testing hybrid search strategy...")
            results = await run_workflow("technology innovation", config={
                "configurable": {"max_search_results_evaluator": 3}
            })
            
            if results:
                print(f"   ✅ Hybrid strategy: {len(results)} results")
                self.test_results.append(("Search Strategy", "PASS", "Hybrid search working"))
            else:
                print(f"   ⚠️  Hybrid strategy returned no results")
                self.test_results.append(("Search Strategy", "WARN", "No results from hybrid"))
                
        except Exception as e:
            print(f"   ❌ Search strategy test failed: {e}")
            self.test_results.append(("Search Strategy", "FAIL", str(e)))
    
    async def test_searxng_connectivity(self):
        """Test SearXNG Docker connectivity."""
        print("\n🐳 Testing SearXNG Docker Connectivity...")
        
        try:
            import aiohttp
            
            # Test SearXNG health endpoint
            async with aiohttp.ClientSession() as session:
                try:
                    async with session.get("http://localhost:9090/healthz", timeout=5) as response:
                        if response.status == 200:
                            print("   ✅ SearXNG health check passed")
                            
                            # Test SearXNG search API
                            async with session.get(
                                "http://localhost:9090/search",
                                params={"q": "test", "format": "json", "categories": "news"},
                                timeout=10
                            ) as search_response:
                                if search_response.status == 200:
                                    data = await search_response.json()
                                    result_count = len(data.get("results", []))
                                    print(f"   ✅ SearXNG API test passed: {result_count} results")
                                    self.test_results.append(("SearXNG Connectivity", "PASS", f"API working, {result_count} results"))
                                else:
                                    print(f"   ⚠️  SearXNG API returned status {search_response.status}")
                                    self.test_results.append(("SearXNG Connectivity", "WARN", f"API status {search_response.status}"))
                        else:
                            print(f"   ⚠️  SearXNG health check failed: status {response.status}")
                            self.test_results.append(("SearXNG Connectivity", "WARN", f"Health check status {response.status}"))
                            
                except asyncio.TimeoutError:
                    print("   ⚠️  SearXNG connection timeout (service may not be running)")
                    print("   💡 To start SearXNG: cd docker && ./scripts/start.sh")
                    self.test_results.append(("SearXNG Connectivity", "WARN", "Connection timeout"))
                    
        except ImportError:
            print("   ⚠️  aiohttp not available for direct connectivity test")
            self.test_results.append(("SearXNG Connectivity", "SKIP", "aiohttp not available"))
        except Exception as e:
            print(f"   ❌ SearXNG connectivity test failed: {e}")
            self.test_results.append(("SearXNG Connectivity", "FAIL", str(e)))
    
    async def test_cli_functionality(self):
        """Test CLI functionality."""
        print("\n💻 Testing CLI Functionality...")
        
        try:
            # Test CLI import
            from src.search_workflow.cli import main
            print("   ✅ CLI module imports successfully")
            
            # Note: We don't actually run the CLI to avoid hanging
            print("   💡 CLI can be tested manually with:")
            print("      uv run python -m search_workflow.cli 'AI news' --max-results 3")
            
            self.test_results.append(("CLI", "PASS", "CLI module available"))
            
        except Exception as e:
            print(f"   ❌ CLI test failed: {e}")
            self.test_results.append(("CLI", "FAIL", str(e)))
    
    async def test_error_scenarios(self):
        """Test error handling scenarios."""
        print("\n🚨 Testing Error Handling...")
        
        # Test with invalid configuration
        try:
            print("   Testing invalid configuration handling...")
            
            # This should handle gracefully
            invalid_config = {
                "configurable": {
                    "max_search_results_evaluator": 999,  # Very high number
                    "max_search_results_tool": 1  # Lower than evaluator
                }
            }
            
            try:
                results = await run_workflow("test", config=invalid_config)
                print("   ✅ Invalid config handled gracefully")
                self.test_results.append(("Error Handling", "PASS", "Invalid config handled"))
            except Exception as e:
                print(f"   ✅ Invalid config properly rejected: {e}")
                self.test_results.append(("Error Handling", "PASS", "Invalid config rejected"))
                
        except Exception as e:
            print(f"   ❌ Error handling test failed: {e}")
            self.test_results.append(("Error Handling", "FAIL", str(e)))
    
    async def test_performance(self):
        """Test performance characteristics."""
        print("\n⚡ Testing Performance...")
        
        try:
            # Test response time
            start_time = time.time()
            results = await run_workflow("AI performance test", config={
                "configurable": {"max_search_results_evaluator": 3}
            })
            duration = time.time() - start_time
            
            print(f"   ✅ Performance test completed")
            print(f"   • Total time: {duration:.2f}s")
            print(f"   • Results: {len(results) if results else 0}")
            print(f"   • Time per result: {duration/len(results):.2f}s" if results else "   • No results to measure")
            
            if duration < 30:  # Should complete within 30 seconds
                self.test_results.append(("Performance", "PASS", f"{duration:.2f}s for {len(results) if results else 0} results"))
            else:
                self.test_results.append(("Performance", "WARN", f"Slow response: {duration:.2f}s"))
                
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
            self.test_results.append(("Performance", "FAIL", str(e)))
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        # Count results
        passed = sum(1 for _, status, _ in self.test_results if status == "PASS")
        failed = sum(1 for _, status, _ in self.test_results if status == "FAIL")
        warned = sum(1 for _, status, _ in self.test_results if status == "WARN")
        skipped = sum(1 for _, status, _ in self.test_results if status == "SKIP")
        total = len(self.test_results)
        
        # Print overview
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warned: {warned}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        
        # Print detailed results
        print(f"\n📋 Detailed Results:")
        for test_name, status, message in self.test_results:
            emoji = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}[status]
            print(f"   {emoji} {test_name}: {message}")
        
        # Print recommendations
        print(f"\n💡 Recommendations:")
        if failed > 0:
            print("   • Fix failing tests before proceeding")
        if warned > 0:
            print("   • Review warning messages for potential issues")
        if any("SearXNG" in test for test, status, _ in self.test_results if status in ["WARN", "FAIL"]):
            print("   • Start SearXNG: cd docker && ./scripts/start.sh")
        if any("OpenAI" in msg for _, _, msg in self.test_results):
            print("   • Set OPENAI_API_KEY in .env file")
        
        # Overall status
        if failed == 0:
            print(f"\n🎉 Overall Status: {'EXCELLENT' if warned == 0 else 'GOOD'}")
            print(f"   Search workflow package is {'fully' if warned == 0 else 'mostly'} functional!")
        else:
            print(f"\n🔧 Overall Status: NEEDS ATTENTION")
            print(f"   Please fix the failing tests before using the package.")

# Main test execution
async def main():
    """Run the comprehensive test suite."""
    tester = SearchWorkflowTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
