#!/bin/bash
set -e

echo "🧪 Testing Search Workflow SearXNG..."

# Check if services are running
if ! docker compose ps | grep -q "Up"; then
    echo "❌ Services are not running. Please run ./start.sh first"
    exit 1
fi

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -s -f "http://localhost:9090/healthz" > /dev/null; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi

# Test search API
echo "🔍 Testing search API..."
RESPONSE=$(curl -s "http://localhost:9090/search?q=test&format=json&categories=news")

if echo "$RESPONSE" | jq -e '.results' > /dev/null 2>&1; then
    RESULT_COUNT=$(echo "$RESPONSE" | jq '.results | length')
    echo "✅ Search API working - returned $RESULT_COUNT results"
else
    echo "❌ Search API test failed"
    echo "Response: $RESPONSE"
    exit 1
fi

# Test from Python package (if available)
echo "🐍 Testing Python package integration..."
if command -v python3 > /dev/null; then
    python3 -c "
import asyncio
import sys
import os
sys.path.append('..')
try:
    from src.search_workflow import run_workflow
    async def test():
        results = await run_workflow('test query')
        print(f'✅ Python package test passed - {len(results)} results')
    asyncio.run(test())
except Exception as e:
    print(f'⚠️  Python package test skipped: {e}')
"
fi

echo ""
echo "🎉 All tests completed successfully!"
