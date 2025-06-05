# Search Workflow

AI-powered search workflow with SearXNG and DuckDuckGo integration.

## Features

- 🔍 **Hybrid Search**: SearXNG primary with DuckDuckGo fallback
- 🤖 **AI Evaluation**: LangChain-powered result ranking
- 🐳 **Docker Integration**: Complete SearXNG setup included
- ⚡ **High Performance**: Redis caching and optimized configuration
- 🛡️ **Self-Hosted**: No external API dependencies

## Quick Start

### Installation

```bash
# Install with UV
uv add search-workflow

# Or with pip
pip install search-workflow

```

## Basic Usage
```python
import asyncio
from search_workflow import run_workflow

async def main():
    results = await run_workflow("AI developments")
    for result in results:
        print(f"• {result['title']}")
        print(f"  {result['link']}")

asyncio.run(main())
```

### Docker Setup
```bash
# Start SearXNG
cd docker
./scripts/start.sh

# Test the API
./scripts/test.sh

# Stop when done
./scripts/stop.sh
```
## Configuration
The package uses intelligent defaults but can be customized:
```python
config = {
    "configurable": {
        "max_search_results_tool": 10,
        "max_search_results_evaluator": 5,
        "searxng_url": "http://localhost:9090"
    }
}

results = await run_workflow("query", config=config)
```
## Development
```bash
# Clone and setup
git clone https://github.com/phj6688/search_workflow.git
cd search-workflow
uv sync

# Run tests
uv run pytest

# Format code
uv run black src/
uv run ruff check src/
```
## License




```text
MIT License

Copyright (c) 2025 Buzzify AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```