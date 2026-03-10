#!/bin/bash
cd "$(dirname "$0")/.." || exit
docker compose down
echo "🛑 Search Workflow SearXNG stopped"