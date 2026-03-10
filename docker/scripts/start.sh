#!/bin/bash
set -e

echo "🚀 Starting Search Workflow SearXNG..."

cd "$(dirname "$0")/.."

# Check if .env exists, create if not
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        echo "⚠️  .env.example not found, creating minimal .env"
        cat > .env << EOF
SEARXNG_SECRET=$(openssl rand -hex 32)
COMPOSE_PROJECT_NAME=search-workflow
SEARXNG_VERSION=2025.5.18
UWSGI_WORKERS=4
UWSGI_THREADS=4
ENVIRONMENT=development
DEBUG=false
EOF
    fi
    
    # Generate random secret if placeholder exists
    if grep -q "your_super_secret_key_here" .env; then
        SECRET=$(openssl rand -hex 32)
        sed -i.bak "s/your_super_secret_key_here_change_this_in_production/$SECRET/g" .env
        rm .env.bak 2>/dev/null || true
    fi
    
    echo "✅ Created .env file with generated secrets"
    echo "⚠️  Please review and update the configuration as needed"
fi

# Ensure config directory exists with proper files
if [ ! -f config/settings.yml ]; then
    echo "❌ config/settings.yml not found!"
    echo "Please ensure docker/config/settings.yml exists"
    exit 1
fi

if [ ! -f config/uwsgi.ini ]; then
    echo "❌ config/uwsgi.ini not found!"
    echo "Please ensure docker/config/uwsgi.ini exists"
    exit 1
fi

# Start services
echo "🐳 Starting Docker services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be healthy..."
for i in {1..30}; do
    if docker compose exec -T searxng wget --quiet --tries=1 --spider http://localhost:8080/healthz 2>/dev/null; then
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ Health check failed after 30 attempts"
        echo "📋 Checking logs..."
        docker compose logs searxng
        exit 1
    fi
    
    echo "  Attempt $i/30..."
    sleep 2
done

echo ""
echo "✅ Search Workflow SearXNG is running!"
echo "🔗 SearXNG Web UI: http://localhost:9090"
echo "🔗 API Endpoint: http://localhost:9090/search"
echo "🔗 Health Check: http://localhost:9090/healthz"
echo ""
echo "📊 Test the API:"
echo "curl \"http://localhost:9090/search?q=test&format=json&categories=news\""
