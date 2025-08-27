#!/bin/bash

# Production startup script for Information Retrieval System
# Usage: ./start.sh

echo "🚀 Starting Information Retrieval System..."

# Change to UI directory
cd ui

# Set default environment variables
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-4}

# Check if running in production
if [ "$ENVIRONMENT" = "production" ]; then
    echo "📦 Starting in PRODUCTION mode..."
    
    # Use Gunicorn for production
    gunicorn main:app \
        --bind ${HOST}:${PORT} \
        --workers ${WORKERS} \
        --worker-class uvicorn.workers.UvicornWorker \
        --timeout 120 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --access-logfile - \
        --error-logfile - \
        --log-level info \
        --preload
else
    echo "🔧 Starting in DEVELOPMENT mode..."
    
    # Use Uvicorn for development
    uvicorn main:app \
        --host ${HOST} \
        --port ${PORT} \
        --reload
fi
