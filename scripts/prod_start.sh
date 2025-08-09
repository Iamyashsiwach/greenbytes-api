#!/usr/bin/env bash
# Production startup script for GreenBytes API

set -e

echo "Starting GreenBytes API in production mode..."

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/Scripts/activate
else
    echo "No virtual environment found. Please create one first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Using defaults."
fi

# Get configuration from environment or defaults
UVICORN_HOST=${UVICORN_HOST:-0.0.0.0}
UVICORN_PORT=${UVICORN_PORT:-8000}

echo "Starting server on ${UVICORN_HOST}:${UVICORN_PORT}"

# Start uvicorn with production settings
uvicorn app.main:app \
    --host "$UVICORN_HOST" \
    --port "$UVICORN_PORT" \
    --workers 2 \
    --access-log \
    --log-level info
