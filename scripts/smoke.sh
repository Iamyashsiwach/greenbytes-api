#!/bin/bash
set -e

echo "🧪 Running GreenBytes API smoke test"

# Test health endpoint
echo "Testing health endpoint..."
RESPONSE=$(curl -sf http://localhost:8000/health 2>/dev/null) || {
    echo "❌ Health check failed - API not responding"
    echo "🔍 Checking service status..."
    sudo systemctl status fastapi --no-pager || true
    echo "🔍 Recent logs..."
    sudo journalctl -u fastapi --no-pager -n 10 || true
    exit 1
}

echo "Response: $RESPONSE"

# Verify JSON response contains expected fields
if echo "$RESPONSE" | grep -q '"ok".*true'; then
    echo "✅ Health check passed - API is healthy"
    echo "🎉 Smoke test successful!"
    exit 0
else
    echo "❌ Health check returned unexpected response"
    echo "Expected: {\"ok\": true}"
    echo "Got: $RESPONSE"
    exit 1
fi
