#!/bin/bash
set -e

echo "🚀 GreenBytes API Bootstrap"

# Ensure python3.11-venv is available
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Create/upgrade virtual environment
echo "Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3.11 -m venv .venv
    echo "✅ Created new virtual environment"
else
    echo "✅ Virtual environment exists"
fi

# Activate and upgrade pip
source .venv/bin/activate
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Copy .env if missing
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env from .env.example"
    else
        echo "⚠️  No .env.example found, creating minimal .env"
        cat > .env << EOF
USE_STUB=true
YOLO_THRESH=0.60
TABNET_THRESH=0.70
MAX_UPLOAD_MB=8
EOF
    fi
else
    echo "✅ .env already exists"
fi

# Restart FastAPI service
echo "Restarting FastAPI service..."
sudo systemctl restart fastapi || echo "⚠️  FastAPI service not yet installed"

echo "🎉 Bootstrap complete!"
