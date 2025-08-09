#!/bin/bash
set -e

echo "🔧 Installing GreenBytes FastAPI systemd service"

# Get the current directory (should be the repo root)
REPO_PATH=$(pwd)
BACKEND_PATH=${BACKEND_PATH:-$REPO_PATH}

echo "Using BACKEND_PATH: $BACKEND_PATH"

# Create systemd service file from template
sudo tee /etc/systemd/system/fastapi.service > /dev/null << EOF
[Unit]
Description=GreenBytes FastAPI Application
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$BACKEND_PATH
Environment="PATH=$BACKEND_PATH/.venv/bin"
ExecStart=$BACKEND_PATH/.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

echo "✅ systemd service file created"

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable fastapi
sudo systemctl start fastapi

echo "✅ FastAPI service enabled and started"
echo "📋 Service status:"
sudo systemctl status fastapi --no-pager
