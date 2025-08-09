# GreenBytes API - FastAPI Backend

Multimodal AI backend for sugarcane disease and pest detection using YOLO + TabNet fusion.

## 🚀 Quick Start

### Local Development

1. **Create virtual environment**:
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env if needed (USE_STUB=true for demo)
```

4. **Run server**:
```bash
python main.py
# Server starts on http://localhost:8000
```

## 🔧 Configuration

### Environment Variables
- `USE_STUB=true` - Enable stub mode (no real models needed)
- `YOLO_THRESH=0.60` - YOLO confidence threshold
- `TABNET_THRESH=0.70` - TabNet probability threshold  
- `MAX_UPLOAD_MB=8` - Maximum upload size

## 📡 API Endpoints

### Health Check
```
GET /health
Response: {"ok": true}
```

### Prediction
```
POST /predict
Content-Type: multipart/form-data

Fields:
- image: JPEG/PNG file (max 8MB)
- mode: "disease" or "pest"  
- answers: JSON array of 10 integers in {-1, 0, 1}

Response: {
  "mode": "disease",
  "answers": {"Q1": -1, "Q2": 0, ...},
  "yolo": {"present": true, "conf": 0.785, ...},
  "tabnet": {"proba": 0.823, "threshold": 0.70, ...},
  "fusion": {"present": true, "rule": "...", ...},
  "ref_img": "deadheart_01.jpg"
}
```

## 🚀 Deployment

### Azure VM Deployment

#### One-time VM Setup (Ubuntu)
```bash
# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev git curl

# Clone repository  
git clone <your-repo-url> /home/azureuser/greenbytes-api
cd /home/azureuser/greenbytes-api
```

#### Install and Start Service
```bash
# Install systemd service
sudo ./scripts/install_systemd.sh

# Bootstrap application
./scripts/bootstrap.sh

# Test deployment
./scripts/smoke.sh
```

#### GitHub Actions Deployment

**Required Secrets** (Settings → Secrets and variables → Actions):
- `AZURE_VM_HOST`: `<vm-ip-or-dns>`
- `AZURE_VM_USER`: `azureuser`  
- `AZURE_VM_SSH_KEY`: Private SSH key content
- `BACKEND_PATH`: `/home/azureuser/greenbytes-api`

**Auto-deploys** on every push to `main` branch.

#### Troubleshooting
```bash
# Check service status
sudo systemctl status fastapi

# View logs
sudo journalctl -u fastapi -f

# Check firewall
sudo ufw status
sudo ufw allow 8000

# Test CORS
curl -H "Origin: https://<frontend-domain>" http://<vm-ip>:8000/health
```

## 🧪 Testing

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# Prediction test
curl -X POST http://localhost:8000/predict \
  -F "mode=disease" \
  -F "answers=[-1,0,1,-1,0,1,-1,0,1,0]" \
  -F "image=@test_image.jpg"
```

### Automated Testing
```bash
python test_api.py
```

## 🏗️ Architecture

- **FastAPI** - Modern Python web framework
- **YOLO Integration** - Computer vision inference  
- **TabNet Integration** - Tabular data processing
- **Multimodal Fusion** - OR-based combination logic
- **Systemd Service** - Production deployment
- **GitHub Actions** - CI/CD pipeline

## 📦 Model Integration

**For Production** (set `USE_STUB=false`):

1. Place model files in `models/`:
   - `deadheart_seg.pt` - YOLO segmentation
   - `esb_det.pt` - YOLO detection  
   - `deadheart_tabnet.pkl` - TabNet for disease
   - `esb_tabnet.pkl` - TabNet for pest

2. Update inference runners:
   - `inference/yolo_runner.py` - Load and run YOLO models
   - `inference/tabnet_runner.py` - Load and run TabNet models

## 🤝 Team GreenBytes

VIT Vellore Hackathon - Agricultural AI Technology

