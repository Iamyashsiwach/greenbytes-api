# GreenBytes API - MVP Backend

Minimal FastAPI backend for multimodal sugarcane AI analysis supporting answers-only, image-only, and combined inference modes.

## 🚀 Quick Start

### Local Development

1. **Create virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Configure environment**:
```bash
cp .env.example .env
# Edit .env if needed (defaults work for MVP)
```

4. **Run server**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# Server starts on http://localhost:8000
```

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
# Response: {"ok": true}
```

### Prediction Modes

#### 1. Answers-Only (JSON)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "pest", 
    "answers": {"Q1": 1, "Q2": 0, "Q3": -1, "Q4": 1}
  }'
```

#### 2. Image-Only (Multipart)
```bash
curl -X POST http://localhost:8000/predict \
  -F "mode=disease" \
  -F "file=@test_image.jpg"
```

#### 3. Combined (Multipart)
```bash
curl -X POST http://localhost:8000/predict \
  -F "mode=pest" \
  -F "file=@test_image.jpg" \
  -F 'answers={"Q1": 1, "Q2": 0, "Q3": 1}'
```

## 🔧 Configuration

Environment variables (in `.env`):
- `USE_STUB=true` - Use TabNet stub (yes/no ratio logic)
- `ALLOWED_ORIGINS=http://localhost:3000` - CORS origins
- `FUSION_YOLO_THRESHOLD=0.60` - YOLO detection threshold
- `FUSION_TABNET_THRESHOLD=0.70` - TabNet detection threshold

## 🧠 AI Models

### TabNet (Stub Mode)
- **Logic**: `confidence = yes_count / (yes_count + no_count)`
- **Ignores**: Unknown answers (-1)
- **Detection**: confidence >= 0.70

### YOLO (Optional)
- **Weights**: Place `esb_yolov8_best.pt` and `disease_yolov8s_seg.pt` in `./models/`
- **Graceful Fallback**: Returns `{"available": false}` if weights missing
- **Enable**: Set `USE_STUB=false` when models ready

### Fusion Logic
```
detected = (yolo_conf >= 0.60) OR (tabnet_conf >= 0.70)
```

## 📁 Response Format

```json
{
  "mode": "pest",
  "used_image": true,
  "yolo": {
    "available": true,
    "conf": 0.85,
    "label": 1,
    "bboxes": [[10, 20, 100, 200]]
  },
  "tabnet": {
    "conf": 0.75,
    "label": 1,
    "top_positive_keys": ["Q1", "Q4"]
  },
  "fusion": {
    "detected": true,
    "reason": "yolo_or_tabnet",
    "thresholds": {"yolo": 0.60, "tabnet": 0.70}
  },
  "trace": {
    "rules": ["detected iff (yolo>=Y_THR) OR (tabnet>=T_THR)", ...],
    "numbers": {"yolo_conf": 0.85, "tabnet_conf": 0.75, "yes_count": 2, "no_count": 1}
  },
  "reference_image_url": "/static/reference/esb/esb_high.jpg"
}
```

## 🧪 Testing

Visit http://localhost:8000/docs for interactive API documentation.

## 📦 Model Integration

**When real models are ready**:
1. Place model files in `./models/` directory
2. Install model dependencies: `pip install ultralytics pytorch-tabnet`
3. Set `USE_STUB=false` in `.env`
4. Restart server

**Current MVP**: Works with stub TabNet and optional YOLO (graceful fallback if weights missing).
