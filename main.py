"""
GreenBytes API - Minimal MVP
FastAPI backend supporting multimodal inference: answers-only, image-only, and combined modes
"""

import os
import json
import tempfile
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import io
import numpy as np
import cv2

# Import inference modules
from inference.yolo_runner import run_yolo
from inference.tabnet_runner import run_tabnet
from inference.fusion import fuse

# Load environment variables
load_dotenv()

app = FastAPI(title="GreenBytes API", description="Multimodal AI for sugarcane analysis", version="1.0.0")

# Configuration from environment
USE_STUB = os.getenv("USE_STUB", "true").lower() == "true"
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
FUSION_YOLO_THRESHOLD = float(os.getenv("FUSION_YOLO_THRESHOLD", "0.60"))
FUSION_TABNET_THRESHOLD = float(os.getenv("FUSION_TABNET_THRESHOLD", "0.70"))
REF_IMAGE_BASE_URL = os.getenv("REF_IMAGE_BASE_URL", "/static/reference")
YOLO_ESB_WEIGHTS = os.getenv("YOLO_ESB_WEIGHTS", "./models/esb_yolov8_best.pt")
YOLO_DISEASE_WEIGHTS = os.getenv("YOLO_DISEASE_WEIGHTS", "./models/disease_yolov8s_seg.pt")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files for reference images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic models
class HealthResponse(BaseModel):
    ok: bool

class PredictJSON(BaseModel):
    mode: str
    answers: Optional[Dict[str, int]] = None

class YOLOResult(BaseModel):
    available: bool
    conf: float
    label: int
    bboxes: List[List[float]]

class TabNetResult(BaseModel):
    conf: float
    label: int
    top_positive_keys: List[str]

class FusionResult(BaseModel):
    detected: bool
    reason: str
    thresholds: Dict[str, float]

class TraceResult(BaseModel):
    rules: List[str]
    numbers: Dict[str, Union[float, int, None]]

class PredictResponse(BaseModel):
    mode: str
    used_image: bool
    yolo: YOLOResult
    tabnet: TabNetResult
    fusion: FusionResult
    trace: TraceResult
    reference_image_url: str

# Global YOLO model cache
_yolo_models = {}

def load_yolo_model(mode: str):
    """Lazy load YOLO model if weights exist"""
    if mode in _yolo_models:
        return _yolo_models[mode]
    
    weights_path = YOLO_ESB_WEIGHTS if mode == "pest" else YOLO_DISEASE_WEIGHTS
    try:
        print(f"[YOLO] Mode={mode} Weights={weights_path} Exists={os.path.exists(weights_path)}")
    except Exception:
        pass
    
    if not os.path.exists(weights_path):
        _yolo_models[mode] = None
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        _yolo_models[mode] = model
        print(f"[YOLO] Loaded model for mode={mode}")
        return model
    except ImportError:
        print(f"ultralytics not installed, YOLO unavailable for {mode}")
        _yolo_models[mode] = None
        return None
    except Exception as e:
        print(f"Failed to load YOLO model for {mode}: {e}")
        _yolo_models[mode] = None
        return None

def run_yolo_inference(mode: str, image_bytes: bytes) -> Dict:
    """Run YOLO inference on image"""
    model = load_yolo_model(mode)
    
    if model is None:
        return {
            "available": False,
            "conf": 0.0,
            "label": 0,
            "bboxes": []
        }
    
    try:
        # Try OpenCV decode first (more tolerant), then PIL fallback
        img_array = None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is not None:
                img_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            img_array = None

        if img_array is None:
            # PIL fallback
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(image)

        # Run inference directly on array (RGB)
        results = model(img_array, conf=FUSION_YOLO_THRESHOLD)

        # Extract results
        max_conf = 0.0
        bboxes = []

        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    conf = float(box.conf.cpu().numpy()[0])
                    max_conf = max(max_conf, conf)
                    xyxy = box.xyxy.cpu().numpy()[0].tolist()
                    bboxes.append(xyxy)

        label = 1 if max_conf >= FUSION_YOLO_THRESHOLD else 0

        return {
            "available": True,
            "conf": max_conf,
            "label": label,
            "bboxes": bboxes
        }

    except Exception as e:
        print(f"YOLO inference failed: {e}")
        return {
            "available": False,
            "conf": 0.0,
            "label": 0,
            "bboxes": []
        }

def run_tabnet_real(answers: Dict[str, int], mode: str) -> Dict:
    """Real TabNet implementation using trained models"""
    try:
        # Use the actual TabNet model
        conf = run_tabnet(answers, mode)
        
        # Get positive answer keys (for interpretability)
        top_positive_keys = [k for k, v in answers.items() if v == 1]
        top_positive_keys.sort()
        
        # Determine label
        label = 1 if conf >= FUSION_TABNET_THRESHOLD else 0
        
        return {
            "conf": conf,
            "label": label,
            "top_positive_keys": top_positive_keys
        }
    except Exception as e:
        print(f"Error in real TabNet: {e}, falling back to stub")
        return run_tabnet_stub(answers)

def run_tabnet_stub(answers: Dict[str, int]) -> Dict:
    """Stub TabNet implementation: conf = yes_count / (yes_count + no_count)"""
    if not answers:
        return {
            "conf": 0.0,
            "label": 0,
            "top_positive_keys": []
        }
    
    # Count yes (1) and no (0), ignore unknown (-1)
    yes_count = sum(1 for v in answers.values() if v == 1)
    no_count = sum(1 for v in answers.values() if v == 0)
    total_known = yes_count + no_count
    
    if total_known > 0:
        conf = yes_count / total_known
    else:
        conf = 0.0
    
    # Get positive answer keys
    top_positive_keys = [k for k, v in answers.items() if v == 1]
    top_positive_keys.sort()
    
    # Determine label
    label = 1 if conf >= FUSION_TABNET_THRESHOLD else 0
    
    return {
        "conf": conf,
        "label": label,
        "top_positive_keys": top_positive_keys
    }

def pick_reference_image(mode: str, yolo_conf: Optional[float], tabnet_conf: float) -> str:
    """Pick reference image based on confidence scores"""
    # Determine overall confidence
    scores = [tabnet_conf]
    if yolo_conf is not None:
        scores.append(yolo_conf)
    
    max_score = max(scores) if scores else 0.0
    
    # Determine bucket
    if max_score >= 0.8:
        bucket = "high"
    elif max_score >= 0.6:
        bucket = "mid"
    else:
        bucket = "low"
    
    # Build URL
    subfolder = "esb" if mode == "pest" else "disease"
    filename = f"{subfolder}_{bucket}.jpg"
    return f"{REF_IMAGE_BASE_URL}/{subfolder}/{filename}"

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(ok=True)

@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    # For multipart/form-data
    mode: Optional[str] = Form(None),
    answers: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Multimodal prediction endpoint supporting:
    1. JSON: answers-only mode
    2. Multipart: image-only mode
    3. Multipart: combined mode (image + answers)
    """
    
    # Determine request type
    content_type = request.headers.get("content-type", "")
    
    if content_type.startswith("application/json"):
        # JSON mode - answers only
        body = await request.body()
        try:
            data = json.loads(body)
            mode_str = data.get("mode")
            answers_dict = data.get("answers", {})
            image_bytes = None
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON payload")
    
    elif content_type.startswith("multipart/form-data"):
        # Multipart mode
        if not mode:
            raise HTTPException(status_code=400, detail="mode field required")
        
        mode_str = mode
        
        # Parse answers if provided
        answers_dict = {}
        if answers:
            try:
                answers_dict = json.loads(answers)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in answers field")
        
        # Read image if provided
        image_bytes = None
        if file:
            image_bytes = await file.read()
            if len(image_bytes) == 0:
                raise HTTPException(status_code=400, detail="Empty image file")
    
    else:
        raise HTTPException(status_code=400, detail="Content-Type must be application/json or multipart/form-data")
    
    # Validate mode
    if mode_str not in ["pest", "disease"]:
        raise HTTPException(status_code=400, detail="mode must be 'pest' or 'disease'")
    
    # Validate inputs
    if not image_bytes and not answers_dict:
        raise HTTPException(status_code=400, detail="Must provide either image file or answers")
    
    # Run YOLO inference if image provided
    yolo_result = None
    if image_bytes:
        yolo_result = run_yolo_inference(mode_str, image_bytes)
    else:
        yolo_result = {
            "available": False,
            "conf": 0.0,
            "label": 0,
            "bboxes": []
        }
    
    # Run TabNet inference (always with answers, empty if not provided)
    if os.getenv("USE_STUB", "true").lower() == "true":
        tabnet_result = run_tabnet_stub(answers_dict)
    else:
        tabnet_result = run_tabnet_real(answers_dict, mode_str)
    
    # Fusion logic
    yolo_conf = yolo_result["conf"] if yolo_result["available"] else None
    tabnet_conf = tabnet_result["conf"]
    
    if image_bytes and answers_dict:
        # Combined mode
        yolo_detected = yolo_conf and yolo_conf >= FUSION_YOLO_THRESHOLD
        tabnet_detected = tabnet_conf >= FUSION_TABNET_THRESHOLD
        detected = yolo_detected or tabnet_detected
        reason = "yolo_or_tabnet"
    elif image_bytes:
        # Image only
        detected = yolo_conf and yolo_conf >= FUSION_YOLO_THRESHOLD if yolo_result["available"] else False
        reason = "image_only"
    else:
        # Answers only
        detected = tabnet_conf >= FUSION_TABNET_THRESHOLD
        reason = "answers_only"
    
    # Count answers for trace
    yes_count = sum(1 for v in answers_dict.values() if v == 1) if answers_dict else 0
    no_count = sum(1 for v in answers_dict.values() if v == 0) if answers_dict else 0
    
    # Build response
    response = PredictResponse(
        mode=mode_str,
        used_image=bool(image_bytes),
        yolo=YOLOResult(
            available=yolo_result["available"],
            conf=yolo_result["conf"],
            label=yolo_result["label"],
            bboxes=yolo_result["bboxes"]
        ),
        tabnet=TabNetResult(
            conf=tabnet_result["conf"],
            label=tabnet_result["label"],
            top_positive_keys=tabnet_result["top_positive_keys"]
        ),
        fusion=FusionResult(
            detected=detected,
            reason=reason,
            thresholds={
                "yolo": FUSION_YOLO_THRESHOLD,
                "tabnet": FUSION_TABNET_THRESHOLD
            }
        ),
        trace=TraceResult(
            rules=[
                "detected iff (yolo>=Y_THR) OR (tabnet>=T_THR)",
                f"Y_THR={FUSION_YOLO_THRESHOLD}, T_THR={FUSION_TABNET_THRESHOLD}",
                "tabnet score = yes/(yes+no), unknown ignored"
            ],
            numbers={
                "yolo_conf": yolo_conf,
                "tabnet_conf": tabnet_conf,
                "yes_count": yes_count,
                "no_count": no_count
            }
        ),
        reference_image_url=pick_reference_image(mode_str, yolo_conf, tabnet_conf)
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)