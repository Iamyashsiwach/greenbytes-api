"""
FastAPI backend for multimodal sugarcane disease/pest detection.
Provides health check and prediction endpoints with YOLO+TabNet fusion.
"""

import os
import io
import json
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from PIL import Image
import numpy as np

# Load environment variables
load_dotenv()

# Import inference modules
from inference.yolo_runner import run_yolo
from inference.tabnet_runner import run_tabnet
from inference.fusion import fuse

# Configuration
USE_STUB = os.getenv("USE_STUB", "true").lower() == "true"
YOLO_THRESH = float(os.getenv("YOLO_THRESH", "0.60"))
TABNET_THRESH = float(os.getenv("TABNET_THRESH", "0.70"))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))

# Initialize FastAPI app
app = FastAPI(title="GreenBytes Multimodal API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Initially allow all, tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response models
class HealthResponse(BaseModel):
    ok: bool


class YOLOResult(BaseModel):
    present: bool
    conf: float
    boxes: List[List[float]]
    mask_rle: Optional[str]


class TabNetResult(BaseModel):
    proba: float
    threshold: float
    pred: bool


class FusionResult(BaseModel):
    rule: str
    yolo_thresh: float
    present: bool


class PredictResponse(BaseModel):
    mode: str
    answers: Dict[str, int]
    yolo: YOLOResult
    tabnet: TabNetResult
    fusion: FusionResult
    ref_img: str


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {"ok": True}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    mode: str = Form(...),
    answers: str = Form(...)
):
    """
    Main prediction endpoint.
    
    Args:
        image: Uploaded image file (JPEG/PNG)
        mode: "disease" or "pest"
        answers: JSON string with array of 10 integers in {-1, 0, 1}
    
    Returns:
        Prediction results with YOLO, TabNet, and fusion outputs
    """
    
    # Validate mode
    if mode not in ["disease", "pest"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Must be 'disease' or 'pest'")
    
    # Validate file type
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=415, detail="Unsupported media type. Upload JPEG or PNG")
    
    # Check file size
    file_size_mb = 0
    image_bytes = await image.read()
    file_size_mb = len(image_bytes) / (1024 * 1024)
    
    if file_size_mb > MAX_UPLOAD_MB:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB}MB")
    
    # Parse and validate answers
    try:
        answers_list = json.loads(answers)
        if not isinstance(answers_list, list):
            raise ValueError("Answers must be a list")
        if len(answers_list) != 10:
            raise ValueError("Exactly 10 answers required")
        if not all(a in [-1, 0, 1] for a in answers_list):
            raise ValueError("Answers must be -1, 0, or 1")
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid answers: {str(e)}")
    
    # Generate deterministic outputs for stub mode
    if USE_STUB:
        # Create deterministic fake outputs based on inputs
        base_seed = sum(answers_list) + len(mode) + len(image_bytes) % 100
        yolo_conf = 0.45 + (base_seed % 40) / 100.0  # Range: 0.45-0.85
        tabnet_proba = 0.55 + ((base_seed * 2) % 40) / 100.0  # Range: 0.55-0.95
        
        # Fake bounding boxes
        boxes = []
        if yolo_conf > YOLO_THRESH:
            boxes = [[100.0, 100.0, 300.0, 300.0]]
        
        yolo_result = {
            "present": yolo_conf > YOLO_THRESH,
            "conf": round(yolo_conf, 3),
            "boxes": boxes,
            "mask_rle": None
        }
        
        tabnet_result = {
            "proba": round(tabnet_proba, 3),
            "threshold": TABNET_THRESH,
            "pred": tabnet_proba >= TABNET_THRESH
        }
    else:
        # Run actual models
        yolo_conf, boxes, mask_rle = run_yolo(image_bytes, mode)
        tabnet_proba = run_tabnet(answers_list, mode)
        
        yolo_result = {
            "present": yolo_conf > YOLO_THRESH,
            "conf": round(yolo_conf, 3),
            "boxes": boxes,
            "mask_rle": mask_rle
        }
        
        tabnet_result = {
            "proba": round(tabnet_proba, 3),
            "threshold": TABNET_THRESH,
            "pred": tabnet_proba >= TABNET_THRESH
        }
    
    # Apply fusion logic
    present, rule = fuse(yolo_result["conf"], tabnet_result["proba"], YOLO_THRESH, TABNET_THRESH)
    
    fusion_result = {
        "rule": rule,
        "yolo_thresh": YOLO_THRESH,
        "present": present
    }
    
    # Map answers to question keys (placeholder mapping for now)
    question_keys = [f"Q{i+1}" for i in range(10)]
    answers_dict = {key: val for key, val in zip(question_keys, answers_list)}
    
    # Select reference image
    ref_images = {
        "disease": ["deadheart_01.jpg", "deadheart_02.jpg", "deadheart_03.jpg"],
        "pest": ["esb_01.jpg", "esb_02.jpg", "esb_03.jpg"]
    }
    ref_img = ref_images[mode][base_seed % 3] if USE_STUB else ref_images[mode][0]
    
    return {
        "mode": mode,
        "answers": answers_dict,
        "yolo": yolo_result,
        "tabnet": tabnet_result,
        "fusion": fusion_result,
        "ref_img": ref_img
    }


# Error handler for unexpected errors
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
