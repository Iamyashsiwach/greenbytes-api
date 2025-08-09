"""
GreenBytes API - Main FastAPI application
Multimodal inference system supporting answers-only, image-only, and combined modes
"""

import logging
import json
from datetime import datetime
from typing import Optional, Dict, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import settings
from .schemas import (
    PredictJSON, PredictResponse, HealthResponse,
    YOLOResult, TabNetResult, FusionResult, TraceResult
)
from .models import yolo_runner, tabnet_runner
from .services.reference import pick_reference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GreenBytes API",
    description="Multimodal AI for sugarcane disease and pest detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files for reference images
app.mount("/static", StaticFiles(directory="app/static"), name="static")

logger.info(f"Started GreenBytes API with origins: {settings.allowed_origins_list}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        ok=True,
        timestamp=datetime.now().isoformat()
    )


def normalize_answers(answers: Union[str, Dict, None]) -> Optional[Dict[str, int]]:
    """Normalize answers from various input formats"""
    if answers is None:
        return None
    
    if isinstance(answers, str):
        try:
            answers = json.loads(answers)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in answers field")
    
    if isinstance(answers, list):
        # Convert list of QAItem to dict
        answers = {item["key"]: item["value"] for item in answers}
    
    if not isinstance(answers, dict):
        raise HTTPException(status_code=400, detail="Answers must be dict or list of QAItem")
    
    return answers


def count_answers(answers: Optional[Dict[str, int]]) -> tuple[int, int]:
    """Count yes and known answers"""
    if not answers:
        return 0, 0
    
    yes_count = sum(1 for v in answers.values() if v == 1)
    known_count = sum(1 for v in answers.values() if v != -1)
    
    return yes_count, known_count


@app.post("/predict")
async def predict(
    request: Request,
    # For multipart/form-data
    mode: Optional[str] = Form(None),
    answers: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    # For JSON payload (will be None if multipart)
    json_payload: Optional[PredictJSON] = None
):
    """
    Multimodal prediction endpoint supporting:
    1. JSON: answers-only mode
    2. Multipart: image-only mode
    3. Multipart: combined mode (image + answers)
    """
    
    # Determine request type and extract data
    content_type = request.headers.get("content-type", "")
    
    if content_type.startswith("application/json"):
        # JSON mode - answers only
        if json_payload is None:
            # Parse JSON manually if not auto-parsed
            try:
                body = await request.body()
                json_payload = PredictJSON.model_validate_json(body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")
        
        mode_str = json_payload.mode
        answers_dict = normalize_answers(json_payload.answers)
        image_bytes = None
        
    elif content_type.startswith("multipart/form-data"):
        # Multipart mode - image and/or answers
        if not mode:
            raise HTTPException(status_code=400, detail="mode field required")
        
        mode_str = mode
        answers_dict = normalize_answers(answers)
        
        # Read image if provided
        image_bytes = None
        if file:
            try:
                image_bytes = await file.read()
                if len(image_bytes) == 0:
                    raise HTTPException(status_code=400, detail="Empty image file")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")
    
    else:
        raise HTTPException(status_code=400, detail="Content-Type must be application/json or multipart/form-data")
    
    # Validate mode
    if mode_str not in ["pest", "disease"]:
        raise HTTPException(status_code=400, detail="mode must be 'pest' or 'disease'")
    
    # Validate that we have at least one input
    if not image_bytes and not answers_dict:
        raise HTTPException(status_code=400, detail="Must provide either image file or answers")
    
    logger.info(f"Prediction request: mode={mode_str}, has_image={bool(image_bytes)}, has_answers={bool(answers_dict)}")
    
    # Run YOLO inference if image provided
    yolo_result = None
    if image_bytes:
        try:
            yolo_result = yolo_runner.predict(
                mode=mode_str,
                image_bytes=image_bytes,
                conf_thr=settings.fusion_yolo_threshold
            )
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            yolo_result = {
                "available": False,
                "label": 0,
                "conf": 0.0,
                "bboxes": []
            }
    else:
        # No image provided
        yolo_result = {
            "available": False,
            "label": 0,
            "conf": 0.0,
            "bboxes": []
        }
    
    # Run TabNet inference if answers provided
    tabnet_result = None
    if answers_dict:
        try:
            tabnet_result = tabnet_runner.predict(
                mode=mode_str,
                answers=answers_dict
            )
        except Exception as e:
            logger.error(f"TabNet inference failed: {e}")
            tabnet_result = {
                "label": 0,
                "conf": 0.0,
                "top_positive_keys": []
            }
    else:
        # No answers provided
        tabnet_result = {
            "label": 0,
            "conf": 0.0,
            "top_positive_keys": []
        }
    
    # Fusion logic
    yolo_conf = yolo_result["conf"] if yolo_result["available"] else None
    tabnet_conf = tabnet_result["conf"]
    
    if image_bytes and answers_dict:
        # BOTH mode
        yolo_detected = yolo_conf and yolo_conf >= settings.fusion_yolo_threshold
        tabnet_detected = tabnet_conf >= settings.fusion_tabnet_threshold
        detected = yolo_detected or tabnet_detected
        reason = "yolo_or_tabnet"
        
    elif image_bytes and not answers_dict:
        # IMAGE ONLY mode
        detected = yolo_conf and yolo_conf >= settings.fusion_yolo_threshold if yolo_result["available"] else False
        reason = "image_only"
        
    elif answers_dict and not image_bytes:
        # ANSWERS ONLY mode
        detected = tabnet_conf >= settings.fusion_tabnet_threshold
        reason = "answers_only"
    
    else:
        # Should not reach here due to earlier validation
        detected = False
        reason = "error"
    
    # Count answers for trace
    yes_count, known_count = count_answers(answers_dict)
    
    # Build response objects
    yolo_response = YOLOResult(
        available=yolo_result["available"],
        conf=yolo_result["conf"],
        label=yolo_result["label"],
        bboxes=yolo_result["bboxes"]
    )
    
    tabnet_response = TabNetResult(
        conf=tabnet_result["conf"],
        label=tabnet_result["label"],
        top_positive_keys=tabnet_result["top_positive_keys"]
    )
    
    fusion_response = FusionResult(
        detected=detected,
        reason=reason,
        thresholds={
            "yolo": settings.fusion_yolo_threshold,
            "tabnet": settings.fusion_tabnet_threshold
        }
    )
    
    trace_response = TraceResult(
        path_taken=reason,
        rules_applied=[
            "detected iff (yolo>=Y_THR) OR (tabnet>=T_THR)",
            f"Y_THR={settings.fusion_yolo_threshold}, T_THR={settings.fusion_tabnet_threshold}",
            "answers score = yes/(yes+no), unknown ignored"
        ],
        numbers={
            "yolo_conf": yolo_conf,
            "tabnet_conf": tabnet_conf,
            "num_yes": yes_count,
            "num_known": known_count
        }
    )
    
    # Pick reference image
    reference_url = pick_reference(
        mode=mode_str,
        yolo_conf=yolo_conf,
        tabnet_conf=tabnet_conf,
        answers=answers_dict
    )
    
    # Build final response
    response = PredictResponse(
        mode=mode_str,
        used_image=bool(image_bytes),
        yolo=yolo_response,
        tabnet=tabnet_response,
        fusion=fusion_response,
        trace=trace_response,
        reference_image_url=reference_url
    )
    
    logger.info(f"Prediction complete: detected={detected}, reason={reason}")
    
    return response


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.uvicorn_host,
        port=settings.uvicorn_port,
        reload=True
    )
