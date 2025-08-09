"""
YOLO inference runner with lazy loading and fallback support
"""

import logging
from typing import Dict, List, Optional
from ..config import settings

logger = logging.getLogger(__name__)

# Global model cache
_yolo_models: Dict[str, any] = {}
_model_availability: Dict[str, bool] = {}

def _load_yolo_model(mode: str) -> Optional[any]:
    """Lazy load YOLO model for the given mode"""
    if mode in _yolo_models:
        return _yolo_models[mode]
    
    # Get weights path based on mode
    if mode == "pest":
        weights_path = settings.yolo_esb_weights
    elif mode == "disease":
        weights_path = settings.yolo_disease_weights
    else:
        logger.error(f"Unknown mode: {mode}")
        return None
    
    # Check if weights exist
    if not settings.model_exists(weights_path):
        logger.warning(f"YOLO weights not found: {weights_path}")
        _model_availability[mode] = False
        return None
    
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        _yolo_models[mode] = model
        _model_availability[mode] = True
        logger.info(f"Loaded YOLO model for {mode}: {weights_path}")
        return model
    except ImportError:
        logger.warning("ultralytics not installed, YOLO unavailable")
        _model_availability[mode] = False
        return None
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        _model_availability[mode] = False
        return None

def predict(mode: str, image_bytes: bytes, conf_thr: float) -> Dict:
    """
    Run YOLO inference on image
    
    Args:
        mode: "pest" or "disease"
        image_bytes: Raw image bytes
        conf_thr: Confidence threshold
        
    Returns:
        Dict with keys: available, label, conf, bboxes
    """
    
    # Try to load model
    model = _load_yolo_model(mode)
    
    if model is None:
        return {
            "available": False,
            "label": 0,
            "conf": 0.0,
            "bboxes": []
        }
    
    try:
        # Convert bytes to temporary file for YOLO
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
        try:
            # Run inference
            results = model(tmp_path, conf=conf_thr)
            
            # Extract predictions
            bboxes = []
            max_conf = 0.0
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        conf = float(box.conf.cpu().numpy()[0])
                        max_conf = max(max_conf, conf)
                        
                        # Convert bbox to list [x1, y1, x2, y2]
                        xyxy = box.xyxy.cpu().numpy()[0].tolist()
                        bboxes.append(xyxy)
            
            # Determine label (1 if detected with sufficient confidence)
            label = 1 if max_conf >= conf_thr else 0
            
            return {
                "available": True,
                "label": label,
                "conf": max_conf,
                "bboxes": bboxes
            }
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        logger.error(f"YOLO prediction failed: {e}")
        return {
            "available": False,
            "label": 0,
            "conf": 0.0,
            "bboxes": []
        }
