"""
YOLO inference runner for disease/pest detection.
Stub implementation included for testing without models.
"""

from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import io
import os


def run_yolo(image_bytes: bytes, mode: str) -> Tuple[float, List[List[float]], Optional[str]]:
    """
    Run YOLO inference on image for specified mode.
    
    Args:
        image_bytes: Raw image bytes
        mode: "disease" or "pest"
    
    Returns:
        Tuple of (confidence, boxes, mask_rle)
        - confidence: float between 0-1
        - boxes: List of [x1, y1, x2, y2] coordinates
        - mask_rle: RLE-encoded mask string (for segmentation) or None
    """
    
    # Check if we're in stub mode
    if os.getenv("USE_STUB", "true").lower() == "true":
        # Return stub values
        return 0.0, [], None
    
    # TODO: Real YOLO implementation
    # This is where you'll load and run the actual YOLO model
    
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Load appropriate model based on mode
        if mode == "disease":
            # Load dead heart segmentation model
            # model = load_yolo_model("models/deadheart_seg.pt")
            pass
        else:  # pest
            # Load ESB detection model
            # model = load_yolo_model("models/esb_det.pt")
            pass
        
        # Run inference
        # results = model(img_array)
        
        # Extract confidence, boxes, and mask
        # For now, return placeholder values
        confidence = 0.0
        boxes = []
        mask_rle = None
        
        return confidence, boxes, mask_rle
        
    except Exception as e:
        print(f"Error in YOLO inference: {e}")
        return 0.0, [], None
