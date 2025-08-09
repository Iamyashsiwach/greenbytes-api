"""
Reference image picker service
"""

from typing import Optional, Dict
from ..config import settings

def pick_reference(
    mode: str, 
    yolo_conf: Optional[float] = None, 
    tabnet_conf: Optional[float] = None, 
    answers: Optional[Dict] = None, 
    base_url: str = None
) -> str:
    """
    Pick appropriate reference image based on confidence scores
    
    Args:
        mode: "pest" or "disease"
        yolo_conf: YOLO confidence score (0-1)
        tabnet_conf: TabNet confidence score (0-1)
        answers: Answer dictionary (for additional context)
        base_url: Base URL for images (defaults to config)
        
    Returns:
        URL to reference image
    """
    
    if base_url is None:
        base_url = settings.ref_image_base_url
    
    # Determine overall confidence score
    scores = []
    if yolo_conf is not None:
        scores.append(yolo_conf)
    if tabnet_conf is not None:
        scores.append(tabnet_conf)
    
    if scores:
        score = max(scores)
    else:
        score = 0.0
    
    # Determine confidence bucket
    if score >= 0.8:
        bucket = "high"
    elif score >= 0.6:
        bucket = "mid"
    else:
        bucket = "low"
    
    # Build filename
    if mode == "pest":
        filename = f"esb_{bucket}.jpg"
        return f"{base_url}/esb/{filename}"
    elif mode == "disease":
        filename = f"disease_{bucket}.jpg"
        return f"{base_url}/disease/{filename}"
    else:
        # Fallback
        return f"{base_url}/disease/disease_low.jpg"
