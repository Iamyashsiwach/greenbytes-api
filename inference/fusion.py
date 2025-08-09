"""
Fusion logic for combining YOLO and TabNet predictions.
Implements the OR rule: present if either model exceeds its threshold.
"""

from typing import Tuple


def fuse(yolo_conf: float, tabnet_proba: float, yolo_thresh: float, tabnet_thresh: float) -> Tuple[bool, str]:
    """
    Fuse YOLO and TabNet predictions using OR logic.
    
    Args:
        yolo_conf: YOLO confidence score (0-1)
        tabnet_proba: TabNet probability (0-1)
        yolo_thresh: Threshold for YOLO
        tabnet_thresh: Threshold for TabNet
    
    Returns:
        Tuple of (present, rule_string)
        - present: True if target is detected
        - rule_string: Human-readable fusion rule
    """
    
    # Apply thresholds
    yolo_present = yolo_conf > yolo_thresh
    tabnet_present = tabnet_proba >= tabnet_thresh
    
    # OR fusion
    present = yolo_present or tabnet_present
    
    # Format rule string
    rule = f"(yolo>{yolo_thresh}) OR (tabnet≥{tabnet_thresh})"
    
    return present, rule
