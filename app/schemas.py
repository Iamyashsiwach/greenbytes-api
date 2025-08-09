"""
Pydantic schemas for API requests and responses
"""

from pydantic import BaseModel
from typing import Dict, List, Literal, Optional, Union


class QAItem(BaseModel):
    """Individual question-answer item"""
    key: str
    value: int  # 1=yes, 0=no, -1=unknown


class PredictJSON(BaseModel):
    """JSON payload for answers-only prediction"""
    mode: Literal["pest", "disease"]
    answers: Optional[Union[List[QAItem], Dict[str, int]]] = None


class YOLOResult(BaseModel):
    """YOLO inference result"""
    available: bool
    conf: float
    label: int
    bboxes: List[List[float]]


class TabNetResult(BaseModel):
    """TabNet inference result"""
    conf: float
    label: int
    top_positive_keys: List[str]


class FusionResult(BaseModel):
    """Fusion logic result"""
    detected: bool
    reason: str
    thresholds: Dict[str, float]


class TraceResult(BaseModel):
    """Decision trace information"""
    path_taken: str
    rules_applied: List[str]
    numbers: Dict[str, Union[float, int, None]]


class PredictResponse(BaseModel):
    """Complete prediction response"""
    mode: str
    used_image: bool
    yolo: YOLOResult
    tabnet: TabNetResult
    fusion: FusionResult
    trace: TraceResult
    reference_image_url: str


class HealthResponse(BaseModel):
    """Health check response"""
    ok: bool
    timestamp: Optional[str] = None
