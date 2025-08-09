"""
TabNet inference runner with stub fallback
"""

import logging
from typing import Dict, List, Optional
from ..config import settings

logger = logging.getLogger(__name__)

# Global model cache
_tabnet_models: Dict[str, any] = {}
_model_availability: Dict[str, bool] = {}

def _load_tabnet_model(mode: str) -> Optional[any]:
    """Lazy load TabNet model for the given mode"""
    if mode in _tabnet_models:
        return _tabnet_models[mode]
    
    # Get model path based on mode
    if mode == "pest":
        model_path = settings.tabnet_esb_path
    elif mode == "disease":
        model_path = settings.tabnet_disease_path
    else:
        logger.error(f"Unknown mode: {mode}")
        return None
    
    # Check if model exists
    if not settings.model_exists(model_path):
        logger.warning(f"TabNet model not found: {model_path}")
        _model_availability[mode] = False
        return None
    
    try:
        # Try to load TabNet model (placeholder for actual implementation)
        # from pytorch_tabnet.tab_model import TabNetClassifier
        # model = TabNetClassifier()
        # model.load_model(model_path)
        
        # For now, mark as unavailable since we don't have real models
        logger.warning(f"TabNet loading not implemented for {model_path}")
        _model_availability[mode] = False
        return None
        
    except ImportError:
        logger.warning("pytorch_tabnet not installed, using stub")
        _model_availability[mode] = False
        return None
    except Exception as e:
        logger.error(f"Failed to load TabNet model: {e}")
        _model_availability[mode] = False
        return None

def _stub_predict(answers: Dict[str, int]) -> Dict:
    """
    Stub prediction based on simple rules
    Score = (#YES answers) / (#answers that are not Unknown)
    """
    
    if not answers:
        return {
            "label": 0,
            "conf": 0.0,
            "top_positive_keys": []
        }
    
    # Count answers
    yes_count = sum(1 for v in answers.values() if v == 1)
    known_count = sum(1 for v in answers.values() if v != -1)
    
    # Calculate confidence score
    if known_count > 0:
        conf = yes_count / known_count
    else:
        conf = 0.0
    
    # Get positive answer keys
    top_positive_keys = [k for k, v in answers.items() if v == 1]
    top_positive_keys.sort()  # Consistent ordering
    
    # Determine label based on threshold
    label = 1 if conf >= settings.fusion_tabnet_threshold else 0
    
    return {
        "label": label,
        "conf": conf,
        "top_positive_keys": top_positive_keys
    }

def predict(mode: str, answers: Dict[str, int]) -> Dict:
    """
    Run TabNet inference on answers
    
    Args:
        mode: "pest" or "disease"
        answers: Dict mapping question keys to answers (1/0/-1)
        
    Returns:
        Dict with keys: label, conf, top_positive_keys
    """
    
    # Try to load real model first
    model = _load_tabnet_model(mode)
    
    if model is not None:
        # TODO: Implement real TabNet prediction
        # For now, fall back to stub
        pass
    
    # Use stub prediction
    logger.info(f"Using stub TabNet prediction for {mode}")
    return _stub_predict(answers)
