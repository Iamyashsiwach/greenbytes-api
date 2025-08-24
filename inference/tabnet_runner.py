"""
TabNet inference runner for tabular data prediction.
Processes questionnaire answers to predict disease/pest presence using trained models.
"""

import os
import numpy as np
import pandas as pd
import joblib
import yaml
from typing import List, Dict, Any, Optional
from pytorch_tabnet.tab_model import TabNetClassifier

# Global model cache
_models = {}
_encoders = {}
_features_config = None

def load_features_config():
    """Load features configuration from YAML"""
    global _features_config
    if _features_config is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "models", "tabnet", "features.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                _features_config = yaml.safe_load(f)
        else:
            print(f"Warning: features.yaml not found at {config_path}")
            _features_config = {}
    return _features_config

def load_tabnet_model(task: str):
    """Load TabNet model and encoders for a specific task"""
    global _models, _encoders
    
    if task in _models:
        return _models[task], _encoders[task]
    
    try:
        # Paths to model files
        base_path = os.path.join(os.path.dirname(__file__), "..", "models", "tabnet", task, "best")
        model_path = os.path.join(base_path, "tabnet_model.zip")
        encoders_path = os.path.join(base_path, "encoders.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(encoders_path):
            print(f"Warning: TabNet model files not found for task {task}")
            return None, None
        
        # Load model
        model = TabNetClassifier()
        model.load_model(model_path)
        
        # Load encoders
        encoders = joblib.load(encoders_path)
        
        # Cache for future use
        _models[task] = model
        _encoders[task] = encoders
        
        print(f"Successfully loaded TabNet model for task: {task}")
        return model, encoders
        
    except Exception as e:
        print(f"Error loading TabNet model for {task}: {e}")
        return None, None

def map_ui_answers_to_features(answers: Dict[str, int], task: str) -> Optional[np.ndarray]:
    """
    Map UI question answers to model features.
    
    Args:
        answers: Dict like {"Q1": 1, "Q2": 0, "Q3": -1}
        task: "larva" (for pest) or "deadheart" (for disease)
        
    Returns:
        Feature array ready for model inference
    """
    try:
        config = load_features_config()
        
        if task not in config.get('categorical', {}):
            print(f"Warning: No feature config found for task {task}")
            return None
        
        # Get the categorical features for this task
        categorical_features = config['categorical'][task]
        numeric_features = config.get('numeric', {}).get(task, [])
        
        # Create feature vector
        features = []
        
        # Add numeric features (confidence - we'll use a default value)
        for feat in numeric_features:
            if feat == 'confidence':
                features.append(0.75)  # Default confidence
            else:
                features.append(0.0)  # Default for other numeric features
        
        # Add categorical features (the questionnaire answers)
        # Map Q1, Q2, ... to actual feature names
        question_mapping = {
            f"Q{i+1}": categorical_features[i] 
            for i in range(min(len(categorical_features), len(answers)))
        }
        
        for feat in categorical_features:
            # Find which question this feature corresponds to
            question_key = None
            for q_key, feat_name in question_mapping.items():
                if feat_name == feat:
                    question_key = q_key
                    break
            
            if question_key and question_key in answers:
                # Map answer: -1 (Unknown) -> 0, 0 (No) -> 1, 1 (Yes) -> 2
                answer_value = answers[question_key]
                if answer_value == -1:  # Unknown
                    mapped_value = 0
                elif answer_value == 0:  # No  
                    mapped_value = 1
                else:  # Yes (1)
                    mapped_value = 2
                features.append(mapped_value)
            else:
                features.append(0)  # Default for missing answers
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        print(f"Error mapping answers to features: {e}")
        return None

def run_tabnet(answers: Dict[str, int], mode: str) -> float:
    """
    Run TabNet inference on questionnaire answers.
    
    Args:
        answers: Dict of question answers like {"Q1": 1, "Q2": 0, "Q3": -1}
        mode: "disease" or "pest"
    
    Returns:
        Probability of target presence (0-1)
    """
    
    # Check if we're in stub mode
    if os.getenv("USE_STUB", "true").lower() == "true":
        print("TabNet running in stub mode")
        return 0.0
    
    try:
        # Map mode to task
        task = "deadheart" if mode == "disease" else "larva"
        
        # Load model and encoders
        model, encoders = load_tabnet_model(task)
        if model is None or encoders is None:
            print(f"Failed to load TabNet model for {task}, falling back to stub")
            return 0.0
        
        # Map UI answers to feature vector
        features = map_ui_answers_to_features(answers, task)
        if features is None:
            print("Failed to map answers to features, falling back to stub")
            return 0.0
        
        # Run inference
        predictions = model.predict_proba(features)
        probability = float(predictions[0][1])  # Probability of positive class
        
        print(f"TabNet inference - Task: {task}, Probability: {probability:.4f}")
        return probability
        
    except Exception as e:
        print(f"Error in TabNet inference: {e}")
        return 0.0

def run_tabnet_legacy(answers: List[int], mode: str) -> float:
    """
    Legacy interface for backward compatibility.
    Converts list format to dict format.
    
    Args:
        answers: List of 10 integers in {-1, 0, 1} representing Unknown/No/Yes
        mode: "disease" or "pest"
    
    Returns:
        Probability of target presence (0-1)
    """
    # Convert list to dict format
    answers_dict = {f"Q{i+1}": val for i, val in enumerate(answers)}
    return run_tabnet(answers_dict, mode)