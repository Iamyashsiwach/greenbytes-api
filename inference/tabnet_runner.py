"""
TabNet inference runner for tabular data prediction.
Processes 10 Yes/No/Unknown answers to predict disease/pest presence.
"""

from typing import List
import numpy as np
import os


def run_tabnet(answers: List[int], mode: str) -> float:
    """
    Run TabNet inference on questionnaire answers.
    
    Args:
        answers: List of 10 integers in {-1, 0, 1} representing Unknown/No/Yes
        mode: "disease" or "pest"
    
    Returns:
        Probability of target presence (0-1)
    """
    
    # Check if we're in stub mode
    if os.getenv("USE_STUB", "true").lower() == "true":
        # Return stub value
        return 0.0
    
    # TODO: Real TabNet implementation
    # This is where you'll load and run the actual TabNet model
    
    try:
        # Convert answers to numpy array
        features = np.array(answers).reshape(1, -1)
        
        # Load appropriate model based on mode
        if mode == "disease":
            # Load dead heart TabNet model
            # model = load_tabnet_model("models/deadheart_tabnet.pkl")
            pass
        else:  # pest
            # Load ESB TabNet model
            # model = load_tabnet_model("models/esb_tabnet.pkl")
            pass
        
        # Run inference
        # predictions = model.predict_proba(features)
        # probability = predictions[0][1]  # Probability of positive class
        
        # For now, return placeholder value
        probability = 0.0
        
        return probability
        
    except Exception as e:
        print(f"Error in TabNet inference: {e}")
        return 0.0
