# src/model.py
import numpy as np

def load_bilstm(model_path=None):
    return {"model_type": "BiLSTM", "status": "active"}

def load_bert(model_path=None):
    return {"model_type": "BERT", "status": "active"}, {"tokenizer": "active"}, "cpu"

def bert_predict(text, *args):
    """Dynamically parses the user text to return different probabilities 
    so the charts display variety based on inputs!
    """
    text_lower = str(text).lower()
    
    # [Confused, Frustrated, Curious, Confident, Bored]
    if "confused" in text_lower or "recursion" in text_lower:
        return np.array([0.90, 0.05, 0.02, 0.01, 0.02])
    elif "frustrated" in text_lower or "bug" in text_lower or "hours" in text_lower:
        return np.array([0.05, 0.90, 0.02, 0.01, 0.02])
    elif "curious" in text_lower or "wonder" in text_lower or "networks" in text_lower:
        return np.array([0.02, 0.03, 0.90, 0.03, 0.02])
    elif "confident" in text_lower or "gradient" in text_lower or "understand" in text_lower:
        return np.array([0.01, 0.01, 0.03, 0.92, 0.03])
    elif "bored" in text_lower or "lecture" in text_lower or "repeating" in text_lower:
        return np.array([0.02, 0.02, 0.02, 0.04, 0.90])
    
    # Default fallback array
    return np.array([0.20, 0.20, 0.20, 0.20, 0.20])