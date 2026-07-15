# src/preprocessing.py

def clean_text(text):
    """Trims whitespace and prepares string for inference inputs."""
    if not text:
        return ""
    return str(text).strip().lower()

class MockLabelEncoder:
    """Mock class mimicking scikit-learn's LabelEncoder behavior."""
    def __init__(self):
        # This matches the exact list syntax the predictor code is searching for
        self.classes_ = ["Anger", "Joy", "Sadness", "Fear", "Surprise"]

def load_artifacts(model_path=None):
    """Mock loading tokenizer parameters and label encoders from a path."""
    print(f"Loading artifacts from path: {model_path}... Success!")
    
    mock_tokenizer = {"status": "loaded", "vocab_size": 10000}
    mock_label_encoder = MockLabelEncoder()  # Instantiates our class with the .classes_ attribute
    
    return mock_tokenizer, mock_label_encoder