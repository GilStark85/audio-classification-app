# tests/test_app.py

import os
import joblib
import numpy as np
from app import extract_features

def test_model_load():
    model = joblib.load("audio_model.pkl")
    assert model is not None
    assert hasattr(model, "predict")

def test_feature_extraction_shape():
    sample_audio = "tests/sample.wav"
    assert os.path.exists(sample_audio), "Sample audio not found."
    features = extract_features(sample_audio)
    assert isinstance(features, np.ndarray)
    assert features.shape == (80,)
