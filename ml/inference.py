# ml/inference.py
import joblib
import numpy as np
import json
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

# Load the trained model and feature columns
MODEL_FILE = ARTIFACTS_DIR / "model.joblib"
_model_artifact = joblib.load(MODEL_FILE)
_model = _model_artifact["model"]
_feature_cols = _model_artifact["features"]

def predict(payload=None):
    """
    Predict function that accepts a dict of feature values.
    Any missing features will default to 0.
    
    Example usage:
    predict({"AGE":50, "GENDER_BIN":1, "PLAN_TYPE":2})
    """
    if payload is None:
        return "Error: No input provided. Provide a dict of feature values."

    # If input is a JSON string, parse it
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as e:
            return f"Error parsing input JSON: {e}"

    # Build input row in correct feature order
    row = [payload.get(col, 0) for col in _feature_cols]
    row = np.array(row).reshape(1, -1)

    # Run prediction
    pred = _model.predict(row)
    return float(pred[0])

# print(predict({"AGE":30, "GENDER_BIN":0, "PLAN_TYPE":1}))
# print(predict({"AGE":65, "GENDER_BIN":1, "PLAN_TYPE":3}))
