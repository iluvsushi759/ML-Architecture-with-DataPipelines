import os
import joblib
import pandas as pd

def load_model(model_path='artifacts/model.joblib'):
    bundle = joblib.load(model_path)
    return bundle['model'], bundle['features']

def predict(model, feature_cols, payload: dict) -> float:
    df = pd.DataFrame([payload])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_cols]
    return float(model.predict(df)[0])

if __name__ == "__main__":
    model, feature_cols = load_model(os.environ.get('MODEL_PATH', 'artifacts/model.joblib'))
    example = {"AGE": 45, "GENDER_BIN": 1, "HOSPITAL_RATING": 4.2,
               "PLAN_TYPE_Standard": 1, "CLAIM_TYPE_Surgery": 1, "STATUS_Approved": 1}
    print(predict(model, feature_cols, example))
