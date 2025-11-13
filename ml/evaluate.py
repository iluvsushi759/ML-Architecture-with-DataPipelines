def evaluate_model(model_path="artifacts/model.joblib"):
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from preprocess import build_training_set

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    X, y, _ = build_training_set()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_val)
    residuals = y_val - preds

    rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
    denom = ((y_val - y_val.mean()) ** 2).sum()
    r2 = float(1 - (((y_val - preds) ** 2).sum() / denom)) if denom != 0 else float("nan")

    return {
        "rmse": rmse,
        "r2": r2,
        "actual": y_val,
        "predicted": preds,
        "residuals": residuals,
        "feature_importance": model.feature_importances_,
        "feature_names": feature_cols
    }
