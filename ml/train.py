import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from preprocess import build_training_set

def train_and_save(model_path='model.joblib', metrics_path='metrics.json'):
    X, y, feature_cols = build_training_set()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))

    # ✅ Safe R² calculation
    denom = ((y_val - y_val.mean()) ** 2).sum()
    if denom == 0:
        r2 = float('nan')
        print("⚠️ Validation set has zero variance in CLAIM_AMOUNT — R² is undefined.")
    else:
        r2 = float(1 - (((y_val - preds) ** 2).sum() / denom))

    # ✅ Debugging info
    print("Validation set size:", len(y_val))
    print("Unique claim amounts in validation:", y_val.nunique())

    os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
    joblib.dump({'model': model, 'features': feature_cols}, model_path)

    with open(metrics_path, 'w') as f:
        f.write(f'{{"rmse": {rmse:.4f}, "r2": {r2:.4f}}}')
    print(f"Saved XGBoost model to {model_path} with RMSE={rmse:.4f}, R2={r2:.4f}")

if __name__ == "__main__":
    model_path = os.environ.get('MODEL_PATH', 'artifacts/model.joblib')
    metrics_path = os.environ.get('METRICS_PATH', 'artifacts/metrics.json')
    train_and_save(model_path, metrics_path)

if __name__ == "__main__":
    model_path = os.environ.get('MODEL_PATH', 'artifacts/model.joblib')
    metrics_path = os.environ.get('METRICS_PATH', 'artifacts/metrics.json')
    print("🚀 Starting training...")
    train_and_save(model_path, metrics_path)
    print("✅ Training complete. Model and metrics saved.")
