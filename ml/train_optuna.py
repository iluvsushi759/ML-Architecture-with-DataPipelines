import optuna
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import build_training_set

# Load training data
X, y, feature_cols = build_training_set()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def rmse(y_true, y_pred):
    # Version-agnostic RMSE: sqrt of MSE (no 'squared' kwarg)
    return np.sqrt(mean_squared_error(y_true, y_pred))

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "random_state": 42,
        "n_jobs": -1,
        # Stronger regularization helps generalization during tuning
        # (Optuna will still adjust them)
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return rmse(y_val, preds)

# Run Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Train final model with best params
best_params = study.best_params
print("✅ Best hyperparameters found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")

final_model = XGBRegressor(**best_params)
final_model.fit(X_train, y_train)

# Evaluate
preds_val = final_model.predict(X_val)
final_rmse = rmse(y_val, preds_val)
final_r2 = r2_score(y_val, preds_val)

print(f"\n📊 Final Model Performance:")
print(f"  RMSE: {final_rmse:.2f}")
print(f"  R² Score: {final_r2:.2f}")

# Save model bundle
bundle = {
    "model": final_model,
    "features": feature_cols
}
joblib.dump(bundle, "artifacts/model.joblib")
print("\n💾 Model saved to artifacts/model.joblib")
