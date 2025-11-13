import streamlit as st
import joblib
import pandas as pd

# Load model
bundle = joblib.load("artifacts/model.joblib")
model = bundle["model"]
feature_cols = bundle["features"]

st.title("Insurance Claim Amount Predictor")

# User inputs
age = st.slider("Age", 18, 90, 45)
gender = st.radio("Gender", ["Male", "Female"])
hospital_rating = st.slider("Hospital Rating", 1.0, 5.0, 4.2)

plan_type = st.selectbox("Plan Type", ["Standard", "Gold", "Silver", "Bronze"])

claim_type = st.selectbox(
    "Claim Type",
    [
        "Surgery", "Consultation", "Therapy", "Emergency", "Radiology",
        "Dental", "Podiatry", "ENT", "Cardiology", "Gastroenterology",
        "Ophthalmology", "Orthopedics", "Pediatrics", "Dermatology",
        "Neurology", "Oncology", "Psychiatry", "Rehabilitation",
        "Obstetrics", "Gynecology"
    ]
)

status = st.selectbox("Status", ["Approved", "Pending", "Denied"])

# Convert inputs to feature vector
payload = {
    "AGE": age,
    "GENDER_BIN": 1 if gender == "Male" else 0,
    "HOSPITAL_RATING": hospital_rating,
    f"PLAN_TYPE_{plan_type}": 1,
    f"CLAIM_TYPE_{claim_type}": 1,
    f"STATUS_{status}": 1,
}

# Ensure all expected features exist
df = pd.DataFrame([payload])
for c in feature_cols:
    if c not in df.columns:
        df[c] = 0
df = df[feature_cols]

# Prediction
if st.button("Predict Claim Amount"):
    prediction = model.predict(df)[0]
    st.success(f"💰 Predicted Claim Amount: ${prediction:,.2f}")

# 🔽 INSERT THIS BLOCK BELOW 🔽
from evaluate import evaluate_model
import matplotlib.pyplot as plt
import seaborn as sns

st.header("📊 Model Evaluation")

results = evaluate_model()

st.write(f"**RMSE:** {results['rmse']:.2f}")
st.write(f"**R² Score:** {results['r2']:.2f}")

# Actual vs Predicted
fig1, ax1 = plt.subplots()
sns.scatterplot(x=results["actual"], y=results["predicted"], ax=ax1)
ax1.set_xlabel("Actual Claim Amount")
ax1.set_ylabel("Predicted Claim Amount")
ax1.set_title("Actual vs Predicted")
st.pyplot(fig1)

# Residuals
fig2, ax2 = plt.subplots()
sns.histplot(results["residuals"], bins=30, kde=True, ax=ax2)
ax2.set_title("Residuals Distribution")
st.pyplot(fig2)

# Feature Importance
fig3, ax3 = plt.subplots()
sns.barplot(x=results["feature_importance"], y=results["feature_names"], ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)


# 🔼 INSERT THIS BLOCK ABOVE 🔼

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from preprocess import build_training_set

if st.button("Show Underfitting vs Overfitting Chart"):
    X, y, feature_cols = build_training_set()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []
    n_estimators_range = [50, 100, 200, 400, 800]

    for n in n_estimators_range:
        model = XGBRegressor(
            n_estimators=n,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds_train = model.predict(X_train)
        preds_val = model.predict(X_val)

        rmse_train = np.sqrt(((preds_train - y_train) ** 2).mean())
        rmse_val = np.sqrt(((preds_val - y_val) ** 2).mean())

        train_errors.append(rmse_train)
        val_errors.append(rmse_val)

    fig, ax = plt.subplots()
    ax.plot(n_estimators_range, train_errors, label="Train RMSE")
    ax.plot(n_estimators_range, val_errors, label="Validation RMSE")
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("RMSE")
    ax.set_title("Underfitting vs Overfitting")
    ax.legend()
    st.pyplot(fig)
