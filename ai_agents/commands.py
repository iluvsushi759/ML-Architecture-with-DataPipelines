# ai_agent/commands.py
import subprocess
import json
from pathlib import Path

ARTIFACTS_DIR = Path("../artifacts")  # adjust if needed

def train_model(params=None):
    cmd = ["python", "../ml/train.py"]
    if params:
        cmd += params
    subprocess.run(cmd)
    return "Training started."

def train_optuna():
    subprocess.run(["python", "../ml/train_optuna.py"])
    return "Optuna hyperparameter search started."

def evaluate_model():
    from ml import evaluate
    return evaluate.evaluate()  # assumes evaluate.py has evaluate() function

def predict(inputs):
    from ml import inference
    return inference.predict(inputs)

def get_metrics():
    metrics_file = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_file) as f:
        return json.load(f)
    
