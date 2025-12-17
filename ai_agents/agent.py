#!/usr/bin/env python3
"""
ai_agents/agent.py
AI Agent entrypoint that ties RAG (retrieval QA) + tools (SQL, train, evaluate, predict) together.

Drop this file into ai_agents/ (replace existing agent.py). No other files are changed.
"""
print("🔵 agent.py STARTED")

import readline
import os
import sys
import json
import subprocess
import threading
import time
from pathlib import Path

# Import your RAG builder and Snowflake config
# Adjust module path if your folder is named differently
from ai_agents.rag_setup import create_rag, SNOWFLAKE_CONFIG

# Optional: try to import inference as python module (preferred)
try:
    from ml import inference as ml_inference_module  # if ml/inference.py exposes predict()
except Exception:
    ml_inference_module = None

# --- CONFIG ---
AGENT_TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", 90))  # seconds for expensive operations

# --- UTILITIES ---
REPO_ROOT = Path(__file__).resolve().parents[1]  # ../
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

def run_subprocess(cmd, timeout=60*10):
    """
    Run a subprocess command (list) and capture stdout/stderr.
    Returns (returncode, stdout+stderr).
    """
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out_lines = []
        start = time.time()
        while True:
            line = proc.stdout.readline()
            if line == "" and proc.poll() is not None:
                break
            if line:
                print(line.rstrip())
                out_lines.append(line)
            # timeout check
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                return -1, "Process killed due to timeout"
        rc = proc.poll()
        return rc, "".join(out_lines)
    except Exception as e:
        return -1, f"Subprocess error: {e}"

# --- TOOLS ---
def sql_query(sql: str, limit_output_lines=200):
    """Execute SQL directly against Snowflake using SNOWFLAKE_CONFIG from rag_setup."""
    import snowflake.connector
    from tabulate import tabulate
    try:
        cfg = SNOWFLAKE_CONFIG.copy()
        # allow environment overrides if present
        for k, envk in [("user","SNOWFLAKE_USER"), ("password","SNOWFLAKE_PASSWORD"),
                        ("account","SNOWFLAKE_ACCOUNT"), ("warehouse","SNOWFLAKE_WAREHOUSE"),
                        ("database","SNOWFLAKE_DATABASE"), ("schema","SNOWFLAKE_SCHEMA")]:
            if os.environ.get(envk):
                cfg[k] = os.environ[envk]
        conn = snowflake.connector.connect(
            user=cfg["user"],
            password=cfg["password"],
            account=cfg["account"],
            warehouse=cfg["warehouse"],
            database=cfg["database"],
            schema=cfg.get("schema", None)
        )
        cs = conn.cursor()
        cs.execute(sql)
        rows = cs.fetchall()
        cols = [d[0] for d in cs.description] if cs.description else []
        cs.close()
        conn.close()
        # pretty print using tabulate
        if not rows:
            return "No results found."
        text = tabulate(rows[:limit_output_lines], headers=cols, tablefmt="pretty")
        if len(rows) > limit_output_lines:
            text += f"\n... (output truncated, {len(rows)-limit_output_lines} more rows)"
        return text
    except Exception as e:
        return f"SQL execution error: {e}"

def train_model():
    """Run ml/train.py as subprocess (non-blocking prints to stdout)."""
    cmd = [sys.executable, str(REPO_ROOT / "ml" / "train.py")]
    rc, out = run_subprocess(cmd, timeout=60*60)  # allow up to 1 hour
    return rc, out

def train_optuna():
    """Run ml/train_optuna.py as subprocess."""
    cmd = [sys.executable, str(REPO_ROOT / "ml" / "train_optuna.py")]
    rc, out = run_subprocess(cmd, timeout=60*60*3)  # allow longer for optuna
    return rc, out

# --- NEW: asynchronous training ---
def train_model_async():
    """Run ml/train.py asynchronously in a separate thread."""
    def target():
        print("→ Starting training (train.py) in background...")
        rc, out = train_model()
        print("\n💾 Training finished.\nOutput:\n", out)
        print("\n🤖 Agent ready for next command.\n>> ", end="", flush=True)
    thread = threading.Thread(target=target)
    thread.start()
    return "Training started in background. Check console for progress."

def train_optuna_async():
    """Run ml/train_optuna.py asynchronously in a separate thread."""
    def target():
        print("→ Starting Optuna hyperparameter search (train_optuna.py) in background...", flush=True)
        rc, out = train_optuna()
        print("\n💾 Optuna training finished.\nOutput:\n", out, flush=True)
        print("\n🤖 Agent ready for next command.\n>> ", end="", flush=True)
    thread = threading.Thread(target=target)
    thread.start()
    return "Optuna training started in background. Check console for progress."


def evaluate_model():
    """Run ml/evaluate.py as subprocess or import evaluate.evaluate() if possible."""
    try:
        from ml import evaluate as ev
        if hasattr(ev, "evaluate"):
            res = ev.evaluate()
            return 0, str(res)
    except Exception:
        pass
    cmd = [sys.executable, str(REPO_ROOT / "ml" / "evaluate.py")]
    rc, out = run_subprocess(cmd, timeout=60*10)
    return rc, out

def predict_with_module(payload=None):
    """Try to call ml.inference.predict(payload) if available"""
    if ml_inference_module and hasattr(ml_inference_module, "predict"):
        try:
            out = ml_inference_module.predict(payload)
            return 0, str(out)
        except Exception as e:
            return -1, f"Error calling ml.inference.predict: {e}"
    cmd = [sys.executable, str(REPO_ROOT / "ml" / "inference.py")]
    rc, out = run_subprocess(cmd, timeout=60*10)
    return rc, out

def read_metrics():
    f = ARTIFACTS_DIR / "metrics.json"
    if not f.exists():
        return "metrics.json not found in artifacts/"
    try:
        return f.read_text()
    except Exception as e:
        return f"Could not read metrics.json: {e}"

# --- AGENT ROUTER / INTERFACE ---
def route_command(user_input: str, qa):
    ui = user_input.strip()
    low = ui.lower()

    # SQL
    if low.startswith("sql:") or low.startswith("sql "):
        sql = ui.split(":", 1)[1] if ":" in ui else ui.split(" ", 1)[1]
        print("→ Running SQL query...")
        return sql_query(sql)

    # show tables
    if "show tables" in low or "list tables" in low or low == "tables":
        return sql_query("SHOW TABLES IN SCHEMA PRESENTATION")

    # train (optuna)
    if low.startswith("train optuna") or low.startswith("train_optuna") or low == "train optuna":
        return train_optuna_async()
    if low.startswith("train") or low == "train model":
        return train_model_async()

    # evaluate
    if "evaluate" in low or low.startswith("eval"):
        rc, out = evaluate_model()
        return out if rc == 0 else f"evaluate exited {rc}\n{out}"

    # predict
    if low.startswith("predict"):
        payload = None
        if " " in ui:
            payload = ui.split(" ", 1)[1].strip()
        rc, out = predict_with_module(payload)
        return out if rc == 0 else f"predict exited {rc}\n{out}"

    # metrics
    if "metrics" in low:
        return read_metrics()

    # explain pipeline
    if "pipeline" in low or "explain pipeline" in low or "what does the pipeline" in low:
        hint = "Summarize the ML and data pipeline present in the repository, step-by-step. Focus on scripts and dataflow."
        query = f"{hint}\n\nQuestion: {ui}"
        result = {}
        def call_qa():
            try:
                resp = qa.invoke({"query": query})
                result["output"] = resp.get("result", str(resp))
            except Exception as e:
                result["output"] = f"RAG error while explaining pipeline: {e}"
        thread = threading.Thread(target=call_qa)
        thread.start()
        thread.join(timeout=AGENT_TIMEOUT)
        if thread.is_alive():
            return f"❌ Timeout ({AGENT_TIMEOUT}s). Possibly not enough CPU/GPU/RAM. Agent ready for next input."
        return result.get("output", "")

    # fallback: RAG
    try:
        print("🤖 Thinking... please wait.")
        resp = qa.invoke({"query": ui})
        return resp.get("result", str(resp))
    except Exception as e:
        return f"RAG invocation error: {e}"


# --- MAIN LOOP ---
def main():
    print("Starting AI Agent...")
    qa = create_rag(project_folder=str(REPO_ROOT))
    print("\n🤖 Agent ready. Type 'exit' to quit.")
    print("Examples: 'train', 'train optuna', 'sql: SELECT count(*) from PRESENTATION.CUSTOMERS', 'predict', 'metrics', or ask natural questions.'\n")

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        out = route_command(user_input, qa)
        # Print output safely; preserve SQL/table formatting
        if isinstance(out, str):
            print(out)
        else:
            try:
                print(out)
            except Exception:
                print(str(out))

if __name__ == "__main__":
    main()
