"""
FastAPI serving layer — loads Production model from MLflow registry.
Endpoints: /predict, /health, /model-info
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import sqlite3, json, os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
MLFLOW_URI  = f"file://{ROOT}/mlflow_store"
MODEL_NAME  = "debutanizer-soft-sensor"
MODEL_STAGE = "Production"
LOG_DB      = ROOT / "results/prediction_log.db"

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

app = FastAPI(
    title="Debutanizer Soft Sensor API",
    description="Predicts butane (C4) content in debutanizer bottom product from process sensor readings.",
    version="1.0.0",
)

_model = None
_feature_names = None

def load_model():
    global _model, _feature_names
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except Exception:
        # Fall back to latest version if no stage set
        _model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/1")
    fn_path = ROOT / "results/registry/feature_names.json"
    _feature_names = json.loads(fn_path.read_text())

def get_model():
    if _model is None:
        load_model()
    return _model, _feature_names

def init_db():
    conn = sqlite3.connect(LOG_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS prediction_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        u1_top_tray_temp REAL, u2_top_temp REAL, u3_reflux_flow REAL,
        u4_feed_flow REAL, u5_6th_tray_temp REAL, u6_bottom_temp REAL,
        u7_pressure REAL, prediction REAL
    )""")
    conn.commit(); conn.close()

init_db()


class SensorReading(BaseModel):
    u1_top_tray_temp: float
    u2_top_temp: float
    u3_reflux_flow: float
    u4_feed_flow: float
    u5_6th_tray_temp: float
    u6_bottom_temp: float
    u7_pressure: float

class PredictionResponse(BaseModel):
    prediction_wt_pct: float
    model_version: str
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "stage": MODEL_STAGE,
            "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    model, feature_names = get_model()
    raw = reading.dict()
    # Build feature vector with engineered features using last known lags (simplified for API)
    row = {f: raw.get(f, 0.0) for f in feature_names}
    # Fill lag/rolling features with raw values as approximation
    for feat in ["u1_top_tray_temp","u2_top_temp","u3_reflux_flow",
                 "u4_feed_flow","u5_6th_tray_temp","u6_bottom_temp","u7_pressure"]:
        for lag in [1,2,3]:
            row[f"{feat}_lag{lag}"] = raw.get(feat, 0.0)
        for w in [6,12]:
            key = f"{feat}_roll{w}"
            if key in row:
                row[key] = raw.get(feat, 0.0)
    row["delta_top_bottom"] = raw["u1_top_tray_temp"] - raw["u6_bottom_temp"]
    row["delta_temp_56"]    = raw["u5_6th_tray_temp"] - raw["u6_bottom_temp"]
    row["reflux_ratio"]     = raw["u3_reflux_flow"] / (raw["u4_feed_flow"] + 1e-6)
    df = pd.DataFrame([{k: row.get(k, 0.0) for k in feature_names}])
    pred = float(model.predict(df)[0])

    # Log to SQLite
    conn = sqlite3.connect(LOG_DB)
    conn.execute("""INSERT INTO prediction_log
        (timestamp,u1_top_tray_temp,u2_top_temp,u3_reflux_flow,u4_feed_flow,
         u5_6th_tray_temp,u6_bottom_temp,u7_pressure,prediction) VALUES (?,?,?,?,?,?,?,?,?)""",
        (datetime.utcnow().isoformat(),
         raw["u1_top_tray_temp"], raw["u2_top_temp"], raw["u3_reflux_flow"],
         raw["u4_feed_flow"], raw["u5_6th_tray_temp"], raw["u6_bottom_temp"],
         raw["u7_pressure"], pred))
    conn.commit(); conn.close()

    return PredictionResponse(
        prediction_wt_pct=round(pred, 4),
        model_version=f"{MODEL_NAME}/{MODEL_STAGE}",
        timestamp=datetime.utcnow().isoformat(),
    )

@app.get("/model-info")
def model_info():
    _, feature_names = get_model()
    metrics = json.loads((ROOT / "results/registry/latest_metrics.json").read_text())
    return {
        "model_name": MODEL_NAME,
        "n_features": len(feature_names),
        "feature_names": feature_names[:10],
        "test_rmse": metrics.get("test_rmse"),
        "test_r2":   metrics.get("test_r2"),
    }
