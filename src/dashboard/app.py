"""
Streamlit monitoring dashboard.
Panel 1: rolling predictions vs actuals
Panel 2: PSI scores per feature (bar chart with threshold line)
Panel 3: retrain history log
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

st.set_page_config(
    page_title="Debutanizer Soft Sensor — MLOps Dashboard",
    page_icon="⚗",
    layout="wide",
)

st.title("Debutanizer Soft Sensor — MLOps Dashboard")
st.caption("Real-time model monitoring | PSI drift detection | Auto-retrain log")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    df = pd.read_parquet(ROOT / "data/processed/debutanizer_features.parquet")
    split = int(len(df)*0.80)
    return df.iloc[split:].reset_index(drop=True)

@st.cache_data
def load_drift_summary():
    return json.loads((ROOT / "results/registry/drift_summary.json").read_text())

@st.cache_data
def load_retrain_log():
    p = ROOT / "results/registry/retrain_log.json"
    return json.loads(p.read_text()) if p.exists() else []

@st.cache_resource
def load_model():
    import mlflow.pyfunc, os, sys
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    sys.path.insert(0, str(ROOT / "src"))
    mlflow.set_tracking_uri(f"file://{ROOT}/mlflow_store")
    try:
        return mlflow.pyfunc.load_model("models:/debutanizer-soft-sensor/1")
    except:
        return None

# ── Metric cards ──────────────────────────────────────────────────────────────
metrics = json.loads((ROOT / "results/registry/latest_metrics.json").read_text())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Test R²",   f"{metrics['test_r2']:.4f}")
c2.metric("Test RMSE", f"{metrics['test_rmse']:.4f} wt%")
c3.metric("Test MAE",  f"{metrics['test_mae']:.4f} wt%")
c4.metric("CV RMSE",   f"{metrics['cv_rmse_mean']:.4f} ± {metrics.get('cv_rmse_std', 0):.4f}")

st.divider()

# ── Panel 1: Predictions ──────────────────────────────────────────────────────
st.subheader("Model predictions — test set")
df_test = load_predictions()
model = load_model()

n_show = st.slider("Samples to display", 50, min(500, len(df_test)), 200)
feat_cols = [c for c in df_test.columns if c not in ["timestamp","y_butane_content"]]

if model is not None:
    preds = model.predict(df_test[feat_cols].values[:n_show])
    actuals = df_test["y_butane_content"].values[:n_show]
    idx = np.arange(n_show)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=idx, y=actuals, name="Actual",    line=dict(color="#1D9E75", width=1.8)))
    fig1.add_trace(go.Scatter(x=idx, y=preds,   name="Predicted", line=dict(color="#D85A30", width=1.8, dash="dash")))
    fig1.update_layout(xaxis_title="Sample", yaxis_title="Butane content (wt%)",
                       legend=dict(orientation="h"), height=300, margin=dict(t=20,b=30))
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Run training to populate model predictions.")

st.divider()

# ── Panel 2: PSI drift scores ─────────────────────────────────────────────────
st.subheader("PSI drift scores by week")
drift_summary = load_drift_summary()
weeks   = [d["week"] for d in drift_summary]
max_psi = [min(d["max_psi"], 6.0) for d in drift_summary]
colors  = ["#D85A30" if d["drift_detected"] else "#1D9E75" for d in drift_summary]

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=weeks, y=max_psi, marker_color=colors, name="Max PSI",
                      text=[f"{p:.2f}" for p in max_psi], textposition="outside"))
fig2.add_hline(y=0.20, line_dash="dash", line_color="#E24B4A",
               annotation_text="PSI threshold (0.20)", annotation_position="top right")
fig2.add_vline(x=6.5, line_dash="dot", line_color="#888780",
               annotation_text="Drift injected →", annotation_position="top left")
fig2.update_layout(xaxis_title="Week", yaxis_title="Max PSI (capped 6.0)",
                   height=320, margin=dict(t=20,b=30))
st.plotly_chart(fig2, use_container_width=True)

# Per-feature PSI breakdown for selected week
week_sel = st.selectbox("Inspect week", list(range(1, 13)), index=11)
report_path = ROOT / f"results/reports/drift_report_week_{week_sel:02d}.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    feat_data = [(f, v["psi"], v["ks_pvalue"], v["drift_detected"])
                 for f, v in report["features"].items()]
    df_feat = pd.DataFrame(feat_data, columns=["Feature","PSI","KS p-value","Drift?"]).sort_values("PSI", ascending=False)
    st.dataframe(df_feat.style.applymap(
        lambda v: "background-color: #FAECE7; color: #993C1D" if v is True else "", subset=["Drift?"]),
        use_container_width=True)

st.divider()

# ── Panel 3: Retrain log ──────────────────────────────────────────────────────
st.subheader("Auto-retrain history")
retrain_log = load_retrain_log()
if retrain_log:
    df_log = pd.DataFrame(retrain_log)
    df_log["delta_rmse"] = (df_log["production_rmse_before"] - df_log["retrain_rmse"]).round(4)
    st.dataframe(df_log[["week","trigger","production_rmse_before","retrain_rmse","delta_rmse","promoted","reason"]],
                 use_container_width=True)
    n_promoted = sum(1 for e in retrain_log if e["promoted"])
    st.caption(f"{n_promoted}/{len(retrain_log)} retrain runs resulted in model promotion.")
else:
    st.info("No retrain events yet — drift will trigger this automatically.")
