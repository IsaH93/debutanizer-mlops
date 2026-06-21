"""
Training script: XGBoost soft sensor with full MLflow tracking.
Logs params, metrics, feature importance, and model artifact.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json, joblib

ROOT = Path(__file__).parent.parent.parent
DATA  = ROOT / "data/processed/debutanizer_features.parquet"
STORE = ROOT / "mlflow_store"
RESULTS = ROOT / "results"

FEATURE_COLS_EXCLUDE = ["timestamp", "y_butane_content"]
TARGET = "y_butane_content"

PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.04,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def load_data():
    df = pd.read_parquet(DATA)
    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]
    X = df[feature_cols].values
    y = df[TARGET].values
    return X, y, feature_cols, df


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}


def plot_feature_importance(model, feature_names, path):
    importance = model.feature_importances_
    top_n = 15
    idx = np.argsort(importance)[-top_n:]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(range(top_n), importance[idx], color="#1D9E75", alpha=0.82)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Feature importance (gain)", fontsize=10)
    ax.set_title("Top 15 features — debutanizer soft sensor", fontsize=11, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(y_true, y_pred, split_label, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n = min(300, len(y_true))
    axes[0].plot(y_true[:n], label="Actual", color="#0F6E56", lw=1.5, alpha=0.9)
    axes[0].plot(y_pred[:n], label="Predicted", color="#D85A30", lw=1.5, alpha=0.85, linestyle="--")
    axes[0].set_title(f"Predicted vs actual — {split_label} (first {n} pts)", fontsize=10)
    axes[0].set_ylabel("Butane content (wt%)")
    axes[0].set_xlabel("Sample")
    axes[0].legend(fontsize=9)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[1].scatter(y_true, y_pred, alpha=0.35, s=12, color="#534AB7")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([mn, mx], [mn, mx], "k--", lw=1, alpha=0.6)
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Parity plot", fontsize=10)
    axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def train_and_log(run_name="initial_training", version_note=""):
    mlflow.set_tracking_uri(f"file://{STORE}")
    mlflow.set_experiment("debutanizer-soft-sensor")

    X, y, feature_names, df = load_data()
    # Temporal split: 80% train, 20% test
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(**PARAMS)

    with mlflow.start_run(run_name=run_name) as run:
        # Cross-validation on training set
        tscv = TimeSeriesSplit(n_splits=5)
        cv_rmses = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = XGBRegressor(**PARAMS)
            m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
            p = m.predict(X_train[val_idx])
            cv_rmses.append(np.sqrt(mean_squared_error(y_train[val_idx], p)))
        cv_rmse_mean = np.mean(cv_rmses)
        cv_rmse_std  = np.std(cv_rmses)

        # Final fit on all training data
        model.fit(X_train, y_train, verbose=False)
        train_metrics = compute_metrics(y_train, model.predict(X_train))
        test_metrics  = compute_metrics(y_test,  model.predict(X_test))

        # Log everything
        mlflow.log_params(PARAMS)
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_size", split)
        mlflow.log_param("test_size", len(X_test))
        if version_note:
            mlflow.log_param("version_note", version_note)

        mlflow.log_metric("cv_rmse_mean", cv_rmse_mean)
        mlflow.log_metric("cv_rmse_std",  cv_rmse_std)
        for k, v in train_metrics.items():
            mlflow.log_metric(f"train_{k}", v)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Plots
        fi_path   = RESULTS / "plots/feature_importance.png"
        pred_path = RESULTS / "plots/predictions.png"
        plot_feature_importance(model, feature_names, fi_path)
        plot_predictions(y_test, model.predict(X_test), "test set", pred_path)
        mlflow.log_artifact(str(fi_path))
        mlflow.log_artifact(str(pred_path))

        # Log model + feature names as artifact
        mlflow.xgboost.log_model(model, "model",
                                  registered_model_name="debutanizer-soft-sensor")
        fn_path = RESULTS / "registry/feature_names.json"
        fn_path.write_text(json.dumps(feature_names))
        mlflow.log_artifact(str(fn_path))

        run_id = run.info.run_id
        print(f"\n{'='*55}")
        print(f"Run: {run_name}  |  Run ID: {run_id}")
        print(f"  CV RMSE:    {cv_rmse_mean:.4f} ± {cv_rmse_std:.4f}")
        print(f"  Train R²:   {train_metrics['r2']:.4f}")
        print(f"  Test  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  Test  R²:   {test_metrics['r2']:.4f}")
        print(f"  Test  MAE:  {test_metrics['mae']:.4f}")
        print(f"  Test  MAPE: {test_metrics['mape']:.2f}%")
        print(f"{'='*55}\n")

        # Save metrics for promotion gate
        metrics_out = RESULTS / "registry/latest_metrics.json"
        metrics_out.write_text(json.dumps({
            "run_id": run_id,
            "run_name": run_name,
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "cv_rmse_mean": cv_rmse_mean,
        }))
        return run_id, test_metrics


if __name__ == "__main__":
    train_and_log(run_name="initial_training_v1",
                  version_note="Baseline XGBoost — lag + rolling features")
