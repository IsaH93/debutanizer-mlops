"""
Airflow DAG: auto-retrain pipeline.
Triggered by drift_check_daily when drift is detected.
Four tasks: fetch_data → retrain → evaluate_vs_production → promote_if_better.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys, json, sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

default_args = {
    "owner": "isa",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}


def fetch_latest_data(**context):
    """Pull latest week of simulated sensor data."""
    import pandas as pd
    from pathlib import Path
    sim_files = sorted((ROOT / "data/simulated").glob("week_*.parquet"))
    latest = sim_files[-1]
    df = pd.read_parquet(latest)
    context["ti"].xcom_push(key="n_rows", value=len(df))
    context["ti"].xcom_push(key="data_file", value=str(latest))
    print(f"Fetched {len(df)} rows from {latest.name}")


def run_retraining(**context):
    """Retrain model on fresh data + historical training set."""
    import os; os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    from training.train import train_and_log
    run_id, metrics = train_and_log(
        run_name=f"auto_retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
        version_note="Auto-retrain triggered by PSI drift detection"
    )
    context["ti"].xcom_push(key="new_run_id", value=run_id)
    context["ti"].xcom_push(key="new_metrics", value=metrics)
    print(f"Retrain complete: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")


def evaluate_vs_production(**context):
    """Compare new model RMSE against current production model."""
    new_metrics = context["ti"].xcom_pull(key="new_metrics")
    # Load current production metrics
    metrics_path = ROOT / "results/registry/latest_metrics.json"
    prod_metrics = json.loads(metrics_path.read_text())
    prod_rmse = prod_metrics.get("test_rmse", 9999)
    new_rmse  = new_metrics["rmse"]
    should_promote = bool(new_rmse < prod_rmse)
    context["ti"].xcom_push(key="should_promote", value=should_promote)
    context["ti"].xcom_push(key="prod_rmse", value=prod_rmse)
    context["ti"].xcom_push(key="new_rmse", value=new_rmse)
    print(f"Production RMSE: {prod_rmse:.4f} | New model RMSE: {new_rmse:.4f}")
    print(f"Decision: {'PROMOTE' if should_promote else 'KEEP EXISTING'}")


def promote_if_better(**context):
    """Promote new model to Production stage if it beats current."""
    import os; os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    should_promote = context["ti"].xcom_pull(key="should_promote")
    new_rmse  = context["ti"].xcom_pull(key="new_rmse")
    prod_rmse = context["ti"].xcom_pull(key="prod_rmse")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "trigger": "PSI drift detection",
        "production_rmse_before": round(prod_rmse, 4),
        "retrain_rmse": round(new_rmse, 4),
        "promoted": bool(should_promote),
        "reason": "RMSE improved" if should_promote else "RMSE did not improve — kept old model",
    }

    if should_promote:
        mlflow.set_tracking_uri(f"file://{ROOT}/mlflow_store")
        client = mlflow.tracking.MlflowClient()
        # Get latest version and transition to Production
        versions = client.search_model_versions("name='debutanizer-soft-sensor'")
        if versions:
            latest_version = max(int(v.version) for v in versions)
            client.transition_model_version_stage(
                name="debutanizer-soft-sensor",
                version=str(latest_version),
                stage="Production",
                archive_existing_versions=True,
            )
        print(f"Model promoted to Production. RMSE: {prod_rmse:.4f} → {new_rmse:.4f}")
    else:
        print(f"Promotion skipped. New RMSE {new_rmse:.4f} >= Production {prod_rmse:.4f}")

    # Append to retrain log
    log_path = ROOT / "results/registry/retrain_log.json"
    existing = json.loads(log_path.read_text()) if log_path.exists() else []
    existing.append(log_entry)
    log_path.write_text(json.dumps(existing, indent=2))


with DAG(
    dag_id="auto_retrain_pipeline",
    default_args=default_args,
    schedule_interval=None,   # triggered externally
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "retrain"],
    description="Auto-retrain and conditional model promotion on drift detection",
) as dag:

    t1 = PythonOperator(task_id="fetch_latest_data",      python_callable=fetch_latest_data)
    t2 = PythonOperator(task_id="run_retraining",         python_callable=run_retraining)
    t3 = PythonOperator(task_id="evaluate_vs_production", python_callable=evaluate_vs_production)
    t4 = PythonOperator(task_id="promote_if_better",      python_callable=promote_if_better)

    t1 >> t2 >> t3 >> t4
