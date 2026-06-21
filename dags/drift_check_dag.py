"""
Airflow DAG: daily drift check.
If drift detected, triggers the retrain DAG via TriggerDagRunOperator.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta
import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

default_args = {
    "owner": "isa",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def check_drift(**context):
    from monitoring.drift_detector import run_drift_check
    # In production: use current week's logged data. Here: simulate via week number.
    exec_date = context["execution_date"]
    week = min(12, max(1, (exec_date.isocalendar()[1] % 12) + 1))
    report = run_drift_check(week)
    context["ti"].xcom_push(key="drift_report", value=report)
    return report

def branch_on_drift(**context):
    report = context["ti"].xcom_pull(key="drift_report")
    if report and report.get("drift_detected"):
        return "trigger_retrain"
    return "no_action"

with DAG(
    dag_id="drift_check_daily",
    default_args=default_args,
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["monitoring", "drift"],
    description="Check for feature drift in incoming process data",
) as dag:

    run_check = PythonOperator(
        task_id="run_drift_check",
        python_callable=check_drift,
    )
    branch = BranchPythonOperator(
        task_id="branch_on_drift_result",
        python_callable=branch_on_drift,
    )
    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id="auto_retrain_pipeline",
        conf={"triggered_by": "drift_check_daily"},
    )
    no_action = EmptyOperator(task_id="no_action")

    run_check >> branch >> [trigger_retrain, no_action]
