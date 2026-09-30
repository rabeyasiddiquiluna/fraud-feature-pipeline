"""
Airflow DAG: fraud_feature_pipeline

Orchestrates the four stages as a daily batch job:

    generate_data  ->  build_features  ->  train_model  ->  validate_metrics
                            \-> benchmark_skew (runs in parallel, non-blocking)

What this shows (interview talking points):
  * Each stage is an idempotent script that reads/writes Parquet, so any task can be
    retried or re-run for a past date without side effects.
  * retries + retry_delay on every task: a transient Spark failure re-runs itself.
  * A quality gate at the end: if PR-AUC on the test window drops below a floor,
    the DAG fails and nothing downstream (model registration) would run.
  * The Spark job is submitted via BashOperator here; on a cluster the same task
    becomes SparkSubmitOperator (YARN) or DatabricksSubmitRunOperator.
  * `catchup=False` so enabling the DAG doesn't backfill every day since start_date.

Run locally:
    export AIRFLOW_HOME=~/airflow
    airflow standalone            # first run prints an admin password
    # copy this file into ~/airflow/dags/, wait ~30 s, open http://localhost:8080
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Where the repo lives on the machine running Airflow.  Override with an env var
# so the same DAG file works on a laptop and on a server.
REPO = os.environ.get("FRAUD_REPO", os.path.expanduser("~/Downloads/fraud-feature-pipeline"))
PY = os.environ.get("FRAUD_PYTHON", f"{REPO}/.venv/bin/python")

PR_AUC_FLOOR = 0.60  # quality gate: fail the run if the model degrades below this

default_args = {
    "owner": "luna",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def check_metrics(**context):
    """Quality gate. Reads results/metrics.json and fails the task if PR-AUC is too low."""
    with open(f"{REPO}/results/metrics.json") as f:
        m = json.load(f)
    pr_auc = m["A_honest_test"]["pr_auc"]
    recall = m["A_honest_test"]["recall@1%FPR"]
    print(f"test PR-AUC={pr_auc:.4f}  recall@1%FPR={recall:.3f}  floor={PR_AUC_FLOOR}")
    if pr_auc < PR_AUC_FLOOR:
        raise ValueError(f"PR-AUC {pr_auc:.4f} below floor {PR_AUC_FLOOR}; blocking model promotion")
    # push to XCom so a downstream "register model" task could read it
    context["ti"].xcom_push(key="pr_auc", value=pr_auc)


with DAG(
    dag_id="fraud_feature_pipeline",
    description="Daily: synthetic txns -> PySpark point-in-time features -> LightGBM -> quality gate",
    default_args=default_args,
    start_date=datetime(2026, 9, 1),
    schedule="0 3 * * *",        # 03:00 daily, after the previous day's transactions have landed
    catchup=False,
    max_active_runs=1,
    tags=["fraud", "spark", "lightgbm"],
) as dag:

    generate_data = BashOperator(
        task_id="generate_data",
        bash_command=f"cd {REPO} && {PY} src/generate_data.py",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {REPO} && {PY} src/build_features.py",
        # on a cluster: SparkSubmitOperator(application='src/build_features.py', conn_id='spark_yarn')
    )

    benchmark_skew = BashOperator(
        task_id="benchmark_skew",
        bash_command=f"cd {REPO} && {PY} src/benchmark_skew.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {REPO} && {PY} src/train.py",
    )

    validate_metrics = PythonOperator(
        task_id="validate_metrics",
        python_callable=check_metrics,
    )

    # dependencies
    generate_data >> build_features >> train_model >> validate_metrics
    generate_data >> benchmark_skew          # independent branch; doesn't gate training
