# Running the pipeline under Airflow

`dags/fraud_pipeline_dag.py` schedules the four stages daily at 03:00 with retries and a
quality gate. Run it on a laptop in about 5 minutes.

```bash
# from the repo root, inside the same .venv
pip install "apache-airflow==2.10.5" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.11.txt"

export AIRFLOW_HOME=~/airflow
export FRAUD_REPO=$(pwd)                     # tells the DAG where the scripts live
airflow standalone                           # starts scheduler + webserver; prints admin password
```

In a second terminal:

```bash
mkdir -p ~/airflow/dags
cp airflow/dags/fraud_pipeline_dag.py ~/airflow/dags/
```

Open http://localhost:8080, log in (user `admin`, password from the first terminal), find
`fraud_feature_pipeline`, toggle it on, click ▶ to trigger. Watch the Graph view: five tasks,
`benchmark_skew` runs in parallel with the training branch.

## DAG shape

```
generate_data ──► build_features ──► train_model ──► validate_metrics
      └──────────► benchmark_skew
```

## Interview points

- **Idempotent tasks**: every stage reads and writes Parquet, so any task can be cleared and re-run.
- **Retries**: `retries=2, retry_delay=5min` on all tasks — a flaky Spark job heals itself.
- **Quality gate**: `validate_metrics` reads `results/metrics.json` and raises if PR-AUC < 0.60,
  so a degraded model never proceeds to registration/deployment.
- **XCom**: the gate pushes `pr_auc` for a downstream task (e.g. MLflow model registration) to read.
- **`catchup=False`, `max_active_runs=1`**: no accidental backfill storm, no overlapping runs.
- **Portable**: `FRAUD_REPO` / `FRAUD_PYTHON` env vars mean the same DAG runs on a laptop or a server.
- **On a cluster**: swap `BashOperator` for `SparkSubmitOperator` (YARN) or
  `DatabricksSubmitRunOperator` / `DatabricksRunNowOperator` to trigger `databricks/job.json`.
