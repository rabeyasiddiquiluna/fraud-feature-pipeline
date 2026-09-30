# Running on Databricks

Two artifacts:

- `build_features_notebook.py` — the feature pipeline as a Databricks notebook (Delta Lake, ZORDER, time travel, Feature Store hook)
- `job.json` — a Workflows job definition that runs it nightly on a job cluster with retries and failure email

## One-time setup (Free Edition / Community Edition / trial)

1. Sign up at databricks.com/learn/free-edition (or use a company workspace).
2. Generate the data locally first: `python src/generate_data.py`.
3. Upload `data/raw/transactions/` and `data/raw/merchants.parquet` to DBFS at `/FileStore/fraud/raw/`
   (Catalog → DBFS → Upload, or `databricks fs cp -r data/raw dbfs:/FileStore/fraud/raw`).
4. Workspace → Import → pick `build_features_notebook.py` (Databricks recognises the
   `# Databricks notebook source` header and renders it as a notebook with cells).
5. Attach a cluster, **Run all**. ~1–2 min on a single-node cluster.

## Create the scheduled job

Via CLI (`pip install databricks-cli`, then `databricks configure`):

```bash
databricks jobs create --json-file databricks/job.json
```

Or in the UI: Workflows → Create Job → paste the same settings. Edit `notebook_path` and
`node_type_id` to match your workspace/cloud.

## What to point at in an interview

| Feature | Where | Why it matters |
|---|---|---|
| Bronze / Silver Delta tables | cells 1–4 | Medallion architecture; ACID writes, schema enforcement |
| `OPTIMIZE … ZORDER BY (card_id)` | cell 4 | Co-locates each card's rows → cheaper window shuffles and point lookups |
| `DESCRIBE HISTORY` / `versionAsOf` | cell 5 | Time travel = reproducible training snapshots |
| Widgets → `base_parameters` | cell 0 + job.json | Same notebook parameterised for dev/prod |
| Feature Store `timestamp_keys` | cell 6 | Native point-in-time joins; offline == online |
| AQE left on | markdown | Databricks auto-splits skewed partitions; local benchmark showed the manual fix |
| Job cluster, retries, failure email | job.json | Production hygiene |
