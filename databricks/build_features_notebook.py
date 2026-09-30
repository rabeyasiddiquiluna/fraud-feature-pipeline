# Databricks notebook source
# MAGIC %md
# MAGIC # Fraud feature pipeline on Databricks
# MAGIC
# MAGIC Same point-in-time feature logic as `src/build_features.py`, adapted to Databricks:
# MAGIC
# MAGIC | Local version | Databricks version |
# MAGIC |---|---|
# MAGIC | `SparkSession.builder.getOrCreate()` | `spark` is pre-created on the cluster |
# MAGIC | Parquet files on disk | **Delta Lake** tables in the Unity Catalog / Hive metastore |
# MAGIC | Manual `repartition` | Delta `OPTIMIZE` + `ZORDER BY (card_id)` for the window shuffle |
# MAGIC | `spark.sql.adaptive.enabled=false` | AQE left **on** (Databricks default) — it auto-handles skew |
# MAGIC | Run with `python src/…` | Run as a **Workflows job** (`databricks/job.json`) |
# MAGIC
# MAGIC Widgets at the top make the notebook parameterised so the same notebook serves dev and prod.

# COMMAND ----------

dbutils.widgets.text("raw_path", "/Volumes/workspace/default/fraud/transactions_flat.parquet", "Raw transactions (Parquet file or folder)")
dbutils.widgets.text("merchants_path", "/Volumes/workspace/default/fraud/merchants.parquet", "Merchant dimension")
dbutils.widgets.text("schema", "fraud", "Target schema")
dbutils.widgets.text("hist_end", "2026-02-01", "Target-encoding history ends")
dbutils.widgets.text("label_cutoff", "2026-03-21", "Labels known as of")

RAW = dbutils.widgets.get("raw_path")
MERCH = dbutils.widgets.get("merchants_path")
SCHEMA = dbutils.widgets.get("schema")
HIST_END = dbutils.widgets.get("hist_end")
LABEL_CUTOFF = dbutils.widgets.get("label_cutoff")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
print(f"schema={SCHEMA}  hist_end={HIST_END}  label_cutoff={LABEL_CUTOFF}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze: land raw Parquet as a Delta table
# MAGIC Bronze = raw, append-only. Delta gives ACID writes, time travel, and schema enforcement for free.

# COMMAND ----------

from pyspark.sql import functions as F, Window

raw = (spark.read.parquet(RAW)
            .withColumn("txn_time", F.col("txn_time").cast("timestamp"))
            .withColumn("label_confirmed_at", F.col("label_confirmed_at").cast("timestamp"))
            .withColumn("txn_date", F.to_date("txn_time")))   # works whether input is flat or day-partitioned

(raw.write.format("delta").mode("overwrite")
    .partitionBy("txn_date")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SCHEMA}.bronze_transactions"))

merchants = spark.read.parquet(MERCH)
merchants.write.format("delta").mode("overwrite").saveAsTable(f"{SCHEMA}.dim_merchant")

display(spark.sql(f"SELECT is_fraud, COUNT(*) n FROM {SCHEMA}.bronze_transactions GROUP BY is_fraud"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver: point-in-time features
# MAGIC Identical logic to the local script. `rangeBetween(-N, -1)` excludes the current row.

# COMMAND ----------

SEC = {"10m": 600, "1h": 3600, "24h": 86400, "7d": 7 * 86400}

df = (spark.table(f"{SCHEMA}.bronze_transactions")
           .withColumn("ts", F.unix_timestamp("txn_time")))

base = Window.partitionBy("card_id").orderBy("ts")
back = lambda s: base.rangeBetween(-s, -1)
w10m, w1h, w24h, w7d = back(SEC["10m"]), back(SEC["1h"]), back(SEC["24h"]), back(SEC["7d"])

df = (df
      .withColumn("card_cnt_1h", F.count("*").over(w1h))
      .withColumn("card_cnt_24h", F.count("*").over(w24h))
      .withColumn("card_cnt_7d", F.count("*").over(w7d))
      .withColumn("card_sum_24h", F.coalesce(F.sum("amount").over(w24h), F.lit(0.0)))
      .withColumn("card_sum_7d", F.coalesce(F.sum("amount").over(w7d), F.lit(0.0)))
      .withColumn("card_mean_7d", F.avg("amount").over(w7d))
      .withColumn("card_nmerch_24h", F.approx_count_distinct("merchant_id").over(w24h))
      .withColumn("card_ncountry_1h", F.approx_count_distinct("txn_country").over(w1h))
      .withColumn("card_declines_10m", F.coalesce(F.sum("declined").over(w10m), F.lit(0)))
      .withColumn("amt_ratio_7d",
                  F.when(F.col("card_mean_7d").isNull(), F.lit(-1.0))
                   .otherwise(F.col("amount") / (F.col("card_mean_7d") + 1e-6)))
      .withColumn("amt_over_card_sum_24h", F.col("amount") / (F.col("card_sum_24h") + F.col("amount")))
      .withColumn("prev_ts", F.lag("ts").over(base))
      .withColumn("secs_since_last", F.coalesce(F.col("ts") - F.col("prev_ts"), F.lit(-1)))
      .withColumn("prev_country", F.lag("txn_country").over(base))
      .withColumn("country_changed",
                  F.when(F.col("prev_country").isNull(), 0)
                   .when(F.col("prev_country") != F.col("txn_country"), 1).otherwise(0))
      .drop("prev_ts", "prev_country"))

# merchant history at hourly grain (avoids per-row HLL on the hot merchant)
hourly = (df.groupBy("merchant_id", F.date_trunc("hour", "txn_time").alias("hr"))
            .agg(F.count("*").alias("n"), F.approx_count_distinct("card_id").alias("nc"))
            .withColumn("hr_ts", F.unix_timestamp("hr")))
mw7d = Window.partitionBy("merchant_id").orderBy("hr_ts").rangeBetween(-SEC["7d"], -1)
mw24 = Window.partitionBy("merchant_id").orderBy("hr_ts").rangeBetween(-SEC["24h"], -1)
hourly = (hourly.withColumn("merch_cnt_7d", F.coalesce(F.sum("n").over(mw7d), F.lit(0)))
                .withColumn("merch_ncards_24h", F.coalesce(F.sum("nc").over(mw24), F.lit(0)))
                .select("merchant_id", "hr", "merch_cnt_7d", "merch_ncards_24h"))
df = (df.withColumn("hr", F.date_trunc("hour", "txn_time"))
        .join(hourly, ["merchant_id", "hr"], "left").drop("hr"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Target encoding — history only, labels as-of `label_cutoff`

# COMMAND ----------

def target_encoding(key, k=20):
    hist = (spark.table(f"{SCHEMA}.bronze_transactions")
                 .filter(F.col("txn_time") < F.lit(HIST_END))
                 .withColumn("known_fraud",
                             F.when((F.col("is_fraud") == 1) &
                                    (F.col("label_confirmed_at") < F.lit(LABEL_CUTOFF)), 1).otherwise(0)))
    g = hist.agg(F.avg("known_fraud")).first()[0]
    stats = (hist.groupBy(key)
                 .agg(F.count("*").alias("n"), F.avg("known_fraud").alias("rate"))
                 .withColumn(f"{key}_te", (F.col("n")*F.col("rate") + k*F.lit(g)) / (F.col("n") + k))
                 .select(key, f"{key}_te"))
    return stats, g

m_te, g = target_encoding("merchant_id")
c_te, _ = target_encoding("mcc")
dim = spark.table(f"{SCHEMA}.dim_merchant").select("merchant_id", "merchant_country", "merchant_risk_tier")

# Databricks auto-broadcasts tables under spark.sql.autoBroadcastJoinThreshold (10 MB default);
# the hint makes the intent explicit and shows up in the Spark UI plan.
df = (df.join(F.broadcast(m_te), "merchant_id", "left")
        .join(F.broadcast(c_te), "mcc", "left")
        .fillna({"merchant_id_te": g, "mcc_te": g})
        .join(F.broadcast(dim), "merchant_id", "left")
        .withColumn("cross_border", (F.col("txn_country") != F.col("merchant_country")).cast("int"))
        .withColumn("is_ecom", (F.col("channel") == "ecom").cast("int"))
        .withColumn("risk_tier_num", F.when(F.col("merchant_risk_tier") == "high", 2)
                                       .when(F.col("merchant_risk_tier") == "med", 1).otherwise(0))
        .withColumn("hour_of_day", F.hour("txn_time"))
        .withColumn("is_weekend", F.dayofweek("txn_time").isin(1, 7).cast("int"))
        .drop("ts", "merchant_country", "merchant_risk_tier"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write Silver as Delta, then OPTIMIZE / ZORDER
# MAGIC ZORDER on `card_id` co-locates each card's rows in the same files, so the next
# MAGIC window computation (or a point lookup for online serving) reads far fewer files.

# COMMAND ----------

(df.write.format("delta").mode("overwrite")
   .partitionBy("txn_date")
   .option("overwriteSchema", "true")
   .saveAsTable(f"{SCHEMA}.silver_features"))

spark.sql(f"OPTIMIZE {SCHEMA}.silver_features ZORDER BY (card_id)")

display(spark.sql(f"""
  SELECT is_fraud,
         ROUND(AVG(card_cnt_1h), 2)       AS cnt_1h,
         ROUND(AVG(secs_since_last), 0)   AS secs_since_last,
         ROUND(AVG(card_declines_10m), 2) AS declines_10m,
         ROUND(AVG(merchant_id_te), 4)    AS merchant_te
  FROM {SCHEMA}.silver_features GROUP BY is_fraud ORDER BY is_fraud"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Delta time travel — the point-in-time story at the table level
# MAGIC Every write is a version. You can train against the exact snapshot a model saw.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {SCHEMA}.silver_features").select("version", "timestamp", "operation"))

# Example: read the previous version (if one exists)
# spark.read.format("delta").option("versionAsOf", 0).table(f"{SCHEMA}.silver_features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register in Databricks Feature Store (optional — needs ML runtime)
# MAGIC Uncomment on an ML cluster. This is what makes offline == online.

# COMMAND ----------

# from databricks.feature_engineering import FeatureEngineeringClient
# fe = FeatureEngineeringClient()
# fe.create_table(
#     name=f"{SCHEMA}.card_txn_features",
#     primary_keys=["txn_id"],
#     timestamp_keys=["txn_time"],       # enables point-in-time lookups
#     df=spark.table(f"{SCHEMA}.silver_features"),
#     description="Point-in-time card/merchant velocity features for fraud scoring",
# )

# COMMAND ----------

dbutils.notebook.exit(f"silver_features written: {spark.table(f'{SCHEMA}.silver_features').count()} rows")
