"""
Step 2 - Build point-in-time-correct fraud features with PySpark.

Every feature answers one question:
    "What did this card / merchant look like at the exact moment of this transaction,
     using ONLY information that existed at that moment?"

Feature families
  A. Window aggregates  - counts / sums / distinct-merchants over 1h, 24h, 7d per card
  B. Velocity           - ratio of this amount vs the card's own 7-day baseline,
                          declines in last 10 min, distinct countries in 1h
  C. Time-since-last    - seconds since the card's previous txn / previous decline
  D. Merchant history   - merchant txn count over 7d (merchant-side window)
  E. Target encoding    - smoothed historical fraud rate per merchant and per MCC,
                          computed from an EARLIER period only, respecting label delay
  F. Static attributes  - broadcast-joined merchant dimension (country, risk tier)

Leakage guards (the interviewer will ask):
  * rangeBetween(-N, -1)             -> window excludes the current row
  * no unboundedFollowing anywhere   -> never looks forward
  * target encoding uses hist period -> labels only from BEFORE the scoring period,
    and only labels whose label_confirmed_at <= cutoff (chargebacks arrive late!)
  * time-based split, not random     -> see train.py

Usage:
    python src/build_features.py --raw data/raw --out data/features \
        --hist-end 2026-02-01 --label-cutoff 2026-03-21

Timeline (data = Jan 1 .. Mar 31, "today" = Mar 21 = label_cutoff):
    Jan 1  - Feb 1   target-encoding history   (labels known by Mar 21 only)
    Feb 1  - Feb 25  training window           (labels mature: 24-48 days old)
    Feb 25 - Mar 5   validation
    Mar 21 - Mar 31  test  (the "next 10 days after deployment")
"""
import argparse
import time

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

SEC = {"10m": 600, "1h": 3600, "24h": 86400, "7d": 7 * 86400}


def spark_session(app="fraud-features"):
    return (SparkSession.builder.appName(app)
            .config("spark.sql.shuffle.partitions", "32")
            .config("spark.sql.adaptive.enabled", "false")   # so the skew demo is honest
            .config("spark.ui.enabled", "false")
            .getOrCreate())


# ----------------------------------------------------------------------------
# A + B + C : per-card rolling history
# ----------------------------------------------------------------------------
def card_window_features(df):
    """Rolling aggregates per card. `ts` is epoch seconds so rangeBetween is in seconds."""
    base = Window.partitionBy("card_id").orderBy("ts")

    def back(seconds):
        # (-N, -1): last N seconds, EXCLUDING the current transaction
        return base.rangeBetween(-seconds, -1)

    w1h, w24h, w7d, w10m = back(SEC["1h"]), back(SEC["24h"]), back(SEC["7d"]), back(SEC["10m"])

    df = (df
          # ---- A. window aggregates ---------------------------------------
          .withColumn("card_cnt_1h", F.count("*").over(w1h))
          .withColumn("card_cnt_24h", F.count("*").over(w24h))
          .withColumn("card_cnt_7d", F.count("*").over(w7d))
          .withColumn("card_sum_24h", F.coalesce(F.sum("amount").over(w24h), F.lit(0.0)))
          .withColumn("card_sum_7d", F.coalesce(F.sum("amount").over(w7d), F.lit(0.0)))
          .withColumn("card_mean_7d", F.avg("amount").over(w7d))
          .withColumn("card_nmerch_24h", F.approx_count_distinct("merchant_id").over(w24h))
          .withColumn("card_ncountry_1h", F.approx_count_distinct("txn_country").over(w1h))
          # ---- B. velocity --------------------------------------------------
          .withColumn("card_declines_10m", F.coalesce(F.sum("declined").over(w10m), F.lit(0)))
          .withColumn("amt_ratio_7d",
                      F.when(F.col("card_mean_7d").isNull(), F.lit(-1.0))   # -1 = no history
                       .otherwise(F.col("amount") / (F.col("card_mean_7d") + 1e-6)))
          .withColumn("amt_over_card_sum_24h",
                      F.col("amount") / (F.col("card_sum_24h") + F.col("amount")))
          # ---- C. time since last -------------------------------------------
          .withColumn("prev_ts", F.lag("ts").over(base))
          .withColumn("secs_since_last", F.coalesce(F.col("ts") - F.col("prev_ts"), F.lit(-1)))
          .withColumn("prev_country", F.lag("txn_country").over(base))
          .withColumn("country_changed",
                      F.when(F.col("prev_country").isNull(), 0)
                       .when(F.col("prev_country") != F.col("txn_country"), 1).otherwise(0))
          .drop("prev_ts", "prev_country"))
    return df


# ----------------------------------------------------------------------------
# D : merchant-side rolling history
# ----------------------------------------------------------------------------
def merchant_window_features(df):
    """
    Merchant-side rolling history.

    Gotcha we hit while building this: `approx_count_distinct(card_id)` over a
    7-day range window on MEGA_MART (538k rows in one partition) rebuilds a
    HyperLogLog sketch per row -> that single task ran for 10+ minutes while
    every other task finished in seconds.  That is key skew inside a window
    function, and salting doesn't help because a window needs the whole key in
    one partition.

    Fix: change the grain.  First aggregate to (merchant, hour) buckets - a tiny
    table - then run the window over hours instead of over rows.  Slight loss
    of precision (counts are bucketed to the hour), huge win in cost.  This is
    the standard trick for merchant-level velocity in production.
    """
    hourly = (df.groupBy("merchant_id", F.date_trunc("hour", "txn_time").alias("hr"))
                .agg(F.count("*").alias("n"), F.approx_count_distinct("card_id").alias("nc"))
                .withColumn("hr_ts", F.unix_timestamp("hr")))
    w7d = Window.partitionBy("merchant_id").orderBy("hr_ts").rangeBetween(-SEC["7d"], -1)
    w24 = Window.partitionBy("merchant_id").orderBy("hr_ts").rangeBetween(-SEC["24h"], -1)
    hourly = (hourly.withColumn("merch_cnt_7d", F.coalesce(F.sum("n").over(w7d), F.lit(0)))
                    .withColumn("merch_ncards_24h", F.coalesce(F.sum("nc").over(w24), F.lit(0)))
                    .select("merchant_id", "hr", "merch_cnt_7d", "merch_ncards_24h"))
    return (df.withColumn("hr", F.date_trunc("hour", "txn_time"))
              .join(hourly, ["merchant_id", "hr"], "left")
              .drop("hr"))


# ----------------------------------------------------------------------------
# E : time-safe, label-delay-aware target encoding
# ----------------------------------------------------------------------------
def target_encoding(spark, raw, key, hist_end, label_cutoff, k=20):
    """
    Smoothed fraud rate per `key` using ONLY transactions from before `hist_end`, and
    ONLY labels that were CONFIRMED before `label_cutoff` (= the day we build features).
    Rows whose chargeback hadn't arrived yet count as non-fraud - because that is
    exactly what production would have known on that day.

        te = (n * rate + k * global_rate) / (n + k)
    """
    hist = raw.filter(F.col("txn_time") < F.lit(hist_end))
    hist = hist.withColumn(
        "known_fraud",
        F.when((F.col("is_fraud") == 1) & (F.col("label_confirmed_at") < F.lit(label_cutoff)), 1).otherwise(0))
    global_rate = hist.agg(F.avg("known_fraud")).first()[0]
    stats = (hist.groupBy(key)
                 .agg(F.count("*").alias("n"), F.avg("known_fraud").alias("rate"))
                 .withColumn(f"{key}_te",
                             (F.col("n") * F.col("rate") + k * F.lit(global_rate)) / (F.col("n") + k))
                 .select(key, f"{key}_te"))
    return stats, global_rate


# ----------------------------------------------------------------------------
def main(raw_path, out_path, hist_end, label_cutoff):
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    t0 = time.time()

    raw = spark.read.parquet(f"{raw_path}/transactions")
    merchants = spark.read.parquet(f"{raw_path}/merchants.parquet")
    print(f"raw partitions on read: {raw.rdd.getNumPartitions()}")

    raw = (raw.withColumn("txn_time", F.col("txn_time").cast("timestamp"))
              .withColumn("label_confirmed_at", F.col("label_confirmed_at").cast("timestamp")))
    # Pre-partition by card so every card's history sits in one partition: the window
    # functions then shuffle once instead of once per feature family.
    df = (raw.withColumn("ts", F.unix_timestamp("txn_time"))
             .repartition(32, "card_id"))

    # A, B, C  (one shuffle on card_id)
    df = card_window_features(df)
    # D        (one shuffle on merchant_id)
    df = merchant_window_features(df)

    # E - two encodings, both from history only
    m_te, g = target_encoding(spark, raw, "merchant_id", hist_end, label_cutoff)
    c_te, _ = target_encoding(spark, raw, "mcc", hist_end, label_cutoff)
    df = (df.join(F.broadcast(m_te), "merchant_id", "left")
            .join(F.broadcast(c_te), "mcc", "left")
            .fillna({"merchant_id_te": g, "mcc_te": g}))   # unseen merchant -> global rate

    # F - static merchant attributes; small table -> broadcast join (no shuffle of big side)
    dim = merchants.select("merchant_id", "merchant_country", "merchant_risk_tier")
    df = (df.join(F.broadcast(dim), "merchant_id", "left")
            .withColumn("cross_border", (F.col("txn_country") != F.col("merchant_country")).cast("int"))
            .withColumn("is_ecom", (F.col("channel") == "ecom").cast("int"))
            .withColumn("risk_tier_num",
                        F.when(F.col("merchant_risk_tier") == "high", 2)
                         .when(F.col("merchant_risk_tier") == "med", 1).otherwise(0))
            .withColumn("hour_of_day", F.hour("txn_time"))
            .withColumn("is_weekend", (F.dayofweek("txn_time").isin(1, 7)).cast("int")))

    df = df.drop("ts", "merchant_country", "merchant_risk_tier")
    df.write.mode("overwrite").partitionBy("txn_date").parquet(out_path)

    feat = spark.read.parquet(out_path)
    print(f"feature rows: {feat.count():,}   written to {out_path}   ({time.time()-t0:.0f}s)")

    # quick sanity print - fraud vs legit means on a few features
    (feat.groupBy("is_fraud")
         .agg(F.round(F.avg("card_cnt_1h"), 2).alias("cnt_1h"),
              F.round(F.avg("secs_since_last"), 0).alias("secs_since_last"),
              F.round(F.avg("amt_ratio_7d"), 2).alias("amt_ratio_7d"),
              F.round(F.avg("card_declines_10m"), 2).alias("declines_10m"),
              F.round(F.avg("merchant_id_te"), 4).alias("merchant_te"))
         .orderBy("is_fraud").show())
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--out", default="data/features")
    p.add_argument("--hist-end", default="2026-02-01",
                   help="target encoding uses only transactions before this date")
    p.add_argument("--label-cutoff", default="2026-03-21",
                   help="'today' when features are built: only labels confirmed before this count")
    a = p.parse_args()
    main(a.raw, a.out, a.hist_end, a.label_cutoff)
