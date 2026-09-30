"""
Step 3 - Spark optimization benchmark: joins under key skew.

The question this answers in an interview:
    "A join is slow because one merchant has 25% of all rows.  What do you do?"

Three ways to join transactions (2M rows, skewed on merchant_id) to a merchant
dimension table (3k rows):

  1. sort_merge_naive  - default sort-merge join. Both sides get shuffled on
                         merchant_id; the MEGA_MART partition is 25x larger than
                         the others -> one straggler task, everyone else waits.
  2. salted            - split the hot key: add a random salt 0..S-1 to the big
                         side, explode the small side S times, join on
                         (merchant_id, salt).  The hot key is now spread over S
                         tasks.  Works for any join, even when the small side is
                         too big to broadcast.
  3. broadcast         - the dimension is tiny (3k rows), so ship it to every
                         executor and do a hash join with NO shuffle of the big
                         side at all.  Best answer when the small side fits in
                         memory (default threshold 10 MB).

Also shows the explain() plans so you can point at "SortMergeJoin" vs
"BroadcastHashJoin" in the interview, and the max-vs-median task size that
proves the skew.

Usage:
    python src/benchmark_skew.py --raw data/raw --salt 16
"""
import argparse
import time

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def spark_session():
    return (SparkSession.builder.appName("skew-benchmark")
            .config("spark.sql.shuffle.partitions", "32")
            .config("spark.sql.adaptive.enabled", "false")          # AQE would auto-fix skew; we show the manual way
            .config("spark.sql.autoBroadcastJoinThreshold", "-1")   # disable auto-broadcast so #1 is really SMJ
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def timed(name, df):
    t = time.time()
    n = df.count()          # count() forces the join to actually run
    dt = time.time() - t
    print(f"\n{name:<18} rows={n:,}   {dt:6.1f}s")
    return dt


def partition_sizes(df, key):
    """How many rows land in each shuffle partition if we hash on `key`?"""
    sizes = (df.groupBy(F.spark_partition_id().alias("pid")).count()
               .select("count").rdd.flatMap(lambda r: r).collect())
    sizes.sort()
    med = sizes[len(sizes) // 2]
    print(f"\n   shuffle partitions on {key}: n={len(sizes)}  median={med:,}  max={sizes[-1]:,}  "
          f"max/median = {sizes[-1]/max(med,1):.1f}x")


def main(raw, salt_n):
    spark = spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    txns = spark.read.parquet(f"{raw}/transactions").select("txn_id", "merchant_id", "amount")
    dim = spark.read.parquet(f"{raw}/merchants.parquet").select("merchant_id", "merchant_country", "merchant_risk_tier")

    print("\n=== key distribution ===")
    (txns.groupBy("merchant_id").count().orderBy(F.desc("count")).limit(3).show())
    print("skew after repartition(merchant_id) - what a shuffle join sees:")
    partition_sizes(txns.repartition(32, "merchant_id"), "merchant_id")

    results = {}

    # ---- 1. naive sort-merge join ---------------------------------------
    print("\n=== 1. sort-merge join (naive) ===")
    j1 = txns.join(dim, "merchant_id")
    j1.explain(mode="simple")
    results["sort_merge_naive"] = timed("sort_merge_naive", j1)

    # ---- 2. salted join ---------------------------------------------------
    print(f"\n=== 2. salted sort-merge join (salt={salt_n}) ===")
    txns_s = txns.withColumn("salt", (F.rand(seed=7) * salt_n).cast("int"))
    dim_s = dim.withColumn("salt", F.explode(F.array([F.lit(i) for i in range(salt_n)])))
    j2 = txns_s.join(dim_s, ["merchant_id", "salt"]).drop("salt")
    print("skew after salting:")
    partition_sizes(txns_s.repartition(32, "merchant_id", "salt"), "(merchant_id, salt)")
    results["salted"] = timed("salted", j2)

    # ---- 3. broadcast join ----------------------------------------------
    print("\n=== 3. broadcast hash join ===")
    j3 = txns.join(F.broadcast(dim), "merchant_id")
    j3.explain(mode="simple")
    results["broadcast"] = timed("broadcast", j3)

    print("\n=== summary ===")
    base = results["sort_merge_naive"]
    for k, v in results.items():
        print(f"{k:<18} {v:6.1f}s   {base/v:4.1f}x faster than naive")
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--salt", type=int, default=16)
    a = p.parse_args()
    main(a.raw, a.salt)
