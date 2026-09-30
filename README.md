# fraud-feature-pipeline

Point-in-time-correct feature engineering for card-fraud detection, built with **PySpark**, evaluated with **LightGBM**, on 2.16M synthetic transactions with a deliberately skewed merchant key.

The repo answers four questions a payments ML team asks every day:

1. How do you turn a raw transaction into features about the card's *recent* behavior — without ever peeking into the future?
2. How do you handle a join when one key holds 25% of all rows?
3. How do you evaluate a model when only 0.4% of rows are positive and labels arrive 3–30 days late?
4. How do you pick a decision threshold that reflects business cost, not 0.5?

## Results

| Metric (test = 10 days after "deployment") | Value |
|---|---|
| Base fraud rate | 0.40% |
| PR-AUC | **0.754** (a random model scores 0.004) |
| ROC-AUC | 0.896 |
| Recall @ 1% false-positive rate | **77%** |
| Recall @ 0.1% false-positive rate | **74%** |
| Cost-optimal threshold | 0.05 → recall 73%, precision 87% |
| Random-split vs time-split PR-AUC gap | +0.01 (no leakage detected) |
| Fraud labels mature at retrain date | 97% |

Join benchmark on the skewed merchant key (2 cores, local):

| Join strategy | Max / median partition size | Time | Speed-up |
|---|---|---|---|
| Sort-merge (naive) | 13.7× | 3.4 s | 1.0× |
| Sort-merge + salting (S=16) | 1.9× | 2.1 s | 1.6× |
| Broadcast hash join | no shuffle | 0.7 s | **5.0×** |

Top features by gain: `secs_since_last` (48%), `merchant_id_te` (13%), then amount/velocity features.

## Pipeline

```
generate_data.py  →  build_features.py  →  train.py
   (pandas)             (PySpark)          (LightGBM)
      │                     │                   │
 data/raw/            data/features/       results/
 transactions/        (Parquet, by day)    metrics.json
 (Parquet, by day)                         feature_importance.csv
 merchants.parquet                         model_lgbm.txt

benchmark_skew.py   (PySpark, standalone)   →  results/benchmark_output.txt
```

### `src/generate_data.py` — synthetic data with planted fraud
20k cards, 90 days, 3k merchants. Fraud episodes are injected as three patterns the features are designed to catch: **velocity bursts**, **card testing** (tiny amounts + declines), **geo-jumps**. One merchant (`MEGA_MART`) receives 25% of traffic so the skew problem is real. Chargebacks are confirmed 3–30 days after the transaction (`label_confirmed_at`).

### `src/build_features.py` — the feature layer
| Family | Features | Spark mechanism |
|---|---|---|
| Card window aggregates | count / sum / mean / distinct merchants over 1h, 24h, 7d | `Window.partitionBy(card).orderBy(ts).rangeBetween(-N, -1)` |
| Velocity | amount ÷ card's 7-day mean, declines in 10 min, distinct countries in 1h | same window, ratio columns |
| Time since last | seconds since previous txn, country changed since last | `F.lag()` over ordered window |
| Merchant history | merchant txn count 7d, distinct cards 24h | window over **hourly buckets** (see gotcha below) |
| Target encoding | smoothed fraud rate per merchant and per MCC | `groupBy` on history period → `broadcast` join |
| Static / context | cross-border, e-com, risk tier, hour, weekend | `broadcast` join on merchant dimension |

**Leakage guards**
- `rangeBetween(-N, -1)`: the window ends *before* the current row.
- No `unboundedFollowing` anywhere.
- Target encoding uses transactions from before `--hist-end` **and** only labels confirmed before `--label-cutoff` ("today"). A chargeback that hadn't arrived yet counts as legit, because that is what production knew.
- `label_confirmed_at` is carried through so `train.py` can build an honest label set.

**Gotcha we hit:** `approx_count_distinct(card_id)` over a 7-day range window on the MEGA_MART partition rebuilt a HyperLogLog sketch per row — one task ran 10+ minutes while every other task took seconds. Salting can't fix a window function (the key must stay in one partition), so we changed the *grain*: aggregate to (merchant, hour) first, then window over hours. 10 min → 78 s for the whole pipeline.

### `src/benchmark_skew.py` — join strategies under skew
Runs the same big-to-small join three ways with AQE and auto-broadcast turned off (so the comparison is honest), prints `explain()` so you can point at `SortMergeJoin` vs `BroadcastHashJoin`, and measures max/median partition size to prove the skew and its fix.

### `src/train.py` — evaluation the way a fraud team does it
- **Time-based split**: TE history (Jan 1–Feb 1) → train (Feb 1–25) → valid (Feb 25–Mar 5) → *gap* → test (Mar 21–31). The gap mimics deploying a model and watching the next 10 days.
- **Label maturity**: the retrain date is Mar 21; only chargebacks confirmed by then are trusted. 97% of training-window fraud is mature by that date; the script prints the number.
- **Imbalance**: `scale_pos_weight = sqrt(n_neg / n_pos)` (~16×; the full ~250× ratio made early stopping fire at tree 1); early stopping on validation PR-AUC; metrics are PR-AUC and recall at fixed FPR. Accuracy is never printed.
- **Leakage check**: trains a second model on a random 70/30 split of the same rows. If random-split PR-AUC ≫ time-split PR-AUC, something leaks. Gap here is +0.01.
- **Business threshold**: sweeps thresholds minimizing `missed_fraud_$ + 5$ × false_declines`.

## Run it

```bash
pip install pyspark lightgbm pandas pyarrow scikit-learn
python src/generate_data.py                 # ~1 min
python src/build_features.py                # ~1.5 min on 2 cores
python src/benchmark_skew.py                # ~1 min
python src/train.py                         # ~2 min
```

## What I'd change for production
- Replace the batch window features with **streaming counters** (Kafka + Spark Structured Streaming or Flink) materialized to an online store (Redis/DynamoDB) via a **feature store** (Feast / Databricks Feature Store) so the same definition serves training and 50 ms scoring — no training-serving skew.
- Compute target encodings per rolling period, not once, and version them.
- Calibrate probabilities after class-weighting before exposing scores to rules.
- Monitor feature drift (PSI on each feature), score distribution, and catch-rate / false-decline-rate; retrain on a schedule *and* on drift triggers; champion/challenger before full rollout.
- Add SHAP reason codes for declined transactions (regulatory + ops).

See `INTERVIEW_TALK_TRACK.md` for the walkthrough script.
