# Fraud Feature Pipeline — Step-by-Step Walkthrough

**For doing the project yourself and explaining it in the Mastercard interview**

Each step has four parts: **Do** (commands), **See** (what the output means), **Understand** (the key code, explained simply), and **Say** (how to explain it to the hiring manager). Do the steps in order. Total time: about 30 minutes of running, plus reading.

---

## Step 0 — Setup

**Where do I type these commands?** In a terminal on your own computer, not in a browser.

- **Mac:** open the **Terminal** app (Cmd+Space, type "Terminal").
- **Windows:** open **PowerShell** (Start menu, type "PowerShell"). Or, if you have it, the Ubuntu / WSL terminal.
- **VS Code (either OS):** Terminal menu → New Terminal. This is the easiest option because you can see the code and the terminal together.

First, go to the folder where the zip was downloaded — usually Downloads:

```bash
cd ~/Downloads                 # Mac / Linux / WSL
cd $HOME\Downloads             # Windows PowerShell
```

**Do (Mac / Linux / WSL)**

```bash
unzip fraud-feature-pipeline.zip
cd fraud-feature-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
java -version
```

**Do (Windows PowerShell)**

```powershell
Expand-Archive fraud-feature-pipeline.zip -DestinationPath .
cd fraud-feature-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
java -version
```

What each line does:

| Line | Meaning |
|---|---|
| `unzip` / `Expand-Archive` | Unpack the project into a folder |
| `cd fraud-feature-pipeline` | Go into that folder — every later command runs from here |
| `python -m venv .venv` | Create a private Python environment so the packages don't mess with your system |
| `activate` | Switch the terminal into that environment (you'll see `(.venv)` at the start of the prompt) |
| `pip install -r requirements.txt` | Install PySpark, LightGBM, pandas, pyarrow, scikit-learn |
| `java -version` | PySpark runs on the JVM, so Java must exist. You want to see `11.x` or `17.x` |

If `java -version` says "not found": install **Temurin JDK 17** from adoptium.net (pick your OS, run the installer, reopen the terminal, try again). On Mac with Homebrew: `brew install openjdk@17`.

If Windows PowerShell refuses to run `activate` ("running scripts is disabled"): run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Every time you open a new terminal later, you need to `cd` back into the folder and run the `activate` line again before running the scripts.

**See**

The folder layout:

```
fraud-feature-pipeline/
├── src/
│   ├── generate_data.py     # step 1 - make the data
│   ├── build_features.py    # step 2 - Spark features
│   ├── benchmark_skew.py    # step 3 - join optimization
│   └── train.py             # step 4 - LightGBM + evaluation
├── results/                 # outputs from my run, for reference
├── README.md                # design doc
└── INTERVIEW_TALK_TRACK.md  # the spoken script
```

**Understand**

A pipeline is a chain: raw data → features → model → evaluation. Each script reads the previous script's output from disk. That's deliberate: in production each stage is a separate job that can be rerun, scheduled, and monitored on its own.

**Say**

> "It's a four-stage batch pipeline. Each stage writes Parquet to disk so any stage can be rerun independently — same shape as a production Airflow or Databricks workflow."

---

## Step 1 — Generate the data

**Do**

```bash
python src/generate_data.py
```

(~1 minute)

**See**

```
transactions : 2,159,920
fraud rows   : 8,100  (0.375%)
MEGA_MART    : 24.9% of rows  (skewed key)
date range   : 2026-01-01 -> 2026-03-31
written to   : data/raw/transactions  (partitioned by txn_date)
```

Open `data/raw/transactions/` in your file browser. You'll see 90 folders named `txn_date=2026-01-01`, `txn_date=2026-01-02`, … Each holds one Parquet file. That's **partitioned Parquet** — the standard layout for big data lakes.

**Understand**

*Why synthetic?* Real card data is private. But random noise teaches nothing, so the generator plants **realistic behavior**:

| What I planted | Why |
|---|---|
| Each card has its own normal spend level and frequency | So "unusual for this card" is a meaningful concept |
| Fraud pattern **burst**: 5–15 purchases in 2 hours, 4× bigger than usual | Stolen card being drained |
| Fraud pattern **card testing**: 8–25 purchases of $0.50–$3 in 15 minutes, half declined | Fraudster checking whether a stolen number works |
| Fraud pattern **geo-jump**: card used in a different country within an hour | Cloned card |
| **MEGA_MART** gets 25% of all traffic | Makes the Spark skew problem real (think Amazon/Walmart) |
| **`label_confirmed_at`** = transaction time + 3–30 days | Chargebacks arrive late; this column lets us respect that |

Key code, simply:

```python
# every card gets its own personality
spend_level = RNG.lognormal(mean=3.2, sigma=0.6, size=n_cards)   # typical $ per purchase
rate        = RNG.gamma(shape=2.0, scale=0.6, size=n_cards)      # purchases per day

# 60% of fraud goes to a small set of "hot" merchants, 40% looks like normal traffic
def pick_merchants(n):
    hot = RNG.random(n) < 0.6
    return np.where(hot, merchant_ids[RNG.choice(hot_merchants, n)],
                         RNG.choice(merchant_ids, n, p=w))

# label delay
df.loc[df.is_fraud == 1, "label_confirmed_at"] = df.txn_time + delay   # 3-30 days
```

*One gotcha you'll hit if you change the code:* pandas writes timestamps in nanoseconds; Spark can only read microseconds. That's why the last lines cast to `datetime64[us]`. Real-world interop problem — worth a sentence in the interview.

**Say**

> "I generated 2 million transactions with three planted fraud patterns — bursts, card testing, geo-jumps — plus two things most toy projects skip: one merchant with 25% of traffic so join skew is real, and chargeback labels that arrive 3 to 30 days late so I have to think about label maturity."

---

## Step 2 — Build the features in Spark

**Do**

```bash
python src/build_features.py
```

(~1.5 minutes on a laptop)

**See**

```
raw partitions on read: 4
feature rows: 2,159,920   written to data/features   (78s)

+--------+------+---------------+------------+------------+-----------+
|is_fraud|cnt_1h|secs_since_last|amt_ratio_7d|declines_10m|merchant_te|
+--------+------+---------------+------------+------------+-----------+
|       0|  0.07|        69540.0|        1.01|         0.0|     0.0026|
|       1|  4.58|        24211.0|        1.23|        1.44|     0.1372|
+--------+------+---------------+------------+------------+-----------+
```

Read that table row by row — it's your proof the features work:

- **cnt_1h** — fraud rows had 4.6 transactions in the previous hour; legit had 0.07. A 65× difference.
- **secs_since_last** — fraud: ~7 hours since last; legit: ~19 hours. (Inside a burst it's seconds.)
- **declines_10m** — fraud: 1.4 declines in the last 10 minutes; legit: 0. That's card testing.
- **merchant_te** — fraud transactions happen at merchants with a 13.7% historical fraud rate; legit at 0.26%.

**Understand — the one idea**

A single transaction tells you almost nothing. **The history around it tells you everything.** So for every transaction, the code asks: *"What did this card look like in the minutes, hours, and days right before this exact moment?"*

And the one rule: **only use information that existed at that moment.** Not one second later.

**Understand — feature family A: rolling windows**

```python
base = Window.partitionBy("card_id").orderBy("ts")

def back(seconds):
    return base.rangeBetween(-seconds, -1)     # last N seconds, EXCLUDING current row

w24h = back(86400)
df = (df.withColumn("card_cnt_24h", F.count("*").over(w24h))
        .withColumn("card_sum_24h", F.sum("amount").over(w24h))
        .withColumn("card_nmerch_24h", F.approx_count_distinct("merchant_id").over(w24h)))
```

Three things to understand here:

1. **`partitionBy("card_id")`** — group by card. Spark shuffles the data so each card's rows land together.
2. **`orderBy("ts")`** — sort each card's rows by time. `ts` is epoch seconds (integer), which is what lets `rangeBetween` work in *seconds* instead of *rows*.
3. **`rangeBetween(-86400, -1)`** — for each row, look at rows whose `ts` is between 86,400 seconds ago and 1 second ago. **The `-1` excludes the current row.** If it were `0`, a $5,000 fraud would count itself in "sum of last 24h" — the feature would already contain the answer. That's leakage.

*Experiment to do yourself:* change `-1` to `0`, rerun, look at the output. Nothing crashes. Nothing looks wrong. That's the lesson — **leakage is silent**. Change it back.

**Understand — feature family B: velocity**

```python
.withColumn("amt_ratio_7d", F.col("amount") / (F.col("card_mean_7d") + 1e-6))
.withColumn("card_declines_10m", F.sum("declined").over(back(600)))
.withColumn("card_ncountry_1h", F.approx_count_distinct("txn_country").over(back(3600)))
```

Raw amount is weak: $500 is normal for one card, crazy for another. So divide by **the card's own 7-day average**. A ratio of 4 means "4× this card's normal." Declines in 10 minutes catches card testing; countries in 1 hour catches geo-jumps.

**Understand — feature family C: time since last**

```python
.withColumn("prev_ts", F.lag("ts").over(base))
.withColumn("secs_since_last", F.coalesce(F.col("ts") - F.col("prev_ts"), F.lit(-1)))
```

`lag("ts")` = "the previous row's timestamp" in this card's ordered history. Subtract to get seconds. The very first transaction has no previous row → null → `coalesce` fills `-1`. That sentinel lets the model learn "brand-new card" is itself a signal. **This became the #1 feature (48% of model gain).**

**Understand — feature family D: merchant history, and the 10-minute hang**

This is the best story in the project. My first version was:

```python
w = Window.partitionBy("merchant_id").orderBy("ts").rangeBetween(-7*86400, -1)
df.withColumn("merch_ncards_24h", F.approx_count_distinct("card_id").over(w))
```

Every partition finished in seconds — except MEGA_MART's. It ran 10+ minutes. **Why:** distinct-count over a range window rebuilds a HyperLogLog sketch for every single row over its whole lookback. 538,000 rows × up to 7 days of rows each, on one core. Quadratic work.

**Why the usual fix doesn't work:** you can't salt a window function. A window needs *all* rows for its key in one partition, in order. Split the key, the window is wrong.

**What I did — change the grain:**

```python
hourly = (df.groupBy("merchant_id", F.date_trunc("hour", "txn_time").alias("hr"))
            .agg(F.count("*").alias("n"), F.approx_count_distinct("card_id").alias("nc")))
w7d = Window.partitionBy("merchant_id").orderBy("hr_ts").rangeBetween(-7*86400, -1)
hourly = hourly.withColumn("merch_cnt_7d", F.sum("n").over(w7d))
df = df.join(hourly, ["merchant_id", "hr"], "left")
```

First aggregate to (merchant, hour) — a tiny table, at most 3,000 × 2,160 rows. Then window over *hours* instead of *rows*. Slight precision loss (bucketed to the hour), 100× less work. Whole pipeline: 10+ min → 78 s.

**Understand — feature family E: target encoding**

3,000 merchants here; millions in real life. One-hot = millions of columns. Instead: replace `merchant_id` with **its historical fraud rate**. One number.

```python
hist = raw.filter(F.col("txn_time") < hist_end)                      # only EARLIER data
hist = hist.withColumn("known_fraud",
        F.when((F.col("is_fraud") == 1) &
               (F.col("label_confirmed_at") < label_cutoff), 1).otherwise(0))   # only labels we HAD
stats = hist.groupBy("merchant_id").agg(F.count("*").alias("n"), F.avg("known_fraud").alias("rate"))
stats = stats.withColumn("merchant_id_te",
        (F.col("n") * F.col("rate") + k * global_rate) / (F.col("n") + k))    # smoothing, k=20
df = df.join(F.broadcast(stats), "merchant_id", "left").fillna({"merchant_id_te": global_rate})
```

Three guards, each one an interview question:

- **Earlier period only** (`txn_time < hist_end`, Jan 1–Feb 1). Training rows come after Feb 1, so no row's own label is in its feature.
- **Only labels we'd have known** (`label_confirmed_at < label_cutoff`). On Feb 1, a fraud whose chargeback arrived Feb 20 was still "legit" to everyone. The code respects that.
- **Smoothing.** A merchant with 1 fraud out of 1 transaction would get 100%. `(n·rate + 20·global) / (n + 20)` pulls small merchants toward the global rate.

Plus `F.broadcast(stats)`: the stats table is tiny, so Spark ships a copy to every executor and does a hash join with no shuffle of the 2M-row side.

**Understand — feature family F: static context**

Cross-border (card country ≠ merchant country), e-commerce vs card-present, merchant risk tier, hour of day, weekend. All from a `broadcast` join on the 3k-row merchant table.

**Understand — the timeline**

```
Jan 1 ──── Feb 1 ──── Feb 25 ─── Mar 5 ·····gap····· Mar 21 ──── Mar 31
 TE history │  TRAIN   │  VALID  │                    │   TEST    │
                                            "today" = Mar 21 = label_cutoff
```

**Say**

> "Every feature answers 'what did this card look like at that exact moment, using only what was knowable then.' Rolling windows with `rangeBetween` bounded at minus one, velocity as a ratio to the card's own baseline, `lag` for time since last, smoothed target encoding from an earlier period that only counts chargebacks that had actually arrived. I hit real skew inside a window function on the hot merchant — you can't salt a window, so I moved to hourly grain and the job went from ten minutes to seventy seconds."

---

## Step 3 — Benchmark the skewed join

**Do**

```bash
python src/benchmark_skew.py
```

(~1 minute)

**See**

```
shuffle partitions on merchant_id: n=32  median=45,751  max=628,235  max/median = 13.7x
sort_merge_naive   rows=2,159,857      3.4s
shuffle partitions on (merchant_id, salt): n=32  median=56,992  max=109,810  max/median = 1.9x
salted             rows=2,159,857      2.1s
broadcast          rows=2,159,857      0.7s

sort_merge_naive      3.4s    1.0x
salted                2.1s    1.6x faster than naive
broadcast             0.7s    5.0x faster than naive
```

And two `explain()` plans. In the first, find these lines:

```
SortMergeJoin [merchant_id], [merchant_id], Inner
   Exchange hashpartitioning(merchant_id, 32)      <- shuffle of the BIG side
   Exchange hashpartitioning(merchant_id, 32)      <- shuffle of the small side
```

In the last:

```
BroadcastHashJoin [merchant_id], [merchant_id], Inner, BuildRight
   BroadcastExchange                               <- only the SMALL side moves
```

**Understand — what skew is, simply**

When Spark joins on `merchant_id`, it sends every row with the same merchant to the same partition (a "shuffle"). MEGA_MART has 538k rows; the median partition has 46k. One task does 13.7× the work while every other core sits idle waiting. That one task is the **straggler**.

**Understand — the three strategies**

| Strategy | What happens | When to use |
|---|---|---|
| **Sort-merge (naive)** | Shuffle both sides, sort, merge. Straggler on the hot key. | Default; fine when keys are balanced |
| **Salting** | Add random `salt` 0–15 to the big side. Explode the small side 16× (one copy per salt). Join on `(merchant_id, salt)`. Hot key now spreads over 16 tasks. | Both sides big, one key hot |
| **Broadcast** | Small side (3k rows) copied to every executor. Hash join locally. **Big side never shuffles.** | Small side fits in memory (default 10 MB, tunable to a few hundred MB) |

Salting code:

```python
txns_s = txns.withColumn("salt", (F.rand(seed=7) * 16).cast("int"))
dim_s  = dim.withColumn("salt", F.explode(F.array([F.lit(i) for i in range(16)])))
j2 = txns_s.join(dim_s, ["merchant_id", "salt"])
```

Broadcast code:

```python
j3 = txns.join(F.broadcast(dim), "merchant_id")
```

**Understand — why I turned off AQE**

Spark 3's Adaptive Query Execution can detect and split skewed partitions automatically. I disabled it (`spark.sql.adaptive.enabled = false`) and auto-broadcast (`autoBroadcastJoinThreshold = -1`) so the benchmark shows what happens *under the hood*. In production I'd leave AQE on — but I need to know what it's doing for me.

**Say**

> "Broadcast first if the small side fits — it removes the shuffle of the big side entirely and was 5× faster here. Salting when both sides are big: it spread the hot key from 13.7× to 1.9× the median. And in Spark 3 I'd enable AQE skew-join, but I benchmarked with it off so I understand what it's doing."

---

## Step 4 — Train and evaluate

**Do**

```bash
python src/train.py
```

(~3 minutes)

**See**

```
train 576,007 | valid 192,135 | test 264,191
label delay: 2,361 true fraud in train window, 2,295 confirmed by 2026-03-21 (97% known at retrain time)

== model A: honest labels (only chargebacks confirmed before cutoff) ==
A test    PR-AUC=0.7535  ROC-AUC=0.8955  recall@1%FPR=0.768  recall@0.1%FPR=0.744

== model B: oracle labels (all fraud, even if not yet confirmed) ==
B test    PR-AUC=0.7593  ROC-AUC=0.8984  recall@1%FPR=0.769  recall@0.1%FPR=0.749

== leakage check: random split on the same rows ==
random-split holdout   PR-AUC=0.7687
PR-AUC gap random - time = +0.0094  (small: no obvious leakage)

cost-optimal threshold = 0.05  recall=0.730  precision=0.867  total cost=$15,183

top features by gain:
   secs_since_last   48.4
    merchant_id_te   12.9
    ...
```

**Understand — the time split**

```python
train = df[(df.txn_time >= "2026-02-01") & (df.txn_time < "2026-02-25")]
valid = df[(df.txn_time >= "2026-02-25") & (df.txn_time < "2026-03-05")]
test  = df[(df.txn_time >= "2026-03-21") & (df.txn_time < "2026-04-01")]
```

Not random. A random split lets the model see Tuesday to predict Monday. The time split mimics reality: train on the past, deploy, watch what happens next. The **gap** between valid and test (Mar 5 → Mar 21) is "the model has been live for two weeks" — performance on test is what the business would actually see.

**Understand — label maturity (the thing most people get wrong)**

"Today" is March 21. Which fraud labels do we truly know? Only those whose chargeback arrived before March 21:

```python
cutoff = pd.Timestamp("2026-03-21")
known = (train.is_fraud == 1) & (train.label_confirmed_at < cutoff)
y_train_honest = known.astype(int)
```

The training window is Feb 1–25, so those transactions are 24–48 days old — chargebacks (3–30 days) have almost all arrived. **97% known.** The other 3% are labeled legit because on March 21 that's what we'd believe.

Model A trains on honest labels; model B on all labels including ones we couldn't have known. They score the same here *because 97% is high*. In an earlier version I trained on the two weeks before "today" — only 13% of labels were mature and model A was much weaker. **That's why the training window sits a month behind the retrain date.**

**Understand — class imbalance, and the surprise**

0.4% positives. My first setting was the textbook one:

```python
scale_pos_weight = n_neg / n_pos      # ~250x
```

Early stopping fired **at tree 1** and PR-AUC was 0.18. The gradients were so dominated by the 2,300 positives that every additional tree made validation worse. Switching to:

```python
scale_pos_weight = (n_neg / n_pos) ** 0.5     # ~16x
```

let it train to ~2,000 trees and PR-AUC went to 0.75. Lesson: **look at what early stopping tells you; the textbook default isn't always right.**

Also: early-stop on validation **PR-AUC**, not ROC-AUC. On 0.4% positives, ROC-AUC dips after the first tree then recovers — a noisy stopping signal.

**Understand — the metrics**

- **PR-AUC 0.754.** A random model scores 0.004 (= the base rate). This is the metric for rare positives.
- **Recall at 1% FPR = 77%.** "If we're allowed to wrongly decline 1 in 100 good customers, we catch 77% of fraud." Business people understand this one.
- **Recall at 0.1% FPR = 74%.** Same at 1 in 1,000.
- **Never accuracy.** A model that says "never fraud" is 99.6% accurate.

**Understand — the leakage smoke test**

```python
mask = rng.random(len(pool)) < 0.7          # random 70/30 on the same rows
```

Train on a random split. If features leak the future, random split scores much higher than time split (the future is in training). Random 0.769 vs time 0.754 → gap **+0.01**. Clean. If it had been +0.20, I'd hunt for a leaky feature.

**Understand — the business threshold**

```python
missed         = test.amount[(~pred) & (y == 1)].sum()     # missed fraud costs the txn amount
false_declines = ((pred) & (y == 0)).sum() * 5.0           # a false decline costs ~$5
cost = missed + false_declines
```

Sweep thresholds 0.05 → 0.95, pick the one minimizing cost. Here it's **0.05** — very low — because missing a $200 fraud costs the same as 40 false declines. The right threshold depends entirely on those two numbers, which come from the business, not the model. Never 0.5.

**Say**

> "Time-based split with a deployment gap, training on data old enough that its chargebacks have arrived — 97% mature. PR-AUC and recall at fixed false-positive rate, never accuracy. A random-versus-time-split comparison as a leakage smoke test — gap of one point, so clean. And the threshold comes from a cost model: missed fraud in dollars versus false declines at about five dollars each."

---

## Step 5 — Push to GitHub

**Do**

```bash
git init
git add .
git commit -m "Fraud feature pipeline: PySpark point-in-time features, skew benchmark, LightGBM"
# create an empty repo on github.com named fraud-feature-pipeline, then:
git remote add origin git@github.com:rabeyasiddiquiluna/fraud-feature-pipeline.git
git branch -M main
git push -u origin main
```

`data/` and the 14 MB model file are already in `.gitignore`. The README's results tables show your numbers without anyone needing to run it.

**Say (if asked "can I see it?")**

> "It's on my GitHub — the README has the results tables and the design decisions, and each script's docstring explains what it does and the traps it avoids."

---

## Step 6 — What to say when they ask "what would you change for production?"

They *will* ask. Have this ready in order:

1. **Streaming counters, not batch windows.** The 1h/24h features must be fresh at scoring time (50 ms). Kafka → Spark Structured Streaming or Flink → per-card counters in Redis/DynamoDB.
2. **Feature store** (Feast, Databricks Feature Store, Tecton). Define `card_cnt_1h` once; it materializes to the offline table for training and the online store for serving, with point-in-time joins. This kills **training-serving skew** — the #1 silent failure in fraud models.
3. **Version target encodings** and recompute per period.
4. **Calibrate** probabilities after class weighting (isotonic) so 0.8 means 80%.
5. **Monitor**: PSI on every feature, score distribution, catch rate, false-decline rate, latency. Retrain on schedule *and* on drift trigger. Champion/challenger in shadow mode first.
6. **SHAP reason codes** on every decline — regulators and ops both need them.

---

## Step 7 — Bridge to your real experience

One sentence each, so the project connects to what you've actually shipped:

> "The same point-in-time instinct showed up in my flood-segmentation work — I found near-duplicate images leaking between train and test with a perceptual-hash and FAISS audit. Different domain, same question: where is the future leaking into training?"

> "At Bayer the rice imager ran in 15 countries. The preprocessing on the device had to match the lab pipeline exactly — that's the same offline/online consistency problem a feature store solves."

---

## Cheat sheet — the five things to remember

| # | The thing | Why it matters |
|---|---|---|
| 1 | `rangeBetween(-N, -1)` | The `-1` excludes the current row. Leakage is silent. |
| 2 | Train on mature labels | Chargebacks arrive late. Only count labels you'd have had. |
| 3 | Broadcast → salt → change grain | Broadcast if small side fits; salt if both big; can't salt a window. |
| 4 | PR-AUC, recall@FPR, cost threshold | Never accuracy, never 0.5. |
| 5 | Random vs time split gap | Your leakage smoke test. |

**Memory hook for the two stories:** *"The window that wouldn't finish, and the weight that wouldn't train."*
