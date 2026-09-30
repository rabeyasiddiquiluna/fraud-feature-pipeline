# Interview Talk Track — Fraud Feature Pipeline

This is how to explain the project out loud. It is written in simple language. Read it a few times, then practice saying the **bold "say this"** lines without looking. The interviewer is the Mastercard hiring manager; they care about feature engineering, Spark, and whether you think about production.

---

## 0. The 30-second opener (say this first)

> **"I built a small end-to-end fraud pipeline to practice the exact problems this role deals with. It takes 2 million card transactions, turns each one into features about the card's recent behavior using PySpark window functions, trains a LightGBM model, and evaluates it the way a fraud team would — PR-AUC and recall at a fixed false-positive rate on a time-based split. Along the way I hit and fixed two real Spark problems: a skewed join key, and a window function that ran for 10 minutes on one partition. Happy to go into any part."**

That's it. Stop and let them pick a direction. If they say "go ahead", walk sections 1 → 5 in order.

---

## 1. The data (1 minute)

**What it is, simply:** I can't use real card data, so I generated it. 20,000 cards, 90 days, 3,000 merchants, about 2.1 million transactions. Every card has its own "normal" — how much it usually spends and how often.

**What I planted on purpose:** fraud is only 0.4% of rows, and it comes in three shapes that real fraud has:

- **Burst** — a stolen card suddenly does 5–15 purchases in two hours, all bigger than usual.
- **Card testing** — 8–25 tiny purchases ($0.50–$3) in 15 minutes, half of them declined. Fraudsters do this to check if a stolen number works.
- **Geo-jump** — the card is used in a different country within an hour.

**Two things I added that most toy projects skip:**

1. **One merchant gets 25% of all traffic** (I called it MEGA_MART). That is realistic — think Amazon or Walmart — and it makes the Spark skew problem real, not staged.
2. **Labels arrive late.** In real life, you find out a transaction was fraud when the customer files a chargeback, 3–30 days later. So every fraud row has a `label_confirmed_at` timestamp. This matters a lot later.

> **Say this:** "I made the synthetic data hard in the two ways that matter in production: a skewed merchant key and delayed labels. Otherwise the project would look good but teach nothing."

---

## 2. The features — how a transaction becomes a row the model can learn from (4 minutes)

**The one idea behind everything:** a single transaction ($40 at a coffee shop) tells you almost nothing. The *history* around it tells you everything. So for each transaction I ask: **"What did this card look like in the minutes, hours, and days right before this moment?"**

And the one rule: **only use information that existed at that exact moment.** Nothing from the future, not even one second later.

### 2a. Rolling window counts (the workhorse)

For every transaction, count what the same card did in the last 1 hour, 24 hours, 7 days: how many transactions, total amount, average amount, how many different merchants.

**How in Spark:**
```python
w_24h = Window.partitionBy("card_id").orderBy("ts").rangeBetween(-86400, -1)
df = df.withColumn("card_cnt_24h", F.count("*").over(w_24h))
```

**The detail the interviewer will test:** the window is `(-86400, -1)`, not `(-86400, 0)`. The `-1` means "stop one second before the current row." If you use `0`, the transaction counts itself — and when it's a $5,000 fraud, the feature "sum in last 24h" already contains that $5,000. That's leakage. The model would look great in training and fail in production.

> **Say this:** "Time-based `rangeBetween` bounded at minus one. That single character is the difference between a real feature and a leaky one."

**The result:** fraud rows average **4.6 transactions in the previous hour**; legit rows average **0.07**. That's a 65× gap from one feature.

### 2b. Velocity — compare the card to itself

Raw amount is a weak feature: $500 is normal for one card and crazy for another. So I divide by the card's own 7-day average: `amt_ratio_7d = amount / mean_amount_7d`. A ratio of 4 means "four times bigger than this card's normal."

Also: number of declines in the last 10 minutes (catches card testing), number of different countries in the last hour (catches geo-jump).

> **Say this:** "Fraud is a change of pace. So I compare the transaction to the card's own baseline instead of to the whole population."

### 2c. Time since last transaction

`F.lag("ts")` gives the previous transaction's timestamp; subtract to get seconds since last. First-ever transaction gets `-1` as a sentinel so the model can learn "brand-new card" is itself a signal.

**This turned out to be the #1 feature (48% of model gain).** Fraud rows average 24,000 seconds since last (~7 hours); legit average 69,000 (~19 hours). And inside a burst, it's often under 60 seconds.

### 2d. Target encoding for merchant and category

There are 3,000 merchants; in real life, millions. One-hot encoding would be millions of columns. Instead I replace `merchant_id` with **"the fraud rate this merchant had historically."** One column.

Two traps and their fixes:

- **Leakage trap:** if I compute the fraud rate from the same rows I train on, each row's own label sneaks into its feature. **Fix:** compute rates only from an *earlier* period (Jan 1–Feb 1), and train on rows after that.
- **Rare-merchant trap:** a merchant with 1 transaction that was fraud gets rate = 100%. **Fix:** smoothing — `(n × rate + k × global_rate) / (n + k)` with k = 20. Small merchants get pulled toward the global average.

**And the label-delay twist:** when I compute "historical fraud rate," I only count chargebacks that had *actually been confirmed* by the day I'm building features. A fraud from Jan 28 whose chargeback came Feb 20 was, on Feb 1, still "legit" as far as anyone knew. My code respects that (`label_confirmed_at < label_cutoff`).

> **Say this:** "Smoothed target encoding from a strictly earlier period, and it only counts labels that had actually arrived by that date. That's point-in-time correctness applied to the label, not just the features."

### 2e. Merchant-side and static features

Merchant transaction count over 7 days, distinct cards in 24h (a merchant suddenly seeing many new cards is suspicious). Plus cheap context: cross-border (card country ≠ merchant country), e-commerce vs card-present, merchant risk tier, hour of day, weekend.

These come from a tiny merchant table (3k rows), so I `broadcast` it — ships the small table to every executor, no shuffle of the 2M-row side.

---

## 3. The Spark problems I actually hit (3 minutes) — THIS is the section that shows seniority

### 3a. Skewed join key

**The problem, simply:** when Spark joins two tables on `merchant_id`, it shuffles rows so all rows with the same merchant land in the same partition. MEGA_MART has 538,000 rows; a typical partition has 45,000. So one task does 13.7× the work of the median. Every other core sits idle waiting for it. That's the "straggler."

**I benchmarked three fixes** (with AQE and auto-broadcast turned off so the comparison is honest):

| Strategy | What it does | Skew | Time |
|---|---|---|---|
| Naive sort-merge join | default | 13.7× | 3.4 s |
| **Salting** | add random number 0–15 to big side, explode small side ×16, join on (key, salt). Hot key now spreads over 16 tasks | 1.9× | 2.1 s |
| **Broadcast** | small table is 3k rows → send a copy to every executor, hash join, **no shuffle of the big side at all** | none | 0.7 s (5×) |

**When to use which:**
- Small side fits in memory (< ~10 MB default, tunable to a few hundred MB)? → **Broadcast.** Always the first choice.
- Both sides big and one key hot? → **Salting** (or Spark 3's AQE skew join, which does salting automatically).

I printed `explain()` for both so you can see `SortMergeJoin` with two `Exchange` (shuffle) nodes vs `BroadcastHashJoin` with only `BroadcastExchange`.

> **Say this:** "Broadcast first if the small side fits. Salting when both sides are big. And in Spark 3 I'd turn on AQE with skew-join enabled, but I benchmarked with it off to understand what it does under the hood."

### 3b. The window function that hung for 10 minutes (my favorite story)

**What happened:** I wrote `approx_count_distinct("card_id")` over a 7-day range window partitioned by merchant. Every partition finished in seconds — except MEGA_MART's. It ran 10+ minutes and I killed it.

**Why:** a range window with a distinct-count has to rebuild a HyperLogLog sketch for every single row over its whole lookback. On 538k rows that's 538k sketches over up to 7 days of rows each. That's quadratic-ish work on one core.

**Why the usual fix doesn't work:** you can't salt a window function. A window needs *all* rows for a key in one partition, in order. Split the key and the window is wrong.

**What I did instead — change the grain:** first `groupBy(merchant, hour)` → count and distinct-cards per hour. That's a tiny table (3k merchants × 2,160 hours max). Then run the window over *hours* instead of *rows*. Lose a little precision (bucketed to the hour), gain 100× speed. Whole pipeline: 10+ min → 78 s.

> **Say this:** "Skew inside a window function can't be salted, so I moved the computation to a coarser grain — hourly buckets — and windowed over those. That's also how you'd do it in a streaming system: maintain hourly counters, not per-row windows."

---

## 4. Training and evaluation — the honest way (3 minutes)

### 4a. Time-based split with a gap

```
Jan 1 ──── Feb 1 ──── Feb 25 ─── Mar 5 ·····gap····· Mar 21 ──── Mar 31
 TE history │  TRAIN   │  VALID  │                    │   TEST    │
```

Not a random split. Random split lets the model see Tuesday to predict Monday. Time split mimics reality: train on the past, deploy, watch the next 10 days. The gap between valid and test is "the model has been live for two weeks."

### 4b. Label maturity

"Today" (retrain date) is Mar 21. The training window is Feb 1–25, so its transactions are 24–48 days old — their chargebacks have mostly arrived. **97% of the fraud labels in the training window are confirmed by Mar 21.** The other 3% I treat as legit, because on Mar 21 that's what we'd believe.

I trained two models to show the effect: one on "honest" labels (only confirmed fraud) and one on "oracle" labels (all fraud, including ones we couldn't have known). They perform the same here because 97% is high. If the training window were last week instead, only ~15% of labels would be mature and the honest model would be much weaker — I saw that in an earlier version, which is why I moved the window back.

> **Say this:** "I don't train on last week. I train on data old enough that its chargebacks have arrived. There's always a lag between 'now' and the newest trustworthy label, and pretending otherwise is a very common leak."

### 4c. Class imbalance

0.4% positives. Three things:
1. `scale_pos_weight` in LightGBM. I first used the full ratio `n_neg/n_pos` (~250×) and early stopping fired at tree 1 — the gradients were so dominated by positives that every extra tree hurt validation. Switching to `sqrt(ratio)` (~16×) fixed it and PR-AUC went from 0.18 to 0.75. Good story: **"the textbook setting isn't always the right one; look at what early stopping tells you."**
2. Metrics: **PR-AUC** (0.754 vs 0.004 for random) and **recall at 1% FPR** (77%) and at 0.1% FPR (74%). Never accuracy — a model that says "never fraud" is 99.6% accurate.
3. Early stopping on validation PR-AUC, not ROC-AUC. ROC-AUC on 0.4% positives dips after the first tree then recovers, so it's a noisy stopping signal.

### 4d. Leakage check

I trained a third model on a *random* 70/30 split of the same rows. If features leak, random split scores much higher than time split (because "the future" is in the training set). Random-split PR-AUC 0.769 vs time-split 0.754. Gap +0.01 — small, explained by drift. If it had been +0.20, I'd go hunting for a leaky feature.

> **Say this:** "The random-vs-time-split gap is my smoke test for leakage. Small gap, clean features."

### 4e. Business threshold, not 0.5

A missed fraud costs the transaction amount. A false decline costs about $5 (lost sale, annoyed customer, support call). I swept thresholds and picked the one minimizing total cost: **0.05** (low, because missing a $200 fraud costs 40 false declines' worth), giving 73% recall at 87% precision. The right threshold depends on the business, so I made it a parameter, not a constant.

---

## 5. "What would you change for production?" (1 minute — they WILL ask)

- **Streaming counters instead of batch windows.** The 1h/24h features need to be fresh at scoring time (50 ms budget). Kafka → Spark Structured Streaming or Flink → Redis/DynamoDB counters per card.
- **Feature store** (Feast, Databricks Feature Store, Tecton). Define `card_cnt_1h` once; it materializes to the offline table (training) and the online store (serving). Point-in-time joins for training. This kills **training-serving skew** — the #1 silent failure in fraud models.
- **Version the target encodings** and recompute per period.
- **Calibrate** probabilities after class weighting (Platt or isotonic) so a 0.8 means 80%.
- **Monitor:** PSI on every feature, score distribution, catch rate, false-decline rate, latency. Retrain on schedule *and* on drift trigger. Champion/challenger in shadow mode before rollout.
- **Explainability:** SHAP reason codes on every decline — regulators and ops both need them.

---

## 6. Likely follow-up questions and short answers

**"Why rangeBetween and not rowsBetween?"**
`rowsBetween(-5, -1)` = last 5 rows regardless of time. `rangeBetween(-3600, -1)` = last hour regardless of row count. For velocity you want time. (rowsBetween is fine for "last 5 transactions" features.)

**"What if the card_id partition itself is skewed?"**
Cards don't get 25% of traffic, so rarely a problem. If one card did (a corporate card?), same trick as merchants: bucket to a coarser grain, or cap history at N rows.

**"Why LightGBM and not a neural net?"**
Tabular data, 22 features, need for reason codes, need for sub-10-ms scoring. Gradient boosting wins on all three. Deep learning earns its place for sequence models over raw transaction histories, which is a next step.

**"How do you know approx_count_distinct is accurate enough?"**
HyperLogLog default error ~2%. For "how many merchants in 24h" that's fine — we care whether it's 1 or 12, not 11 vs 12.

**"Why 32 shuffle partitions?"**
Local laptop, 2 cores. In production: roughly 2–3× total cores, and target 100–200 MB per partition. Or enable AQE and let it coalesce.

**"How would you backfill these features for 2 years of history?"**
Same Spark job, partition by day, run per month, write to the offline feature store. The point-in-time logic is already in the code so backfill is safe.

**"Your PR-AUC is 0.75 — is that realistic?"**
It's high because synthetic fraud is cleaner than real fraud. Real fraud models land anywhere from 0.3 to 0.7 depending on the product. The number matters less than the method being honest — time split, mature labels, leakage check — and those are what transfer.

**"What was the hardest part?"**
Two things. The window-function skew — salting didn't apply, and the fix was a modeling decision (hourly grain), not a Spark config. And the class-weight surprise — the textbook `n_neg/n_pos` weight broke early stopping. Both are reminders that "the fix" is often upstream of the tool.

---

## 7. Connect it to your real experience (Bayer / SunEdison)

Have one sentence ready that bridges to what you've actually shipped:

> "The same point-in-time discipline showed up in my flood-segmentation work — I found near-duplicate images leaking between train and test with a perceptual-hash + FAISS audit, and IoU dropped from inflated to honest. Different domain, same instinct: check where the future is leaking into training."

> "At Bayer the rice imager ran in 15 countries; the offline/online consistency problem there was the same shape as a feature store — the preprocessing that ran in the lab had to be byte-for-byte what ran on the device."

---

## 8. If you only remember five things

1. `rangeBetween(-N, -1)` — the `-1` is the whole point.
2. Labels arrive late; train on mature data; only count labels you'd have known.
3. Broadcast if small side fits; salt if both big; can't salt a window — change the grain.
4. PR-AUC and recall@FPR, never accuracy; threshold from business cost.
5. Random-split vs time-split gap = leakage smoke test.
