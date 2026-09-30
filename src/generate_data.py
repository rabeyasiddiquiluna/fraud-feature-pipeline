"""
Step 1 - Generate a synthetic card-transaction dataset and write it as
date-partitioned Parquet.

Why synthetic?  Real card data can't be shared.  The generator plants the
fraud behaviors the feature pipeline is designed to catch:
  * velocity bursts        - a compromised card fires many txns in minutes
  * card testing           - a run of tiny amounts, some declined
  * geo-jump               - two countries within an hour
  * hot merchants          - a few merchants attract most fraud
  * late labels            - the chargeback is confirmed 3-30 days AFTER the txn

One merchant ("MEGA_MART") gets ~25% of all traffic on purpose so the
join-skew problem in step 3 is real, not staged.

Usage:
    python src/generate_data.py --n-cards 20000 --days 90 --out data/raw
"""
import argparse
import os

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

COUNTRIES = ["US", "US", "US", "US", "US", "CA", "GB", "MX", "DE", "BR", "IN", "AE"]
MCC = {  # merchant category codes, roughly realistic fraud propensity
    5411: ("grocery", 0.0005), 5812: ("restaurant", 0.0008), 5541: ("gas", 0.0010),
    5732: ("electronics", 0.0060), 5944: ("jewelry", 0.0080), 7995: ("gambling", 0.0120),
    5967: ("direct_marketing", 0.0150), 4814: ("telecom", 0.0030), 5311: ("dept_store", 0.0015),
    5999: ("misc_retail", 0.0025),
}


def build_merchants(n_merchants: int) -> pd.DataFrame:
    mcc_codes = np.array(list(MCC.keys()))
    codes = RNG.choice(mcc_codes, size=n_merchants)
    m = pd.DataFrame({
        "merchant_id": [f"M{i:06d}" for i in range(n_merchants)],
        "mcc": codes,
        "mcc_desc": [MCC[c][0] for c in codes],
        "merchant_country": RNG.choice(COUNTRIES, size=n_merchants),
        "merchant_risk_tier": RNG.choice(["low", "med", "high"], size=n_merchants, p=[0.8, 0.15, 0.05]),
    })
    m.loc[0, "merchant_id"] = "MEGA_MART"          # the skewed key
    m.loc[0, ["mcc", "mcc_desc"]] = [5311, "dept_store"]
    return m


def generate(n_cards: int, days: int, n_merchants: int, out: str):
    merchants = build_merchants(n_merchants)
    merchant_ids = merchants["merchant_id"].to_numpy()
    mcc_by_merchant = dict(zip(merchants.merchant_id, merchants.mcc))

    # merchant traffic: MEGA_MART gets 25%, the rest Zipf-ish
    w = 1.0 / np.arange(1, n_merchants + 1) ** 0.9
    w[0] = 0.0
    w = w / w.sum() * 0.75
    w[0] = 0.25

    start = pd.Timestamp("2026-01-01")
    end = start + pd.Timedelta(days=days)

    cards = [f"C{i:07d}" for i in range(n_cards)]
    home_country = RNG.choice(COUNTRIES, size=n_cards)
    spend_level = RNG.lognormal(mean=3.2, sigma=0.6, size=n_cards)  # per-card typical amount
    rate = RNG.gamma(shape=2.0, scale=0.6, size=n_cards)           # txns per day

    rows = []
    # ---- normal traffic -------------------------------------------------
    for i, c in enumerate(cards):
        n = RNG.poisson(rate[i] * days)
        if n == 0:
            continue
        ts = start + pd.to_timedelta(RNG.uniform(0, days * 86400, size=n), unit="s")
        amt = RNG.lognormal(np.log(spend_level[i]), 0.5, size=n).round(2)
        mid = RNG.choice(merchant_ids, size=n, p=w)
        rows.append(pd.DataFrame({
            "card_id": c, "txn_time": ts, "amount": amt, "merchant_id": mid,
            "txn_country": np.where(RNG.random(n) < 0.97, home_country[i], RNG.choice(COUNTRIES, size=n)),
            "channel": RNG.choice(["card_present", "ecom"], size=n, p=[0.6, 0.4]),
            "declined": (RNG.random(n) < 0.02).astype(int),
            "fraud_pattern": "none",
        }))
    df = pd.concat(rows, ignore_index=True)

    # ---- inject fraud episodes -----------------------------------------
    n_fraud_cards = int(n_cards * 0.03)
    fraud_cards = RNG.choice(np.arange(n_cards), size=n_fraud_cards, replace=False)
    hot_merchants = RNG.choice(np.arange(1, n_merchants), size=30, replace=False)

    def pick_merchants(n):
        """60% of fraud hits a small set of 'hot' merchants, 40% looks like normal traffic."""
        hot = RNG.random(n) < 0.6
        return np.where(hot, merchant_ids[RNG.choice(hot_merchants, n)], RNG.choice(merchant_ids, n, p=w))
    fraud_rows = []
    for i in fraud_cards:
        c = cards[i]
        t0 = start + pd.Timedelta(seconds=float(RNG.uniform(7 * 86400, (days - 1) * 86400)))
        pattern = RNG.choice(["burst", "card_test", "geo_jump"], p=[0.5, 0.3, 0.2])
        if pattern == "burst":
            n = RNG.integers(5, 15)
            ts = t0 + pd.to_timedelta(np.sort(RNG.uniform(0, 3600 * 2, n)), unit="s")
            amt = RNG.lognormal(np.log(spend_level[i] * 4), 0.6, n).round(2)
            mid = pick_merchants(n)
            ctry, ch, dec = home_country[i], "ecom", (RNG.random(n) < 0.15).astype(int)
        elif pattern == "card_test":
            n = RNG.integers(8, 25)
            ts = t0 + pd.to_timedelta(np.sort(RNG.uniform(0, 900, n)), unit="s")
            amt = RNG.uniform(0.5, 3.0, n).round(2)
            mid = pick_merchants(n)
            ctry, ch, dec = home_country[i], "ecom", (RNG.random(n) < 0.5).astype(int)
        else:  # geo_jump
            n = RNG.integers(3, 8)
            ts = t0 + pd.to_timedelta(np.sort(RNG.uniform(0, 3600, n)), unit="s")
            amt = RNG.lognormal(np.log(spend_level[i] * 3), 0.6, n).round(2)
            mid = pick_merchants(n)
            other = RNG.choice([x for x in COUNTRIES if x != home_country[i]])
            ctry, ch, dec = other, "card_present", (RNG.random(n) < 0.1).astype(int)
        fraud_rows.append(pd.DataFrame({
            "card_id": c, "txn_time": ts, "amount": amt, "merchant_id": mid,
            "txn_country": ctry, "channel": ch, "declined": dec, "fraud_pattern": pattern,
        }))
    fr = pd.concat(fraud_rows, ignore_index=True)
    fr["is_fraud"] = 1
    df["is_fraud"] = 0
    # sprinkle a little random fraud into normal traffic too
    rand = RNG.random(len(df)) < 0.0008
    df.loc[rand, "is_fraud"] = 1
    df.loc[rand, "fraud_pattern"] = "random"

    df = pd.concat([df, fr], ignore_index=True)
    df = df[df.txn_time < end].sort_values("txn_time").reset_index(drop=True)
    df["txn_id"] = [f"T{i:09d}" for i in range(len(df))]
    df["mcc"] = df.merchant_id.map(mcc_by_merchant).astype(int)

    # label delay: chargebacks confirmed 3-30 days later (simplified; real tail is longer); legit rows never get a label time
    delay = pd.to_timedelta(RNG.uniform(3, 30, len(df)) * 86400, unit="s")
    df["label_confirmed_at"] = pd.NaT
    df.loc[df.is_fraud == 1, "label_confirmed_at"] = df.txn_time + delay
    df["txn_date"] = df.txn_time.dt.date.astype(str)

    os.makedirs(out, exist_ok=True)
    cols = ["txn_id", "card_id", "txn_time", "amount", "merchant_id", "mcc", "txn_country",
            "channel", "declined", "is_fraud", "fraud_pattern", "label_confirmed_at", "txn_date"]
    # Spark can't read pandas' nanosecond timestamps -> coerce to microseconds
    df["txn_time"] = df.txn_time.astype("datetime64[us]")
    df["label_confirmed_at"] = df.label_confirmed_at.astype("datetime64[us]")
    df[cols].to_parquet(f"{out}/transactions", partition_cols=["txn_date"], index=False,
                        coerce_timestamps="us", allow_truncated_timestamps=True)
    merchants.to_parquet(f"{out}/merchants.parquet", index=False)

    print(f"transactions : {len(df):,}")
    print(f"fraud rows   : {int(df.is_fraud.sum()):,}  ({df.is_fraud.mean():.3%})")
    print(f"MEGA_MART    : {(df.merchant_id == 'MEGA_MART').mean():.1%} of rows  (skewed key)")
    print(f"date range   : {df.txn_time.min()} -> {df.txn_time.max()}")
    print(f"written to   : {out}/transactions  (partitioned by txn_date)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-cards", type=int, default=20000)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--n-merchants", type=int, default=3000)
    p.add_argument("--out", default="data/raw")
    a = p.parse_args()
    generate(a.n_cards, a.days, a.n_merchants, a.out)
