"""
Step 4 - Train a LightGBM fraud model on the Spark features, evaluated the way a
fraud team actually evaluates: PR-AUC and recall at a fixed false-positive rate,
on a TIME-based split.

Also includes two leakage checks you can talk about:
  1. Time split vs random split.  If random-split AUC >> time-split AUC, features
     are leaking future info (or the world drifts).  We print both.
  2. Label-delay check.  Only rows whose label was CONFIRMED before the training
     cutoff are trusted as "known fraud" during training.  Fraud rows whose
     chargeback hadn't arrived yet are what production would see as "not fraud
     (yet)".  We train on the honest version and report how many labels that costs.

Split (data spans Jan 1 - Mar 31; "today" / retrain date = Mar 21):
    TE history : Jan 1  - Feb 1    (used inside build_features.py, never trained on)
    train      : Feb 1  - Feb 25   labels are 24-48 days old on Mar 21 -> mostly mature
    valid      : Feb 25 - Mar 5    early stopping + threshold
    test       : Mar 21 - Mar 31   what happens in the 10 days after deployment

Usage:
    python src/train.py --features data/features --out results
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

FEATURES = [
    "amount", "card_cnt_1h", "card_cnt_24h", "card_cnt_7d", "card_sum_24h", "card_sum_7d",
    "card_nmerch_24h", "card_ncountry_1h", "card_declines_10m", "amt_ratio_7d",
    "amt_over_card_sum_24h", "secs_since_last", "country_changed", "merch_cnt_7d",
    "merch_ncards_24h", "merchant_id_te", "mcc_te", "cross_border", "is_ecom",
    "risk_tier_num", "hour_of_day", "is_weekend",
]

TRAIN_START, TRAIN_END = "2026-02-01", "2026-02-25"
VALID_END = "2026-03-05"
TEST_START, TEST_END = "2026-03-21", "2026-04-01"
RETRAIN_DATE = "2026-03-21"   # "today": only labels confirmed before this are known


def recall_at_fpr(y, p, fpr_target):
    fpr, tpr, thr = roc_curve(y, p)
    i = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[max(i, 0)]), float(thr[max(i, 0)])


def evaluate(name, y, p):
    r1, t1 = recall_at_fpr(y, p, 0.01)
    r01, _ = recall_at_fpr(y, p, 0.001)
    out = {
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "recall@1%FPR": round(r1, 4),
        "recall@0.1%FPR": round(r01, 4),
        "threshold@1%FPR": round(t1, 4),
        "n": int(len(y)), "n_fraud": int(y.sum()),
    }
    print(f"{name:<22} PR-AUC={out['pr_auc']:.4f}  ROC-AUC={out['roc_auc']:.4f}  "
          f"recall@1%FPR={out['recall@1%FPR']:.3f}  recall@0.1%FPR={out['recall@0.1%FPR']:.3f}  "
          f"(n={out['n']:,}, fraud={out['n_fraud']:,})")
    return out


def fit(X, y, Xv, yv, seed=42):
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        # Class imbalance: full n_neg/n_pos (~250x) made every tree chase the positives
        # and early-stopping fired at iteration 1.  sqrt of the ratio (~16x) is the
        # usual compromise: positives still matter a lot, gradients stay sane.
        scale_pos_weight=(n_neg / n_pos) ** 0.5,
        random_state=seed, verbose=-1,
    )
    # early-stop on validation PR-AUC (average_precision).  ROC-AUC is a poor stopping
    # signal at 0.4% positives: it dips after the first tree and then recovers.
    model.fit(X, y, eval_set=[(Xv, yv)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
    return model


def main(feat_path, out):
    os.makedirs(out, exist_ok=True)
    df = pd.read_parquet(feat_path)
    df["txn_time"] = pd.to_datetime(df.txn_time)
    df["label_confirmed_at"] = pd.to_datetime(df.label_confirmed_at)
    df = df.sort_values("txn_time").reset_index(drop=True)
    print(f"loaded {len(df):,} rows, {len(FEATURES)} features")

    # ------------------------------------------------------------------ split
    train = df[(df.txn_time >= TRAIN_START) & (df.txn_time < TRAIN_END)]
    valid = df[(df.txn_time >= TRAIN_END) & (df.txn_time < VALID_END)]
    test = df[(df.txn_time >= TEST_START) & (df.txn_time < TEST_END)]
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    # ------------------------------------------------- label-delay honesty
    # On the retrain date, which fraud labels do we truly know?
    cutoff = pd.Timestamp(RETRAIN_DATE)
    known = (train.is_fraud == 1) & (train.label_confirmed_at < cutoff)
    y_train_honest = known.astype(int).to_numpy()
    y_train_oracle = train.is_fraud.to_numpy()
    print(f"label delay: {y_train_oracle.sum():,} true fraud in train window, "
          f"{y_train_honest.sum():,} confirmed by {RETRAIN_DATE} "
          f"({y_train_honest.sum()/y_train_oracle.sum():.0%} known at retrain time)")

    results = {}
    X_tr, X_va, X_te = train[FEATURES], valid[FEATURES], test[FEATURES]
    y_va, y_te = valid.is_fraud.to_numpy(), test.is_fraud.to_numpy()

    # ------------------------------------------------ main model (honest labels)
    print("\n== model A: honest labels (only chargebacks confirmed before cutoff) ==")
    m = fit(X_tr, y_train_honest, X_va, y_va)
    results["A_honest_valid"] = evaluate("A valid", y_va, m.predict_proba(X_va)[:, 1])
    results["A_honest_test"] = evaluate("A test", y_te, m.predict_proba(X_te)[:, 1])
    results["A_best_iter"] = int(m.best_iteration_)

    # ---------------------------------------------- oracle labels (for contrast)
    print("\n== model B: oracle labels (all fraud, even if not yet confirmed) ==")
    mb = fit(X_tr, y_train_oracle, X_va, y_va)
    results["B_oracle_test"] = evaluate("B test", y_te, mb.predict_proba(X_te)[:, 1])

    # -------------------------------------------- leakage check: random split
    print("\n== leakage check: random split on the same rows ==")
    pool = pd.concat([train, valid, test])
    rng = np.random.default_rng(0)
    mask = rng.random(len(pool)) < 0.7
    mr = fit(pool[FEATURES][mask], pool.is_fraud[mask], pool[FEATURES][~mask], pool.is_fraud[~mask])
    results["random_split_holdout"] = evaluate("random-split holdout", pool.is_fraud[~mask].to_numpy(),
                                               mr.predict_proba(pool[FEATURES][~mask])[:, 1])
    # compare like with like: random split used oracle labels, so compare to model B
    gap = results["random_split_holdout"]["pr_auc"] - results["B_oracle_test"]["pr_auc"]
    print(f"PR-AUC gap random - time = {gap:+.4f}  "
          f"({'small: no obvious leakage' if gap < 0.05 else 'LARGE: investigate leakage / drift'})")

    # ------------------------------------------------- business threshold
    # cost model: a missed fraud costs the txn amount; a false decline costs $5 (lost sale / friction)
    p_te = m.predict_proba(X_te)[:, 1]
    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        pred = p_te >= thr
        missed = test.amount[(~pred) & (y_te == 1)].sum()
        false_declines = int((pred & (y_te == 0)).sum()) * 5.0
        cost = missed + false_declines
        if best is None or cost < best["cost"]:
            best = {"threshold": round(float(thr), 2), "cost": round(float(cost), 2),
                    "missed_fraud_$": round(float(missed), 2), "false_decline_$": round(false_declines, 2),
                    "recall": round(float((pred & (y_te == 1)).sum() / y_te.sum()), 4),
                    "precision": round(float((pred & (y_te == 1)).sum() / max(pred.sum(), 1)), 4)}
    results["business_threshold"] = best
    print(f"\ncost-optimal threshold = {best['threshold']}  recall={best['recall']:.3f}  "
          f"precision={best['precision']:.3f}  total cost=${best['cost']:,.0f}")

    # ------------------------------------------------- feature importance
    imp = pd.DataFrame({"feature": FEATURES, "gain": m.booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["gain_pct"] = (imp.gain / imp.gain.sum() * 100).round(1)
    print("\ntop features by gain:")
    print(imp.head(10).to_string(index=False))
    imp.to_csv(f"{out}/feature_importance.csv", index=False)

    with open(f"{out}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    m.booster_.save_model(f"{out}/model_lgbm.txt")
    print(f"\nsaved {out}/metrics.json, feature_importance.csv, model_lgbm.txt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="data/features")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    main(a.features, a.out)
