"""
add_metadata.py — test whether the excluded METADATA columns improve the model:
    player_name, game_time, quarter, score_margin, team_id, closest_def_name

Encoding: player_name / closest_def_name / team_id are IDENTITIES → given to
XGBoost as native categoricals (enable_categorical=True), so the tree can
partition on the player/defender/team directly (a stronger form of "identity"
than the earlier OOF target-encodings). game_time / quarter / score_margin are
numeric game context. `description` and `game_id` stay excluded (description
leaks the outcome; game_id is the split group).

Split is game-disjoint (matches feature_ablation) so a player seen in training
is scored on *different* games in test — legitimate identity use, not leakage.

Run:  python src/experiments/add_metadata.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import feature_ablation as fa
from sklearn.model_selection import GroupShuffleSplit

DATA = "data/shot_features_valid2.csv"
CAT = ["player_name", "closest_def_name", "team_id"]
NUM = ["game_time", "quarter", "score_margin"]
META = CAT + NUM
PARAMS = {**fa.XGB_PARAMS, "n_estimators": 1500}   # capped for speed; early stopping usually stops sooner


def fit_eval(df, cols, tr, va, te, seed=42):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
    Xtr, ytr = df.loc[tr, cols], df.loc[tr, fa.TARGET]
    Xva, yva = df.loc[va, cols], df.loc[va, fa.TARGET]
    Xte, yte = df.loc[te, cols], df.loc[te, fa.TARGET]
    m = xgb.XGBClassifier(**PARAMS, random_state=seed,
                          early_stopping_rounds=fa.EARLY_STOP, enable_categorical=True)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    return dict(brier=brier_score_loss(yte, p), log_loss=log_loss(yte, p),
                roc_auc=roc_auc_score(yte, p), n=len(cols),
                best_iter=int(getattr(m, "best_iteration", 0) or 0))


def main():
    df = pd.read_csv(DATA)
    df = df[df[fa.TARGET].isin([0, 1])].reset_index(drop=True)
    for c in CAT:                      # shared category set across splits
        df[c] = df[c].astype("category")

    gss = GroupShuffleSplit(1, test_size=0.20, random_state=42)
    trv, te = next(gss.split(df, groups=df[fa.GROUP]))
    trvd = df.iloc[trv]
    gss2 = GroupShuffleSplit(1, test_size=0.15 / 0.8, random_state=42)
    a, b = next(gss2.split(trvd, groups=trvd[fa.GROUP]))
    trI, vaI, teI = trvd.index[a], trvd.index[b], df.index[te]

    baseline = [c for c in df.columns if c not in fa.METADATA + [fa.TARGET]]

    def run(cols, label):
        m = fit_eval(df, cols, trI, vaI, teI)
        print(f"  {label:<40} brier={m['brier']:.5f}  logloss={m['log_loss']:.5f}  auc={m['roc_auc']:.5f}  (nfeat={m['n']}, trees={m['best_iter']})")
        return m

    print(f"Baseline features: {len(baseline)} | adding metadata: {META}\n")
    b0 = run(baseline, "current model (no metadata)")
    b1 = run(baseline + NUM, "  + context (game_time, quarter, margin)")
    b2 = run(baseline + CAT, "  + identity (player, defender, team)")
    b3 = run(baseline + META, "  + ALL metadata")
    print(f"\n  gain vs current — context: Δauc {b1['roc_auc']-b0['roc_auc']:+.5f}  Δbrier {b0['brier']-b1['brier']:+.5f}")
    print(f"  gain vs current — identity: Δauc {b2['roc_auc']-b0['roc_auc']:+.5f}  Δbrier {b0['brier']-b2['brier']:+.5f}")
    print(f"  gain vs current — ALL:      Δauc {b3['roc_auc']-b0['roc_auc']:+.5f}  Δbrier {b0['brier']-b3['brier']:+.5f}")


if __name__ == "__main__":
    main()
