"""
zone_comparison.py — per shooting-zone comparison of three models:
  * CENTRALIZED — trained on the pooled training set (all teams).
  * FEDERATED   — FedXgbBagging, team partition, seed 42, round selected on
                  federated validation (evaluate_federated.py):
                  results/federated/xgb_federated_bagging_team_seed42_model_selected.json
  * LOCAL       — trained on a SINGLE team's shots only (the largest silo alone).

All three use the federated XGBoost params, the same features, early stopping /
round selection on validation games, and are scored on the SAME held-out global
test set (task.py), so the only thing that varies is the data regime (pooled vs
federated-partitioned vs single team).

Reports, per zone: n, actual make%, and each model's mean predicted P(make),
Brier, and calibration gap (mean_pred − actual).

Run:  python src/experiments/zone_comparison.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
import evaluate_federated as ev          # noqa: E402  (matched baselines)
from nba_federated import task           # noqa: E402  (canonical split)

FED_MODEL = "results/federated/xgb_federated_bagging_team_seed42_model_selected.json"
SEED = 42
TARGET, GROUP = "made_shot", "game_id"
OUT = "results/zone_model_comparison.csv"
ZONES = ["Rim (<4ft)", "Close (4-10)", "Mid (10-16)", "Long 2 (16+)",
         "Corner 3", "Above-break 3", "Deep 3"]


def zone_of(r):
    d, three = r["dist"], r["is_3_pointer"] == 1
    if three:
        return "Corner 3" if d < 23.2 else ("Above-break 3" if d < 26.5 else "Deep 3")
    if d < 4:  return "Rim (<4ft)"
    if d < 10: return "Close (4-10)"
    if d < 16: return "Mid (10-16)"
    return "Long 2 (16+)"


def main():
    # federated model (round selected on federated validation by evaluate_federated.py)
    fed = xgb.Booster(); fed.load_model(FED_MODEL)
    print(f"Federated model: {os.path.basename(FED_MODEL)} | {fed.num_boosted_rounds()} trees")

    # Same held-out games and the same matched baselines as evaluate_federated.py:
    # centralized = union of the clients' train splits, local = the largest team alone,
    # both with the federated params and early stopping on their own validation games.
    df = task.load_full_dataset()
    _, te_idx = task._get_global_split()
    test = df.iloc[te_idx]
    feats = task._get_feature_cols(df)
    splits = ev.client_splits("team", SEED)
    X_tr, X_va, y_tr, y_va = ev.pooled(splits)
    params = ev.xgb_params(SEED)
    cen = ev.train_early_stopped(params, X_tr, y_tr, X_va, y_va)
    top = int(np.argmax([len(s[0]) for s in splits]))
    xt, xv, yt, yv = splits[top]
    loc = ev.train_early_stopped(params, xt, yt.to_numpy(float), xv, yv.to_numpy(float))
    print(f"Test {len(test):,} shots ({test[GROUP].nunique()} games) | central {cen.num_boosted_rounds()} trees"
          f" | local silo = client {top} ({len(xt):,} train shots, {loc.num_boosted_rounds()} trees)\n")

    dtest = xgb.DMatrix(test[feats], label=test[TARGET])
    ytest = test[TARGET].values

    preds = {"centralized": cen.predict(dtest),
             "federated":   fed.predict(dtest),
             "local":       loc.predict(dtest)}

    def overall(p):
        return brier_score_loss(ytest, p), roc_auc_score(ytest, p)
    print("=== overall (held-out test) ===")
    for k, p in preds.items():
        b, a = overall(p)
        print(f"  {k:<12} Brier={b:.5f}  AUC={a:.5f}")

    # per-zone
    test = test.copy(); test["zone"] = test.apply(zone_of, axis=1)
    rows = []
    for z in ZONES:
        m = (test["zone"] == z).values
        if m.sum() == 0:
            continue
        row = {"zone": z, "n": int(m.sum()), "actual": round(float(ytest[m].mean()), 3)}
        for k, p in preds.items():
            row[f"{k}_pred"] = round(float(p[m].mean()), 3)
            row[f"{k}_brier"] = round(float(brier_score_loss(ytest[m], p[m])), 4)
            row[f"{k}_gap"] = round(float(p[m].mean() - ytest[m].mean()), 3)
        rows.append(row)
    zt = pd.DataFrame(rows)
    zt.to_csv(OUT, index=False)

    print("\n=== per-zone Brier (lower = better) ===")
    print(f"{'zone':<15}{'n':>6}{'actual':>8} | {'central':>8}{'fed':>8}{'local':>8}")
    for _, r in zt.iterrows():
        print(f"{r['zone']:<15}{int(r['n']):>6}{r['actual']:>8.3f} | "
              f"{r['centralized_brier']:>8.4f}{r['federated_brier']:>8.4f}{r['local_brier']:>8.4f}")

    print("\n=== per-zone calibration gap (mean predicted − actual make%) ===")
    print(f"{'zone':<15}{'actual':>8} | {'central':>8}{'fed':>8}{'local':>8}")
    for _, r in zt.iterrows():
        print(f"{r['zone']:<15}{r['actual']:>8.3f} | "
              f"{r['centralized_gap']:>+8.3f}{r['federated_gap']:>+8.3f}{r['local_gap']:>+8.3f}")

    # plot: per-zone Brier grouped bars
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = np.arange(len(zt)); w = 0.26
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - w, zt["centralized_brier"], w, label="centralized", color="#1D428A")
        ax.bar(x,     zt["federated_brier"],   w, label="federated",   color="#C8102E")
        ax.bar(x + w, zt["local_brier"],       w, label="local (1 team)", color="#888888")
        ax.set_xticks(x); ax.set_xticklabels(zt["zone"], rotation=30, ha="right")
        ax.set_ylabel("Brier (lower = better)"); ax.set_title("Shot-model calibration by zone: centralized vs federated vs single-team local")
        ax.legend(); fig.tight_layout(); fig.savefig("results/zone_model_comparison.png", dpi=150)
        print("\nSaved results/zone_model_comparison.png and", OUT)
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
