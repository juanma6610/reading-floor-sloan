"""
sim_bagging.py — fast, dependency-light simulation of FedXgbBagging.

Reproduces Flower's FedXgbBagging WITHOUT flwr/ray so we can sweep XGBoost configs
quickly and see which (if any) narrows the federation gap to the centralized
baseline. It reuses nba_federated/task.py so the global test split and the 30 team
(or IID) partitions are IDENTICAL to the real pipeline — results are directly
comparable to results/federated/*.

Faithful bagging semantics (local-epochs = 1):
  * every client starts each round from the SAME global model,
  * fits ONE tree on its silo (warm-started from the current global margin),
  * the round's new trees from all 30 clients are SUMMED into the global model
    (that's what FedXgbBagging.aggregate does — tree concatenation == margin sum).
We track the global margin on the train pool and on the test set, add each new
tree's margin contribution (learning_rate already baked in), and evaluate AUC/Brier
each round — exactly the server-side metric the real runs log.

NOTE: this is a research proxy for config search, not a replacement for the real
Flower runs. Validate it against a known config first (`--validate`), then sweep.
Clients here train on the full silo (the real client_app holds out 20% for local
eval); this shifts absolute numbers by <~0.003 AUC and does not change config
ORDERING, which is what a sweep needs.

Run:
  python src/federated/sim_bagging.py --validate                 # check vs known 0.685
  python src/federated/sim_bagging.py --sweep                    # config sweep table
  python src/federated/sim_bagging.py --max-depth 3 --eta 0.03   # one config
"""
from __future__ import annotations
import os, sys, argparse, json
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nba_federated"))
import task  # noqa: E402

ROUNDS = 20           # the federated model peaks by ~round 3-5; 20 is plenty
NCLIENTS = 30


def _logit(p): return float(np.log(p / (1.0 - p)))
def _sigmoid(m): return 1.0 / (1.0 + np.exp(-m))


def load_pool(strategy="team"):
    """Return pooled train (X,y, silo row-index lists) and the global test (X,y)."""
    df = task.load_full_dataset()
    feats = task._get_feature_cols(df)
    tr_idx, te_idx = task._get_global_split()
    train = df.iloc[tr_idx].reset_index(drop=True)
    test = df.iloc[te_idx]
    Xtr, ytr = train[feats].values, train[task.TARGET_COL].values.astype(float)
    Xte, yte = test[feats].values, test[task.TARGET_COL].values.astype(float)

    if strategy == "team":
        silos = [np.flatnonzero(train["team_id"].values == t)
                 for t in sorted(train["team_id"].unique())]
    else:  # iid
        rng = np.random.default_rng(42)
        order = rng.permutation(len(train))
        silos = [np.asarray(c) for c in np.array_split(order, NCLIENTS)]
    return Xtr, ytr, silos, Xte, yte, feats


def sim(params, strategy="team", rounds=ROUNDS, seed=42, verbose=False):
    Xtr, ytr, silos, Xte, yte, feats = load_pool(strategy)
    m0 = _logit(ytr.mean())
    Gtr = np.full(len(ytr), m0, dtype=np.float64)
    Gte = np.full(len(yte), m0, dtype=np.float64)

    # reusable DMatrices for PURE-tree contribution (base_margin = 0)
    d_tr0 = xgb.DMatrix(Xtr); d_tr0.set_base_margin(np.zeros(len(ytr)))
    d_te0 = xgb.DMatrix(Xte); d_te0.set_base_margin(np.zeros(len(yte)))

    p = dict(objective="binary:logistic", eval_metric="logloss",
             tree_method="hist", seed=seed, **params)
    hist = []
    for r in range(1, rounds + 1):
        for c, rows in enumerate(silos):
            dc = xgb.DMatrix(Xtr[rows], label=ytr[rows])
            dc.set_base_margin(Gtr[rows])                 # warm-start from shared global
            bst = xgb.train(dict(p, seed=seed + c), dc, num_boost_round=1)
            Gtr += bst.predict(d_tr0, output_margin=True)  # += eta * tree(x)
            Gte += bst.predict(d_te0, output_margin=True)
        auc = roc_auc_score(yte, _sigmoid(Gte))
        brier = brier_score_loss(yte, _sigmoid(Gte))
        hist.append((r, r * NCLIENTS, auc, brier))
        if verbose:
            print(f"  round {r:2d}  trees {r*NCLIENTS:4d}  AUC {auc:.4f}  Brier {brier:.4f}")
    peak = max(hist, key=lambda h: h[2])
    return dict(peak_auc=peak[2], peak_round=peak[0], peak_trees=peak[1],
                brier_at_peak=peak[3], hist=hist)


REG = dict(max_depth=4, eta=0.01, subsample=0.8, colsample_bytree=0.8,
           min_child_weight=10, reg_lambda=2.0)
STRONG = dict(max_depth=5, eta=0.05, subsample=0.8, colsample_bytree=0.8,
              min_child_weight=10, reg_lambda=2.0)
CENT_AUC, CENT_BRIER = 0.732, 0.204


def validate():
    print("Validating simulator against the known real runs (team partition):")
    for name, cfg, real in [("regularized (real ~0.685)", REG, 0.685),
                            ("strong (real 0.685)", STRONG, 0.685)]:
        r = sim(cfg, "team")
        print(f"  {name:26s}  sim peak AUC {r['peak_auc']:.4f} @round {r['peak_round']:>2} "
              f"({r['peak_trees']} trees)  Brier {r['brier_at_peak']:.4f}   [real ~{real}]")


def sweep():
    grid = {
        "strong (baseline of sweep)": STRONG,
        "regularized":                REG,
        "shallow d3 eta.03":          dict(STRONG, max_depth=3, eta=0.03),
        "shallow d3 eta.01":          dict(STRONG, max_depth=3, eta=0.01),
        "d2 stumps eta.05":           dict(STRONG, max_depth=2, eta=0.05),
        "d4 eta.02 strongL2":         dict(STRONG, max_depth=4, eta=0.02, reg_lambda=5.0),
        "heavy reg (mcw30,L2=8)":     dict(STRONG, max_depth=4, min_child_weight=30, reg_lambda=8.0),
        "low subsample .5/.5":        dict(STRONG, subsample=0.5, colsample_bytree=0.5),
        "d3 eta.02 mcw20 L2=5":       dict(max_depth=3, eta=0.02, subsample=0.7,
                                           colsample_bytree=0.7, min_child_weight=20, reg_lambda=5.0),
    }
    print(f"{'config':30s} {'partition':9s} {'peakAUC':>8} {'rnd':>4} {'trees':>6} {'Brier':>7} "
          f"{'ΔAUC':>7} {'ΔBrier%':>8}")
    rows = []
    for name, cfg in grid.items():
        for strat in ("team", "iid"):
            r = sim(cfg, strat)
            dauc = r["peak_auc"] - CENT_AUC
            dbri = (r["brier_at_peak"] / CENT_BRIER - 1) * 100
            print(f"{name:30s} {strat:9s} {r['peak_auc']:8.4f} {r['peak_round']:4d} "
                  f"{r['peak_trees']:6d} {r['brier_at_peak']:7.4f} {dauc:+7.4f} {dbri:+7.1f}%")
            rows.append(dict(config=name, partition=strat, peak_auc=r["peak_auc"],
                             peak_round=r["peak_round"], peak_trees=r["peak_trees"],
                             brier=r["brier_at_peak"], dAUC=dauc, dBrier_pct=dbri))
    import pandas as pd
    pd.DataFrame(rows).to_csv("results/federated/config_sweep_sim.csv", index=False)
    print(f"\ncentralized baseline: AUC {CENT_AUC}, Brier {CENT_BRIER}")
    print("wrote results/federated/config_sweep_sim.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--strategy", default="team")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    for k, v in STRONG.items():
        ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=None)
    a = ap.parse_args()
    if a.validate: validate(); return
    if a.sweep: sweep(); return
    cfg = dict(STRONG)
    for k in STRONG:
        v = getattr(a, k)
        if v is not None: cfg[k] = v
    r = sim(cfg, a.strategy, rounds=a.rounds, verbose=True)
    print(f"\npeak AUC {r['peak_auc']:.4f} @round {r['peak_round']} ({r['peak_trees']} trees) "
          f"Brier {r['brier_at_peak']:.4f}  | centralized {CENT_AUC}/{CENT_BRIER}")


if __name__ == "__main__":
    main()
