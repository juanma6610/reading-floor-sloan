"""
XGBoost Hyperparameter Tuning (group-aware, calibration-friendly)

Random search over a focused space, scored by mean log-loss across a
GroupKFold(5) over game_id on the trainval portion. Test set is split off
first (game-disjoint) and never touched until the final report.

Why log-loss as the search objective?
  Brier and log-loss both reward calibration; log-loss penalizes overconfident
  wrong predictions harder, which is what you want for a model whose
  probabilities feed POE/EPV downstream.

Run from project root:
    python src/tune_xgboost.py --n-iter 30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from train_xgboost import DATA_PATH, RANDOM_STATE, TEST_SIZE, VAL_SIZE, group_split, load_data


# ------------------------------------------------------------
# Search space
# ------------------------------------------------------------
FIXED_PARAMS = dict(
    enable_categorical=True,
    tree_method='hist',
    objective='binary:logistic',
    eval_metric='logloss',
    n_jobs=-1,
)

SEARCH_SPACE = dict(
    max_depth=[4, 5, 6, 7, 8],
    min_child_weight=[3, 5, 10, 20, 50],
    subsample=[0.7, 0.8, 0.9, 1.0],
    colsample_bytree=[0.6, 0.7, 0.8, 0.9, 1.0],
    learning_rate=[0.01, 0.02, 0.03, 0.05],
    gamma=[0.0, 0.5, 1.0, 2.0, 5.0],
    reg_lambda=[0.5, 1.0, 2.0, 5.0],
    reg_alpha=[0.0, 0.5, 1.0, 2.0],
)

N_FOLDS         = 5
DEFAULT_N_ITER  = 30
MAX_BOOST_ROUNDS = 1500
EARLY_STOP      = 50


# ------------------------------------------------------------
# Sampling + evaluation
# ------------------------------------------------------------

def sample_params(rng):
    """Independent uniform draw from each list in SEARCH_SPACE."""
    return {k: rng.choice(v).item() if hasattr(rng.choice(v), 'item') else rng.choice(v)
            for k, v in SEARCH_SPACE.items()}


def evaluate_candidate(params, X, y, groups, n_folds=N_FOLDS):
    """
    Run GroupKFold(n_folds). In each fold: fit with early stopping using
    the fold's val set; record best logloss and the chosen tree count.
    Return mean/std logloss across folds and mean tree count.
    """
    losses, trees = [], []
    gkf = GroupKFold(n_splits=n_folds)

    for tr_idx, va_idx in gkf.split(X, y, groups):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = xgb.XGBClassifier(
            **FIXED_PARAMS, **params,
            n_estimators=MAX_BOOST_ROUNDS,
            early_stopping_rounds=EARLY_STOP,
            random_state=RANDOM_STATE,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

        # predict_proba on a fitted model with early_stopping_rounds is
        # automatically truncated to best_iteration in modern XGBoost.
        probs = model.predict_proba(X_va)[:, 1]
        losses.append(log_loss(y_va, probs, labels=[0, 1]))
        trees.append((model.best_iteration or MAX_BOOST_ROUNDS - 1) + 1)

    return float(np.mean(losses)), float(np.std(losses)), int(np.mean(trees))


def random_search(X, y, groups, n_iter, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    best = {'logloss': float('inf'), 'params': None, 'n_trees': None}
    history = []

    for i in range(1, n_iter + 1):
        params = sample_params(rng)
        mean_ll, std_ll, mean_trees = evaluate_candidate(params, X, y, groups)
        history.append({
            **params,
            'cv_logloss_mean': mean_ll,
            'cv_logloss_std':  std_ll,
            'best_n_trees':    mean_trees,
        })

        marker = ''
        if mean_ll < best['logloss']:
            best.update(logloss=mean_ll, params=params, n_trees=mean_trees)
            marker = '  <-- new best'
        print(f"[{i:3d}/{n_iter}] cv_logloss={mean_ll:.4f}+/-{std_ll:.4f}  "
              f"trees={mean_trees}  depth={params['max_depth']}  "
              f"lr={params['learning_rate']}{marker}")

    return best, history


# ------------------------------------------------------------
# Final fit + held-out test eval
# ------------------------------------------------------------

def fit_final_model(best, X_trainval, y_trainval, g_trainval, X_test, y_test):
    """
    Refit the best params on the full trainval, with a small game-disjoint
    val carved off for one last round of early stopping. Then score on test.
    """
    X_train, X_val, y_train, y_val, _, _ = group_split(
        X_trainval, y_trainval, g_trainval,
        test_size=VAL_SIZE, random_state=RANDOM_STATE,
    )
    budget = max(int(best['n_trees'] * 1.5), 200)

    model = xgb.XGBClassifier(
        **FIXED_PARAMS, **best['params'],
        n_estimators=budget,
        early_stopping_rounds=EARLY_STOP,
        random_state=RANDOM_STATE,
    )
    print(f"\nRefitting best params on trainval (budget={budget} trees, "
          f"early stopping on a {len(X_val)}-shot game-disjoint val)...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    metrics = dict(
        brier=float(brier_score_loss(y_test, probs)),
        log_loss=float(log_loss(y_test, probs, labels=[0, 1])),
        roc_auc=float(roc_auc_score(y_test, probs)),
        pr_auc=float(average_precision_score(y_test, probs)),
        accuracy=float(accuracy_score(y_test, preds)),
    )
    return model, metrics


# ------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default=DATA_PATH)
    parser.add_argument('--n-iter', type=int, default=DEFAULT_N_ITER)
    parser.add_argument('--out-dir', default='results')
    parser.add_argument('--model-out', default='data/xgb_shot_model_tuned.json')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Splitting test set out (game-disjoint, untouched until the final eval)...")
    X, y, groups, feature_cols = load_data(args.csv)
    X_trainval, X_test, y_trainval, y_test, g_trainval, g_test = group_split(
        X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"Trainval: {len(X_trainval)} ({g_trainval.nunique()} games) | "
          f"Test: {len(X_test)} ({g_test.nunique()} games)")

    print(f"\nRandom search: {args.n_iter} candidates, {N_FOLDS}-fold GroupKFold over game_id")
    best, history = random_search(X_trainval, y_trainval, g_trainval, n_iter=args.n_iter)

    pd.DataFrame(history).sort_values('cv_logloss_mean').to_csv(
        out_dir / 'tuning_history.csv', index=False
    )
    print(f"\nFull search history -> {out_dir/'tuning_history.csv'}")

    print("\n" + "=" * 56)
    print("BEST HYPERPARAMETERS")
    print("=" * 56)
    for k, v in best['params'].items():
        print(f"  {k:<18} {v}")
    print(f"  best_n_trees       {best['n_trees']}  (mean across folds)")
    print(f"  CV log-loss        {best['logloss']:.4f}")

    final_model, test_metrics = fit_final_model(
        best, X_trainval, y_trainval, g_trainval, X_test, y_test
    )

    print("\n" + "=" * 56)
    print("HELD-OUT TEST PERFORMANCE  (best-tuned model)")
    print("=" * 56)
    for k, v in test_metrics.items():
        print(f"  {k:<10} {v:.4f}")

    final_model.save_model(args.model_out)
    with open(out_dir / 'best_params.json', 'w') as f:
        json.dump({
            'params':       best['params'],
            'n_trees':      best['n_trees'],
            'cv_logloss':   best['logloss'],
            'test_metrics': test_metrics,
        }, f, indent=2)
    print(f"\nFinal model -> {args.model_out}")
    print(f"Best params + test metrics -> {out_dir/'best_params.json'}")


if __name__ == '__main__':
    main()
