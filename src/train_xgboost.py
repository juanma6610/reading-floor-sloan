"""
Shot Prediction XGBoost Model

Trains a Gradient Boosted Tree to predict shot success probability.
Optimized for tabular spatial/tracking features.

Key design decisions:
- Group-aware splitting on game_id so no game straddles train/val/test.
- Validation set is carved from the training portion. The test set is
  held out from early stopping and used only for the final report.
- No scale_pos_weight: classes are ~balanced (~45% makes) and the
  downstream POE/EPV pipeline depends on calibrated probabilities.
- Evaluation focuses on probability quality (Brier, log loss, PR-AUC)
  and a reliability diagram, not on a tunable accuracy threshold.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    brier_score_loss,
    log_loss,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
import matplotlib
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# Canonical dataset: valid2 + PBP shot-type flags (stype_*). Every model in the
# paper — centralized, federated, POE — trains on this file.
DATA_PATH = 'data/shot_features_valid2_type.csv'

# Never model inputs. Must match METADATA_COLS in federated/nba_federated/task.py
# so the centralized and federated models see the same features.
METADATA_COLS = ['player_name', 'game_time', 'quarter', 'score_margin',
                 'team_id', 'game_id', 'description', 'closest_def_name']

# Held-out test games. Same GroupShuffleSplit (15% of games, seed 42) as the
# federated global test set, so every reported number is on the same games.
TEST_SIZE = 0.15
VAL_SIZE = 0.15   # fraction of the training games used for early stopping

# ============================================================
# 1. Data Loading & Preparation
# ============================================================

def load_data(csv_path=DATA_PATH):
    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    target_col = 'made_shot'

    feature_cols = [c for c in df.columns if c not in METADATA_COLS + [target_col]]

    X = df[feature_cols]
    y = df[target_col]
    groups = df['game_id']  # used only by GroupShuffleSplit

    num_makes = (y == 1).sum()
    print(f"Number of features: {len(feature_cols)} | Number of shots: {len(df)}")
    print(f"Dataset: {len(df)} shots | {len(feature_cols)} features | {df['game_id'].nunique()} games")
    print(f"Make Rate: {(num_makes / len(df)) * 100:.1f}%")

    return X, y, groups, feature_cols


# ============================================================
# 2. Group-aware splitting (no game leaks across splits)
# ============================================================

def group_split(X, y, groups, test_size, random_state):
    """Single GroupShuffleSplit by game_id."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        groups.iloc[train_idx], groups.iloc[test_idx],
    )


# ============================================================
# 3. Model Training
# ============================================================

def train_xgboost(X, y, groups):
    # 1) Hold out the test set by game (15%, identical to the federated test set).
    X_trainval, X_test, y_trainval, y_test, g_trainval, g_test = group_split(
        X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # 2) Carve a validation set out of training (also by game) for early stopping.
    X_train, X_val, y_train, y_val, _, _ = group_split(
        X_trainval, y_trainval, g_trainval, test_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    # Sanity check: no game appears in more than one split.
    assert set(g_trainval).isdisjoint(set(g_test)), "Game leak between trainval and test!"

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("\nInitializing XGBoost...")
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        n_estimators=3000,          # max trees; early stopping decides actual count
        learning_rate=0.02,         # step-size shrinkage
        max_depth=5,                # shallow trees to limit spatial overfitting
        min_child_weight=5,         # require more data per split (fights noise)
        subsample=0.8,              # row subsampling
        colsample_bytree=0.8,       # column subsampling
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training with validation-based early stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # validation drives early stopping, NOT test
        verbose=100,
    )

    return model, X_val, y_val, X_test, y_test


# ============================================================
# 4. Evaluation: probability quality, not threshold accuracy
# ============================================================

def evaluate_model(model, X_test, y_test, feature_cols, out_dir='.'):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)  # 0.5 reported as a reference only

    # Probability-quality metrics (these are what POE/EPV depend on).
    brier = brier_score_loss(y_test, y_probs)
    ll = log_loss(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)
    base_rate = float(np.mean(y_test))
    acc = accuracy_score(y_test, y_preds)

    # Brier baseline = variance of y under a constant base-rate predictor.
    brier_baseline = base_rate * (1.0 - base_rate)

    print("\n" + "=" * 56)
    print("HELD-OUT TEST SET PERFORMANCE  (game-disjoint from training)")
    print("=" * 56)
    print(f"Test shots:      {len(y_test)}   Make rate: {base_rate*100:.1f}%")
    print(f"Brier score:     {brier:.4f}    (lower = better; constant-baseline = {brier_baseline:.4f})")
    print(f"Log loss:        {ll:.4f}     (lower = better)")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"PR-AUC:          {pr_auc:.4f}    (baseline = {base_rate:.4f})")
    print(f"Accuracy @0.5:   {acc*100:.2f}%   (reference only — not the optimization target)")
    print("-" * 56)
    print(classification_report(y_test, y_preds, target_names=['Miss', 'Make']))

    # --- Reliability diagram (the plot that matters for a probability model) ---
    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=15, strategy='quantile')
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='o', lw=2, label='Model')
    ax.set_xlabel('Mean predicted probability (per bin)')
    ax.set_ylabel('Empirical make rate (per bin)')
    ax.set_title(f'Reliability diagram — Brier {brier:.3f}, LogLoss {ll:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{out_dir}/reliability_diagram.png', dpi=150)
    plt.close(fig)
    print(f"Saved reliability diagram to '{out_dir}/reliability_diagram.png'")

    # --- Predicted-probability distribution by true outcome (sanity check) ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_probs[y_test == 0], bins=40, alpha=0.55, label='Misses', color='tab:red')
    ax.hist(y_probs[y_test == 1], bins=40, alpha=0.55, label='Makes',  color='tab:blue')
    ax.set_xlabel('Predicted P(Make)')
    ax.set_ylabel('Count')
    ax.set_title('Predicted probability distribution by outcome')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{out_dir}/probability_distribution.png', dpi=150)
    plt.close(fig)
    print(f"Saved probability distribution to '{out_dir}/probability_distribution.png'")

    # --- Feature importance (gain) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances.sort_values(ascending=True).tail(40).plot(kind='barh', ax=ax)
    ax.set_title('Top 40 XGBoost Feature Importances (Gain)')
    ax.set_xlabel('Relative Importance')
    fig.tight_layout()
    fig.savefig(f'{out_dir}/xgboost_feature_importance.png', dpi=150)
    plt.close(fig)
    
    print(f"Saved feature importance plot to '{out_dir}/xgboost_feature_importance.png'")

    return {
        'brier': brier,
        'log_loss': ll,
        'roc_auc': auc,
        'pr_auc': pr_auc,
        'accuracy_at_0.5': acc,
        'base_rate': base_rate,
    }


if __name__ == '__main__':
    X, y, groups, feature_cols = load_data()
    model, X_val, y_val, X_test, y_test = train_xgboost(X, y, groups)
    metrics = evaluate_model(model, X_test, y_test, feature_cols)

    model.save_model('data/xgb_shot_model.json')
    print("Model saved to 'data/xgb_shot_model.json'")
