"""
Points Over Expectation (POE) Pipeline

For every shot, computes:
  - xMake: predicted P(make) from a model that NEVER saw this shot's game
           during training (out-of-fold via GroupKFold over game_id).
  - shot_value: 2 or 3 points, derived from the PBP 'description' string
                if available, else from the geometric is_3_pointer flag.
  - expected_points = xMake * shot_value
  - actual_points   = made_shot * shot_value
  - POE             = actual_points - expected_points

Aggregating POE per player gives a defensible "shot maker" ranking,
because no player can be flattered by predictions on shots their game
contributed to during training.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from train_xgboost import DATA_PATH, load_data

RANDOM_STATE = 42
N_FOLDS = 5
MIN_SHOTS = 100  # reporting threshold; below this, POE is too noisy to interpret

# Fixed tree count (no early stopping inside each fold — we don't waste a
# val slice from each fold's training data; ~1150 is where the main model
# plateaued in the held-out training run, train_xgboost.py best_iteration 1163).
XGB_PARAMS = dict(
    enable_categorical=True,
    tree_method='hist',
    n_estimators=1150,
    learning_rate=0.02,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# ============================================================
# 1. Shot value (2 vs 3 points)
# ============================================================

def infer_shot_value(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Decide whether each shot is worth 2 or 3 points.

    Priority:
      1. PBP 'description' regex on '3PT' (most authoritative — handles corner 3s).
      2. The geometric is_3_pointer flag (will miss corner 3s if extractor uses 23.75ft).
    """
    if 'description' in df_raw.columns:
        is_three = (
            df_raw['description'].fillna('').str.contains(r'3PT', case=False, regex=True)
        )
        return np.where(is_three, 3, 2).astype(int)
    if 'is_3_pointer' in df_raw.columns:
        return np.where(df_raw['is_3_pointer'] == 1, 3, 2).astype(int)
    raise ValueError("Need 'description' or 'is_3_pointer' to determine shot value.")


# ============================================================
# 2. Out-of-fold predictions (GroupKFold over game_id)
# ============================================================

def out_of_fold_predict(X, y, groups, n_folds=N_FOLDS):
    """
    GroupKFold over game_id. For each fold, train on the other (n_folds - 1)
    folds and predict P(make) on the held-out fold. Returns one prediction
    per row aligned with X's original order.
    """
    oof_probs = np.full(len(X), np.nan, dtype=float)
    gkf = GroupKFold(n_splits=n_folds)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups), start=1):
        # Sanity check: no game crosses train/val inside a fold.
        train_games = set(groups.iloc[train_idx])
        val_games = set(groups.iloc[val_idx])
        assert train_games.isdisjoint(val_games), f"Fold {fold_idx} has game leak!"

        print(f"--- Fold {fold_idx}/{n_folds} | train={len(train_idx)} predict={len(val_idx)} "
              f"({len(val_games)} games held out) ---")

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)

        oof_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    assert not np.isnan(oof_probs).any(), "Some rows missed an OOF prediction"
    return oof_probs


# ============================================================
# 3. POE pipeline
# ============================================================

def compute_poe(csv_path=DATA_PATH,
                n_folds=N_FOLDS,
                min_shots=MIN_SHOTS,
                out_dir='results'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build the same X/y/groups the training pipeline uses.
    X, y, groups, feature_cols = load_data(csv_path)

    # 2) Out-of-fold xMake for every shot.
    print(f"\nGenerating out-of-fold predictions ({n_folds}-fold GroupKFold over game_id)...")
    oof_probs = out_of_fold_predict(X, y, groups, n_folds=n_folds)

    # 3) Quick OOF quality report — these should be in the same ballpark as the
    #    held-out test metrics from train_xgboost.py. If much better → leak.
    oof_brier = brier_score_loss(y, oof_probs)
    oof_ll = log_loss(y, oof_probs)
    oof_auc = roc_auc_score(y, oof_probs)
    print(f"\nOOF quality: Brier={oof_brier:.4f}  LogLoss={oof_ll:.4f}  ROC-AUC={oof_auc:.4f}")
    print("(Should be close to held-out test metrics. Much better = leakage; much worse = under-trained.)")

    # 4) Re-read raw to recover columns stripped from X (player_name, description, x/y).
    df_raw = pd.read_csv(csv_path)
    assert len(df_raw) == len(X), "Raw and feature row counts diverged."

    shot_value = infer_shot_value(df_raw)

    eval_df = pd.DataFrame({
        'player_name': df_raw['player_name'].values,
        'game_id':     df_raw['game_id'].values,
        'x':           df_raw['x'].values,
        'y':           df_raw['y'].values,
        'dist':        df_raw['dist'].values,
        'made_shot':   y.values,
        'xMake':       oof_probs,
        'shot_value':  shot_value,
    })
    eval_df['expected_points'] = eval_df['xMake'] * eval_df['shot_value']
    eval_df['actual_points']   = eval_df['made_shot'] * eval_df['shot_value']
    eval_df['POE']             = eval_df['actual_points'] - eval_df['expected_points']

    # 5) Aggregate per player.
    agg = eval_df.groupby('player_name').agg(
        total_shots=('made_shot', 'count'),
        actual_pts=('actual_points', 'sum'),
        expected_pts=('expected_points', 'sum'),
        total_POE=('POE', 'sum'),
    ).reset_index()

    # 6) Apply minimum-sample threshold and compute the per-100 rate.
    leaderboard = agg[agg['total_shots'] >= min_shots].copy()
    leaderboard['POE_per_100'] = (leaderboard['total_POE'] / leaderboard['total_shots']) * 100

    # Round display fields.
    for col in ['actual_pts', 'expected_pts', 'total_POE']:
        leaderboard[col] = leaderboard[col].round(1)
    leaderboard['POE_per_100'] = leaderboard['POE_per_100'].round(2)

    leaderboard = leaderboard.sort_values('total_POE', ascending=False).reset_index(drop=True)

    # 7) Persist.
    eval_df.to_csv(out_dir / 'poe_per_shot.csv', index=False)
    leaderboard.to_csv(out_dir / 'poe_leaderboard.csv', index=False)
    print(f"\nSaved {len(eval_df)} per-shot POE rows to '{out_dir/'poe_per_shot.csv'}'")
    print(f"Saved {len(leaderboard)} qualifying players to '{out_dir/'poe_leaderboard.csv'}' "
          f"(min {min_shots} shots)")

    return eval_df, leaderboard


# ============================================================
# 4. Reporting helpers
# ============================================================

def print_leaderboard(leaderboard: pd.DataFrame, n: int = 10):
    cols = ['player_name', 'total_shots', 'actual_pts', 'expected_pts', 'total_POE', 'POE_per_100']

    print("\n" + "=" * 78)
    print(f"TOP {n}: most points OVER expectation (out-of-fold)")
    print("=" * 78)
    print(leaderboard[cols].head(n).to_string(index=False))

    print("\n" + "=" * 78)
    print(f"BOTTOM {n}: most points UNDER expectation (out-of-fold)")
    print("=" * 78)
    print(leaderboard[cols].tail(n).iloc[::-1].to_string(index=False))


def plot_player_poe(eval_df: pd.DataFrame, target_player: str, out_dir='.'):
    """
    Spatial scatter of a single player's shots, colored by per-shot POE.
    Blue = made a hard shot (positive POE). Red = missed an easy one (negative POE).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = eval_df[eval_df['player_name'] == target_player]
    if len(pdf) == 0:
        print(f"Player '{target_player}' not found.")
        return

    # Symmetric color limits so the diverging colormap is centered on zero.
    vmax = float(np.nanpercentile(np.abs(pdf['POE']), 95))
    vmax = max(vmax, 0.5)

    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(
        pdf['x'], pdf['y'],
        c=pdf['POE'],
        cmap='coolwarm_r',  # red = bad outcome, blue = good outcome
        s=40, alpha=0.85, edgecolors='white', linewidths=0.5,
        vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(sc, ax=ax, label='Points Over Expectation (per shot)')
    total_poe = pdf['POE'].sum()
    ax.set_title(
        f"{target_player} — {len(pdf)} shots — total POE = {total_poe:+.1f}",
        fontsize=14,
    )
    ax.set_aspect('equal')
    ax.grid(False)

    fname = out_dir / f"{target_player.replace(' ', '_')}_POE_chart.png"
    fig.tight_layout()
    fig.savefig(fname, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved chart to {fname}")


# ============================================================
# 5. Entry point
# ============================================================

if __name__ == '__main__':
    eval_df, leaderboard = compute_poe()
    print_leaderboard(leaderboard, n=10)

    # Optional spatial charts — toggle as needed.
    # for player in ['Stephen Curry', 'Kevin Durant', 'Russell Westbrook', 'Kobe Bryant']:
    #     plot_player_poe(eval_df, player, out_dir='figures')
