"""
Thesis Results Generator

Trains the model (or reuses an already-trained one) and produces
thesis-grade figures and tables for the Results section.

Run from the project root:
    python src/thesis_results.py

Produces
--------
results/headline_metrics.csv      Model vs constant + distance-only baselines.
results/per_zone_metrics.csv      Brier / LogLoss / AUC / Accuracy by zone.
results/poe_leaderboard.csv       Top-10 + bottom-10 ranked by total POE.
figures/calibration_diagram.png   Reliability diagram with Brier annotated.
figures/feature_importance_top20.png
figures/probability_distribution.png
figures/poe_spatial_<player>.png  One per case-study player.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve

from train_xgboost import DATA_PATH, load_data, train_xgboost
from poe.compute_poe import compute_poe, infer_shot_value

# ------------------------------------------------------------
# Style — neutral, print-friendly. Override in the thesis template if needed.
# ------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIG_DIR = Path('figures')
RES_DIR = Path('results')
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_PATH   # canonical dataset (valid2 + PBP shot type), see train_xgboost.py
CASE_STUDY_PLAYERS = ['Stephen Curry', 'Russell Westbrook']


# ============================================================
# 1. Baselines
# ============================================================

def constant_baseline_metrics(y_test):
    """Always predicts P(make) = base rate."""
    p = float(np.mean(y_test))
    n = len(y_test)
    probs = np.full(n, p)
    return {
        'brier':    brier_score_loss(y_test, probs),
        'log_loss': log_loss(y_test, probs, labels=[0, 1]),
        'roc_auc':  0.5,
        'pr_auc':   p,
        'accuracy': max(p, 1.0 - p),
    }


def distance_only_metrics(X_train, y_train, X_test, y_test):
    """Logistic regression on `dist` alone — the classical xG-from-distance baseline."""
    clf = LogisticRegression()
    clf.fit(X_train[['dist']], y_train)
    probs = clf.predict_proba(X_test[['dist']])[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        'brier':    brier_score_loss(y_test, probs),
        'log_loss': log_loss(y_test, probs, labels=[0, 1]),
        'roc_auc':  roc_auc_score(y_test, probs),
        'pr_auc':   average_precision_score(y_test, probs),
        'accuracy': accuracy_score(y_test, preds),
    }


def full_model_metrics(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        'brier':    brier_score_loss(y_test, probs),
        'log_loss': log_loss(y_test, probs, labels=[0, 1]),
        'roc_auc':  roc_auc_score(y_test, probs),
        'pr_auc':   average_precision_score(y_test, probs),
        'accuracy': accuracy_score(y_test, preds),
    }


def headline_metrics_table(model, X_train, y_train, X_test, y_test):
    rows = [
        {'model': 'Constant (base rate)', **constant_baseline_metrics(y_test)},
        {'model': 'Distance-only logistic', **distance_only_metrics(X_train, y_train, X_test, y_test)},
        {'model': 'XGBoost (full features)', **full_model_metrics(model, X_test, y_test)},
    ]
    df = pd.DataFrame(rows)
    df = df[['model', 'brier', 'log_loss', 'roc_auc', 'pr_auc', 'accuracy']]
    return df


# ============================================================
# 2. Per-zone breakdown
# ============================================================

def assign_zone(dist):
    if dist <= 4.0:
        return 'Restricted area'
    elif dist <= 14.0:
        return 'Paint (non-RA)'
    elif dist < 22.0:
        return 'Mid-range'
    return '3-pointer'


def per_zone_metrics(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df = X_test.copy()
    df['_y'] = y_test.values
    df['_p'] = probs
    df['_yhat'] = preds
    df['_zone'] = df['dist'].apply(assign_zone)

    rows = []
    zone_order = ['Restricted area', 'Paint (non-RA)', 'Mid-range', '3-pointer']
    for zone in zone_order:
        sub = df[df['_zone'] == zone]
        if len(sub) == 0:
            continue
        rows.append({
            'zone': zone,
            'n_shots': int(len(sub)),
            'make_rate': float(sub['_y'].mean()),
            'brier':    brier_score_loss(sub['_y'], sub['_p']),
            'log_loss': log_loss(sub['_y'], sub['_p'], labels=[0, 1]),
            'roc_auc':  roc_auc_score(sub['_y'], sub['_p']) if sub['_y'].nunique() == 2 else np.nan,
            'accuracy': accuracy_score(sub['_y'], sub['_yhat']),
        })
    return pd.DataFrame(rows)


# ============================================================
# 3. Plots
# ============================================================

def plot_calibration(y_test, probs, brier, out_path):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=15, strategy='quantile')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], '--', color='gray', lw=1.2, label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, marker='o', lw=2, color='#1f4e79', label='XGBoost (full)')
    ax.set_xlabel('Mean predicted probability of make (per bin)')
    ax.set_ylabel('Empirical make rate (per bin)')
    ax.set_title(f'Reliability diagram (Brier = {brier:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance_top20(model, feature_cols, out_path):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=True).tail(25)

    # Categorize features so the chart is readable.
    def category(name):
        n = name.lower()
        if n.startswith('release_') or n in ('shot_angle',):
            return 'Release mechanics'
        if 'def' in n or n.startswith('time_to_contest') or n.startswith('second_'):
            return 'Defender pressure'
        if 'primary_creator' in n or 'spacer' in n or 'mid-interior' in n or 'rim_center' in n:
            return 'Shooter archetype'      
        if 'paint_anchors' in n or 'perimeter guards' in n or 'def_liability' in n or 'switch_wing' in n:  
            return 'Defender archetype'
        if 'vel' in n or 'acc' in n or n in ('touch_time',):
            return 'Shooter kinematics'
        if n.startswith('stype_'):
            return 'Shot type (PBP)'
        if n in ('dist', 'x', 'y', 'is_3_pointer', 'shot_clock'):
            return 'Shot context'
        return 'Other'

    palette = {
        'Shot context':       '#1f4e79',
        'Defender pressure':  '#c0504d',
        'Shooter kinematics': '#9bbb59',
        'Release mechanics':  '#8064a2',
        'Shooter archetype':   '#f79646',
        "Defender archetype": '#4bacc6',
        'Shot type (PBP)':    '#d4a017',
        'Other':              '#7f7f7f',
    }
    cats = [category(name) for name in top.index]
    colors = [palette[c] for c in cats]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top.index, top.values, color=colors)
    ax.set_xlabel('Relative gain')
    ax.set_title('Top 25 features by gain')

    # Manual legend covering only the categories that appear in the top 20.
    seen = list(dict.fromkeys(cats))  # preserve order
    handles = [patches.Patch(color=palette[c], label=c) for c in seen]
    ax.legend(handles=handles, loc='lower right', frameon=False)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_probability_distribution(y_test, probs, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(probs[y_test == 0], bins=40, alpha=0.55, label='Missed shots', color='#c0504d')
    ax.hist(probs[y_test == 1], bins=40, alpha=0.55, label='Made shots',   color='#1f4e79')
    ax.set_xlabel('Predicted P(make)')
    ax.set_ylabel('Number of shots')
    ax.set_title('Predicted probability distribution by outcome')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_half_court(ax, color='black', lw=1.2):
    """
    Draws a to-scale NBA offensive half-court (47 x 50 ft) on the given axes.
    Frame: x in [47, 94] (baseline at x=94), y in [0, 50]; hoop centre at (88.75, 25).
    All dimensions in feet: lane 16 x 19, FT line 19 ft off the baseline, hoop
    5.25 ft off the baseline, backboard 4 ft off the baseline, 3-pt line 23.75 ft
    (22 ft in the corners).
    """
    # Half-court boundary
    ax.add_patch(patches.Rectangle((47, 0), 47, 50, fill=False, edgecolor=color, lw=lw))
    # Paint / lane: 16 ft wide x 19 ft long, anchored on the baseline up to the FT line (x=75)
    ax.add_patch(patches.Rectangle((75, 17), 19, 16, fill=False, edgecolor=color, lw=lw))
    # Free-throw circle (r=6) centred on the FT line (75, 25): solid toward mid-court, dashed toward basket
    ax.add_patch(patches.Arc((75, 25), 12, 12, theta1=90, theta2=270, color=color, lw=lw))
    ax.add_patch(patches.Arc((75, 25), 12, 12, theta1=270, theta2=90, color=color, lw=lw, linestyle='--'))
    # Backboard (6 ft wide, 4 ft off the baseline -> x=90) and rim (r=0.75 around 88.75)
    ax.plot([90, 90], [22, 28], color=color, lw=lw)
    ax.add_patch(patches.Circle((88.75, 25), radius=0.75, fill=False, edgecolor=color, lw=lw))
    # Restricted-area arc (r=4) around the hoop, opening toward mid-court
    ax.add_patch(patches.Arc((88.75, 25), 8, 8, theta1=90, theta2=270, color=color, lw=lw))
    # 3-point line: corner lines 3 ft from each sideline (y=3, y=47) + arc r=23.75 around the hoop
    r = 23.75
    x_corner = 88.75 - np.sqrt(r ** 2 - 22.0 ** 2)   # where the arc meets the corner lines
    ax.plot([94, x_corner], [3, 3], color=color, lw=lw)
    ax.plot([94, x_corner], [47, 47], color=color, lw=lw)
    span = np.degrees(np.arcsin(22.0 / r))           # ~67.9 deg half-width of the arc about 180 deg
    ax.add_patch(patches.Arc((88.75, 25), 2 * r, 2 * r,
                             theta1=180 - span, theta2=180 + span,
                             color=color, lw=lw))
    ax.set_xlim(47, 94)
    ax.set_ylim(0, 50)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ('top', 'right', 'bottom', 'left'):
        ax.spines[s].set_visible(False)


def plot_poe_spatial(eval_df, target_player, out_path):
    pdf = eval_df[eval_df['player_name'] == target_player]
    if len(pdf) == 0:
        print(f"  [skip] {target_player}: no shots in eval set.")
        return

    vmax = float(np.nanpercentile(np.abs(pdf['POE']), 95))
    vmax = max(vmax, 0.5)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    draw_half_court(ax, color='black', lw=1.2)
    sc = ax.scatter(
        pdf['x'], pdf['y'],
        c=pdf['POE'],
        cmap='coolwarm_r',
        s=42, alpha=0.85, edgecolors='white', linewidths=0.5,
        vmin=-vmax, vmax=vmax, zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('Points Over Expectation (per shot)')
    total_poe = pdf['POE'].sum()
    n_shots = len(pdf)
    ax.set_title(
        f"{target_player} — {n_shots} shots — total POE = {total_poe:+.1f} "
        f"({total_poe / n_shots * 100:+.2f} per 100)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ============================================================
# 4. LaTeX-friendly table writer
# ============================================================

def write_table(df, base_path, float_fmt='%.4f'):
    """Write CSV (always) and LaTeX (best-effort) for a dataframe."""
    csv_path = base_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False, float_format=float_fmt)
    print(f"  Wrote {csv_path}")
    try:
        tex_path = base_path.with_suffix('.tex')
        df.to_latex(tex_path, index=False, float_format=float_fmt, escape=True)
        print(f"  Wrote {tex_path}")
    except Exception as e:
        print(f"  (skipping LaTeX for {base_path.name}: {e})")


# ============================================================
# 5. Orchestrator
# ============================================================

def main():
    print("\n[1/6] Loading data and training model on game-disjoint splits...")
    X, y, groups, feature_cols = load_data(CSV_PATH)
    model, X_val, y_val, X_test, y_test = train_xgboost(X, y, groups)

    # Reconstruct X_train for the distance-only baseline.
    train_idx = ~X.index.isin(X_test.index) & ~X.index.isin(X_val.index)
    X_train, y_train = X[train_idx], y[train_idx]

    probs_test = model.predict_proba(X_test)[:, 1]
    brier = brier_score_loss(y_test, probs_test)

    print("\n[2/6] Headline metrics vs baselines...")
    headline = headline_metrics_table(model, X_train, y_train, X_test, y_test)
    print(headline.to_string(index=False))
    write_table(headline, RES_DIR / 'headline_metrics')

    print("\n[3/6] Per-zone metrics...")
    zone_df = per_zone_metrics(model, X_test, y_test)
    print(zone_df.to_string(index=False))
    write_table(zone_df, RES_DIR / 'per_zone_metrics')

    print("\n[4/6] Calibration / feature importance / probability histogram...")
    plot_calibration(y_test, probs_test, brier, FIG_DIR / 'calibration_diagram.png')
    plot_feature_importance_top20(model, feature_cols, FIG_DIR / 'feature_importance_top20.png')
    plot_probability_distribution(y_test, probs_test, FIG_DIR / 'probability_distribution.png')
    print(f"  Saved 3 figures into {FIG_DIR}/")

    print("\n[5/6] POE pipeline (out-of-fold)...")
    eval_df, leaderboard = compute_poe(CSV_PATH, n_folds=5, min_shots=100, out_dir=str(RES_DIR))

    # Save a slimmer top10/bottom10 view alongside the full leaderboard.
    top10 = leaderboard.head(10).assign(rank_type='top')
    bot10 = leaderboard.tail(10).iloc[::-1].assign(rank_type='bottom')
    poe_summary = pd.concat([top10, bot10], ignore_index=True)
    write_table(poe_summary, RES_DIR / 'poe_top_bottom_10')

    print("\n[6/6] Case-study spatial charts...")
    for player in CASE_STUDY_PLAYERS:
        out = FIG_DIR / f"poe_spatial_{player.replace(' ', '_')}.png"
        plot_poe_spatial(eval_df, player, out)
        print(f"  Saved {out}")

    print("\nAll thesis artifacts generated:")
    print(f"  Figures → {FIG_DIR.resolve()}")
    print(f"  Results → {RES_DIR.resolve()}")


if __name__ == '__main__':
    main()
