"""
Defender Points Over Expectation (DEF-POE) and Matchup Heatmap

Two products, both built on top of the per-shot OOF POE table from compute_poe.py:

  1. DEF-POE leaderboard. For each defender (the closest defender on the shot),
     sum (expected_points - actual_points) over their shots. Positive DEF-POE
     means the defender systematically suppresses outcomes below shot quality.
     Sign convention is the mirror of shooter POE.

  2. Archetype matchup heatmap. Mean POE binned by (offensive archetype,
     defensive archetype) using argmax over the cluster probability columns.
     Reveals which matchups systematically beat or fall short of their context.

The leaderboard requires `closest_def_name` in the raw CSV (added to the
extractor). The matchup heatmap only requires the cluster columns and works
on any CSV that already has them.

Run after compute_poe.py:
    python src/compute_def_poe.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

POE_PER_SHOT  = 'results/poe_per_shot.csv'
RAW_CSV       = 'data/shot_features_valid2_type.csv'
MIN_DEF_SHOTS = 100

N_OFF_CLUSTERS = 4
N_DEF_CLUSTERS = 4

FIG_DIR = Path('figures')
RES_DIR = Path('results')


# ============================================================
# 1. Join OOF POE with defender + archetype info
# ============================================================

def load_per_shot_with_extras(poe_path=POE_PER_SHOT, raw_path=RAW_CSV):
    poe = pd.read_csv(poe_path)
    raw = pd.read_csv(raw_path)

    if len(poe) != len(raw):
        raise ValueError(
            f"Row count mismatch: poe_per_shot has {len(poe)}, raw CSV has {len(raw)}. "
            "Re-run compute_poe.py against the current shot_features_valid2_type.csv."
        )

    if 'closest_def_name' in raw.columns:
        poe['closest_def_name'] = raw['closest_def_name'].values
    else:
        poe['closest_def_name'] = np.nan
        print("[note] 'closest_def_name' missing from raw CSV — DEF-POE leaderboard skipped.")
        print("       Re-extract features after the shot_features.py edit, then re-run compute_poe.py.")

    off_cols = ["Primary_Creator", "Spacer", "Mid-Interior", "Rim_Center"]
    def_cols = ["Paint_Anchors", "Perimeter Guards", "Def_liability", "Switch_Wing"]
    poe['off_archetype'] = (raw[off_cols].values.argmax(axis=1) + 1) if all(c in raw.columns for c in off_cols) else np.nan
    poe['def_archetype'] = (raw[def_cols].values.argmax(axis=1) + 1) if all(c in raw.columns for c in def_cols) else np.nan
    return poe


# ============================================================
# 2. DEF-POE leaderboard
# ============================================================

def def_poe_leaderboard(poe, min_shots=MIN_DEF_SHOTS):
    if poe['closest_def_name'].isna().all():
        return None
    df = poe.dropna(subset=['closest_def_name']).copy()

    # Defender perspective: positive = scored *less* than expected (good defense).
    df['DEF_POE_per_shot'] = df['expected_points'] - df['actual_points']

    agg = df.groupby('closest_def_name').agg(
        shots_defended=('DEF_POE_per_shot', 'count'),
        actual_pts_allowed=('actual_points', 'sum'),
        expected_pts_allowed=('expected_points', 'sum'),
        total_DEF_POE=('DEF_POE_per_shot', 'sum'),
    ).reset_index()

    agg = agg[agg['shots_defended'] >= min_shots].copy()
    agg['DEF_POE_per_100'] = (agg['total_DEF_POE'] / agg['shots_defended']) * 100

    for col in ['actual_pts_allowed', 'expected_pts_allowed', 'total_DEF_POE']:
        agg[col] = agg[col].round(1)
    agg['DEF_POE_per_100'] = agg['DEF_POE_per_100'].round(2)

    return agg.sort_values('total_DEF_POE', ascending=False).reset_index(drop=True)


# ============================================================
# 3. Archetype matchup heatmap
# ============================================================

def matchup_pivots(poe):
    if poe['off_archetype'].isna().all() or poe['def_archetype'].isna().all():
        return None
    grp = poe.groupby(['off_archetype', 'def_archetype'])['POE'].agg(['mean', 'count']).reset_index()
    pivot_mean  = grp.pivot(index='off_archetype', columns='def_archetype', values='mean')
    pivot_count = grp.pivot(index='off_archetype', columns='def_archetype', values='count')
    return pivot_mean, pivot_count


def plot_matchup_heatmap(pivot_mean, pivot_count, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    vmax = float(np.nanpercentile(np.abs(pivot_mean.values), 95))
    vmax = max(vmax, 0.05)
    im = ax.imshow(pivot_mean.values, cmap='coolwarm', vmin=-vmax, vmax=vmax, origin='lower')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Mean shooter POE per shot')

    for i, _ in enumerate(pivot_mean.index):
        for j, _ in enumerate(pivot_mean.columns):
            v = pivot_mean.iloc[i, j]
            n = pivot_count.iloc[i, j]
            if pd.notna(v) and n > 0:
                ax.text(j, i, f"{v:+.2f}\n(n={int(n)})",
                        ha='center', va='center', fontsize=9,
                        color='black' if abs(v) < vmax * 0.6 else 'white')

    ax.set_xticks(range(len(pivot_mean.columns)))
    ax.set_yticks(range(len(pivot_mean.index)))
    ax.set_xticklabels([f'D{c}' for c in pivot_mean.columns])
    ax.set_yticklabels([f'O{r}' for r in pivot_mean.index])
    ax.set_xlabel('Defensive archetype')
    ax.set_ylabel('Offensive archetype')
    ax.set_title('Mean shooter POE by archetype matchup')
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ============================================================
# 4. Entry point
# ============================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading per-shot POE + extras...")
    poe = load_per_shot_with_extras()

    # --- DEF-POE leaderboard ---
    print("\n" + "=" * 60)
    print("DEFENDER POE LEADERBOARD")
    print("=" * 60)
    lb = def_poe_leaderboard(poe, min_shots=MIN_DEF_SHOTS)
    if lb is not None:
        cols = ['closest_def_name', 'shots_defended', 'actual_pts_allowed',
                'expected_pts_allowed', 'total_DEF_POE', 'DEF_POE_per_100']
        print(f"\nTop 10 defenders (most points suppressed below expectation):")
        print(lb[cols].head(10).to_string(index=False))
        print(f"\nBottom 10 defenders (most points conceded above expectation):")
        print(lb[cols].tail(10).iloc[::-1].to_string(index=False))
        lb.to_csv(RES_DIR / 'def_poe_leaderboard.csv', index=False)
        print(f"\nSaved DEF-POE leaderboard to {RES_DIR/'def_poe_leaderboard.csv'}")
    else:
        print("[skip] Need closest_def_name in the CSV. "
              "Edit was added to shot_features.py — re-run extraction first.")

    # --- Matchup heatmap (works on existing CSVs) ---
    print("\n" + "=" * 60)
    print("ARCHETYPE MATCHUP HEATMAP")
    print("=" * 60)
    pivots = matchup_pivots(poe)
    if pivots is not None:
        pm, pc = pivots
        print("\nMean POE per shot, rows=offense, cols=defense:")
        print(pm.round(3).to_string())
        print("\nShot counts per cell:")
        print(pc.astype('Int64').to_string())
        plot_matchup_heatmap(pm, pc, FIG_DIR / 'matchup_heatmap.png')
        pm.to_csv(RES_DIR / 'matchup_mean_poe.csv')
        pc.to_csv(RES_DIR / 'matchup_counts.csv')
        print(f"\nSaved heatmap to {FIG_DIR/'matchup_heatmap.png'}")
    else:
        print("[skip] Need Prob_Cluster_* and Def_Prob_Cluster_* columns.")


if __name__ == '__main__':
    main()
