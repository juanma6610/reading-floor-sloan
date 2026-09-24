"""
Per-zone POE decomposition figure for the Results section.

For each qualifying player, splits their total Points Over Expectation into
contributions from four court zones (Restricted Area / Paint non-RA /
Mid-range / 3-pointer). Same zone definitions as Section 4.5.

The figure is a diverging stacked horizontal bar chart: each player's row
shows their per-zone POE contributions as colored segments. Positive segments
extend right of zero, negative left. A black tick marks total POE. The shape
of the row reveals where each player creates or sheds expected points.

Reuses the out-of-fold predictions written by compute_poe.py, so no
additional model training is required.

Run from project root:
    python src/plot_per_zone_poe.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ------------------------------------------------------------
# Style
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

POE_PATH       = Path('results/poe_per_shot.csv')
FIG_OUT        = Path('figures/per_zone_poe.png')
TABLE_OUT      = Path('results/per_zone_poe.csv')
MIN_SHOTS      = 150     # tighter than the overall 100 because we're slicing 4 ways
TOP_N          = 12
BOTTOM_N       = 8

# Distance cutoffs match Section 4.5 of the thesis.
ZONES = ['Restricted area', 'Paint (non-RA)', 'Mid-range', '3-pointer']
ZONE_COLORS = {
    'Restricted area': '#1f4e79',   # dark blue   — closest to basket
    'Paint (non-RA)':  '#5b9bd5',   # mid blue
    'Mid-range':       '#f4b183',   # warm tan
    '3-pointer':       '#c0504d',   # red         — furthest from basket
}


def assign_zone(dist: float) -> str:
    if dist <= 4.0:
        return 'Restricted area'
    if dist <= 14.0:
        return 'Paint (non-RA)'
    if dist < 22.0:
        return 'Mid-range'
    return '3-pointer'


# ============================================================
# 1. Per-(player, zone) aggregation
# ============================================================

def build_per_zone_table(poe_path=POE_PATH, min_shots=MIN_SHOTS):
    poe = pd.read_csv(poe_path)
    poe['zone'] = poe['dist'].apply(assign_zone)

    # POE per (player, zone)
    grid = (poe.groupby(['player_name', 'zone'])['POE']
                .sum()
                .unstack(fill_value=0.0)
                .reindex(columns=ZONES, fill_value=0.0))
    # Shot counts per (player, zone) for the table; total volume for the filter.
    n_grid = (poe.groupby(['player_name', 'zone']).size()
                .unstack(fill_value=0)
                .reindex(columns=ZONES, fill_value=0))

    grid['total_POE']    = grid[ZONES].sum(axis=1)
    grid['total_shots']  = n_grid.sum(axis=1)
    for z in ZONES:
        grid[f'n_{z}']   = n_grid[z]

    grid = grid[grid['total_shots'] >= min_shots].copy()
    grid = grid.sort_values('total_POE', ascending=False)
    return grid


# ============================================================
# 2. Plot
# ============================================================

def plot_per_zone_heatmap(grid, top_n=TOP_N, bottom_n=BOTTOM_N, out_path=FIG_OUT):
    """
    Player x Zone heatmap of POE contributions, with a side panel showing
    total POE per player. Diverging RdBu colormap centered on zero:
    blue cell = positive POE in that zone, red cell = negative.

    The heatmap is the decomposition; the side bars give the total so the
    "same total, different profile" reading is immediate.
    """
    top = grid.head(top_n)
    bot = grid.tail(bottom_n)
    sub = pd.concat([top, bot])
    # Most positive at the top, most negative at the bottom of the chart.
    sub = sub.sort_values('total_POE', ascending=False)

    matrix = sub[ZONES].to_numpy()
    totals = sub['total_POE'].to_numpy()

    # Symmetric colour limits so zero sits at the centre of the colormap;
    # 98th percentile so one outlier shot can't blow out the scale.
    vmax = float(np.nanpercentile(np.abs(matrix), 98))
    vmax = max(vmax, 5.0)

    fig, (ax_main, ax_total) = plt.subplots(
        1, 2,
        figsize=(12, max(6, len(sub) * 0.42)),
        gridspec_kw={'width_ratios': [3.5, 1.0], 'wspace': 0.04},
    )

    # ---- Main heatmap: 4 zones across the columns ----
    im = ax_main.imshow(matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if pd.notna(v):
                # Light text on saturated cells, dark text near the centre.
                text_color = 'white' if abs(v) > vmax * 0.55 else 'black'
                ax_main.text(j, i, f'{v:+.1f}',
                             ha='center', va='center',
                             fontsize=10, color=text_color)

    ax_main.set_xticks(range(len(ZONES)))
    ax_main.set_xticklabels(ZONES, rotation=18, ha='right')
    ax_main.set_yticks(range(len(sub)))
    ax_main.set_yticklabels(sub.index)
    ax_main.set_title(
        f'Per-zone POE decomposition — top {top_n} + bottom {bottom_n} '
        f'by total POE (min {MIN_SHOTS} shots, OOF predictions)'
    )
    # Thin white grid between cells for readability.
    ax_main.set_xticks(np.arange(-0.5, len(ZONES), 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, len(sub), 1), minor=True)
    ax_main.grid(which='minor', color='white', linewidth=1.5)
    ax_main.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.01)
    cbar.set_label('POE in zone')

    # ---- Side panel: total POE bars (mirror the row order of the heatmap) ----
    bar_colors = ['#1f4e79' if t >= 0 else '#c0504d' for t in totals]
    ax_total.barh(range(len(sub)), totals,
                  color=bar_colors, edgecolor='black', linewidth=0.4)
    ax_total.invert_yaxis()  # imshow uses top-down rows; mirror it here
    ax_total.axvline(0, color='black', lw=0.6)
    ax_total.set_yticks([])
    ax_total.set_xlabel('Total POE')
    ax_total.set_title('Total')
    for spine in ('top', 'right'):
        ax_total.spines[spine].set_visible(False)

    # Numeric labels next to each bar.
    x_lo, x_hi = ax_total.get_xlim()
    span = x_hi - x_lo
    pad = span * 0.02
    for i, t in enumerate(totals):
        ax_total.text(t + (pad if t >= 0 else -pad), i,
                      f'{t:+.1f}',
                      ha='left' if t >= 0 else 'right',
                      va='center', fontsize=9)
    ax_total.set_xlim(x_lo - 0.18 * span, x_hi + 0.18 * span)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved figure -> {out_path}")


# ============================================================
# 3. Entry point
# ============================================================

def main():
    print(f"Loading {POE_PATH}...")
    grid = build_per_zone_table()
    print(f"Qualifying players (>= {MIN_SHOTS} shots): {len(grid)}")

    # Persist the full per-player per-zone table — useful as an appendix.
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.round(2).to_csv(TABLE_OUT)
    print(f"Wrote full per-zone table -> {TABLE_OUT}")

    # The figure.
    plot_per_zone_heatmap(grid)

    # Console summary: the same top/bottom slice that appears in the figure.
    print("\nTop 5 by total POE (per-zone breakdown):")
    print(grid[ZONES + ['total_POE', 'total_shots']].head(5).round(2).to_string())
    print("\nBottom 5 by total POE (per-zone breakdown):")
    print(grid[ZONES + ['total_POE', 'total_shots']].tail(5).round(2).to_string())


if __name__ == '__main__':
    main()
