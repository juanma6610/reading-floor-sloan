"""
plot_leakage_abstract.py — print-sized leakage figure for the Sloan abstract.

Three panels, one per sentence of the abstract's privacy claim: what an honest-but-curious
server reconstructs of each team's shooting, region by region, from what it receives.

  1. Tree bagging, tree as sent            -> exact (leakage_attack.py, plain)
  2. Tree bagging + DP noise on leaf values -> still exact (structural attack, ε = 1)
  3. Histogram aggregation + SecAgg+        -> only league sums (leakage_attack_hist.py)

Beside each panel, the player silhouette (image.png) fades as protection is added
(VISIBILITY, a visual cue only); in panel 3 it is also blurred.

Reads results/federated/leakage_attack_leaves.csv and leakage_attack_hist_regions.csv.

Run from the project root:  python src/federated/plot_leakage_abstract.py
Writes: assets/leakage_abstract.pdf (for LaTeX) and assets/leakage_abstract.png (preview)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from plot_leakage_silhouette import player_cutout

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FED = ROOT / "results" / "federated"
ASSETS = ROOT / "assets"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
EXPOSED, SAFE = "#d9403f", "#1a9e6f"
MIN_SHOTS = 10          # regions smaller than this have an unstable true rate
TEXTWIDTH_IN = 6.77     # abstract.tex: a4paper, margin 1.9cm
VISIBILITY = (1.0, 0.6, 0.12)   # silhouette opacity per panel (a visual cue, not a measured value)


def load():
    tree = pd.read_csv(RESULTS_FED / "leakage_attack_leaves.csv")
    hist = pd.read_csv(RESULTS_FED / "leakage_attack_hist_regions.csv")
    hist = hist.assign(rate_hat=hist["makes_hat"] / hist["n_hat"])
    panels = [
        (tree[tree["scenario"] == "plain, subsample=1.0"], EXPOSED,
         "Tree bagging", "tree sent as is"),
        (tree[tree["scenario"] == "DP eps=1, subsample=1.0"], EXPOSED,
         "+ DP noise on leaves", "ε = 1, noise on leaf values only"),
        (hist[hist["scenario"] == "histogram + SecAgg+"], SAFE,
         "Histogram aggregation", "+ secure aggregation"),
    ]
    out = []
    for d, *rest in panels:
        d = d[d["n_true"] >= MIN_SHOTS]
        true = 100 * d["makes_true"] / d["n_true"]
        est = 100 * d["rate_hat"].clip(0, 1)
        exact = (d["n_hat"].round() == d["n_true"]) & (d["makes_hat"].round() == d["makes_true"])
        out.append((true.to_numpy(), est.to_numpy(), exact.mean(), *rest))
    return out


def main():
    plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "pdf.fonttype": 42})
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 2.45))
    gs = fig.add_gridspec(1, 8, width_ratios=[1, 0.34, 0.16, 1, 0.34, 0.16, 1, 0.34],
                          left=0.075, right=0.995, top=0.78, bottom=0.16, wspace=0.05)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 3, 6)]
    players = [fig.add_subplot(gs[0, i]) for i in (1, 4, 7)]
    cutout, _ = player_cutout()
    rng = np.random.default_rng(0)
    for ax, pax, vis, (true, est, exact, colour, title, sub) in zip(axes, players, VISIBILITY, load()):
        _player(pax, cutout, visible=vis, blur=exact < 0.5)
        if len(true) > 3000:                     # histogram cells: thin for legibility
            keep = rng.choice(len(true), 3000, replace=False)
            true_s, est_s = true[keep], est[keep]
        else:
            true_s, est_s = true, est
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.scatter(true_s, est_s, s=5, color=colour, alpha=0.45, lw=0, zorder=2, rasterized=True)
        ax.text(0, 1.20, title, transform=ax.transAxes, fontsize=8.5, color=INK,
                fontweight="bold", va="bottom")
        ax.text(0, 1.105, sub, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")
        verdict = (f"{exact:.0%} of regions exact" if exact > 0.5
                   else "sees league totals only")
        ax.text(0, 1.02, verdict, transform=ax.transAxes, fontsize=7.5, color=colour,
                fontweight="bold", va="bottom")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.set_xlabel("Team's true FG% in region", color=INK2)
        ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, length=2, pad=1.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Server's estimate (FG%)", color=INK2)
    ASSETS.mkdir(exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(ASSETS / f"leakage_abstract.{ext}", facecolor="white", **kw)
        print(f"Wrote {ASSETS / f'leakage_abstract.{ext}'}")
    plt.close(fig)


def _player(ax, cutout, visible, blur):
    """Silhouette beside a panel at the given opacity, blurred if the data is hidden."""
    cutout = np.pad(cutout, 40)                            # room for the blur to spread
    alpha = gaussian_filter(cutout, 14) if blur else cutout
    rgba = np.zeros(cutout.shape + (4,)); rgba[..., 3] = alpha * visible
    ax.imshow(rgba, interpolation="bilinear")
    ax.set_anchor("C"); ax.axis("off")


if __name__ == "__main__":
    main()
