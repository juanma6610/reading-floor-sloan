"""
plot_threat_model.py — one round of the histogram protocol, and who sees what.

Draws the data flow for the paper's threat-model section: shots stay on each team,
only noisy binned sums leave, secure aggregation exposes just their total, and the
server grows one shared tree that it broadcasts back.

Run from the project root:  python src/federated/plot_threat_model.py
Writes: results/federated/threat_model.png (copy to assets/ for the README/paper)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RESULTS_FED = Path(__file__).resolve().parents[2] / "results" / "federated"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
TEAM, SEC, SERVER, LEAK = "#2a78d6", "#1baf7a", "#0b0b0b", "#e34948"


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.02, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p, q, colour=INK2, lw=1.6, style="-|>", ls="-", z=3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=colour,
                                 lw=lw, ls=ls, shrinkA=2, shrinkB=2, zorder=z))


def main():
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── teams ────────────────────────────────────────────────────────────────
    ax.text(0.005, 0.95, "Each of the 30 teams", fontsize=10, color=INK, fontweight="bold")
    rows = [(0.70, "Team 1"), (0.44, "Team 2"), (0.06, "Team 30")]
    for y, name in rows:
        box(ax, 0.005, y, 0.30, 0.20, "#eaf2fc", TEAM)
        ax.text(0.02, y + 0.145, name, fontsize=9.5, color=INK, fontweight="bold")
        ax.text(0.02, y + 0.10, "raw shots never leave", fontsize=8.5, color=LEAK, style="italic")
        ax.text(0.02, y + 0.055, "bin on the public grid → per-bin (G, H)", fontsize=8.5, color=INK2)
        ax.text(0.02, y + 0.015, "clip per player · add N(0, σ²/(K−c)) · mask", fontsize=8.5, color=INK2)
        arrow(ax, (0.315, y + 0.10), (0.40, 0.52), TEAM)
    ax.text(0.08, 0.335, "⋮", fontsize=16, color=INK2)

    # ── secure aggregation ───────────────────────────────────────────────────
    box(ax, 0.40, 0.24, 0.14, 0.56, "#e8f7f0", SEC)
    ax.text(0.47, 0.72, "SecAgg+", fontsize=10, color=INK, fontweight="bold", ha="center")
    ax.text(0.47, 0.60, "masks cancel\nonly the sum\nis revealed", fontsize=8.5, color=INK2,
            ha="center", linespacing=1.5)
    ax.text(0.47, 0.42, "Σ over teams\n+ noise σ", fontsize=9, color=SEC, ha="center",
            fontweight="bold", linespacing=1.4)
    arrow(ax, (0.545, 0.52), (0.635, 0.52), SEC)

    # ── server ───────────────────────────────────────────────────────────────
    box(ax, 0.64, 0.30, 0.245, 0.44, "#f2f2f0", SERVER)
    ax.text(0.7625, 0.665, "Server (league office)", fontsize=10, color=INK,
            fontweight="bold", ha="center")
    ax.text(0.655, 0.575, "honest but curious", fontsize=8.5, color=LEAK, style="italic")
    for i, step in enumerate(["1. add the 30 histograms",
                              "2. pick each split on the public grid",
                              "3. extend the one shared tree"]):
        ax.text(0.655, 0.495 - i * 0.055, step, fontsize=8.5, color=INK2)
    ax.text(0.655, 0.33, "never sees a single team's vector", fontsize=8.5, color=SEC)

    # ── broadcast loop ───────────────────────────────────────────────────────
    arrow(ax, (0.7625, 0.30), (0.7625, 0.20), INK2)
    box(ax, 0.30, 0.005, 0.59, 0.15, "#fdf6e3", "#eda100", lw=1.2)
    ax.text(0.585, 0.105, "Broadcast back to every team: the new tree, the next split points, "
            "the noise scale", fontsize=8.5, color=INK2, ha="center")
    ax.text(0.585, 0.045, "all public — one round per tree level, repeated until the model is grown",
            fontsize=8.5, color=INK2, ha="center", style="italic")
    arrow(ax, (0.30, 0.08), (0.17, 0.06), INK2, ls=(0, (4, 2)))

    # ── what an adversary would get without each layer ───────────────────────
    ax.text(0.40, 0.955, "Remove secure aggregation and the server reads each team's own\n"
            "histogram: exact shots and makes, region by region. Remove the\n"
            "noise and the released model still memorises its training shots.",
            fontsize=8.5, color=LEAK, va="top", linespacing=1.5)

    fig.suptitle("One round of the histogram protocol: what leaves a team, and who sees it",
                 x=0.005, y=0.995, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    out = RESULTS_FED / "threat_model.png"
    fig.savefig(out, facecolor=SURFACE); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
