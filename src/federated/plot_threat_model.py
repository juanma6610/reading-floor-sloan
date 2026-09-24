"""
plot_threat_model.py — data-flow + threat diagram for the paper's threat-model section.

One round of the histogram protocol, drawn the way a threat model is usually drawn:
trust boundaries (dashed), processes (P), data stores (D), numbered data flows (F) and
the threats (T) that sit on them, coloured by status. The numbering matches the table in
docs/threat_model.md.

  boundary 1  the team silo (×30)   raw shots never cross it
  boundary 2  secure aggregation    only the sum of the 30 vectors is ever revealed
  boundary 3  the league server     honest but curious; everything here is public to all teams

Run from the project root:  python src/federated/plot_threat_model.py
Writes: results/federated/threat_model.png (and assets/threat_model.png)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
RESULTS_FED, ASSETS = ROOT / "results" / "federated", ROOT / "assets"

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
TEAM, SEC, SERVER = "#2a78d6", "#1baf7a", "#3f3f3d"
OK, PARTIAL, OPEN = "#1baf7a", "#e0a106", "#e34948"       # threat status
FILL = {TEAM: "#eaf2fc", SEC: "#e8f7f0", SERVER: "#f0efec"}


def box(ax, x, y, w, h, ec, title, lines=(), fc=None, lw=1.3, fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.015",
                                fc=fc or FILL[ec], ec=ec, lw=lw, zorder=2))
    ax.text(x + 0.012, y + h - 0.045, title, fontsize=fs + 0.6, color=INK, fontweight="bold",
            zorder=3)
    for i, line in enumerate(lines):
        ax.text(x + 0.012, y + h - 0.092 - i * 0.043, line, fontsize=fs, color=INK2, zorder=3)


def boundary(ax, x, y, w, h, colour, label):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                fc="none", ec=colour, lw=1.3, ls=(0, (5, 4)), zorder=1))
    ax.text(x + 0.008, y + h + 0.022, label, fontsize=8.4, color=colour, style="italic", zorder=3)


def arrow(ax, p, q, colour=INK2, lw=1.5, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, color=colour, lw=lw,
                                 ls=ls, shrinkA=2, shrinkB=2, zorder=3,
                                 connectionstyle=f"arc3,rad={rad}"))


def flow(ax, x, y, text, colour=INK2):
    ax.text(x, y, text, fontsize=7.8, color=colour, ha="center", va="center", zorder=4,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none"))


def threat(ax, x, y, label, status):
    ax.plot(x, y, marker="o", ms=15, mfc=status, mec=SURFACE, mew=1.2, zorder=5)
    ax.text(x, y, label, fontsize=7.4, color="white", fontweight="bold",
            ha="center", va="center", zorder=6)


def main():
    fig, ax = plt.subplots(figsize=(12.6, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.004, 0.965, "Threat model: one round of the histogram protocol",
            fontsize=12, color=INK, fontweight="bold")
    ax.text(0.004, 0.925, "Trust boundaries are dashed. Threats T1–T10 are listed in "
            "docs/threat_model.md; colour is their status.", fontsize=8.6, color=INK2)

    # ── boundary 1: the team silo ────────────────────────────────────────────
    boundary(ax, 0.012, 0.215, 0.275, 0.585, TEAM, "Trust boundary 1 — team silo (×30)")
    box(ax, 0.028, 0.655, 0.243, 0.115, TEAM, "D1  raw tracking & shots",
        ["never crosses the boundary"])
    box(ax, 0.028, 0.505, 0.243, 0.115, TEAM, "P1  bin on the public grid",
        ["per-bin gradient sums (G, H)"])
    box(ax, 0.028, 0.355, 0.243, 0.115, TEAM, "P2  clip per player, add noise",
        ["N(0, σ²/(K−c)) — this team's share"])
    box(ax, 0.028, 0.240, 0.243, 0.080, TEAM, "P3  mask the vector", [])
    for y0, y1 in ((0.655, 0.620), (0.505, 0.470), (0.355, 0.320)):
        arrow(ax, (0.150, y0), (0.150, y1), TEAM)
    threat(ax, 0.283, 0.562, "9", PARTIAL)      # public grid set after inspecting data
    threat(ax, 0.283, 0.412, "5", OK)           # collusion / dropout budget
    threat(ax, 0.283, 0.262, "8", OPEN)         # malicious client contributions

    # ── boundary 2: secure aggregation ───────────────────────────────────────
    boundary(ax, 0.345, 0.300, 0.150, 0.420, SEC, "Trust boundary 2 — SecAgg+")
    box(ax, 0.357, 0.330, 0.127, 0.360, SEC, "P4  aggregate",
        ["masks cancel", "", "Σ over the 30 teams", "+ full noise σ", "", "no single team's",
         "vector is revealed"])
    arrow(ax, (0.288, 0.280), (0.352, 0.430), TEAM, rad=-0.12)
    flow(ax, 0.318, 0.235, "F2  ×30 masked, noisy histograms")
    for by, label in ((0.455, "3"), (0.395, "6"), (0.335, "7")):
        threat(ax, 0.520, by, label, OK)        # no SecAgg → per-team vector, ε inflation, lattice

    # ── boundary 3: the server / public zone ─────────────────────────────────
    boundary(ax, 0.555, 0.215, 0.432, 0.505, SERVER,
             "Trust boundary 3 — league server, honest but curious (everything here is public "
             "to all 30 teams)")
    box(ax, 0.570, 0.560, 0.200, 0.130, SERVER, "P5  pick the split",
        ["on the public grid, from Σ"])
    box(ax, 0.570, 0.390, 0.200, 0.130, SERVER, "P6  extend the shared tree",
        ["one tree, all 30 teams"])
    box(ax, 0.570, 0.232, 0.200, 0.120, SERVER, "D2  public state",
        ["tree, split points, σ"])
    box(ax, 0.800, 0.390, 0.175, 0.300, SERVER, "Released model",
        ["every team receives it,", "rivals included", "", "whatever it memorises", "is available to all"],
        fc="#faf4f4", lw=1.3)
    arrow(ax, (0.497, 0.510), (0.566, 0.625), SEC, rad=-0.1)
    flow(ax, 0.527, 0.585, "F3  Σ only")
    arrow(ax, (0.670, 0.560), (0.670, 0.525), SERVER)
    arrow(ax, (0.670, 0.390), (0.670, 0.355), SERVER)
    arrow(ax, (0.775, 0.470), (0.796, 0.470), SERVER)
    threat(ax, 0.887, 0.370, "4", OK)           # membership inference on the released model
    threat(ax, 0.960, 0.700, "10", PARTIAL)     # league structure revealed by design

    # broadcast back to every team (F4)
    arrow(ax, (0.660, 0.226), (0.150, 0.178), SERVER, ls=(0, (5, 3)), rad=0.06)
    flow(ax, 0.420, 0.172, "F4  broadcast to all 30 teams: new tree, next split points, noise "
                           "scale — all public", SERVER)
    arrow(ax, (0.150, 0.175), (0.150, 0.235), SERVER)

    # ── the counterfactual: sending the tree itself ──────────────────────────
    box(ax, 0.012, 0.012, 0.560, 0.140, OPEN, "If a team sent its own tree instead (tree bagging)",
        ["the server recovers that team's exact shot and make counts, region by region,",
         "and noise on the leaf values does not stop it"], fc="#fdf0ef", lw=1.3)
    threat(ax, 0.500, 0.112, "1", OPEN)
    threat(ax, 0.537, 0.112, "2", OPEN)

    # ── legend ───────────────────────────────────────────────────────────────
    handles = [Line2D([], [], ls=(0, (5, 4)), color=INK2, label="trust boundary"),
               Line2D([], [], color=INK2, label="data flow (F)"),
               Line2D([], [], marker="o", ls="", mfc=OK, mec=SURFACE, ms=9,
                      label="threat: mitigated"),
               Line2D([], [], marker="o", ls="", mfc=PARTIAL, mec=SURFACE, ms=9,
                      label="threat: residual"),
               Line2D([], [], marker="o", ls="", mfc=OPEN, mec=SURFACE, ms=9,
                      label="threat: out of scope / protocol we reject")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2, ncol=2,
              labelcolor=INK2, handletextpad=0.6, columnspacing=1.8,
              bbox_to_anchor=(1.0, 0.01))

    RESULTS_FED.mkdir(parents=True, exist_ok=True)
    out = RESULTS_FED / "threat_model.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight"); print(f"Wrote {out}")
    fig.savefig(ASSETS / "threat_model.png", facecolor=SURFACE, bbox_inches="tight")
    print(f"Wrote {ASSETS / 'threat_model.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
