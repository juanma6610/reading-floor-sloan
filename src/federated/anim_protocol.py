"""
anim_protocol.py — hero animation of the histogram protocol, for the README / slides.

Four beats, looped:
  1. every team bins its own shots locally            (real per-team distance histograms)
  2. each team sends a MASKED histogram               (on its own, indistinguishable from noise)
  3. the masks cancel in the sum                      (only the league total is ever revealed)
  4. each team's 1/sqrt(30) noise share adds up       (distributed Gaussian DP, ε = 1)
  5. one shared tree grows from that sum              (Brier 0.2047 vs 0.2050 centralized)

Run from the project root:  python src/federated/anim_protocol.py
Writes: assets/hero_privacy.gif and assets/hero_privacy.png (still of the last beat)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nba_federated import task                                  # noqa: E402

ROOT = SCRIPT_DIR.parents[1]
ASSETS = ROOT / "assets"

BG, PANEL, EDGE = "#0b1220", "#161f33", "#2b3a55"
TEXT, MUTED = "#eaf0f9", "#93a3ba"
BLUE, GREEN, RED = "#6fb7ff", "#35d39a", "#ff7a7a"

NBINS = 10                       # distance bins, 0–32 ft
FPS, N_FRAMES = 14, 100
BEATS = (0, 16, 40, 56, 72, 100)  # bin | masked send | sum | DP noise | tree grows

# layout, in axes units (0–100 x, 0–32 y)
GRID_X0, GRID_Y0, GRID_DX, GRID_DY = 3.5, 4.6, 4.2, 4.0        # 6 x 5 team silos
SILO_W, SILO_H = 3.4, 3.2
SERVER_X, SERVER_Y, SERVER_W, SERVER_H = 38.0, 6.0, 20.0, 17.5
TREE_X, TREE_Y = 79.0, 21.0


def team_histograms():
    """Per-team shot-distance histograms (training pool), normalised, plus the league sum."""
    df = task.load_full_dataset()
    train_idx, _ = task._get_global_split()
    pool = df.iloc[train_idx]
    rows = []
    for team_id in sorted(pool["team_id"].unique()):
        h = np.histogram(pool.loc[pool["team_id"] == team_id, "dist"].clip(0, 32),
                         bins=NBINS, range=(0, 32))[0].astype(float)
        rows.append(h)
    hists = np.array(rows)
    return hists / hists.max(axis=1, keepdims=True), hists.sum(0) / hists.sum(0).max()


def ease(t):
    return t * t * (3 - 2 * t)


def bars(ax, x, y, w, h, vals, colour, alpha=1.0, lw=0.0):
    """Tiny bar glyph anchored at (x, y), width w, height h."""
    bw = 0.72 * w / len(vals)
    out = []
    for i, v in enumerate(vals):
        out.append(ax.add_patch(plt.Rectangle((x + i * w / len(vals), y), bw, max(h * v, h * 0.06),
                                              color=colour, alpha=alpha, lw=lw, zorder=4)))
    return out


class Scene:
    def __init__(self, hists, league):
        self.hists, self.league = hists, league
        self.rng = np.random.default_rng(7)
        self.fig, self.ax = plt.subplots(figsize=(9.6, 3.2), dpi=100)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 100); self.ax.set_ylim(0, 32); self.ax.axis("off")
        self.fig.patch.set_facecolor(BG); self.ax.set_facecolor(BG)
        self.dp = np.clip(league + self.rng.normal(0, 0.07, NBINS), 0.03, 1.1)   # ε = 1 share
        self.silos = [(GRID_X0 + c * GRID_DX, GRID_Y0 + r * GRID_DY)
                      for r in range(5) for c in range(6)]
        self.teams = task.team_abbrs()

    # ── static furniture ────────────────────────────────────────────
    def board(self, x, y, w, h, ec=EDGE, fc=PANEL, lw=1.0, alpha=1.0):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                         fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=2))

    def draw(self, frame):
        ax = self.ax
        ax.clear(); ax.set_xlim(0, 100); ax.set_ylim(0, 32); ax.axis("off")
        ax.set_facecolor(BG)
        beat = int(np.searchsorted(BEATS, frame, side="right")) - 1
        span = BEATS[beat + 1] - BEATS[beat]
        t = ease(min((frame - BEATS[beat]) / max(span - 6, 1), 1.0))

        ax.text(3.5, 29.6, "Histogram aggregation under SecAgg+", color=TEXT, fontsize=14,
                fontweight="bold", va="center")
        captions = ["Every team bins its own shots. Nothing leaves yet.",
                    "Each team sends a masked histogram — alone, it is noise.",
                    "The masks cancel: only the league total is revealed.",
                    "Every team also adds its 1/\u221a30 share of Gaussian noise: ε = 1 for every shot.",
                    "One shared tree grows from the noisy sum — every shot covered at ε = 1."]
        ax.text(3.5, 26.8, captions[beat], color=MUTED, fontsize=10.5, va="center")

        # ── the 30 teams ────────────────────────────────────────────
        for i, (x, y) in enumerate(self.silos):
            self.board(x, y, SILO_W, SILO_H, alpha=0.95)
            grown = ease(np.clip((frame - BEATS[0] - i * 0.28) / 7, 0, 1)) if beat == 0 else 1.0
            bars(ax, x + 0.45, y + 0.5, SILO_W - 0.9, (SILO_H - 1.1) * grown, self.hists[i],
                 BLUE, alpha=0.9)
        ax.text(GRID_X0, 2.4, "30 franchises · raw shots stay home", color=MUTED, fontsize=9.5)

        # ── packets in flight (beat 1) ──────────────────────────────
        if beat == 1:
            for i, (x, y) in enumerate(self.silos):
                p = np.clip(t * 1.8 - 0.55 * i / len(self.silos), 0, 1)
                if p <= 0:
                    continue
                tx = SERVER_X - 1.0
                ty = SERVER_Y + 3.0 + (i % 6) * (SERVER_H - 6.0) / 5
                px = x + SILO_W / 2 + (tx - x - SILO_W / 2) * p
                py = y + SILO_H / 2 + (ty - y - SILO_H / 2) * p
                ax.plot([x + SILO_W / 2, px], [y + SILO_H / 2, py], color=BLUE, lw=0.5,
                        alpha=0.18 * (1 - p), zorder=1)
                noise = self.rng.random(NBINS)
                fade = np.clip((1 - p) / 0.2, 0, 1)            # absorbed by the server
                bars(ax, px - 0.9, py - 0.6, 1.8, 1.5, noise, RED, alpha=0.9 * fade)
            ax.text(SERVER_X - 10.5, 3.4, "masked share + noise share", color=RED, fontsize=9.5)

        # ── the server ──────────────────────────────────────────────
        self.board(SERVER_X, SERVER_Y, SERVER_W, SERVER_H, ec=EDGE)
        ax.text(SERVER_X + SERVER_W / 2, SERVER_Y + SERVER_H - 1.6, "coordinating server",
                color=MUTED, fontsize=9.5, ha="center", va="center")
        if beat == 0:
            shown, colour, alpha = np.zeros(NBINS), RED, 0.0   # nothing has arrived yet
        elif beat == 1:
            shown, colour, alpha = self.rng.random(NBINS), RED, 0.35 + 0.55 * t
        elif beat == 2:
            mix = t
            shown = (1 - mix) * self.rng.random(NBINS) + mix * self.league
            colour, alpha = (GREEN if mix > 0.6 else RED), 0.95
        else:
            mix = t if beat == 3 else 1.0                      # DP noise fades in, then stays
            shown = (1 - mix) * self.league + mix * self.dp
            colour, alpha = GREEN, 0.95
        bars(ax, SERVER_X + 2.0, SERVER_Y + 2.4, SERVER_W - 4.0, SERVER_H - 6.0, shown,
             colour, alpha=alpha)
        if beat >= 2:
            note = "league totals only" if beat == 2 else "league totals + DP noise  (ε = 1)"
            ax.text(SERVER_X + SERVER_W / 2, SERVER_Y + 0.9, note,
                    color=GREEN, fontsize=9, ha="center", va="center")

        # ── the shared tree ─────────────────────────────────────────
        if beat >= 4:
            self.tree(t)
        footer = ("Brier 0.2047 vs 0.2050 centralized" if beat < 3 else
                  "Brier 0.2047 no DP  ·  0.2183 at ε = 1  ·  centralized 0.2050")
        ax.text(96.5, 2.4, footer, color=MUTED, fontsize=9, ha="right")

    def tree(self, t):
        ax = self.ax
        levels = [[0.0], [-7.0, 7.0], [-10.5, -3.5, 3.5, 10.5]]
        for d, xs in enumerate(levels):
            vis = np.clip(t * 3 - d, 0, 1)
            if vis <= 0:
                continue
            y = TREE_Y - d * 6.3
            for k, dx in enumerate(xs):
                x = TREE_X + dx
                if d:
                    px = TREE_X + levels[d - 1][k // 2]
                    ax.plot([px, x], [y + 6.3, y], color=EDGE, lw=1.2, alpha=vis, zorder=2)
                ax.add_patch(plt.Circle((x, y), 1.3, fc=GREEN if d else BLUE, ec=BG, lw=1.2,
                                        alpha=vis, zorder=3))
        ax.text(TREE_X, TREE_Y + 4.4, "shared tree", color=MUTED, fontsize=9.5, ha="center")


def main():
    hists, league = team_histograms()
    scene = Scene(hists, league)
    anim = FuncAnimation(scene.fig, scene.draw, frames=N_FRAMES, interval=1000 / FPS)
    gif = ASSETS / "hero_privacy.gif"
    anim.save(gif, writer=PillowWriter(fps=FPS), savefig_kwargs={"facecolor": BG})
    print(f"Wrote {gif}")
    scene.draw(N_FRAMES - 1)
    still = ASSETS / "hero_privacy.png"
    scene.fig.savefig(still, facecolor=BG, dpi=200)
    print(f"Wrote {still}")
    plt.close(scene.fig)


if __name__ == "__main__":
    main()
