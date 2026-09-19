"""
epv_report.py — the metrics table + figure for the EPV model comparison.

Reads what the training scripts already wrote (no retraining):
  results/epv/epv_baselines.json        B0/B1/B2/B3/full, 3 seeds, game-disjoint
  results/epv/epv_causal_sweep.json     the lambda / hidden / lr / refresh sweep
and writes:
  results/epv/epv_model_comparison.png  3-panel figure
  results/epv/epv_metrics_table.md      the table, ready to paste into docs

Colour: the validated reference palette (blue sequential for the single-measure
bars; categorical slots 1/2/3/7 for the multi-series panels — an order whose
adjacent pairs clear the CVD and normal-vision floors). Every series is also
direct-labelled, so identity never rests on hue alone.

Run: python src/epv/epv_report.py
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

OUT_DIR = "results/epv"
BASE_JSON = os.path.join(OUT_DIR, "epv_baselines.json")
SWEEP_JSON = os.path.join(OUT_DIR, "epv_causal_sweep.json")

# reference palette
INK, SEC, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURF = "#fcfcfb"
BLUE_LIGHT, BLUE, BLUE_DARK = "#86b6ef", "#2a78d6", "#1c5cab"
ORDER = ["B0 constant PPP", "B1 XGBoost value", "B2 GNN spatial-only",
         "B3 GNN temporal-only", "GNN-EPV causal (full)"]
# Colour follows the ENTITY, not its rank or its position in a filtered list:
# a model keeps the same hue in every panel even when a panel omits some models.
COLOR = dict(zip(ORDER, ["#898781",    # B0 is the reference floor -> muted, not a series hue
                         "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
TAG = {"B0 constant PPP": "B0", "B1 XGBoost value": "B1",
       "B2 GNN spatial-only": "B2", "B3 GNN temporal-only": "B3",
       "GNN-EPV causal (full)": "full"}
SHORT = {"B0 constant PPP": "B0  constant PPP",
         "B1 XGBoost value": "B1  XGBoost (hand-crafted)",
         "B2 GNN spatial-only": "B2  GNN spatial-only",
         "B3 GNN temporal-only": "B3  GNN temporal-only",
         "GNN-EPV causal (full)": "GNN-EPV causal (full)"}


def _style(ax):
    ax.set_facecolor(SURF); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)


def figure(B, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tbl = B["table"]
    models = [m for m in ORDER if m in tbl]
    fig = plt.figure(figsize=(14.5, 4.6), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)

    # ── A: terminal-frame RMSE (one measure -> one hue, winner emphasised) ──
    ax = fig.add_subplot(gs[0]); _style(ax)
    vals = [tbl[m]["term_rmse"][0] for m in models]
    errs = [tbl[m]["term_rmse"][1] for m in models]
    best = int(np.argmin(vals))
    cols = [BLUE_DARK if i == best else BLUE_LIGHT for i in range(len(models))]
    pad = max(np.array(vals) + np.array(errs)) * 0.035
    y = np.arange(len(models))[::-1]
    ax.barh(y, vals, height=0.6, color=cols, xerr=errs, capsize=3,
            error_kw=dict(ecolor=SEC, lw=1.2))
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + pad, yi, f"{v:.4f}", va="center", ha="left",
                color=INK, fontsize=9, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels([SHORT[m] for m in models], fontsize=9, color=SEC)
    ax.set_xlim(0, max(np.array(vals) + np.array(errs)) * 1.24)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("terminal-frame EPV RMSE (points) — lower is better", color=MUTED, fontsize=9)
    ax.set_title("A  Held-out accuracy at the terminal frame", loc="left",
                 fontsize=10.5, color=INK, pad=8)

    # ── B: calibration, as deviation from the diagonal ─────────────────────
    # Four near-identical diagonals carry no information; plotting
    # (realized - predicted) against predicted separates the models and puts
    # "perfectly calibrated" on a flat zero line where deviations are readable.
    ax = fig.add_subplot(gs[1]); _style(ax)
    cal = B.get("calibration_seed0", {})
    shown = [m for m in models if m in cal and len(cal[m]) > 1]
    ax.axhline(0, color=AXIS, ls=(0, (5, 4)), lw=1.2, zorder=1)
    ax.annotate("perfectly calibrated", xy=(0, 0),
                xycoords=("axes fraction", "data"), xytext=(4, 4),
                textcoords="offset points", ha="left", va="bottom",
                color=MUTED, fontsize=8)
    for i, m in enumerate(shown):
        c = np.array([[a, b] for a, b, _ in cal[m]])
        # x = decile RANK, not the bin's predicted value: the deciles bunch hard
        # around league PPP, so plotting against value crushes 8 of 10 bins into
        # a sliver. Rank spaces them evenly; the bins are ordered by value anyway.
        xr = np.arange(1, len(c) + 1)
        ax.plot(xr, c[:, 1] - c[:, 0], color=COLOR[m], lw=2, marker="o",
                ms=4.5, zorder=3, label=SHORT[m])
        ax.annotate(TAG[m], (xr[-1], c[-1, 1] - c[-1, 0]), textcoords="offset points",
                    xytext=(6, (-1) ** i * 8), fontsize=8.5, color=COLOR[m],
                    fontweight="bold")
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlim(0.4, 11.0); ax.set_xticks(range(1, 11))
    ax.set_xlabel("predicted-EPV decile (low → high)", color=MUTED, fontsize=9)
    ax.set_ylabel("realized − predicted (points)", color=MUTED, fontsize=9)
    ax.set_title("B  Per-frame calibration error (decile bins)", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    ax.legend(fontsize=8, frameon=False, labelcolor=SEC, loc="upper left")

    # ── C: RMSE by frame position ───────────────────────────────────────────
    ax = fig.add_subplot(gs[2]); _style(ax)
    pr = B.get("pos_rmse_seed0", {})
    # 5 series -> legend only; direct labels at this density collide.
    for m in [m for m in models if m in pr]:
        v = np.asarray(pr[m], float)
        ax.plot(np.arange(len(v)), v, color=COLOR[m], lw=2, label=SHORT[m], zorder=3)
    ax.grid(color=GRID, lw=0.8)
    ax.set_xlabel("frame index within possession (terminal at right)", color=MUTED, fontsize=9)
    ax.set_ylabel("RMSE vs realized points", color=MUTED, fontsize=9)
    ax.set_title("C  Where the value signal appears", loc="left",
                 fontsize=10.5, color=INK, pad=8)
    ax.legend(fontsize=8, frameon=False, labelcolor=SEC, loc="lower left")

    n = B.get("per_seed", {}).get(models[0], [{}])[0].get("n", "?")
    fig.suptitle("Per-moment EPV — model comparison on game-disjoint held-out games "
                 f"(3 seeds, mean ± std; test n≈{n} possessions)",
                 fontsize=12, color=INK, x=0.006, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=145, facecolor=SURF)
    print(f"wrote {out}")


def table_md(B, out, sweep=None):
    tbl = B["table"]
    models = [m for m in ORDER if m in tbl]
    f = lambda m, k, p=4: f"{tbl[m][k][0]:.{p}f} ± {tbl[m][k][1]:.{p}f}"
    L = ["# EPV model comparison",
         "",
         f"Game-disjoint 70/15/15 splits, seeds {B['seeds']}, mean ± std. "
         f"Model selection on the val split only; every number below is the TEST split.",
         "",
         "| model | terminal-frame RMSE | per-frame RMSE | skill vs constant PPP | "
         "corr(V_T, R) | mean V(0) | calib. MAE | calib. slope |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        L.append(f"| {SHORT[m]} | {f(m,'term_rmse')} | {f(m,'frame_rmse')} | "
                 f"{f(m,'skill')} | {f(m,'term_corr',3)} | {f(m,'first_frame_epv',3)} | "
                 f"{f(m,'cal_mae',4)} | {f(m,'cal_slope',3)} |")
    if sweep:
        L += ["", "## Sweep (selection on VAL terminal RMSE)", "",
              "| λ | GRU hidden | lr | target refresh | epochs | val RMSE | test RMSE |",
              "|---|---|---|---|---|---|---|"]
        for r in sorted(sweep, key=lambda r: r["val_rmse"]):
            L.append(f"| {r['lam']} | {r['gru_h']} | {r['lr']} | {r['refresh']} | "
                     f"{r['epochs']} | {r['val_rmse']:.4f} | {r['test_rmse']:.4f} |")
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", default=BASE_JSON)
    ap.add_argument("--sweep", default=SWEEP_JSON)
    a = ap.parse_args()
    B = json.load(open(a.baselines))
    S = json.load(open(a.sweep)) if os.path.exists(a.sweep) else None
    figure(B, os.path.join(OUT_DIR, "epv_model_comparison.png"))
    table_md(B, os.path.join(OUT_DIR, "epv_metrics_table.md"), S)
