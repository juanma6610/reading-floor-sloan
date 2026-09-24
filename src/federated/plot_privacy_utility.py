"""
plot_privacy_utility.py — the privacy/utility figure: test AUC and Brier of the
histogram protocol with distributed DP against ε, protecting one shot vs one player.

Reads (results/federated/): hist_dp_fixed_summary.csv (sim_histogram.py --fixed / --merge),
eval_summary.csv and eval_local_only.csv (evaluate_federated.py).
Writes: results/federated/privacy_utility.png

Run from the project root:  python src/federated/plot_privacy_utility.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

RESULTS_FED = Path(__file__).resolve().parents[2] / "results" / "federated"

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SHOT, PLAYER = "#2a78d6", "#eb6834"          # categorical slots 1, 2
CONSTANT_BRIER = 0.2472                      # base-rate prediction on the test games (headline_metrics)


def main():
    fx = pd.read_csv(RESULTS_FED / "hist_dp_fixed_summary.csv")
    ev = pd.read_csv(RESULTS_FED / "eval_summary.csv").set_index("config")
    local = pd.read_csv(RESULTS_FED / "eval_local_only.csv")[["auc", "brier"]].mean()
    refs = {  # label → (auc, brier, colour)
        "No DP (= centralized)": (ev.loc["histogram_team", "fed_auc_mean"], ev.loc["histogram_team", "fed_brier_mean"], INK),
        "Tree bagging, no privacy": (ev.loc["bagging_team", "fed_auc_mean"], ev.loc["bagging_team", "fed_brier_mean"], INK2),
        "One team alone": (local["auc"], local["brier"], INK2),
    }
    series = [
        ("Protect each shot", fx[fx["headline"] & (fx["unit"] == "shot")], SHOT, "-", "o", True),
        ("Protect each player", fx[fx["headline"] & (fx["unit"] == "player")], PLAYER, "-", "o", True),
        ("Protect each player, 3,200 trees (sensitivity)",
         fx[(fx["unit"] == "player") & (fx["n_trees"] == 3200)], PLAYER, (0, (4, 2)), "o", False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric, better in ((axes[0], "auc", "higher is better"), (axes[1], "brier", "lower is better")):
        ax.set_facecolor(SURFACE)
        for label, (a, b, col) in refs.items():
            v = a if metric == "auc" else b
            ax.axhline(v, color=col, lw=1.2, ls=(0, (1, 2)))
            ax.text(1.01, v, " " + label, transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=col)
        if metric == "brier":
            ax.axhline(CONSTANT_BRIER, color=INK2, lw=1.2, ls=(0, (1, 2)))
            ax.text(1.01, CONSTANT_BRIER, " League-average guess", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=INK2)
        for label, d, col, ls, mk, filled in series:
            d = d.sort_values("eps")
            eb = ax.errorbar(d["eps"], d[f"test_{metric}_mean"], yerr=d[f"test_{metric}_sd"], color=col, ls=ls,
                             lw=2, marker=mk, ms=6, mfc=col if filled else SURFACE,
                             mec=col if not filled else SURFACE, mew=1.5, capsize=0, elinewidth=1.2,
                             label=label, zorder=3)
            eb[2][0].set_linestyle("-")                     # solid error bars on the dashed series too
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0.5, 1, 2, 4, 8, 16]))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("Privacy budget ε (smaller = more private)", color=INK2, fontsize=9)
        ax.set_ylabel(f"Test {'ROC-AUC' if metric == 'auc' else 'Brier score'} ({better})", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, ls=ls, lw=2, marker=mk, ms=6, mfc=col if filled else SURFACE, mec=col)
               for _, _, col, ls, mk, filled in series]
    axes[1].legend(handles, [s[0] for s in series], frameon=False, fontsize=8, loc="upper right",
                   labelcolor=INK, handlelength=3)
    fig.suptitle("Privacy/utility trade-off of the federated histogram protocol (distributed DP + SecAgg+, δ = 10⁻⁵)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.text(0.01, 0.905, "One configuration for every ε, no tuning on private data · mean ± SD over 3 noise seeds · "
             "95 held-out games", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3)
    out = RESULTS_FED / "privacy_utility.png"
    fig.savefig(out, facecolor=SURFACE)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
