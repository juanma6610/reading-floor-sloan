"""
per_team_benefit.py — what each of the 30 teams gets out of joining the federation.

Every team is scored on ITS OWN shots in the held-out games, under four models:

    alone        trained on that team's shots only (its local train split, early
                 stopped on its own validation games)
    federated    the histogram protocol (shared trees from summed gradient
                 histograms) — xgb_federated_hist_team_seed<S>_model_selected.json
    bagging      FedXgbBagging, the protocol the thesis started with
    centralized  the matched centralized model (upper bound: pooled raw data)

Averaged over the seeds, with a game-level cluster bootstrap over that team's own
test games for the federated − alone gap. This is the incentive question: a team
only joins if the shared model beats the one it can train by itself.

Teams with fewer than --min-test-games held-out games (HOU has 1, POR and DAL have 2)
are scored on too few shots for a meaningful per-team interval and are excluded from
the reported statistics and the figure; they stay in the CSV with included = False.
Training coverage, by contrast, is even across teams (32-43 games each), so no team's
solo model is handicapped.

Run after evaluate_federated.py and sim_histogram.py --utility, from the project root:
    python src/federated/per_team_benefit.py
    python src/federated/per_team_benefit.py --seeds 42 7 --n-boot 500

Outputs (results/federated/):
    per_team_benefit.csv          one row per team: metrics of all four models + gap CI
    per_team_benefit_per_seed.csv one row per (team, seed)
    per_team_benefit.png          AUC alone vs federated, per team
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_federated as ev                 # noqa: E402  (splits, metrics, bootstrap)
from nba_federated import task                  # noqa: E402

MODELS = ["alone", "federated", "bagging", "centralized"]


def team_of_test_rows():
    df = task.load_full_dataset()
    _, test_idx = task._get_global_split()
    return df.iloc[test_idx]["team_id"].to_numpy()


def predictions(seed: int, X_te: pd.DataFrame, splits) -> dict[str, np.ndarray]:
    """Test-set predictions of every model for one seed (None if a run is missing)."""
    d_te = xgb.DMatrix(X_te)
    params = ev.xgb_params(seed)
    X_tr, X_va, y_tr, y_va = ev.pooled(splits)
    out = {"centralized": ev.train_early_stopped(params, X_tr, y_tr, X_va, y_va).predict(d_te)}
    for name, tag in (("federated", "hist_team"), ("bagging", "bagging_team")):
        path = RESULTS_FED / f"xgb_federated_{tag}_seed{seed}_model_selected.json"
        if path.exists():
            b = xgb.Booster(); b.load_model(str(path))
            out[name] = b.predict(d_te)
    out["alone"] = {}
    for pid, (xt, xv, yt, yv) in enumerate(splits):
        out["alone"][pid] = ev.train_early_stopped(params, xt, yt.to_numpy(float),
                                                   xv, yv.to_numpy(float)).predict(d_te)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: seeds of the federated runs")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-test-games", type=int, default=3,
                    help="exclude teams with fewer held-out games from the reported stats")
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()

    X_te, y_te, games = ev.global_test()
    team_col = team_of_test_rows()
    team_ids, abbrs = task.get_team_ids(), task.team_abbrs()
    seeds = args.seeds or sorted({s for _, s in ev.discover_runs(None)})
    print(f"Teams: {len(team_ids)} | test shots {len(y_te)} | seeds {seeds}")

    rows, p_by_seed = [], {}
    for seed in seeds:
        splits = ev.client_splits("team", seed)
        preds = predictions(seed, X_te, splits)
        p_by_seed[seed] = preds
        for pid, tid in enumerate(team_ids):
            m = team_col == tid
            r = {"team": abbrs[pid], "partition": pid, "seed": seed, "test_shots": int(m.sum()),
                 "train_shots": len(splits[pid][0])}
            for name in MODELS:
                p = preds[name][pid] if name == "alone" else preds.get(name)
                if p is None:
                    continue
                r.update({f"{name}_{k}": fn(y_te[m], p[m]) for k, fn in ev.METRICS.items()})
            rows.append(r)
        print(f"  seed {seed}: done")

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(RESULTS_FED / "per_team_benefit_per_seed.csv", index=False)

    # per-team means, plus a cluster bootstrap (over that team's test games) of the
    # federated − alone gap, averaged over seeds like evaluate_federated does
    summary = []
    for pid, tid in enumerate(team_ids):
        g = per_seed[per_seed["partition"] == pid]
        m = team_col == tid
        ci = ev.cluster_bootstrap_gap(
            y_te[m], pd.factorize(games[m])[0],
            [p_by_seed[s]["federated"][m] for s in seeds if "federated" in p_by_seed[s]],
            [p_by_seed[s]["alone"][pid][m] for s in seeds if "federated" in p_by_seed[s]],
            args.n_boot, args.boot_seed)
        row = {"team": abbrs[pid], "partition": pid, "test_shots": int(m.sum()),
               "test_games": int(pd.Series(games[m]).nunique()),
               "train_shots": g["train_shots"].mean(), "n_seeds": len(g)}
        for name in MODELS:
            for k in ev.METRICS:
                if f"{name}_{k}" in g:
                    row[f"{name}_{k}"] = g[f"{name}_{k}"].mean()
        for k in ev.METRICS:
            row[f"gap_{k}"] = row.get(f"federated_{k}", np.nan) - row[f"alone_{k}"]
            row[f"gap_{k}_lo"], row[f"gap_{k}_hi"] = ci[k]
        summary.append(row)
    summary = pd.DataFrame(summary).sort_values("gap_auc", ascending=False)
    summary["included"] = summary["test_games"] >= args.min_test_games
    summary.to_csv(RESULTS_FED / "per_team_benefit.csv", index=False)
    dropped = summary[~summary["included"]]
    summary_all, summary = summary, summary[summary["included"]]

    won = (summary["gap_auc"] > 0).sum()
    sig = ((summary["gap_auc_lo"] > 0) | (summary["gap_auc_hi"] < 0)).sum()
    print(f"\n{'team':>5} {'shots':>6} {'alone':>7} {'federated':>10} {'Δ AUC':>8}  {'95% CI':>18}  {'bagging':>8}")
    for _, r in summary_all.iterrows():
        mark = "" if r["included"] else "  (excluded: too few games)"
        print(f"{r['team']:>5} {r['test_shots']:>6d} {r['alone_auc']:>7.3f} {r['federated_auc']:>10.3f} "
              f"{r['gap_auc']:>+8.3f}  [{r['gap_auc_lo']:+.3f}, {r['gap_auc_hi']:+.3f}]  "
              f"{r.get('bagging_auc', np.nan):>8.3f}{mark}")
    if not dropped.empty:
        print(f"\n  Excluded (< {args.min_test_games} held-out games, interval not meaningful): "
              + ", ".join(f"{r['team']} ({int(r['test_games'])}g, {int(r['test_shots'])} shots, "
                          f"Δ{r['gap_auc']:+.3f})" for _, r in dropped.iterrows()))
        a = summary_all
        print(f"  Including them all 30 teams still gain: mean Δ AUC {a['gap_auc'].mean():+.3f}, "
              f"worst {a['gap_auc'].min():+.3f}")
    print(f"\nFederated beats training alone for {won}/{len(summary)} teams "
          f"({sig} with a 95% CI excluding 0); mean Δ AUC {summary['gap_auc'].mean():+.3f}, "
          f"worst {summary['gap_auc'].min():+.3f}")
    print(f"Mean Δ AUC vs bagging: {(summary['bagging_auc'] - summary['alone_auc']).mean():+.3f}")
    plot(summary)


def plot(summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    alone_c, fed_c, bag_c = "#52514e", "#2a78d6", "#eb6834"
    d = summary[summary["included"]].sort_values("federated_auc") if "included" in summary else summary.sort_values("federated_auc")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(7.5, 8), dpi=200)
    fig.patch.set_facecolor(surface); ax.set_facecolor(surface)
    ax.hlines(y, d["alone_auc"], d["federated_auc"], color=grid, lw=2.5, zorder=1)
    ax.scatter(d["alone_auc"], y, s=42, color=alone_c, zorder=3, ec=surface, lw=1.5)
    ax.scatter(d["bagging_auc"], y, s=36, color=bag_c, zorder=3, ec=surface, lw=1.5)
    ax.scatter(d["federated_auc"], y, s=42, color=fed_c, zorder=3, ec=surface, lw=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t}  ({n})" for t, n in zip(d["team"], d["test_shots"])], fontsize=8)
    ax.set_xlabel("ROC-AUC on the team's own held-out shots (higher is better) · (n shots)", color=ink2, fontsize=9)
    ax.set_title("What each team gains by joining the federation\n"
                 "mean over seeds · its own shots in the held-out games · teams with < 3 such games excluded",
                 loc="left", fontsize=10.5, color=ink)
    ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    handles = [Line2D([], [], ls="", marker="o", ms=7, color=c, label=lab)
               for c, lab in ((alone_c, "Trained alone"), (bag_c, "Federated (tree bagging)"),
                              (fed_c, "Federated (histogram protocol)"))]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right", labelcolor=ink)
    fig.tight_layout()
    out = RESULTS_FED / "per_team_benefit.png"
    fig.savefig(out, facecolor=surface); plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
