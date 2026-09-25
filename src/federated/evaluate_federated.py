"""
evaluate_federated.py — every reported federated-vs-centralized number, from one script.

Run after run_seeds.py, from the project root:
    python src/federated/evaluate_federated.py
    python src/federated/evaluate_federated.py --n-boot 2000 --seeds 42 7

Protocol (all four rules apply identically to every model compared):

  1. Round selection on FEDERATED VALIDATION, never on the test set.
     Each client holds out 20% of its games (task.partition_frames). The round
     reported for a run is the one minimising validation log loss over the union
     of those held-out splits. A size-weighted mean of per-client log losses is
     exactly the pooled log loss, so this is what Flower's federated evaluation
     would aggregate from scalar client metrics — no raw data needed.
     Every round's global model is a prefix of the final booster, so the selected
     model is recovered by slicing (saved as *_model_selected.json).

  2. Matched centralized baseline. Same XGBoost params (pyproject.toml), trained
     on the union of the 30 clients' local train splits, early-stopped on the
     union of their local eval splits (same seed & partition). The only thing
     that differs from the federated run is the training protocol.

  3. Local-only baseline (team partition). Each team alone, same params, early-
     stopped on its own eval split. Scored on the global test set.

  Histogram protocol runs (xgb_federated_hist_team_seed<S>_model_selected.json, from
  sim_histogram.py --utility or Flower protocol = "histogram") arrive with their tree
  count already chosen on the same federated validation, and are compared with the
  same matched centralized models ("histogram_team").

  4. Uncertainty. Game-level cluster bootstrap on the global test set (shots in
     a game are correlated). For each draw the gap is the mean over seeds of
     (federated_s − centralized_s), so the CI covers test-set sampling of the
     seed-averaged gap; the seed SD is reported separately.

Outputs (results/federated/):
    eval_per_run.csv       one row per (config, seed): selected round, test metrics, matched central
    eval_summary.csv       one row per config: means, SDs, gaps with 95% CIs
    eval_summary.tex       LaTeX table
    eval_curves.csv        per-round validation + test curves
    eval_local_only.csv    per-team local-only results
    federated_convergence.png
    xgb_federated_<cfg>_seed<S>_model_selected.json
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
from nba_federated import task  # noqa: E402

CONFIGS    = ["histogram_team", "bagging_team", "bagging_iid", "cyclic_team", "cyclic_iid"]
NUM_CLIENTS = 30
EPS = 1e-15


# ──────────────────────────────────────────────────────────────
# Metrics (weighted, so the cluster bootstrap can reuse them)
# ──────────────────────────────────────────────────────────────
def brier(y, p, w=None):
    return float(np.average((p - y) ** 2, weights=w))


def logloss(y, p, w=None):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


def auc(y, p, w=None):
    return float(roc_auc_score(y, p, sample_weight=w))


METRICS = {"brier": brier, "logloss": logloss, "auc": auc}


def _sigmoid(m):
    return 1.0 / (1.0 + np.exp(-m))


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────
def xgb_params(seed: int) -> dict:
    """The federated XGBoost params from pyproject.toml, plus the global prior."""
    cfg = tomllib.loads((SCRIPT_DIR / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    params = {k.removeprefix("params."): v for k, v in cfg.items() if k.startswith("params.")}
    params["base_score"] = task.get_global_make_rate()
    params["seed"] = seed
    params["nthread"] = 4          # more threads thrash on small silos and starve concurrent Flower runs
    return params


def client_splits(strategy: str, seed: int):
    """The 30 clients' (X_train, X_eval, y_train, y_eval), exactly as the clients see them."""
    return [task.partition_frames(pid, NUM_CLIENTS, strategy, random_state=seed)
            for pid in range(NUM_CLIENTS)]


def pooled(splits):
    X_tr = pd.concat([s[0] for s in splits]); X_va = pd.concat([s[1] for s in splits])
    y_tr = pd.concat([s[2] for s in splits]); y_va = pd.concat([s[3] for s in splits])
    return X_tr, X_va, y_tr.to_numpy(float), y_va.to_numpy(float)


def global_test():
    df = task.load_full_dataset()
    _, test_idx = task._get_global_split()
    test = df.iloc[test_idx]
    X = test[task._get_feature_cols(df)]
    y = test[task.TARGET_COL].to_numpy(float)
    games = pd.factorize(test[task.GROUP_COL])[0]
    return X, y, games


# ──────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────
def train_early_stopped(params, X_tr, y_tr, X_va, y_va, max_rounds=5000, patience=200):
    """Train with early stopping on validation log loss; return best-iteration booster."""
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dva = xgb.DMatrix(X_va, label=y_va)
    bst = xgb.train(params, dtr, max_rounds, evals=[(dva, "val")],
                    early_stopping_rounds=patience, verbose_eval=False)
    return bst[: bst.best_iteration + 1]


def cumulative_margins(bst: xgb.Booster, X: pd.DataFrame, iter_bounds: list[int]) -> np.ndarray:
    """Margins of the prefix models ending at each bound in `iter_bounds` (shape: rounds × rows).

    Adds each round's trees incrementally, so cost is linear in the number of trees.
    """
    d_full = xgb.DMatrix(X)
    d_zero = xgb.DMatrix(X); d_zero.set_base_margin(np.zeros(len(X)))
    out = np.empty((len(iter_bounds), len(X)))
    margin, prev = None, 0
    for i, end in enumerate(iter_bounds):
        if margin is None:
            margin = bst.predict(d_full, iteration_range=(0, end), output_margin=True)
        else:
            margin = margin + bst.predict(d_zero, iteration_range=(prev, end), output_margin=True)
        out[i], prev = margin, end
    return out


# ──────────────────────────────────────────────────────────────
# Per-run evaluation
# ──────────────────────────────────────────────────────────────
RUN_RE = re.compile(r"^xgb_federated_(?P<cfg>(bagging|cyclic)_(team|iid))_seed(?P<seed>\d+)_model\.json$")


def discover_runs(seeds: list[int] | None) -> list[tuple[str, int]]:
    runs = []
    for p in sorted(RESULTS_FED.glob("xgb_federated_*_seed*_model.json")):
        m = RUN_RE.match(p.name)
        if m and (seeds is None or int(m["seed"]) in seeds):
            runs.append((m["cfg"], int(m["seed"])))
    return runs


def evaluate_run(cfg: str, seed: int, X_va, y_va, X_te, y_te):
    """Select the round on federated validation; return (row, curve, test_probs)."""
    bst = xgb.Booster(); bst.load_model(str(RESULTS_FED / f"xgb_federated_{cfg}_seed{seed}_model.json"))
    server_log = pd.read_csv(RESULTS_FED / f"federated_{cfg}_seed{seed}_metrics.csv")
    bounds = server_log["n_trees"].astype(int).tolist()        # trees in the global model after each round

    m_va = cumulative_margins(bst, X_va, bounds)
    m_te = cumulative_margins(bst, X_te, bounds)
    val_ll = np.array([logloss(y_va, _sigmoid(m)) for m in m_va])
    k = int(np.argmin(val_ll))

    bst[: bounds[k]].save_model(str(RESULTS_FED / f"xgb_federated_{cfg}_seed{seed}_model_selected.json"))
    p_te = _sigmoid(m_te[k])

    curve = server_log[["round", "n_trees", "auc", "brier", "logloss"]].rename(
        columns={"auc": "test_auc", "brier": "test_brier", "logloss": "test_logloss"})
    curve.insert(0, "seed", seed); curve.insert(0, "config", cfg)
    curve["val_logloss"] = val_ll

    row = {"config": cfg, "seed": seed,
           "selected_round": int(server_log["round"].iloc[k]), "selected_trees": bounds[k],
           "val_logloss": float(val_ll[k]),
           **{f"fed_{m}": fn(y_te, p_te) for m, fn in METRICS.items()}}
    return row, curve, p_te


# ──────────────────────────────────────────────────────────────
# Cluster bootstrap
# ──────────────────────────────────────────────────────────────
def cluster_bootstrap_gap(y, games, p_fed: list[np.ndarray], p_cen: list[np.ndarray],
                          n_boot: int, seed: int, alpha: float = 0.05) -> dict:
    """Game-level bootstrap CI of mean_s [metric(fed_s) − metric(central_s)] for each metric."""
    rng = np.random.default_rng(seed)
    n_games = games.max() + 1
    draws = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_games, n_games), minlength=n_games)[games].astype(float)
        for m, fn in METRICS.items():
            draws[m][b] = np.mean([fn(y, pf, w) - fn(y, pc, w) for pf, pc in zip(p_fed, p_cen)])
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {m: tuple(np.percentile(d, q)) for m, d in draws.items()}


# ──────────────────────────────────────────────────────────────
# Figure
# ──────────────────────────────────────────────────────────────
def plot_convergence(curves: pd.DataFrame, summary: pd.DataFrame, central_brier: float,
                     local_brier: float, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    color = {"bagging": "#2a78d6", "cyclic": "#eb6834", "histogram": "#1baf7a"}   # categorical slots 1-3
    label = {"histogram_team": "Histogram protocol · team silos",
             "bagging_team": "Bagging · team silos", "bagging_iid": "Bagging · IID control",
             "cyclic_team": "Cyclic · team silos", "cyclic_iid": "Cyclic · IID control"}

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    mean = curves.groupby(["config", "n_trees"], as_index=False)["test_brier"].mean()
    for cfg in CONFIGS:
        c = mean[mean["config"] == cfg]
        if c.empty:
            continue
        agg, part = cfg.split("_")
        ax.plot(c["n_trees"], c["test_brier"], color=color[agg], lw=2,
                ls="-" if part == "team" else (0, (4, 2)), label=label[cfg])
        sel = summary.loc[summary["config"] == cfg]
        if not sel.empty:
            t = sel["selected_trees_mean"].iloc[0]; b = sel["fed_brier_mean"].iloc[0]
            ax.plot(t, b, "o", ms=8, color=color[agg], mec="#fcfcfb", mew=2, zorder=5)

    ax.axhline(central_brier, color=ink, lw=1.5, ls=(0, (1, 2)))
    ax.axhline(local_brier, color=ink2, lw=1.5, ls=(0, (1, 2)))
    ax.text(1.0, central_brier, "  Centralized (same params)", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color=ink)
    ax.text(1.0, local_brier, "  Single team alone", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=8, color=ink2)

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 10, 30, 100, 300, 1000]); ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
    ax.set_xlabel("Trees in the global model (log scale)", color=ink2, fontsize=9)
    ax.set_ylabel("Brier score on held-out games (lower is better)", color=ink2, fontsize=9)
    ax.set_title("Federated convergence vs centralized training\n"
                 "mean over seeds · dots = round selected on federated validation",
                 loc="left", fontsize=10, color=ink)
    lo = min(central_brier, curves["test_brier"].min()) - 0.003
    ax.set_ylim(lo, min(curves["test_brier"].max(), local_brier + 0.02) + 0.003)
    ax.grid(axis="y", color=grid, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 0.14), labelcolor=ink)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def fmt_ci(pt, lo, hi, d=4):
    return f"{pt:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: every seed found")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()

    runs = discover_runs(args.seeds)
    if not runs:
        sys.exit(f"No federated runs found in {RESULTS_FED}. Run run_seeds.py first.")
    X_te, y_te, games = global_test()
    print(f"Global test set: {len(y_te)} shots, {games.max() + 1} games, make rate {y_te.mean():.4f}")
    print(f"Runs: {len(runs)}")

    rows, curves, p_fed, p_cen, local_rows = [], [], {}, {}, []
    central_cache: dict[tuple[str, int], tuple[dict, np.ndarray]] = {}

    for cfg, seed in runs:
        part = cfg.split("_")[1]
        splits = client_splits(part, seed)
        X_tr, X_va, y_tr, y_va = pooled(splits)

        # Matched centralized baseline — one per (partition, seed)
        if (part, seed) not in central_cache:
            bst_c = train_early_stopped(xgb_params(seed), X_tr, y_tr, X_va, y_va)
            pc = bst_c.predict(xgb.DMatrix(X_te))
            central_cache[(part, seed)] = (
                {"central_trees": bst_c.num_boosted_rounds(),
                 **{f"central_{m}": fn(y_te, pc) for m, fn in METRICS.items()}}, pc)
            # Local-only baseline — team partition only
            if part == "team":
                for pid, (xt, xv, yt, yv) in enumerate(splits):
                    bst_l = train_early_stopped(xgb_params(seed), xt, yt.to_numpy(float),
                                                xv, yv.to_numpy(float))
                    pl = bst_l.predict(xgb.DMatrix(X_te))
                    local_rows.append({"seed": seed, "client": pid, "trees": bst_l.num_boosted_rounds(),
                                       **{m: fn(y_te, pl) for m, fn in METRICS.items()}})

        row, curve, pf = evaluate_run(cfg, seed, X_va, y_va, X_te, y_te)
        cen_row, pc = central_cache[(part, seed)]
        row.update(cen_row)
        rows.append(row); curves.append(curve)
        p_fed.setdefault(cfg, []).append(pf); p_cen.setdefault(cfg, []).append(pc)
        print(f"  {cfg:13s} seed={seed:<5d} round {row['selected_round']:>4d} ({row['selected_trees']:>4d} trees)"
              f"  fed Brier {row['fed_brier']:.4f} AUC {row['fed_auc']:.4f}"
              f"  | central Brier {row['central_brier']:.4f} AUC {row['central_auc']:.4f}")

    # Histogram protocol (sim_histogram.py --utility, or Flower protocol = "histogram"): its
    # tree count was already chosen on federated validation; same matched central model.
    for seed in sorted({s for _, s in runs}):
        path = RESULTS_FED / f"xgb_federated_hist_team_seed{seed}_model_selected.json"
        if not path.exists() or ("team", seed) not in central_cache:
            continue
        bst = xgb.Booster(); bst.load_model(str(path))
        pf = bst.predict(xgb.DMatrix(X_te))
        cen_row, pc = central_cache[("team", seed)]
        rows.append({"config": "histogram_team", "seed": seed, "selected_round": np.nan,
                     "selected_trees": bst.num_boosted_rounds(), "val_logloss": np.nan,
                     **{f"fed_{m}": fn(y_te, pf) for m, fn in METRICS.items()}, **cen_row})
        p_fed.setdefault("histogram_team", []).append(pf); p_cen.setdefault("histogram_team", []).append(pc)
        print(f"  histogram_team seed={seed:<5d} {bst.num_boosted_rounds():>4d} trees"
              f"  fed Brier {rows[-1]['fed_brier']:.4f} AUC {rows[-1]['fed_auc']:.4f}")
    hist_curve = RESULTS_FED / "hist_curve.csv"
    if hist_curve.exists():
        hc = pd.read_csv(hist_curve).rename(columns={"trees": "n_trees"})
        hc = hc[hc["seed"].isin({s for _, s in runs})]
        curves.append(hc.assign(config="histogram_team", round=hc["n_trees"]))

    per_run = pd.DataFrame(rows)
    curves = pd.concat(curves, ignore_index=True)
    local = pd.DataFrame(local_rows)
    local_mean = local[list(METRICS)].mean() if not local.empty else None

    summary_rows = []
    for cfg in [c for c in CONFIGS if c in p_fed]:
        r = per_run[per_run["config"] == cfg]
        ci = cluster_bootstrap_gap(y_te, games, p_fed[cfg], p_cen[cfg], args.n_boot, args.boot_seed)
        s = {"config": cfg, "n_seeds": len(r),
             "selected_round_mean": r["selected_round"].mean(),
             "selected_trees_mean": r["selected_trees"].mean(),
             "central_trees_mean": r["central_trees"].mean()}
        for m in METRICS:
            gap = r[f"fed_{m}"] - r[f"central_{m}"]
            s.update({f"fed_{m}_mean": r[f"fed_{m}"].mean(), f"fed_{m}_sd": r[f"fed_{m}"].std(ddof=1),
                      f"central_{m}_mean": r[f"central_{m}"].mean(),
                      f"gap_{m}": gap.mean(), f"gap_{m}_lo": ci[m][0], f"gap_{m}_hi": ci[m][1],
                      f"gap_{m}_seed_sd": gap.std(ddof=1)})
        cb = s["central_brier_mean"]
        s.update({"gap_brier_rel_pct": 100 * s["gap_brier"] / cb,
                  "gap_brier_rel_pct_lo": 100 * s["gap_brier_lo"] / cb,
                  "gap_brier_rel_pct_hi": 100 * s["gap_brier_hi"] / cb})
        if local_mean is not None:
            s.update({f"local_{m}_mean": local_mean[m] for m in METRICS})
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)

    RESULTS_FED.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(RESULTS_FED / "eval_per_run.csv", index=False)
    summary.to_csv(RESULTS_FED / "eval_summary.csv", index=False)
    curves.to_csv(RESULTS_FED / "eval_curves.csv", index=False)
    if not local.empty:
        local.to_csv(RESULTS_FED / "eval_local_only.csv", index=False)

    # ── console report ──
    print(f"\n=== Summary (test set, game-level cluster bootstrap, B={args.n_boot}) ===")
    for _, s in summary.iterrows():
        print(f"\n{s['config']}  (n={s['n_seeds']}, selected round {s['selected_round_mean']:.1f}, "
              f"{s['selected_trees_mean']:.0f} trees; central {s['central_trees_mean']:.0f} trees)")
        for m in METRICS:
            print(f"  {m:8s} fed {s[f'fed_{m}_mean']:.4f} ± {s[f'fed_{m}_sd']:.4f}   central {s[f'central_{m}_mean']:.4f}"
                  f"   gap {fmt_ci(s[f'gap_{m}'], s[f'gap_{m}_lo'], s[f'gap_{m}_hi'])}"
                  f"  (seed SD {s[f'gap_{m}_seed_sd']:.4f})")
        print(f"  relative Brier gap {fmt_ci(s['gap_brier_rel_pct'], s['gap_brier_rel_pct_lo'], s['gap_brier_rel_pct_hi'], 2)} %")
    if local_mean is not None:
        print(f"\nLocal-only (one team alone, mean over {local['client'].nunique()} teams × "
              f"{local['seed'].nunique()} seeds): Brier {local_mean['brier']:.4f}  AUC {local_mean['auc']:.4f}  "
              f"LogLoss {local_mean['logloss']:.4f}")

    # ── LaTeX ──
    names = {"histogram_team": "Histogram protocol, team silos", "bagging_team": "Bagging, team silos", "bagging_iid": "Bagging, IID",
             "cyclic_team": "Cyclic, team silos", "cyclic_iid": "Cyclic, IID"}
    lines = [f"% Auto-generated by src/federated/evaluate_federated.py (B={args.n_boot}, game-level cluster bootstrap)",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"Model & Brier & Log loss & ROC-AUC & $\Delta$Brier (95\% CI) \\", r"\midrule"]
    # The team and IID partitions have DIFFERENT matched centralized models (same params,
    # different local train/eval splits). Each row's gap already uses its own baseline; this
    # single display row is the team one, the primary experiment — not summary.iloc[0], which
    # was whichever config happened to sort first.
    team_rows = summary[summary["config"].str.endswith("_team")]
    c0 = (team_rows if not team_rows.empty else summary).iloc[0]
    lines.append(f"Centralized (same params, team split) & {c0['central_brier_mean']:.4f} "
                 f"& {c0['central_logloss_mean']:.4f} & {c0['central_auc_mean']:.4f} & -- \\\\")
    for _, s in summary.iterrows():
        lines.append(f"{names[s['config']]} & {s['fed_brier_mean']:.4f} & {s['fed_logloss_mean']:.4f} & "
                     f"{s['fed_auc_mean']:.4f} & ${s['gap_brier_rel_pct']:+.1f}\\%$ "
                     f"$[{s['gap_brier_rel_pct_lo']:+.1f},\\,{s['gap_brier_rel_pct_hi']:+.1f}]$ \\\\")
    if local_mean is not None:
        lines.append(f"Single team alone (mean) & {local_mean['brier']:.4f} & {local_mean['logloss']:.4f} "
                     f"& {local_mean['auc']:.4f} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (RESULTS_FED / "eval_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if local_mean is not None:
        # Team baseline, not the mean over configs: the team and IID partitions have
        # different matched centralized models, and averaging them draws a line that is
        # neither.
        team_rows = summary[summary["config"].str.endswith("_team")]
        central_ref = float((team_rows if not team_rows.empty else summary)["central_brier_mean"].iloc[0])
        plot_convergence(curves, summary, central_ref,
                         float(local_mean["brier"]), RESULTS_FED / "federated_convergence.png")
    print(f"\nWrote eval_per_run.csv, eval_summary.csv/.tex, eval_curves.csv, "
          f"eval_local_only.csv, federated_convergence.png → {RESULTS_FED.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
