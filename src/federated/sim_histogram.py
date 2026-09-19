"""
sim_histogram.py — histogram-based federated boosting (nba_federated/hist_gbdt.py)
on the 30 team silos: accuracy without DP, and the privacy/utility frontier with
distributed DP.

Same data protocol as evaluate_federated.py: the 30 team partitions with their
local train / validation games (task.partition_frames), the global test games,
selection on pooled client validation, and the matched centralized baseline
numbers from eval_per_run.csv for comparison.

Run from the project root:
    python src/federated/sim_histogram.py --utility            # no DP, full protocol params
    python src/federated/sim_histogram.py --dp-sweep           # ε grid × split mode × configs
    python src/federated/sim_histogram.py --dp-sweep --eps 1 4 --seeds 42 7
    python src/federated/sim_histogram.py --dp-sweep --extend --modes random --eps 1 4 --tag random_c
                                  # larger random-mode grid (T=1600, η=0.05); run for ε ∈ {1, 4}
    python src/federated/sim_histogram.py --merge   # combine all sweep parts → hist_dp_frontier.csv

Outputs (results/federated/):
    hist_utility.csv, hist_curve.csv, xgb_federated_hist_team_seed<S>_model_selected.json
    hist_dp_sweep.csv          every (ε, mode, config, noise seed): val + test metrics
    hist_dp_frontier.csv       per (ε, mode): config chosen on validation, test mean ± SD over noise seeds

Caveats reported with the DP numbers: the configuration for each ε is chosen on
validation data, and that tuning step is not charged to the privacy budget.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_federated as ev                      # noqa: E402  (splits, metrics)
from nba_federated import hist_gbdt as HG, task       # noqa: E402

PROTOCOL = dict(eta=0.05, max_depth=5, reg_lambda=2.0, min_child_weight=10.0,
                colsample_bytree=0.8, subsample=0.8)   # = pyproject.toml federated params


def _sigmoid(m):
    return 1.0 / (1.0 + np.exp(-m))


class Setup:
    """Clients' binned local splits + pooled validation + global test, for one seed and grid."""

    def __init__(self, seed: int, bin_stride: int = 1):
        self.seed = seed
        self.p0 = task.get_global_make_rate()
        self.m0 = float(np.log(self.p0 / (1 - self.p0)))
        self.splits = ev.client_splits("team", seed)
        self.spec = HG.BinSpec(list(self.splits[0][0].columns), bin_stride)
        _, X_va, _, y_va = ev.pooled(self.splits)
        X_te, y_te, _ = ev.global_test()
        self.b_va, self.y_va = self.spec.bin(X_va), y_va
        self.b_te, self.y_te, self.X_te = self.spec.bin(X_te), y_te, X_te
        self.client_bins = [(self.spec.bin(s[0]), s[2].to_numpy()) for s in self.splits]

    def clients(self, noise_seed: int):
        return [HG.Client(b, y, self.m0, self.spec, seed=noise_seed * 1000 + c)
                for c, (b, y) in enumerate(self.client_bins)]


def train(setup: Setup, params: HG.Params, noise_seed: int, patience: int | None = None, curve: bool = True):
    """Grow trees, tracking pooled-validation and test margins. Returns (trainer, curve DataFrame).

    curve=False scores only the final model (one row), which is all the DP sweep needs.
    """
    trainer = HG.Trainer(setup.clients(noise_seed), setup.spec, replace(params, seed=noise_seed))
    m_va = np.full(len(setup.y_va), setup.m0); m_te = np.full(len(setup.y_te), setup.m0)
    rows, best = [], (np.inf, 0)
    for t in range(params.n_trees):
        tree = trainer.grow_tree()
        m_va += tree.predict_bins(setup.b_va, setup.spec.n_bins)
        m_te += tree.predict_bins(setup.b_te, setup.spec.n_bins)
        if not curve and t + 1 < params.n_trees:
            continue
        pv, pt = _sigmoid(m_va), _sigmoid(m_te)
        rows.append({"trees": t + 1, "val_logloss": ev.logloss(setup.y_va, pv),
                     "test_brier": ev.brier(setup.y_te, pt), "test_logloss": ev.logloss(setup.y_te, pt),
                     "test_auc": ev.auc(setup.y_te, pt)})
        if rows[-1]["val_logloss"] < best[0]:
            best = (rows[-1]["val_logloss"], t + 1)
        if patience is not None and t + 1 - best[1] >= patience:
            break
    return trainer, pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# No DP: does the histogram protocol close the gap?
# ──────────────────────────────────────────────────────────────
def utility(seeds: list[int]):
    out, curves = [], []
    for seed in seeds:
        setup = Setup(seed)
        params = HG.Params(n_trees=3000, **PROTOCOL)
        t0 = time.time()
        trainer, curve = train(setup, params, noise_seed=seed, patience=200)
        k = int(curve["val_logloss"].idxmin())
        sel = curve.iloc[k]
        HG.to_xgboost(trainer.trees[: int(sel["trees"])], setup.spec, setup.p0).save_model(
            str(RESULTS_FED / f"xgb_federated_hist_team_seed{seed}_model_selected.json"))
        out.append({"seed": seed, "selected_trees": int(sel["trees"]), "test_brier": sel["test_brier"],
                    "test_logloss": sel["test_logloss"], "test_auc": sel["test_auc"],
                    "minutes": (time.time() - t0) / 60})
        curves.append(curve.assign(seed=seed))
        print(f"  seed {seed}: {int(sel['trees'])} trees  Brier {sel['test_brier']:.4f}  "
              f"LogLoss {sel['test_logloss']:.4f}  AUC {sel['test_auc']:.4f}  ({out[-1]['minutes']:.1f} min)")
    res = pd.DataFrame(out)
    res.to_csv(RESULTS_FED / "hist_utility.csv", index=False)
    pd.concat(curves).to_csv(RESULTS_FED / "hist_curve.csv", index=False)

    ref = RESULTS_FED / "eval_per_run.csv"
    if ref.exists():
        r = pd.read_csv(ref)
        r = r[(r["config"] == "bagging_team") & r["seed"].isin(seeds)]
        if not r.empty:
            print(f"\n  same seeds — matched centralized: Brier {r['central_brier'].mean():.4f} AUC {r['central_auc'].mean():.4f}"
                  f" | FedXgbBagging: Brier {r['fed_brier'].mean():.4f} AUC {r['fed_auc'].mean():.4f}")
    print(f"  histogram protocol:              Brier {res['test_brier'].mean():.4f} AUC {res['test_auc'].mean():.4f}")


# ──────────────────────────────────────────────────────────────
# DP: privacy / utility frontier
# ──────────────────────────────────────────────────────────────
GRID = {
    "hist": dict(n_trees=[25, 50, 100], max_depth=[3, 4], eta=[0.1, 0.3],
                 colsample_bytree=[0.25], bin_stride=[4, 8], reg_lambda=[10.0]),
    "random": dict(n_trees=[50, 100, 200, 400, 800], max_depth=[4, 6], eta=[0.1, 0.3, 0.5],
                   colsample_bytree=[0.8], bin_stride=[4], reg_lambda=[10.0]),
}


# --extend: the random-mode optimum sat at the grid edge (T=800), so ε ∈ {1, 4} were
# re-swept on this larger grid (part "random_c").
GRID_EXTEND_RANDOM = dict(n_trees=[800, 1600], max_depth=[4, 6], eta=[0.05, 0.1],
                          colsample_bytree=[0.8], bin_stride=[4], reg_lambda=[10.0])


def run_dp(setup, params, cfg, eps, mode, stride, s):
    trainer, curve = train(setup, params, noise_seed=s, curve=False)
    last = curve.iloc[-1]                                # DP: final model, no early stopping
    return {"eps": eps, "mode": mode, "noise_seed": s, "bin_stride": stride, **cfg,
            "sigma": trainer.sigma, "releases": trainer.releases_per_shot(),
            "val_logloss": last["val_logloss"], "test_brier": last["test_brier"],
            "test_logloss": last["test_logloss"], "test_auc": last["test_auc"]}


def dp_sweep(eps_list: list[float], seeds: list[int], modes: list[str], extra_seeds: list[int], tag: str):
    """Writes hist_dp_sweep_part_<tag>.csv; run several in parallel, then --merge."""
    out = RESULTS_FED / f"hist_dp_sweep_part_{tag}.csv"
    setups: dict[int, Setup] = {}
    rows = []
    for mode in modes:
        keys, values = zip(*GRID[mode].items())
        for eps in eps_list:
            for combo in itertools.product(*values):
                cfg = dict(zip(keys, combo))
                stride = cfg.pop("bin_stride")
                setup = setups.setdefault(stride, Setup(42, stride))
                params = HG.Params(split_mode=mode, dp_epsilon=eps, subsample=0.8, min_child_weight=10.0,
                                   bin_stride=stride, **cfg)
                for s in seeds:
                    rows.append(run_dp(setup, params, cfg, eps, mode, stride, s))
            # re-run the validation-chosen config with extra noise seeds for error bars
            g = pd.DataFrame([r for r in rows if r["eps"] == eps and r["mode"] == mode])
            best = dict(zip(keys, g.groupby(list(keys))["val_logloss"].mean().idxmin()))
            stride = best.pop("bin_stride")
            params = HG.Params(split_mode=mode, dp_epsilon=eps, subsample=0.8, min_child_weight=10.0,
                               bin_stride=stride, **best)
            for s in extra_seeds:
                rows.append(run_dp(setups[stride], params, best, eps, mode, stride, s))
            print(f"  ε={eps:<5g} {mode:6s}: best on val {best | {'bin_stride': stride}}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)


def merge_frontier():
    """Combine every hist_dp_sweep_part_*.csv; per (ε, mode) take the config with the best mean
    validation loss and report its test metrics over noise seeds."""
    parts = sorted(RESULTS_FED.glob("hist_dp_sweep_part_*.csv"))
    sweep = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    sweep.to_csv(RESULTS_FED / "hist_dp_sweep.csv", index=False)
    modes = sorted(sweep["mode"].unique())
    cfg_cols = [c for c in sweep.columns if c in set(itertools.chain(*[GRID[m] for m in modes]))]
    frontier = []
    for (eps, mode), g in sweep.groupby(["eps", "mode"]):
        # Select on the sweep seed only (every config has it), then report all noise seeds
        # of that config; averaging first would favour configs that were run once.
        first = g[g["noise_seed"] == g["noise_seed"].iloc[0]]
        best_cfg = first.groupby(cfg_cols, dropna=False)["val_logloss"].mean().idxmin()
        sel = g.groupby(cfg_cols, dropna=False).get_group(best_cfg)
        frontier.append({"eps": eps, "mode": mode, **dict(zip(cfg_cols, best_cfg)),
                         "sigma": sel["sigma"].iloc[0], "n_noise_seeds": len(sel),
                         **{f"test_{m}_mean": sel[f"test_{m}"].mean() for m in ("brier", "logloss", "auc")},
                         **{f"test_{m}_sd": sel[f"test_{m}"].std(ddof=1) for m in ("brier", "auc")}})
    frontier = pd.DataFrame(frontier).sort_values(["mode", "eps"])
    frontier.to_csv(RESULTS_FED / "hist_dp_frontier.csv", index=False)
    print("\n=== DP frontier (config chosen on validation; test mean over noise seeds) ===")
    for _, r in frontier.iterrows():
        print(f"  ε={r['eps']:<5g} {r['mode']:6s}  Brier {r['test_brier_mean']:.4f} ± {r['test_brier_sd']:.4f}"
              f"  AUC {r['test_auc_mean']:.4f} ± {r['test_auc_sd']:.4f}   σ={r['sigma']:.1f}  "
              f"T={int(r['n_trees'])} depth={int(r['max_depth'])} η={r['eta']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utility", action="store_true")
    ap.add_argument("--dp-sweep", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16])
    ap.add_argument("--modes", nargs="+", default=["random", "hist"])
    ap.add_argument("--extra-seeds", type=int, nargs="+", default=[7, 123],
                    help="noise seeds added for the validation-chosen config (error bars)")
    ap.add_argument("--tag", default="all", help="suffix of this sweep part (parallel runs)")
    ap.add_argument("--merge", action="store_true", help="merge sweep parts into hist_dp_frontier.csv")
    ap.add_argument("--extend", action="store_true", help="use the extended random-mode grid")
    args = ap.parse_args()
    RESULTS_FED.mkdir(parents=True, exist_ok=True)
    if args.utility:
        utility(args.seeds)
    if args.extend:
        GRID["random"] = GRID_EXTEND_RANDOM
    if args.dp_sweep:
        dp_sweep(args.eps, args.seeds, args.modes, args.extra_seeds, args.tag)
    if args.merge:
        merge_frontier()


if __name__ == "__main__":
    main()
