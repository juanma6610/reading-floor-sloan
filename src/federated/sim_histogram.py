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
    python src/federated/sim_histogram.py --fixed --seeds 42 7 123 --tag all
                                  # HEADLINE: one fixed config, every ε, shot- and player-level DP
    python src/federated/sim_histogram.py --fixed --units player --eps 4 --clip-grid --seeds 42 7 123 --tag clip

Outputs (results/federated/):
    hist_utility.csv, hist_curve.csv, xgb_federated_hist_team_seed<S>_model_selected.json
    hist_dp_sweep.csv          every (ε, mode, config, noise seed): val + test metrics
    hist_dp_frontier.csv       per (ε, mode): config chosen on validation, test mean ± SD over noise seeds

The headline DP numbers (--fixed → hist_dp_fixed_part_*.csv) use ONE configuration
fixed in advance for all ε, so no tuning step touches private data; the per-ε tuned
sweep (--dp-sweep → hist_dp_frontier.csv) chooses configs on validation data without
charging that to the budget, and is reported only as an upper bound.
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
    """Clients' binned local splits + pooled validation + global test, for one seed and grid.
    The initial prediction is the PUBLIC league FG% (nothing about the teams' data is
    released to set it); each client also keeps its shooters' names (never sent).
    """

    def __init__(self, seed: int, bin_stride: int = 1, discrete: bool = False, lattice: float = 0.0):
        self.seed, self.discrete, self.lattice = seed, discrete, lattice
        self.p0 = task.PUBLIC_LEAGUE_FG_PCT
        self.m0 = float(np.log(self.p0 / (1 - self.p0)))
        self.player_teams = task.teams_per_player()
        self.players = [task.partition_frames(pid, ev.NUM_CLIENTS, "team", random_state=seed,
                                              return_players=True)[4] for pid in range(ev.NUM_CLIENTS)]
        self.splits = ev.client_splits("team", seed)
        self.spec = HG.BinSpec(list(self.splits[0][0].columns), bin_stride)
        _, X_va, _, y_va = ev.pooled(self.splits)
        X_te, y_te, _ = ev.global_test()
        self.b_va, self.y_va = self.spec.bin(X_va), y_va
        self.b_te, self.y_te, self.X_te = self.spec.bin(X_te), y_te, X_te
        self.client_bins = [(self.spec.bin(s[0]), s[2].to_numpy()) for s in self.splits]

    def clients(self, noise_seed: int):
        return [HG.Client(b, y, self.m0, self.spec, seed=noise_seed * 1000 + c,
                          players=self.players[c], player_teams=self.player_teams,
                          discrete=self.discrete, lattice=self.lattice)
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
    summarize_fixed()
    frontier.to_csv(RESULTS_FED / "hist_dp_frontier.csv", index=False)
    print("\n=== DP frontier (config chosen on validation; test mean over noise seeds) ===")
    for _, r in frontier.iterrows():
        print(f"  ε={r['eps']:<5g} {r['mode']:6s}  Brier {r['test_brier_mean']:.4f} ± {r['test_brier_sd']:.4f}"
              f"  AUC {r['test_auc_mean']:.4f} ± {r['test_auc_sd']:.4f}   σ={r['sigma']:.1f}  "
              f"T={int(r['n_trees'])} depth={int(r['max_depth'])} η={r['eta']}")


# ──────────────────────────────────────────────────────────────
# Fixed configuration (headline): no data-dependent tuning, every release charged
# ──────────────────────────────────────────────────────────────
# One configuration for every ε and both privacy units, fixed before the DP runs.
# (It is the random-mode setting the exploratory sweep favoured at ε = 2–8; the sweep
# itself is reported separately as an unaccounted upper bound.) Player clip norms are
# set a priori from public shot volumes, not tuned: a rotation player takes ~120 sampled
# shots per tree, |g| ≈ 0.5 with random sign → per-leaf-vector norm ≈ 0.5·√120 ≈ 5.5;
# h ≈ 0.24 over ~8 occupied leaves of ~15 shots → norm ≈ 0.24·√(8·15²) ≈ 10. (In the
# data players concentrate in fewer leaves, H norms ≈ 16, so most players are scaled
# down; --clip-grid reports the sensitivity to this choice instead of retuning it.)
FIXED = dict(split_mode="random", n_trees=800, max_depth=4, eta=0.1, reg_lambda=10.0,
             colsample_bytree=0.8, subsample=0.8, min_child_weight=10.0, bin_stride=4,
             clip_g=5.0, clip_h=10.0)

# Params fields written to every hist_dp_fixed_part_*.csv row. Read off the Params
# object the run actually used, never off FIXED: FIXED is the intent, and anything
# derived from it (player_sampling, set when dp_amplify is on at the player level)
# would otherwise be recorded wrong. `seed` is deliberately absent — train() does
# replace(params, seed=noise_seed) internally, so params.seed is still the default
# here; the noise seed is recorded separately as `noise_seed`.
RECORDED = ("split_mode", "n_trees", "max_depth", "eta", "reg_lambda", "colsample_bytree",
            "subsample", "min_child_weight", "bin_stride", "leaf_clip", "dp_delta",
            "clip_g", "clip_h", "dp_amplify", "player_sampling",
            "dp_tolerated_collusion", "dp_discrete")


def summarize_fixed():
    """hist_dp_fixed_summary.csv: mean ± SD over noise seeds for every fixed-config cell
    (headline = n_trees 800, η 0.1, clips 5/10; other rows are sensitivity analyses)."""
    parts = sorted(RESULTS_FED.glob("hist_dp_fixed_part_*.csv"))
    if not parts:
        return
    defaults = {"subsample": 0.8, "dp_amplify": False, "player_sampling": False,
                "dp_tolerated_collusion": 0, "dp_discrete": False}
    d = pd.concat([pd.read_csv(p) for p in parts])
    for k, v in defaults.items():
        d[k] = d[k].fillna(v) if k in d else v
    keys = ["unit", "n_trees", "eta", "clip_g", "clip_h", "subsample", "dp_amplify",
            "player_sampling", "dp_tolerated_collusion", "dp_discrete", "eps"]
    d = d.drop_duplicates(keys + ["noise_seed"])
    g = d.groupby(keys)
    out = pd.concat([g.size().rename("n_noise_seeds"), g["sigma_g"].first(),
                     g[["test_brier", "test_logloss", "test_auc"]].mean().add_suffix("_mean"),
                     g[["test_brier", "test_auc"]].std(ddof=1).add_suffix("_sd")], axis=1).reset_index()
    out["headline"] = ((out["n_trees"] == FIXED["n_trees"]) & (out["eta"] == FIXED["eta"])
                       & (out["clip_g"] == FIXED["clip_g"]) & (out["clip_h"] == FIXED["clip_h"])
                       & (out["subsample"] == FIXED["subsample"]) & ~out["dp_amplify"].astype(bool)
                       & (out["dp_tolerated_collusion"] == 0) & ~out["dp_discrete"].astype(bool))
    out.to_csv(RESULTS_FED / "hist_dp_fixed_summary.csv", index=False)
    print("\n=== Fixed configuration (no tuning; every release charged) ===")
    print(out[out["headline"]][["unit", "eps", "test_brier_mean", "test_auc_mean", "test_auc_sd"]].round(4).to_string(index=False))


def _parse_value(v: str):
    """CLI override value → bool / int / float / str."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def fixed_frontier(eps_list, units, noise_seeds, tag, clips=None):
    """Test metrics of the fixed configuration at each ε and privacy unit (final model, no
    early stopping, no per-tree validation). `clips` overrides (clip_g, clip_h) — used only
    for the clip-sensitivity table. FIXED may have been changed first via --override; every
    RECORDED field is written to the output from the Params the run used, not from FIXED."""
    setup = Setup(42, FIXED["bin_stride"], FIXED.get("dp_discrete", False),
                  FIXED.get("dp_lattice", HG.Params.dp_lattice))
    rows = []
    for unit in units:
        for cg, ch in (clips or [(FIXED["clip_g"], FIXED["clip_h"])]):
            for eps in eps_list:
                params = HG.Params(**{**FIXED, "clip_g": cg, "clip_h": ch}, dp_epsilon=eps, dp_unit=unit)
                if unit == "player" and params.dp_amplify:
                    params = replace(params, player_sampling=True)
                for s in noise_seeds:
                    trainer, curve = train(setup, params, noise_seed=s, curve=False)
                    last = curve.iloc[-1]
                    rows.append({"unit": unit, "eps": eps, "noise_seed": s,
                                 **{k: getattr(params, k) for k in RECORDED},
                                 # sigma_* is the CALIBRATED noise the accountant priced, i.e. what
                                 # the K - c honest teams carry between them. sigma_*_total is what
                                 # all K teams actually put on the aggregate — larger by
                                 # sqrt(K/(K-c)) whenever collusion is tolerated.
                                 "sigma_g": trainer.sigma, "sigma_h": trainer.sigma_h,
                                 "sigma_g_total": trainer._client_sd()[0] * np.sqrt(len(trainer.clients)),
                                 "sigma_h_total": trainer._client_sd()[1] * np.sqrt(len(trainer.clients)),
                                 "test_brier": last["test_brier"], "test_logloss": last["test_logloss"],
                                 "test_auc": last["test_auc"]})
                g = pd.DataFrame(rows[-len(noise_seeds):])
                print(f"  {unit:6s} ε={eps:<5g} clip=({cg:g},{ch:g})  Brier {g['test_brier'].mean():.4f} ± "
                      f"{g['test_brier'].std(ddof=1):.4f}  AUC {g['test_auc'].mean():.4f} ± {g['test_auc'].std(ddof=1):.4f}"
                      f"   σ_G={trainer.sigma:.1f}", flush=True)
                pd.DataFrame(rows).to_csv(RESULTS_FED / f"hist_dp_fixed_part_{tag}.csv", index=False)


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
    ap.add_argument("--fixed", action="store_true", help="headline: FIXED config at every ε (no tuning)")
    ap.add_argument("--units", nargs="+", default=["shot", "player"], help="--fixed: privacy units")
    ap.add_argument("--clip-grid", action="store_true", help="--fixed: player clip-norm sensitivity table")
    ap.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                    help="--fixed: change FIXED entries, e.g. subsample=0.2 dp_amplify=true n_trees=1600")
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
    for kv in args.override:
        k, v = kv.split("=", 1)
        FIXED[k] = _parse_value(v)
        print(f"  override {k} = {FIXED[k]!r}")
    if args.fixed:
        clips = [(2.5, 5.0), (5.0, 10.0), (10.0, 20.0)] if args.clip_grid else None
        fixed_frontier(args.eps, args.units, args.seeds, args.tag, clips)


if __name__ == "__main__":
    main()
