"""
team_heterogeneity.py — how non-IID are the 30 team silos, really?

The thesis calls the team partition "non-IID by construction", but team silos scored
the same as the IID control in every federated experiment. This quantifies the
heterogeneity, always next to the IID partition of the same shots as the "pure
sampling noise" reference:

  1. Label shift   each team's make rate; is the spread across teams larger than
                   shuffling the team labels would give? (permutation test)
  2. Feature shift Jensen-Shannon divergence between a team's feature distribution
                   and the league's, per feature, on the public bin grid
  3. Task shift    transfer matrix: train on one team, score every team's held-out
                   shots. If a team's own model is no better on its own shots than
                   the other 29 teams' models are, the silos differ in what shots
                   they take, not in what makes a shot go in.

Run from the project root:
    python src/federated/team_heterogeneity.py
    python src/federated/team_heterogeneity.py --seed 42 --n-perm 2000

Outputs (results/federated/):
    team_heterogeneity.csv           per team: shots, make rate, mean JSD, transfer AUCs
    team_heterogeneity_features.csv  per feature: mean JSD (team silos vs IID control)
    team_heterogeneity.png           make rates + the features that differ most
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
import evaluate_federated as ev                          # noqa: E402
from nba_federated import hist_gbdt as HG, task          # noqa: E402

BIN_STRIDE = 4          # 16 public bins per continuous feature


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint)."""
    p, q = p / max(p.sum(), 1e-12), q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        ok = a > 0
        return float(np.sum(a[ok] * np.log2(a[ok] / b[ok])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def feature_divergences(splits, spec: HG.BinSpec) -> np.ndarray:
    """(clients × features) JSD between each client's binned feature distribution and the pool's."""
    bins = [spec.bin(s[0]) for s in splits]
    pooled = np.concatenate(bins)
    out = np.zeros((len(bins), len(spec.features)))
    for f in range(len(spec.features)):
        nb = spec.n_bins[f] + 1
        league = np.bincount(pooled[:, f], minlength=nb)
        for c, b in enumerate(bins):
            out[c, f] = jsd(np.bincount(b[:, f], minlength=nb), league)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    abbrs = task.team_abbrs()
    team_splits = ev.client_splits("team", args.seed)
    iid_splits = ev.client_splits("iid", args.seed)
    spec = HG.BinSpec(list(team_splits[0][0].columns), BIN_STRIDE)

    # ── 1. label shift ──
    y = [s[2].to_numpy(float) for s in team_splits]
    rates = np.array([v.mean() for v in y])
    sizes = np.array([len(v) for v in y])
    pooled_y = np.concatenate(y)
    obs_sd = rates.std(ddof=1)
    perm_sd = np.empty(args.n_perm)
    for i in range(args.n_perm):
        sh = rng.permutation(pooled_y)
        cuts = np.cumsum(sizes)[:-1]
        perm_sd[i] = np.array([v.mean() for v in np.split(sh, cuts)]).std(ddof=1)
    p_val = (1 + np.sum(perm_sd >= obs_sd)) / (1 + args.n_perm)
    iid_rates = np.array([s[2].mean() for s in iid_splits])

    print(f"1. LABEL SHIFT — make rate per team (seed {args.seed})")
    print(f"   league {pooled_y.mean():.4f} | teams {rates.min():.4f}–{rates.max():.4f}, SD {obs_sd:.4f}"
          f" | shuffling teams gives SD {perm_sd.mean():.4f} (permutation p = {p_val:.4f})")
    print(f"   IID control: SD {iid_rates.std(ddof=1):.4f}")
    print(f"   → teams differ by ~{100 * (rates.max() - rates.min()):.1f} FG% points, "
          f"{obs_sd / perm_sd.mean():.1f}× the spread of a random split")

    # ── 2. feature shift ──
    jsd_team = feature_divergences(team_splits, spec)
    jsd_iid = feature_divergences(iid_splits, spec)
    feats = pd.DataFrame({"feature": spec.features,
                          "jsd_team_mean": jsd_team.mean(0), "jsd_team_max": jsd_team.max(0),
                          "jsd_iid_mean": jsd_iid.mean(0)}).sort_values("jsd_team_mean", ascending=False)
    feats["ratio_team_over_iid"] = feats["jsd_team_mean"] / feats["jsd_iid_mean"].clip(lower=1e-9)
    feats.to_csv(RESULTS_FED / "team_heterogeneity_features.csv", index=False)
    print(f"\n2. FEATURE SHIFT — Jensen-Shannon divergence vs the league, mean over teams (bits)")
    print(f"   all features: team silos {jsd_team.mean():.4f} | IID control {jsd_iid.mean():.4f}"
          f"  ({jsd_team.mean() / max(jsd_iid.mean(), 1e-9):.1f}×)")
    for _, r in feats.head(6).iterrows():
        print(f"     {r['feature']:<24} {r['jsd_team_mean']:.4f}  (IID {r['jsd_iid_mean']:.4f}, "
              f"{r['ratio_team_over_iid']:.1f}×, worst team {r['jsd_team_max']:.4f})")

    # ── 3. task shift: train on one team, score every team ──
    X_te, y_te, _ = ev.global_test()
    df = task.load_full_dataset()
    _, test_idx = task._get_global_split()
    team_col = df.iloc[test_idx]["team_id"].to_numpy()
    d_te = xgb.DMatrix(X_te)
    params = ev.xgb_params(args.seed)
    preds = []
    for xt, xv, yt, yv in team_splits:
        preds.append(ev.train_early_stopped(params, xt, yt.to_numpy(float), xv, yv.to_numpy(float)).predict(d_te))
    ids = task.get_team_ids()
    M = np.full((len(ids), len(ids)), np.nan)              # M[trained_on, scored_on]
    for j, tid in enumerate(ids):
        m = team_col == tid
        if len(np.unique(y_te[m])) < 2:
            continue
        for i in range(len(ids)):
            M[i, j] = ev.auc(y_te[m], preds[i][m])
    own = np.diag(M)
    others = np.array([np.nanmean(np.delete(M[:, j], j)) for j in range(len(ids))])
    print(f"\n3. TASK SHIFT — one team's model scored on each team's own held-out shots")
    print(f"   own model {np.nanmean(own):.4f} | the other 29 teams' models {np.nanmean(others):.4f}"
          f" | difference {np.nanmean(own - others):+.4f}")
    print(f"   → a team's own model is {'no better' if abs(np.nanmean(own - others)) < 0.01 else 'better'} "
          f"on its own shots than its rivals' models are: the shot-making relationship transfers")

    pd.DataFrame({"team": abbrs, "train_shots": sizes, "make_rate": rates,
                  "jsd_mean": jsd_team.mean(1), "own_model_auc": own,
                  "other_teams_models_auc": others}).to_csv(RESULTS_FED / "team_heterogeneity.csv", index=False)
    np.savetxt(RESULTS_FED / "team_heterogeneity_transfer_matrix.csv", M, delimiter=",")
    plot(abbrs, rates, sizes, pooled_y.mean(), iid_rates, feats)


def plot(abbrs, rates, sizes, league, iid_rates, feats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    team_c, iid_c = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=200)
    fig.patch.set_facecolor(surface)

    ax = axes[0]; ax.set_facecolor(surface)
    o = np.argsort(rates)
    se = np.sqrt(rates[o] * (1 - rates[o]) / sizes[o])
    ax.errorbar(rates[o], np.arange(len(o)), xerr=1.96 * se, ls="", marker="o", ms=5, color=team_c,
                elinewidth=1.2, capsize=0, mec=surface, mew=1)
    ax.axvline(league, color=ink, lw=1.2, ls=(0, (1, 2)))
    ax.text(league, len(o) - 0.3, " league", color=ink, fontsize=8, va="top")
    ax.set_yticks(np.arange(len(o))); ax.set_yticklabels(np.array(abbrs)[o], fontsize=7)
    ax.set_xlabel("Make rate on the team's training shots (95% binomial CI)", color=ink2, fontsize=9)
    ax.set_title("Teams convert at different rates", loc="left", fontsize=10.5, color=ink,
                 fontweight="bold", pad=20)
    ax.text(0, 1.012, "label shift: FG% on each team's own shots, vs the league",
            transform=ax.transAxes, fontsize=8, color=ink2)

    ax = axes[1]; ax.set_facecolor(surface)
    top = feats.head(10).iloc[::-1]
    yy = np.arange(len(top)); h = 0.38
    ax.barh(yy + h / 2, top["jsd_team_mean"], h, color=team_c, label="Real teams")
    ax.barh(yy - h / 2, top["jsd_iid_mean"], h, color=iid_c, label="Random split (control)")
    ax.set_yticks(yy); ax.set_yticklabels(top["feature"], fontsize=7.5)
    ax.set_xlabel("Distance from the league's distribution — Jensen-Shannon divergence (bits)",
                  color=ink2, fontsize=9)
    ax.set_title("Teams take different kinds of shots", loc="left", fontsize=10.5, color=ink,
                 fontweight="bold", pad=20)
    ax.text(0, 1.012, "feature shift: the 10 features that differ most between teams "
            "(0 = same as the league)", transform=ax.transAxes, fontsize=8, color=ink2)
    ax.legend(frameon=False, fontsize=8, labelcolor=ink, loc="lower right")

    for ax in axes:
        ax.grid(axis="x", color=grid, lw=0.8); ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
    fig.tight_layout()
    out = RESULTS_FED / "team_heterogeneity.png"
    fig.savefig(out, facecolor=surface); plt.close(fig)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
