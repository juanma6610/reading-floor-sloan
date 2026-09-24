"""
leakage_attack_hist.py — the leakage_attack.py reconstruction, rerun against what the
HISTOGRAM protocol actually sends (nba_federated/hist_gbdt.py), first tree, all 30 teams.

In tree 1 every shot starts from the same public prior p0 (league FG%), so any sum of
gradients g = p0 − y and hessians h = p0(1 − p0) over a set of shots gives

    n     = H / (p0 (1 − p0))        shots in that region
    makes = n · p0 − G               made shots in that region

The regions are what the protocol releases: in "hist" mode the (feature, bin) cells of
the root histogram (each team's shooting in every public bin of every released
feature); in "random" mode (the DP setting) the leaves of the public random tree.

What the server gets to see:
  no SecAgg   each team's own vector (+ its 1/√K share of the DP noise, if any)
  SecAgg+     only the sum over the 30 teams (+ the full DP noise). The best it can
              say about one team is the league figure for that region.

Without SecAgg, one tree understates the risk: a team's release carries only its 1/√K
share of the DP noise and the server sees every tree, so the script also prints the
formal per-team ε in that case (ε = 1 with SecAgg+ becomes ε ≈ 6 without it).

Scored per team and region against the team's true (n, makes), for regions with at
least --min-shots shots, next to the error of guessing the league-average make rate.
Row subsampling is off (subsample = 1) so exact recovery is checkable; with the
protocol's 0.8 the counts describe an unobserved 80% sample instead.

Run from the project root:
    python src/federated/leakage_attack_hist.py

Outputs (results/federated/): leakage_attack_hist_summary.csv, leakage_attack_hist_regions.csv
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
import sim_histogram as S                                   # noqa: E402  (setup, fixed DP config)
from nba_federated import hist_gbdt as HG                    # noqa: E402

# (label, split mode, ε, privacy unit, SecAgg+)
SCENARIOS = [
    ("histogram, no SecAgg",               "hist",   0.0, "shot",   False),
    ("histogram + SecAgg+",                "hist",   0.0, "shot",   True),
    ("DP ε=1 (shot), no SecAgg",           "random", 1.0, "shot",   False),
    ("DP ε=1 (shot) + SecAgg+",            "random", 1.0, "shot",   True),
    ("DP ε=8 (player), no SecAgg",         "random", 8.0, "player", False),
    ("DP ε=8 (player) + SecAgg+",          "random", 8.0, "player", True),
]


def params_for(mode: str, eps: float, unit: str) -> HG.Params:
    if mode == "hist":                                       # the no-DP protocol settings
        return HG.Params(**{**S.PROTOCOL, "subsample": 1.0})
    f = {k: v for k, v in S.FIXED.items()}                   # the headline DP configuration
    return HG.Params(**{**f, "subsample": 1.0}, dp_epsilon=eps, dp_unit=unit)


def released_sums(setup: S.Setup, p: HG.Params, noise_seed: int):
    """Each team's first-tree release (G, H) as the client code produces it, and the
    region each of its shots falls in (bin index per released feature, or leaf)."""
    trainer = HG.Trainer(setup.clients(noise_seed), setup.spec, replace(p, seed=noise_seed))
    rng = np.random.default_rng(noise_seed)
    feats = np.sort(rng.choice(len(setup.spec.features), trainer.n_features_per_tree(), replace=False))
    sd_g, sd_h = trainer._client_sd()
    builder = HG.TreeBuilder(setup.spec, p, feats, trainer.sigma, rng)
    sent, region_of = [], []
    width = setup.spec.n_bins[feats] + 1
    offs = np.concatenate([[0], np.cumsum(width)[:-1]])
    for c in trainer.clients:
        c.begin_tree(p.subsample)
        if p.split_mode == "hist":
            sent.append([a.ravel() for a in c.histograms(1, feats, sd_g, sd_h)])
            region_of.append(offs[None, :] + c.bins[:, feats])          # one region per (feature, bin)
        else:
            clip = trainer._clip()
            sent.append(list(c.leaf_sums(builder.tree, sd_g, sd_h, clip)))
            region_of.append(builder.tree.route(c.bins, setup.spec.n_bins)[:, None])
    return sent, region_of


def attack(setup, label, mode, eps, unit, secagg, noise_seed, min_shots):
    p = params_for(mode, eps, unit)
    sent, region_of = released_sums(setup, p, noise_seed)
    p0 = setup.p0; q = p0 * (1 - p0)
    K = len(sent)
    if secagg:                                  # server sees only the aggregate
        G_sum, H_sum = sum(s[0] for s in sent), sum(s[1] for s in sent)
        n_league = H_sum / q
        rate_league = (n_league * p0 - G_sum) / np.where(np.abs(n_league) > 1e-9, n_league, np.nan)
    rows, per_region = [], []
    for t, ((G, H), reg) in enumerate(zip(sent, region_of)):
        y = setup.client_bins[t][1]
        size = len(G)
        n_true = np.zeros(size); m_true = np.zeros(size)
        for col in range(reg.shape[1]):
            n_true += np.bincount(reg[:, col], minlength=size)
            m_true += np.bincount(reg[:, col], weights=y, minlength=size)
        if secagg:
            n_hat, rate_hat = n_league / K, rate_league
            m_hat = n_hat * rate_hat
        else:
            n_hat = H / q
            m_hat = n_hat * p0 - G
            rate_hat = m_hat / np.where(np.abs(n_hat) > 1e-9, n_hat, np.nan)
        keep = n_true >= min_shots
        true_rate = m_true[keep] / n_true[keep]
        exact = (np.round(n_hat[keep]) == n_true[keep]) & (np.round(m_hat[keep]) == m_true[keep])
        err = np.abs(np.clip(rate_hat[keep], 0, 1) - true_rate)
        rows.append({"scenario": label, "team": t, "regions": int(keep.sum()), "exact": int(exact.sum()),
                     "median_abs_err_make_rate": float(np.nanmedian(err)),
                     "median_abs_err_guess_league_avg": float(np.median(np.abs(true_rate - p0)))})
        per_region.append(pd.DataFrame({"scenario": label, "team": t, "n_true": n_true[keep],
                                        "makes_true": m_true[keep], "n_hat": n_hat[keep], "makes_hat": m_hat[keep]}))
    return rows, per_region


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-seed", type=int, default=42)
    ap.add_argument("--min-shots", type=int, default=10, help="score regions with at least this many shots")
    args = ap.parse_args()

    setups = {"hist": S.Setup(42, 1), "random": S.Setup(42, S.FIXED["bin_stride"])}
    summary, regions = [], []
    for label, mode, eps, unit, secagg in SCENARIOS:
        r, pr = attack(setups[mode], label, mode, eps, unit, secagg, args.noise_seed, args.min_shots)
        summary += r; regions += pr
    summary = pd.DataFrame(summary)
    RESULTS_FED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_FED / "leakage_attack_hist_summary.csv", index=False)
    pd.concat(regions).to_csv(RESULTS_FED / "leakage_attack_hist_regions.csv", index=False)

    print(f"First tree, 30 teams, regions with ≥ {args.min_shots} shots. p0 = public league FG% {setups['hist'].p0}\n")
    print(f"{'scenario':32s} {'regions':>8} {'exactly recovered':>18} {'median |err| FG%':>17} {'(guess avg)':>12}")
    for label, g in summary.groupby("scenario", sort=False):
        print(f"{label:32s} {g['regions'].sum():>8d} {g['exact'].sum():>9d} ({100 * g['exact'].sum() / g['regions'].sum():5.1f}%)"
              f" {100 * g['median_abs_err_make_rate'].median():>16.1f}% {100 * g['median_abs_err_guess_league_avg'].median():>11.1f}%")
    print("\n  With SecAgg+ the attacker's per-team estimate is the league figure for the region —\n"
          "  the same thing the released model tells everyone; no team-specific information remains.")

    # A one-tree attack understates the risk without SecAgg: each team's release then carries
    # only its 1/√K share of the noise, and the server sees every tree. Formal per-team ε:
    print("\n  Formal guarantee against the server (whole run, δ = 1e-5):")
    from nba_federated import dp
    for label, mode, eps, unit, _ in SCENARIOS:
        if eps <= 0 or "no SecAgg" not in label:
            continue
        p = params_for(mode, eps, unit)
        z = dp.gaussian_z_for_epsilon(eps, p.dp_delta, HG.releases_per_shot(p))
        eps_team = dp.rdp_gaussian_epsilon(z / np.sqrt(len(setups[mode].client_bins)), HG.releases_per_shot(p), p.dp_delta)
        print(f"    target ε = {eps:g} ({unit}):  with SecAgg+ ε = {eps:g};  without SecAgg ε = {eps_team:.1f}")


if __name__ == "__main__":
    main()
