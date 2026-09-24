"""
dp_audit.py — does the trained model actually hide who is in the training data?

The accountant promises a bound; this measures what a real attacker gets. For each
privacy setting we train the histogram protocol, then run a membership-inference
attack on the released model:

    members      every client's local TRAINING shots
    non-members  every client's local VALIDATION shots (same teams, different games,
                 never trained on)
    attack       per-shot loss  ℓ = −log p(y);  small loss ⇒ "member"

Reported per setting:
  MIA AUC          0.5 = the attack cannot tell members from non-members
  ε (empirical)    a lower bound from the attack's best (TPR, FPR) operating point,
                   ε ≥ log((TPR − δ)/FPR), with one-sided 95% Clopper-Pearson bounds
                   on TPR/FPR (Jagielski, Ullman & Oprea 2020). It is only a LOWER
                   bound, and a loss-threshold attack is weak, so a large gap to the
                   analytic ε is expected — the audit catches accounting or
                   implementation errors, it does not tighten the guarantee.

Run from the project root (a few minutes per setting):
    python src/federated/dp_audit.py
    python src/federated/dp_audit.py --settings no-dp shot:1 --seeds 42

Outputs (results/federated/): dp_audit.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_federated as ev                      # noqa: E402
import sim_histogram as S                            # noqa: E402
from nba_federated import hist_gbdt as HG            # noqa: E402

DEFAULT_SETTINGS = ["no-dp-hist", "no-dp", "shot:4", "shot:1", "player:8", "player:4"]


def params_for(name: str) -> HG.Params:
    """`no-dp-hist` is the real no-DP protocol (data-driven splits, the setting whose
    accuracy the paper reports); every other setting uses the fixed DP configuration."""
    if name == "no-dp-hist":
        return HG.Params(**S.PROTOCOL, n_trees=600)
    unit, eps = ("shot", 0.0) if name == "no-dp" else (name.split(":")[0], float(name.split(":")[1]))
    return HG.Params(**S.FIXED, dp_epsilon=eps, dp_unit=unit)


def empirical_epsilon(scores_in: np.ndarray, scores_out: np.ndarray, delta: float = 1e-5,
                      alpha: float = 0.05) -> float:
    """Best ε lower bound over thresholds, with one-sided 95% Clopper-Pearson bounds."""
    n_in, n_out = len(scores_in), len(scores_out)
    best = 0.0
    for thr in np.quantile(np.concatenate([scores_in, scores_out]), np.linspace(0.001, 0.999, 200)):
        tp, fp = int((scores_in >= thr).sum()), int((scores_out >= thr).sum())
        # conservative: lower bound on TPR, upper bound on FPR
        tpr_lo = beta.ppf(alpha, tp, n_in - tp + 1) if 0 < tp else 0.0
        fpr_hi = beta.ppf(1 - alpha, fp + 1, n_out - fp) if fp < n_out else 1.0
        if fpr_hi > 0 and tpr_lo > delta:
            best = max(best, np.log((tpr_lo - delta) / fpr_hi))
        # and the mirrored direction (predicting "non-member")
        tnr_lo = beta.ppf(alpha, n_out - fp, fp + 1) if fp < n_out else 0.0
        fnr_hi = beta.ppf(1 - alpha, n_in - tp + 1, tp) if tp > 0 else 1.0
        if fnr_hi > 0 and tnr_lo > delta:
            best = max(best, np.log((tnr_lo - delta) / fnr_hi))
    return float(max(best, 0.0))


def audit_one(setup: S.Setup, name: str, noise_seed: int, members, non_members):
    params = params_for(name)
    trainer, curve = S.train(setup, params, noise_seed=noise_seed, curve=False)
    m0 = setup.m0
    margins = []
    for bins in (members[0], non_members[0]):
        m = np.full(len(bins), m0)
        for t in trainer.trees:
            m += t.predict_bins(bins, setup.spec.n_bins)
        margins.append(m)
    losses = []
    for m, y in zip(margins, (members[1], non_members[1])):
        p = np.clip(1.0 / (1.0 + np.exp(-m)), 1e-12, 1 - 1e-12)
        losses.append(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
    score_in, score_out = -losses[0], -losses[1]          # higher score ⇒ "member"
    y_true = np.concatenate([np.ones(len(score_in)), np.zeros(len(score_out))])
    return {"unit": params.dp_unit, "eps": params.dp_epsilon, "trees": params.n_trees,
            "split_mode": params.split_mode, "bin_stride": params.bin_stride, "noise_seed": noise_seed,
            "test_auc": float(curve.iloc[-1]["test_auc"]), "test_brier": float(curve.iloc[-1]["test_brier"]),
            "mia_auc": ev.auc(y_true, np.concatenate([score_in, score_out])),
            "mean_loss_members": float(losses[0].mean()), "mean_loss_non_members": float(losses[1].mean()),
            "eps_empirical": empirical_epsilon(score_in, score_out, params.dp_delta)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help='e.g. no-dp shot:1 player:8')
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123], help="DP noise seeds")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--merge", action="store_true", help="concatenate dp_audit_*.csv into dp_audit.csv")
    args = ap.parse_args()

    if args.merge:
        parts = sorted(p for p in RESULTS_FED.glob("dp_audit_*.csv") if p.name != "dp_audit.csv")
        d = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(["setting", "noise_seed"])
        d.to_csv(RESULTS_FED / "dp_audit.csv", index=False)
        g = d.groupby("setting").agg(test_auc=("test_auc", "mean"), mia_auc=("mia_auc", "mean"),
                                     mia_sd=("mia_auc", "std"), eps_emp=("eps_empirical", "max"))
        print(g.round(4).to_string())
        return

    # The bin grid is a property of the SETTING, not of the settings list: `no-dp-hist`
    # is the 64-bin no-DP protocol, every DP setting the coarser FIXED grid. Deriving it
    # from the list ran one on the other's grid whenever both were requested, so take it
    # from what the setting itself declares and build one Setup per distinct grid.
    grids: dict[int, tuple] = {}

    def grid_for(name: str):
        """(setup, members, non_members) on the bin grid this setting declares."""
        stride = params_for(name).bin_stride
        if stride not in grids:
            st = S.Setup(42, stride)
            members = (np.concatenate([b for b, _ in st.client_bins]),
                       np.concatenate([y for _, y in st.client_bins]))
            grids[stride] = (st, members, (st.b_va, st.y_va))
            print(f"  [bin stride {stride}: {int(st.spec.n_bins.max())} bins per continuous feature] "
                  f"members {len(members[1])} training shots | non-members {len(st.y_va)} validation shots")
        return grids[stride]

    print(f"fixed config: {S.FIXED['n_trees']} trees, depth {S.FIXED['max_depth']}, η {S.FIXED['eta']}\n")

    rows = []
    for name in args.settings:
        setup, members, non_members = grid_for(name)
        for s in args.seeds:
            rows.append({"setting": name, **audit_one(setup, name, s, members, non_members)})
            pd.DataFrame(rows).to_csv(RESULTS_FED / f"dp_audit_{args.tag}.csv", index=False)
        g = pd.DataFrame([r for r in rows if r["setting"] == name])
        print(f"  {name:10s} test AUC {g['test_auc'].mean():.4f} | MIA AUC {g['mia_auc'].mean():.4f} "
              f"± {g['mia_auc'].std(ddof=1):.4f} | ε empirical ≥ {g['eps_empirical'].max():.3f} "
              f"| loss members {g['mean_loss_members'].mean():.4f} vs non-members {g['mean_loss_non_members'].mean():.4f}",
              flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(RESULTS_FED / f"dp_audit_{args.tag}.csv", index=False)
    print("\n  MIA AUC 0.5 = attack no better than chance. ε empirical is a LOWER bound from a "
          "loss-threshold attack;\n  a gap to the analytic ε is expected and does not tighten the guarantee.")


if __name__ == "__main__":
    main()
