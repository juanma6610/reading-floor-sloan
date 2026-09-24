"""
leakage_attack.py — what an honest-but-curious server learns from ONE federated tree.

In FedXgbBagging every client sends, in round 1, a tree trained on its private
shots, serialised as XGBoost JSON. Besides the split structure, that JSON carries
per-node statistics XGBoost stores for its own bookkeeping:

    sum_hessian[i]   H_i = Σ p(1−p) over the node's shots
    base_weights[i]  w_i = −G_i / (H_i + λ)          (internal nodes, unscaled)
    leaf value       η · w_i                         (leaves; = base_weights at leaves)
    loss_changes[i]  G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_P²/(H_P+λ)

where G = Σ (p − y). In round 1 every shot starts from the same prior p0 (the
global make rate, sent by the server), so for every node

    n     = H / (p0 (1 − p0))        number of the team's shots in that region
    makes = n · p0 − G               how many of them went in

Attack A (plain protocol): read H and the leaf value of each leaf -> exact
    (n, makes) for every region of feature space the tree carves out, e.g.
    "shots from 21.9+ ft with the closest defender < 2.9 ft: 66 shots, 15 makes".

Attack B (DP prototype on, nba_federated/dp.py): the DP layer noises leaf values
    only. Every internal node's weight, every split gain and every sum_hessian is
    still sent in the clear, so each leaf's G is recovered from its parent:
      * sibling is internal:  G_leaf = G_parent − G_sibling
      * sibling is a leaf:    solve the loss_changes quadratic for G_L; of its two
                              roots, keep the one giving an integer makes count
    The leaf noise is simply ignored. DP at ε = 1 leaks exactly as much as no DP,
    with one blind spot: two sibling leaves with the SAME shot count make the
    equation symmetric, so the server learns both make counts but not which
    leaf has which (flagged `swap_ambiguous`).

Attack C (layer-1 hardening on, nba_federated/hardening.py): with sum_hessian,
    gains and internal weights zeroed, counts are gone. But a leaf value still is
    w = n(r − p0)/(n·p0(1−p0) + λ), so r ≈ p0 + p0(1−p0)·w: the region's make
    rate, shrunk toward the league average. Compared against simply guessing the
    league average, this measures what hardening alone still leaks.

Ground truth is scored by routing the client's own training shots through the
tree. XGBoost's row subsample (0.8 in pyproject.toml) is not observable, so the
exact check uses subsample = 1.0; with the protocol's 0.8 the recovered counts
describe the sampled 80% and the recovered make rate is compared to the true
rate over all of the team's shots in that region.

Run from the project root:
    python src/federated/leakage_attack.py
    python src/federated/leakage_attack.py --clients 0 5 --dp-epsilon 1 10

Outputs (results/federated/):
    leakage_attack_summary.csv   one row per (client, scenario)
    leakage_attack_leaves.csv    every recovered region: rule, n, makes, truth
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"

sys.path.insert(0, str(SCRIPT_DIR))
from nba_federated import dp, hardening, task                             # noqa: E402
from nba_federated.client_app import _build_xgb_params, _model_to_params  # noqa: E402

NUM_CLIENTS = 30


# ──────────────────────────────────────────────────────────────
# Client side: exactly what client_app.fit sends in round 1
# ──────────────────────────────────────────────────────────────
def run_config() -> dict:
    cfg = tomllib.loads((SCRIPT_DIR / "pyproject.toml").read_text())["tool"]["flwr"]["app"]["config"]
    return {**cfg, "params.base_score": task.get_global_make_rate(), "dp-num-rounds": cfg["num-server-rounds"]}


def client_round1_message(pid: int, cfg: dict, seed: int, subsample: float | None,
                          dp_epsilon: float, dp_rng: np.random.Generator, hardened: bool = False):
    """Train the client's round-1 tree as client_app.fit does; return (bytes sent, local train data)."""
    X_tr, _, y_tr, _ = task.partition_frames(pid, NUM_CLIENTS, "team", random_state=seed)
    params = _build_xgb_params({**cfg, "seed": seed})
    if subsample is not None:
        params["subsample"] = subsample
    edges = hardening.public_edges(list(X_tr.columns)) if hardened else None
    X_fit = hardening.bin_to_public_edges(X_tr, edges) if hardened else X_tr
    bst = xgb.train(params, xgb.DMatrix(X_fit, label=y_tr), num_boost_round=1)
    if hardened:
        bst = hardening.snap_thresholds(bst, edges)
    bst = dp.maybe_dp(bst, 1, {**cfg, "dp-epsilon": dp_epsilon}, dp_rng)
    if hardened:
        bst = hardening.strip_bookkeeping(bst)
    return _model_to_params(bst).tensors[0], X_tr, y_tr.to_numpy()


# ──────────────────────────────────────────────────────────────
# Server side: uses ONLY the received bytes + public config (p0, λ, η)
# ──────────────────────────────────────────────────────────────
def _tree(message: bytes) -> tuple[dict, list[str]]:
    model = json.loads(message)["learner"]
    return model["gradient_booster"]["model"]["trees"][0], model["feature_names"]


def gradient_sums_plain(t: dict, lam: float, eta: float) -> dict[int, float]:
    """Attack A: G at each leaf from its (η-scaled) leaf value and H."""
    return {i: -(t["split_conditions"][i] / eta) * (t["sum_hessian"][i] + lam)
            for i, c in enumerate(t["left_children"]) if c == -1}


def gradient_sums_structural(t: dict, lam: float, p0: float) -> tuple[dict[int, float], set[int]]:
    """Attack B: G at each leaf WITHOUT reading any leaf value (defeats leaf-only DP).

    Also returns the leaves in equal-count sibling pairs, whose two values are known
    but whose assignment to left/right is not.
    """
    L, R, H, W, gain = (t["left_children"], t["right_children"], t["sum_hessian"],
                        t["base_weights"], t["loss_changes"])
    q = p0 * (1 - p0)
    G_internal = {i: -W[i] * (H[i] + lam) for i, c in enumerate(L) if c != -1}
    out, ambiguous = {}, set()
    for parent, G_p in G_internal.items():
        a, b = L[parent], R[parent]
        if a in G_internal or b in G_internal:          # one child internal → subtraction
            if a not in G_internal:
                out[a] = G_p - G_internal[b]
            if b not in G_internal:
                out[b] = G_p - G_internal[a]
            continue
        # Both children are leaves: G_a²/ha + (G_p − G_a)²/hb = gain + G_p²/hp
        ha, hb, hp = H[a] + lam, H[b] + lam, H[parent] + lam
        k = gain[parent] + G_p ** 2 / hp
        roots = np.roots([1 / ha + 1 / hb, -2 * G_p / hb, G_p ** 2 / hb - k]).real
        n_a = H[a] / q

        def off_integer(g):           # makes = n·p0 − G must be an integer in [0, n]
            m = n_a * p0 - g
            return abs(m - round(m)) + (0 if -0.5 <= m <= n_a + 0.5 else 1e9)
        g_a = min(roots, key=off_integer)
        out[a], out[b] = g_a, G_p - g_a
        if H[a] > 0 and round(H[a] / q) == round(H[b] / q):
            ambiguous |= {a, b}
    return out, ambiguous


def leaf_rules(t: dict, feature_names: list[str]) -> dict[int, str]:
    """Human-readable region for each leaf (conjunction of the splits on its path)."""
    L, R, feat, thr, dflt = (t["left_children"], t["right_children"], t["split_indices"],
                             t["split_conditions"], t["default_left"])
    rules, stack = {}, [(0, [])]
    while stack:
        i, conds = stack.pop()
        if L[i] == -1:
            rules[i] = " & ".join(conds) or "(all shots)"
            continue
        f, v = feature_names[feat[i]], thr[i]
        miss_l, miss_r = (" or missing", "") if dflt[i] else ("", " or missing")
        stack.append((L[i], conds + [f"{f} < {v:.3g}{miss_l}"]))
        stack.append((R[i], conds + [f"{f} >= {v:.3g}{miss_r}"]))
    return rules


def recover(message: bytes, p0: float, lam: float, eta: float, attack: str) -> pd.DataFrame:
    """attack: "plain" (A), "structural" (B) or "rate_only" (C, leaf values alone)."""
    t, names = _tree(message)
    rules = leaf_rules(t, names)
    q = p0 * (1 - p0)
    if attack == "rate_only":
        return pd.DataFrame([{"leaf": i, "rule": rules[i], "n_hat": np.nan, "makes_hat": np.nan,
                              "rate_hat": p0 + q * t["split_conditions"][i] / eta, "swap_ambiguous": False}
                             for i, c in enumerate(t["left_children"]) if c == -1])
    G, ambiguous = (gradient_sums_structural(t, lam, p0) if attack == "structural"
                    else (gradient_sums_plain(t, lam, eta), set()))
    rows = []
    for leaf, g in G.items():
        n = t["sum_hessian"][leaf] / q
        makes = n * p0 - g
        rows.append({"leaf": leaf, "rule": rules[leaf], "n_hat": n, "makes_hat": makes,
                     "rate_hat": makes / n if n > 0 else np.nan, "swap_ambiguous": leaf in ambiguous})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Scoring against the client's private data
# ──────────────────────────────────────────────────────────────
def ground_truth(message: bytes, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    bst = xgb.Booster(); bst.load_model(bytearray(message))
    leaf_of = bst.predict(xgb.DMatrix(X), pred_leaf=True).astype(int).ravel()
    df = pd.DataFrame({"leaf": leaf_of, "y": y})
    return df.groupby("leaf")["y"].agg(n_true="size", makes_true="sum").reset_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clients", type=int, nargs="+", default=list(range(NUM_CLIENTS)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dp-epsilon", type=float, nargs="+", default=[1.0, 0.1],
                    help="total ε for the DP scenarios (the DP prototype in nba_federated/dp.py)")
    args = ap.parse_args()

    cfg = run_config()
    p0 = cfg["params.base_score"]
    lam, eta = float(cfg["params.reg_lambda"]), float(cfg["params.eta"])
    teams = task.team_abbrs()
    print(f"Prior p0 = {p0:.4f}  λ = {lam}  η = {eta}  clients = {len(args.clients)}  seed = {args.seed}\n")

    e1 = args.dp_epsilon[0]
    # (label, row subsample, dp ε, attack, layer-1 hardening)
    scenarios = [("plain, subsample=1.0", 1.0, 0.0, "plain", False),
                 ("plain, protocol subsample", None, 0.0, "plain", False)]
    scenarios += [(f"DP eps={e:g}, subsample=1.0", 1.0, e, "structural", False) for e in args.dp_epsilon]
    scenarios += [(f"DP eps={e1:g}, leaf-value attack", 1.0, e1, "plain", False),
                  ("hardened, count attack (A)", 1.0, 0.0, "plain", True),
                  ("hardened, structural attack (B)", 1.0, 0.0, "structural", True),
                  ("hardened, rate-from-leaf (C)", 1.0, 0.0, "rate_only", True),
                  (f"hardened + DP eps={e1:g}, (C)", 1.0, e1, "rate_only", True)]

    summary, leaves = [], []
    for label, subsample, eps, attack, hardened in scenarios:
        for pid in args.clients:
            msg, X, y = client_round1_message(pid, cfg, args.seed, subsample, eps,
                                              np.random.default_rng(1000 + pid), hardened)
            rec = recover(msg, p0, lam, eta, attack).merge(ground_truth(msg, X, y), on="leaf", how="left")
            rec["n_true"] = rec["n_true"].fillna(0); rec["makes_true"] = rec["makes_true"].fillna(0)
            exact_mask = (rec["n_hat"].round() == rec["n_true"]) & (rec["makes_hat"].round() == rec["makes_true"])
            true_rate = rec["makes_true"] / rec["n_true"].clip(lower=1)
            rate_err = (rec["rate_hat"] - true_rate).abs().replace([np.inf, -np.inf], np.nan)
            summary.append({
                "scenario": label, "client": pid, "team": teams[pid], "leaves": len(rec),
                "team_shots": len(y), "shots_in_leaves_hat": rec["n_hat"].sum(),
                "leaves_exact": int(exact_mask.sum()),
                "leaves_swap_ambiguous": int(rec["swap_ambiguous"].sum()),
                "max_abs_err_n": float((rec["n_hat"] - rec["n_true"]).abs().max()),
                "max_abs_err_makes": float((rec["makes_hat"] - rec["makes_true"]).abs().max()),
                "median_abs_err_make_rate": float(rate_err.median()) if rate_err.notna().any() else np.nan,
                "median_abs_err_guess_league_avg": float((p0 - true_rate).abs().median()),
                "n_hat_over_n_true": (float(rec["n_hat"].sum() / rec["n_true"].sum())
                                      if rec["n_hat"].fillna(0).sum() > 0 else np.nan),
            })
            leaves.append(rec.assign(scenario=label, client=pid, team=teams[pid]))

    summary = pd.DataFrame(summary)
    leaves = pd.concat(leaves, ignore_index=True)
    RESULTS_FED.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_FED / "leakage_attack_summary.csv", index=False)
    leaves.to_csv(RESULTS_FED / "leakage_attack_leaves.csv", index=False)

    print(f"{'scenario':32s} {'leaves':>7} {'exactly recovered':>18} {'swap-ambig.':>12} "
          f"{'median |err| FG%':>17} {'(guess avg)':>12} {'shots seen':>11}")
    for label, g in summary.groupby("scenario", sort=False):
        err = g["median_abs_err_make_rate"].median(skipna=True) if g["median_abs_err_make_rate"].notna().any() else np.nan
        err_s = f"{100 * err:>16.1f}%" if np.isfinite(err) else f"{'n/a':>17}"
        seen = g["n_hat_over_n_true"].mean()
        seen_s = f"{100 * seen:>10.0f}%" if np.isfinite(seen) else f"{'n/a':>11}"
        print(f"{label:32s} {g['leaves'].sum():>7d} {g['leaves_exact'].sum():>9d} "
              f"({100 * g['leaves_exact'].sum() / g['leaves'].sum():5.1f}%) "
              f"{g['leaves_swap_ambiguous'].sum():>12d} {err_s} "
              f"{100 * g['median_abs_err_guess_league_avg'].median():>11.1f}% {seen_s}")
    print("  (guess avg) = error of just guessing the league-average make rate for every region.")
    print("  swap-ambig. = leaves in equal-count sibling pairs: both make counts recovered, order unknown.\n"
          "  shots seen  = recovered shot count / team's shots in those regions. With the protocol's row\n"
          "  subsample (0.8) counts describe the unobserved 80% sample, so they are not exact, but each\n"
          "  region's make rate is still recovered up to sampling noise.")

    ex = leaves[(leaves["scenario"] == scenarios[2][0]) & (leaves["client"] == args.clients[0])]
    ex = ex.sort_values("n_true", ascending=False).head(5)
    print(f"\nExample — {teams[args.clients[0]]}, recovered from its DP-noised (ε={args.dp_epsilon[0]:g}) round-1 tree:")
    for _, r in ex.iterrows():
        print(f"  {r['rule']}\n      server: {r['n_hat']:.0f} shots, {r['makes_hat']:.0f} makes"
              f"   |   truth: {r['n_true']:.0f} shots, {r['makes_true']:.0f} makes")
    print(f"\nWrote {(RESULTS_FED / 'leakage_attack_summary.csv').relative_to(PROJECT_ROOT)} and leakage_attack_leaves.csv")


if __name__ == "__main__":
    main()
