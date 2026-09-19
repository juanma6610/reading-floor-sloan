"""
dp.py — differential privacy for the federated XGBoost protocol (leaf perturbation).

STATUS: the Rényi-DP accountant below (rdp_gaussian_epsilon, gaussian_z_for_epsilon)
is what the histogram protocol (hist_gbdt.py) uses. The leaf-perturbation path
(maybe_dp / perturb_*) for tree bagging is SUPERSEDED and is not a valid DP
guarantee: it noises leaf values only, while split structure, thresholds,
sum_hessian, split gains and internal-node weights are sent unperturbed, from
which leakage_attack.py recovers each team's exact shot and make counts.

The natural DP upgrade to the *structural* privacy of federation: instead of
transmitting a client's newly-trained tree(s) verbatim, clip and noise their leaf
outputs before they leave the client, so the server sees only a
differentially-private version of each contribution.

Two mechanisms (choose with `dp-mechanism`):

  * "gaussian" (DEFAULT, modern) — Gaussian noise + Renyi-DP / moments accountant.
    A record touches one leaf per tree, so the L2 sensitivity of a tree's leaf
    vector to one record is <= 2C. The Gaussian mechanism with std sigma = z*2C
    is (alpha, alpha/(2 z^2))-RDP per round; over R rounds RDP adds, and converts
    to (eps, delta) via  eps = min_alpha [ R*alpha/(2 z^2) + ln(1/delta)/(alpha-1) ].
    We invert that to pick z for a target (eps, delta). This is far tighter than
    basic composition.

  * "laplace" (legacy, loose) — pure eps-DP via Laplace with basic composition:
    scale = 2C / (eps_total / R).

Scope: local, record-level DP (protects one shot). Player-level (user-level) DP
— grouping a player's shots before clipping — is the more meaningful unit and a
documented extension. Leaf values live in `split_conditions[i]` where
`left_children[i] == -1` (mirrored into `base_weights[i]`).
"""
from __future__ import annotations
import json, math
import numpy as np
import xgboost as xgb

_RDP_ORDERS = [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]


# ── Renyi-DP (moments) accountant for the Gaussian mechanism ──────────────────
def rdp_gaussian_epsilon(z: float, rounds: int, delta: float) -> float:
    """(eps) at fixed delta for `rounds` compositions of a Gaussian mech, noise
    multiplier z (= sigma / L2-sensitivity)."""
    if z <= 0:
        return float("inf")
    best = float("inf")
    for a in _RDP_ORDERS:
        eps = rounds * a / (2.0 * z * z) + math.log(1.0 / delta) / (a - 1.0)
        best = min(best, eps)
    return best


def gaussian_z_for_epsilon(eps: float, delta: float, rounds: int,
                           lo: float = 1e-3, hi: float = 1e4, iters: int = 100) -> float:
    """Smallest noise multiplier z achieving (eps, delta) over `rounds` rounds
    (epsilon is monotone-decreasing in z)."""
    if rdp_gaussian_epsilon(hi, rounds, delta) > eps:
        return hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)                     # geometric bisection (z spans decades)
        if rdp_gaussian_epsilon(mid, rounds, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


def gaussian_sigma_basic(eps: float, delta: float, rounds: int, clip: float) -> float:
    """Noise sigma under the LEGACY accounting: basic (linear) composition of the
    classical analytic Gaussian mechanism. eps_round = eps/rounds must be <= 1 for
    the classical bound to be valid. Used only to contrast with RDP."""
    eps_r, delta_r = eps / rounds, delta / rounds
    return (2.0 * clip) * math.sqrt(2.0 * math.log(1.25 / delta_r)) / eps_r


# ── leaf perturbation ─────────────────────────────────────────────────────────
def _perturb(booster, k, clip, rng, mechanism, noise):
    if k <= 0 or noise <= 0:
        return booster
    j = json.loads(bytes(booster.save_raw("json")))
    trees = j["learner"]["gradient_booster"]["model"]["trees"]
    if not trees:
        return booster
    for t in trees[-k:]:
        left, sc, bw = t["left_children"], t["split_conditions"], t["base_weights"]
        for i, lc in enumerate(left):
            if lc == -1:
                base = float(np.clip(float(sc[i]), -clip, clip))
                n = rng.laplace(0.0, noise) if mechanism == "laplace" else rng.normal(0.0, noise)
                sc[i] = bw[i] = base + float(n)
    nb = xgb.Booster()
    nb.load_model(bytearray(json.dumps(j).encode("utf-8")))
    return nb


def perturb_gaussian(booster, k, sigma, clip, rng):
    return _perturb(booster, k, clip, rng, "gaussian", sigma)


def perturb_laplace(booster, k, eps_round, clip, rng):
    return _perturb(booster, k, clip, rng, "laplace", (2.0 * clip) / eps_round)


# backward-compatible alias (older callers used the Laplace path)
def perturb_last_trees(booster, k, eps_round, clip, rng):
    return perturb_laplace(booster, k, eps_round, clip, rng)


def maybe_dp(booster: xgb.Booster, num_local_round: int, config: dict,
             rng: np.random.Generator) -> xgb.Booster:
    """Apply local DP to the newly-added trees if `dp-epsilon` > 0, else no-op.

    Config: dp-epsilon (<=0 disables), dp-clip (C, default 0.3),
            dp-num-rounds (R), dp-mechanism ("gaussian"|"laplace"),
            dp-delta (for gaussian, default 1e-5).
    """
    try:
        eps_total = float(config.get("dp-epsilon", 0.0) or 0.0)
    except (TypeError, ValueError):
        eps_total = 0.0
    if eps_total <= 0.0:
        return booster                                   # DP disabled — exact no-op
    clip = float(config.get("dp-clip", 0.3) or 0.3)
    R = int(config.get("dp-num-rounds", config.get("num-server-rounds", 50)) or 50)
    mech = str(config.get("dp-mechanism", "gaussian")).lower()
    if mech == "laplace":
        return perturb_laplace(booster, num_local_round, eps_total / max(R, 1), clip, rng)
    delta = float(config.get("dp-delta", 1e-5) or 1e-5)
    sigma = gaussian_z_for_epsilon(eps_total, delta, R) * (2.0 * clip)   # L2 sensitivity 2C
    return perturb_gaussian(booster, num_local_round, sigma, clip, rng)
