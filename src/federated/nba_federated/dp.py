"""
dp.py — the differential-privacy primitives the federated protocols use.

Two unrelated things live here.

1. The accountants and noise used by the HISTOGRAM protocol (hist_gbdt.py), which is
   the protocol the paper reports:
     * Renyi-DP accountant for the Gaussian mechanism, composed over every release a
       unit takes part in, inverted to pick the noise multiplier z for a target
       (eps, delta)  ->  rdp_gaussian_epsilon, gaussian_z_for_epsilon
     * the same for the POISSON-SUBSAMPLED Gaussian, so the subsampling clients
       already do can be credited (only sound under secure aggregation, where the
       server never learns who took part)  ->  rdp_sampled_gaussian,
       sampled_gaussian_epsilon, sampled_gaussian_z_for_epsilon
     * exact discrete Gaussian sampling, so the noise lives on the lattice that
       secure aggregation actually sums  ->  discrete_gaussian, lattice_gaussian

2. Leaf perturbation for the TREE-BAGGING protocol (maybe_dp), kept for one reason:
   it is NOT a valid DP guarantee, and leakage_attack.py uses it to demonstrate that.
   It noises the leaf values of the trees a client transmits while the split
   structure, thresholds, sum_hessian, split gains and internal-node weights go out
   untouched, so each leaf's gradient sum is recoverable from its parent and the
   noise is bypassed. Do not use it as a privacy mechanism.
"""
from __future__ import annotations

import json
import math

import numpy as np
import xgboost as xgb

# ──────────────────────────────────────────────────────────────────────────────
# Renyi-DP accountant — Gaussian mechanism
# ──────────────────────────────────────────────────────────────────────────────
_RDP_ORDERS = [1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256]


def rdp_gaussian_epsilon(z: float, rounds: int, delta: float) -> float:
    """(eps) at fixed delta for `rounds` compositions of a Gaussian mechanism with
    noise multiplier z (= sigma / L2 sensitivity).

    One release is (alpha, alpha/(2 z^2))-RDP; RDP adds over rounds and converts with
    eps = min_alpha [ rounds*alpha/(2 z^2) + ln(1/delta)/(alpha-1) ].
    """
    if z <= 0:
        return float("inf")
    return min(rounds * a / (2.0 * z * z) + math.log(1.0 / delta) / (a - 1.0) for a in _RDP_ORDERS)


def gaussian_z_for_epsilon(eps: float, delta: float, rounds: int,
                           lo: float = 1e-3, hi: float = 1e4, iters: int = 100) -> float:
    """Smallest noise multiplier z achieving (eps, delta) over `rounds` rounds.

    Geometric bisection: epsilon is monotone decreasing in z, and z spans decades.
    """
    if rdp_gaussian_epsilon(hi, rounds, delta) > eps:
        return hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if rdp_gaussian_epsilon(mid, rounds, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


# ──────────────────────────────────────────────────────────────────────────────
# Renyi-DP accountant — Poisson-subsampled Gaussian mechanism
# ──────────────────────────────────────────────────────────────────────────────
# Mironov, Talwar & Zhang (2019), "Renyi Differential Privacy of the Sampled Gaussian
# Mechanism": for integer alpha >= 2, sampling rate q and noise multiplier z,
#   RDP_alpha <= 1/(alpha-1) * log( sum_k C(alpha,k) (1-q)^(alpha-k) q^k e^(k(k-1)/(2 z^2)) ).
_SGM_ORDERS = list(range(2, 257))


def rdp_sampled_gaussian(q: float, z: float, alpha: int) -> float:
    """RDP at integer order `alpha` of one q-subsampled Gaussian release."""
    if z <= 0:
        return float("inf")
    if q >= 1.0:
        return alpha / (2.0 * z * z)
    if q <= 0.0:
        return 0.0
    terms = [math.log(math.comb(alpha, k)) + (alpha - k) * math.log1p(-q)
             + k * math.log(q) + k * (k - 1) / (2.0 * z * z) for k in range(alpha + 1)]
    hi = max(terms)
    return (hi + math.log(sum(math.exp(t - hi) for t in terms))) / (alpha - 1)


def sampled_gaussian_epsilon(q: float, z: float, rounds: int, delta: float) -> float:
    """(eps) at fixed delta for `rounds` compositions of a q-subsampled Gaussian."""
    if z <= 0:
        return float("inf")
    return min(rounds * rdp_sampled_gaussian(q, z, a) + math.log(1.0 / delta) / (a - 1)
               for a in _SGM_ORDERS)


def sampled_gaussian_z_for_epsilon(eps: float, delta: float, rounds: int, q: float,
                                   lo: float = 1e-2, hi: float = 1e4, iters: int = 60) -> float:
    """Smallest z reaching (eps, delta) over `rounds` q-subsampled rounds."""
    if q >= 1.0:
        return gaussian_z_for_epsilon(eps, delta, rounds)
    if sampled_gaussian_epsilon(q, hi, rounds, delta) > eps:
        return hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if sampled_gaussian_epsilon(q, mid, rounds, delta) > eps:
            lo = mid
        else:
            hi = mid
    return hi


# ──────────────────────────────────────────────────────────────────────────────
# Discrete Gaussian noise (for secure aggregation on a lattice)
# ──────────────────────────────────────────────────────────────────────────────
def discrete_gaussian(sigma: float, size, rng: np.random.Generator) -> np.ndarray:
    """Exact samples from the discrete Gaussian N_Z(0, sigma^2) (Canonne, Kamath &
    Steinke 2020, Alg. 3): propose a discrete Laplace — the difference of two
    geometrics — and accept with prob exp(-(|Y| - sigma^2/t)^2 / (2 sigma^2)).

    Secure aggregation sums integers on a lattice, and the sum of independent discrete
    Gaussians is (very nearly) a discrete Gaussian, which is what makes the distributed
    guarantee provable on that lattice (Kairouz et al. 2021), unlike a continuous
    Gaussian quantised after the fact.
    """
    size = int(np.prod(size)) if np.iterable(size) else int(size)
    t = math.floor(sigma) + 1
    p_geom = -math.expm1(-1.0 / t)                      # 1 - exp(-1/t)
    out = np.empty(size)
    todo = np.arange(size)
    while todo.size:
        y = rng.geometric(p_geom, todo.size) - rng.geometric(p_geom, todo.size)
        accept = rng.random(todo.size) < np.exp(-((np.abs(y) - sigma ** 2 / t) ** 2) / (2 * sigma ** 2))
        out[todo[accept]] = y[accept]
        todo = todo[~accept]
    return out


def lattice_gaussian(sigma: float, size, rng: np.random.Generator, step: float) -> np.ndarray:
    """Discrete Gaussian of std `sigma` on the lattice `step`·Z (the SecAgg+ grid)."""
    return step * discrete_gaussian(sigma / step, size, rng)


# ──────────────────────────────────────────────────────────────────────────────
# Leaf perturbation for tree bagging — NOT a valid guarantee, see the module docstring
# ──────────────────────────────────────────────────────────────────────────────
def perturb_leaves(booster: xgb.Booster, k: int, sigma: float, clip: float,
                   rng: np.random.Generator) -> xgb.Booster:
    """Clip the last `k` trees' leaf values to +/- clip and add N(0, sigma^2).

    Leaf values live in `split_conditions[i]` where `left_children[i] == -1`, mirrored
    into `base_weights[i]`. Everything else in the tree is transmitted untouched.
    """
    if k <= 0 or sigma <= 0:
        return booster
    model = json.loads(bytes(booster.save_raw("json")))
    trees = model["learner"]["gradient_booster"]["model"]["trees"]
    for tree in trees[-k:]:
        left, sc, bw = tree["left_children"], tree["split_conditions"], tree["base_weights"]
        for i, child in enumerate(left):
            if child == -1:
                sc[i] = bw[i] = float(np.clip(float(sc[i]), -clip, clip) + rng.normal(0.0, sigma))
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(model).encode("utf-8")))
    return out


def maybe_dp(booster: xgb.Booster, num_local_round: int, config: dict,
             rng: np.random.Generator) -> xgb.Booster:
    """Apply leaf perturbation to the newly added trees if `dp-epsilon` > 0, else a no-op.

    Config: dp-epsilon (<= 0 disables), dp-clip (C, default 0.3), dp-num-rounds (R),
    dp-delta (default 1e-5). A record touches one leaf per tree, so the L2 sensitivity of
    a tree's leaf vector is <= 2C and sigma = z(eps, delta, R) * 2C — which is why the
    noise looks calibrated even though the surrounding statistics give the leaf values
    away anyway (leakage_attack.py).
    """
    try:
        eps_total = float(config.get("dp-epsilon", 0.0) or 0.0)
    except (TypeError, ValueError):
        eps_total = 0.0
    if eps_total <= 0.0:
        return booster                                   # DP disabled — exact no-op
    clip = float(config.get("dp-clip", 0.3) or 0.3)
    rounds = int(config.get("dp-num-rounds", config.get("num-server-rounds", 50)) or 50)
    delta = float(config.get("dp-delta", 1e-5) or 1e-5)
    sigma = gaussian_z_for_epsilon(eps_total, delta, rounds) * (2.0 * clip)
    return perturb_leaves(booster, num_local_round, sigma, clip, rng)
