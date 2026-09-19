"""
hardening.py — layer-1 defences against reconstruction from transmitted trees.

Two client-side measures, both switchable from pyproject.toml:

  harden-strip  Zero the per-node bookkeeping XGBoost serialises but prediction
                never reads: `sum_hessian`, `loss_changes` and the weights of
                internal nodes (`base_weights`). These give the server each
                region's exact shot count and make count (leakage_attack.py).
                Predictions are unchanged.

  public-bins   Split thresholds come from a fixed, PUBLIC grid instead of each
                team's own data quantiles (XGBoost `hist` builds its candidate
                cuts from local data, so every threshold otherwise reveals a
                value of the team's feature distribution). The grid is equal-
                width bins over ranges set by physical limits (court size,
                angles, human speed), agreed before training — see PUBLIC_RANGES.
                The client rounds each feature down to its bin's lower edge
                before training, then snaps every threshold up to the next
                public edge. Because every binned value is itself an edge,
                `binned(x) < t` and `x < edge(t)` select the same shots, so the
                tree gives identical predictions on raw features downstream.

Neither measure is a formal guarantee: leaf values are still the team's (shrunk)
make rate per region. That needs calibrated noise on the leaf sums (DP).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

N_BINS = 64          # equal-width bins per continuous feature

# (low, high) public ranges from physical / definitional limits, NOT from data.
# Court: 94 x 50 ft, hoop-side half court for shots; speeds: ~30 ft/s human sprint;
# accelerations: noisy Savitzky-Golay derivatives, bounded generously.
# Values outside a range fall into the end bins (clamping is data-independent).
PUBLIC_RANGES: dict[str, tuple[float, float]] = {
    "dist": (0, 47), "x": (47, 94), "y": (0, 50), "shot_angle": (-180, 180),
    "closest_def_dist": (0, 30), "closest_def_angle": (0, 180),
    "second_closest_def_dist": (0, 50), "second_closest_def_time": (0, 5),
    "time_to_contest": (0, 5), "shot_clock": (0, 24), "touch_time": (0, 12),
    "shooter_par_vel": (-30, 30), "shooter_perp_vel": (0, 30),
    "def_par_vel": (-30, 30), "def_perp_vel": (0, 30),
    "shooter_par_acc": (-150, 150), "shooter_perp_acc": (0, 150),
    "def_par_acc": (-150, 150), "def_perp_acc": (0, 150),
    "ratio_off_def_hull": (0, 30),
    "release_height": (0, 12), "release_speed": (0, 60), "release_angle": (-90, 90),
    "release_x": (47, 94), "release_y": (0, 50),
}
# Small-integer counts: one edge per value.
INTEGER_FEATURES = {"def_very_tight": 5, "def_tight": 5, "def_open": 5}
# Anything else is a flag or a probability in [0, 1] (stype_*, archetype soft labels).
UNIT_INTERVAL_BINS = 20


def public_edges(feature_names: list[str]) -> dict[str, np.ndarray]:
    """Bin lower edges per feature, from public ranges only."""
    edges = {}
    for f in feature_names:
        if f in PUBLIC_RANGES:
            lo, hi = PUBLIC_RANGES[f]
            edges[f] = np.linspace(lo, hi, N_BINS + 1)[:-1]
        elif f in INTEGER_FEATURES:
            edges[f] = np.arange(INTEGER_FEATURES[f] + 1, dtype=float)
        else:
            edges[f] = np.linspace(0, 1, UNIT_INTERVAL_BINS + 1)
    return edges


def bin_to_public_edges(X: pd.DataFrame, edges: dict[str, np.ndarray]) -> pd.DataFrame:
    """Replace each value by the lower edge of its public bin (NaN stays NaN)."""
    out = X.copy()
    for f in X.columns:
        e = edges[f]
        v = X[f].to_numpy(float)
        k = np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(e) - 1)
        out[f] = np.where(np.isnan(v), np.nan, e[k])
    return out


def _edit_trees(booster: xgb.Booster, edit) -> xgb.Booster:
    model = json.loads(bytes(booster.save_raw("json")))
    learner = model["learner"]
    for tree in learner["gradient_booster"]["model"]["trees"]:
        edit(tree, learner["feature_names"])
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(model).encode("utf-8")))
    return out


def snap_thresholds(booster: xgb.Booster, edges: dict[str, np.ndarray]) -> xgb.Booster:
    """Move every split threshold up to the smallest public edge >= it."""
    def edit(tree, names):
        for i, left in enumerate(tree["left_children"]):
            if left == -1:
                continue
            e = edges[names[tree["split_indices"][i]]]
            t = tree["split_conditions"][i]
            j = int(np.searchsorted(e, t, side="left"))
            if j > 0 and abs(e[j - 1] - t) <= 1e-5 * max(1.0, abs(t)):   # float32 round-off of an edge
                j -= 1
            tree["split_conditions"][i] = float(e[min(j, len(e) - 1)])
    return _edit_trees(booster, edit)


def strip_bookkeeping(booster: xgb.Booster) -> xgb.Booster:
    """Zero every field prediction does not need (keeps structure + leaf values)."""
    def edit(tree, _names):
        n = len(tree["left_children"])
        tree["sum_hessian"] = [0.0] * n
        tree["loss_changes"] = [0.0] * n
        tree["base_weights"] = [tree["split_conditions"][i] if tree["left_children"][i] == -1 else 0.0
                                for i in range(n)]
    return _edit_trees(booster, edit)
