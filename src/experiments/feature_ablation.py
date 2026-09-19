"""
feature_ablation.py — Iterative feature study for the shot-quality model.

Purpose
-------
Measure, step by step, how much each *group* of features contributes to the
calibrated shot model, and quantify the performance gained by dropping the
portability constraint and adding SHOOTER INFORMATION (identity / skill).

Two studies are produced, both on a game-disjoint split (mirrors
train_xgboost.py so numbers are comparable to the thesis baseline):

  1. CUMULATIVE  — start from geometry only, add one group at a time, log the
                   metric after each addition. Shows marginal gains and makes
                   the "portable vs. +shooter" jump explicit.
  2. LEAVE-ONE-OUT (LOO) — from the full feature set, drop each group in turn
                   and measure the degradation → marginal value *given the rest*.


Plugging in scraped stats
-------------------------
Any column in the CSV whose name starts with ``ext_`` is automatically picked
up as an ``external_scraped`` group. To add nba.com / Basketball-Reference
stats: merge them onto the shot table on ``player_name`` (or player_id),
prefix the new columns with ``ext_`` (e.g. ``ext_usg_pct``, ``ext_efg_c3``,
``ext_prior_season_3p``), save the CSV, and re-run — they will appear in both
studies with no code change.

Usage
-----
    python -m src.experiments.feature_ablation \
        --data data/shot_features_valid2.csv --out results/ablation
"""

from __future__ import annotations
import argparse, json, os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (brier_score_loss, log_loss, roc_auc_score,
                             average_precision_score)
import xgboost as xgb

TARGET = "made_shot"
GROUP  = "game_id"

# ── columns that are never model inputs ──────────────────────────────────────
METADATA = ["player_name", "game_time", "quarter", "score_margin", "team_id",
            "game_id", "description", "closest_def_name"]

# ── XGBoost config (aligned with train_xgboost.py; calibration-first) ────────
XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=1.5, objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", base_score=0.45, n_jobs=-1,
)
EARLY_STOP = 50


# ─────────────────────────────────────────────────────────────────────────────
# Leakage-safe priors
# ─────────────────────────────────────────────────────────────────────────────
def oof_prior(df, tr_idx, va_idx, te_idx, key, target, alpha,
              mask_col=None, n_splits=5):
    """Empirical-Bayes shrunk group prior.

    - val/test rows  : mapped from a prior fit on ALL training rows.
    - train rows      : out-of-fold (GroupKFold on game_id) so a shot never
                        contributes to the prior applied to itself.
    ``mask_col`` (0/1): if given, only rows with mask==1 contribute to the
                        prior, but the prior is attached to every row by ``key``
                        (used for conditional skill, e.g. 3-pt make rate).
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    tr = df.loc[tr_idx]

    def build(frame):
        base = frame if mask_col is None else frame[frame[mask_col] == 1]
        gm = base[target].mean()
        st = base.groupby(key)[target].agg(["mean", "count"])
        pr = (st["mean"] * st["count"] + gm * alpha) / (st["count"] + alpha)
        return pr.to_dict(), gm

    full_pr, full_gm = build(tr)
    for idx in (va_idx, te_idx):
        out.loc[idx] = df.loc[idx, key].map(full_pr).fillna(full_gm)

    gkf = GroupKFold(n_splits=n_splits)
    for a, b in gkf.split(tr, groups=tr[GROUP]):
        idx_a, idx_b = tr.index[a], tr.index[b]
        pr, gm = build(tr.loc[idx_a])
        out.loc[idx_b] = df.loc[idx_b, key].map(pr).fillna(gm)
    return out


def add_shooter_features(df, tr_idx, va_idx, te_idx, a_skill=200, a_rate=100):
    """Attach engineered shooter/defender features. Returns (df, group_dict)."""
    df = df.copy()
    df["_is2"] = 1 - df["is_3_pointer"].astype(int)

    df["shooter_prior"]  = oof_prior(df, tr_idx, va_idx, te_idx, "player_name",     TARGET, a_skill)
    df["def_prior"]      = oof_prior(df, tr_idx, va_idx, te_idx, "closest_def_name", TARGET, a_skill)
    df["shooter_3p_prior"] = oof_prior(df, tr_idx, va_idx, te_idx, "player_name", TARGET, a_rate, mask_col="is_3_pointer")
    df["shooter_2p_prior"] = oof_prior(df, tr_idx, va_idx, te_idx, "player_name", TARGET, a_rate, mask_col="_is2")
    df["shooter_3par"]     = oof_prior(df, tr_idx, va_idx, te_idx, "player_name", "is_3_pointer", a_rate)

    # experience: log(#shots by player in TRAIN); volume, not target → map to all
    cnt = df.loc[tr_idx].groupby("player_name").size()
    df["shooter_experience"] = np.log1p(df["player_name"].map(cnt).fillna(0.0))

    df.drop(columns=["_is2"], inplace=True)
    groups = OrderedDict([
        ("shooter_skill",   ["shooter_prior", "def_prior"]),
        ("shooter_profile", ["shooter_3p_prior", "shooter_2p_prior",
                              "shooter_3par", "shooter_experience"]),
    ])
    return df, groups


# ─────────────────────────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────────────────────────
def build_groups(df):
    G = OrderedDict()
    G["geometry"]   = ["dist", "x", "y", "shot_angle", "is_3_pointer"]
    G["pressure"]   = ["closest_def_dist", "closest_def_angle", "time_to_contest",
                        "second_closest_def_dist", "second_closest_def_time",
                        "def_very_tight", "def_tight", "def_open"]
    G["kinematics"] = ["shooter_par_vel", "shooter_perp_vel", "def_par_vel", "def_perp_vel",
                        "shooter_par_acc", "shooter_perp_acc", "def_par_acc", "def_perp_acc"]
    G["release"]    = ["release_height", "release_speed", "release_angle", "release_x", "release_y"]
    G["tempo"]      = ["shot_clock", "touch_time", "is_catch_and_shoot"]
    G["spacing"]    = ["ratio_off_def_hull"]
    G["archetypes"] = ["Primary_Creator", "Spacer", "Mid-Interior", "Rim_Center",
                        "Paint_Anchors", "Perimeter Guards", "Def_liability", "Switch_Wing"]
    # keep only columns that actually exist
    for k in list(G):
        G[k] = [c for c in G[k] if c in df.columns]
    # PBP shot-type flags (non-portable) become a proper group when present
    stype = [c for c in df.columns if c.startswith("stype_")]
    if stype:
        G["shot_type"] = stype
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Train / evaluate
# ─────────────────────────────────────────────────────────────────────────────
def fit_eval(df, cols, tr_idx, va_idx, te_idx, seed):
    Xtr, ytr = df.loc[tr_idx, cols], df.loc[tr_idx, TARGET]
    Xva, yva = df.loc[va_idx, cols], df.loc[va_idx, TARGET]
    Xte, yte = df.loc[te_idx, cols], df.loc[te_idx, TARGET]
    model = xgb.XGBClassifier(**XGB_PARAMS, random_state=seed,
                              early_stopping_rounds=EARLY_STOP)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    p = model.predict_proba(Xte)[:, 1]
    return {
        "brier":   round(float(brier_score_loss(yte, p)), 5),
        "log_loss": round(float(log_loss(yte, p)), 5),
        "roc_auc": round(float(roc_auc_score(yte, p)), 5),
        "pr_auc":  round(float(average_precision_score(yte, p)), 5),
        "n_feat":  len(cols),
        "best_iter": int(getattr(model, "best_iteration", 0) or 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/shot_features_valid2_type.csv")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.20)
    ap.add_argument("--val-size", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df[df[TARGET].isin([0, 1])].reset_index(drop=True)
    print(f"Loaded {len(df):,} shots | base make-rate {df[TARGET].mean():.4f} "
          f"| {df[GROUP].nunique()} games")

    # ── game-disjoint split: test, then val out of the remainder ─────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    trv_idx, te_idx = next(gss.split(df, groups=df[GROUP]))
    trv = df.iloc[trv_idx]
    gss2 = GroupShuffleSplit(n_splits=1,
                             test_size=args.val_size / (1 - args.test_size),
                             random_state=args.seed)
    tr_rel, va_rel = next(gss2.split(trv, groups=trv[GROUP]))
    tr_idx = trv.index[tr_rel]; va_idx = trv.index[va_rel]; te_idx = df.index[te_idx]
    assert set(df.loc[tr_idx, GROUP]).isdisjoint(df.loc[te_idx, GROUP])
    assert set(df.loc[va_idx, GROUP]).isdisjoint(df.loc[te_idx, GROUP])
    print(f"Split: train {len(tr_idx):,} | val {len(va_idx):,} | test {len(te_idx):,}")

    # ── features ─────────────────────────────────────────────────────────────
    df, shooter_groups = add_shooter_features(df, tr_idx, va_idx, te_idx)
    groups = build_groups(df)
    ext = [c for c in df.columns if c.startswith("ext_")]
    for k, v in shooter_groups.items():
        groups[k] = v
    if ext:
        groups["external_scraped"] = ext
        print(f"Detected {len(ext)} external scraped columns → group 'external_scraped'")

    NONPORTABLE = {"shooter_skill", "shooter_profile", "external_scraped", "shot_type"}
    portable = [g for g in groups if g not in NONPORTABLE]
    addon    = [g for g in groups if g in NONPORTABLE]
    print("Portable groups:", portable)
    print("Non-portable add-on groups:", addon)

    # ── 1) CUMULATIVE ────────────────────────────────────────────────────────
    print("\n=== CUMULATIVE (add one group at a time) ===")
    order = portable + addon
    cum_cols, cum_rows, prev = [], [], None
    for g in order:
        cum_cols += groups[g]
        m = fit_eval(df, cum_cols, tr_idx, va_idx, te_idx, args.seed)
        d_brier = "" if prev is None else round(prev["brier"] - m["brier"], 5)
        d_auc   = "" if prev is None else round(m["roc_auc"] - prev["roc_auc"], 5)
        row = {"step": f"+{g}", **m, "d_brier": d_brier, "d_auc": d_auc}
        cum_rows.append(row); prev = m
        print(f"  +{g:<17} brier={m['brier']:.5f}  logloss={m['log_loss']:.5f}  "
              f"auc={m['roc_auc']:.5f}  (nfeat={m['n_feat']})")
    cum = pd.DataFrame(cum_rows)
    cum.to_csv(f"{args.out}/cumulative.csv", index=False)

    # portable-only vs full headline
    portable_cols = sum((groups[g] for g in portable), [])
    full_cols     = sum((groups[g] for g in order), [])
    m_port = fit_eval(df, portable_cols, tr_idx, va_idx, te_idx, args.seed)
    m_full = fit_eval(df, full_cols,     tr_idx, va_idx, te_idx, args.seed)
    headline = {
        "portable_only": m_port, "full_with_shooter": m_full,
        "delta_brier": round(m_port["brier"] - m_full["brier"], 5),
        "delta_auc":   round(m_full["roc_auc"] - m_port["roc_auc"], 5),
        "delta_logloss": round(m_port["log_loss"] - m_full["log_loss"], 5),
    }

    # ── 2) LEAVE-ONE-GROUP-OUT (from full) ───────────────────────────────────
    print("\n=== LEAVE-ONE-GROUP-OUT (drop from full set) ===")
    loo_rows = []
    for g in order:
        cols = sum((groups[k] for k in order if k != g), [])
        m = fit_eval(df, cols, tr_idx, va_idx, te_idx, args.seed)
        row = {"dropped": g, **m,
               "brier_cost": round(m["brier"] - m_full["brier"], 5),
               "auc_cost":   round(m_full["roc_auc"] - m["roc_auc"], 5)}
        loo_rows.append(row)
        print(f"  -{g:<17} brier={m['brier']:.5f}  auc={m['roc_auc']:.5f}  "
              f"Δbrier(vs full)=+{row['brier_cost']:+.5f}")
    loo = pd.DataFrame(loo_rows).sort_values("brier_cost", ascending=False)
    loo.to_csv(f"{args.out}/leave_one_out.csv", index=False)

    with open(f"{args.out}/headline.json", "w") as f:
        json.dump(headline, f, indent=2)
    print("\n=== HEADLINE: portability dropped → non-portable features (shooter + shot type) added ===")
    print(f"  portable-only : brier {m_port['brier']:.5f}  logloss {m_port['log_loss']:.5f}  auc {m_port['roc_auc']:.5f}")
    print(f"  +shooter info : brier {m_full['brier']:.5f}  logloss {m_full['log_loss']:.5f}  auc {m_full['roc_auc']:.5f}")
    print(f"  gain          : Δbrier {headline['delta_brier']:+.5f}  "
          f"Δlogloss {headline['delta_logloss']:+.5f}  Δauc {headline['delta_auc']:+.5f}")

    # ── plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax1 = plt.subplots(figsize=(10, 5))
        x = range(len(cum)); labels = cum["step"]
        ax1.plot(x, cum["brier"], "o-", color="#C8102E", label="Brier (↓)")
        ax1.set_ylabel("Brier", color="#C8102E"); ax1.set_xticks(list(x))
        ax1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax2 = ax1.twinx()
        ax2.plot(x, cum["roc_auc"], "s--", color="#1D428A", label="ROC-AUC (↑)")
        ax2.set_ylabel("ROC-AUC", color="#1D428A")
        # shade the shooter region
        first_shooter = len(portable) - 0.5
        ax1.axvspan(first_shooter, len(cum) - 0.5, color="#FCE4D6", alpha=0.6)
        ax1.set_title("Cumulative feature study — shaded = non-portable features added")
        fig.tight_layout(); fig.savefig(f"{args.out}/cumulative.png", dpi=150)
        print(f"\nSaved plot → {args.out}/cumulative.png")
    except Exception as e:
        print("plot skipped:", e)

    print(f"Saved: {args.out}/cumulative.csv, leave_one_out.csv, headline.json")


if __name__ == "__main__":
    main()
