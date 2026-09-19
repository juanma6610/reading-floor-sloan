"""
add_shot_type.py — test whether PBP shot-TYPE (layup/dunk/hook/floater/step-back/
pullup/... vs jump shot) adds predictive power to the make model.

Shot type is a release-time mechanical descriptor (like the existing
is_dunk_or_tip / is_catch_and_shoot), parsed from the PBP `description`. It is
LEAKAGE-SAFE only if we take the type tokens and NOTHING else: the same
description also contains the outcome ("MISS", "(24 PTS)"), which we never touch.
(Sanity: 'MISS' rows have make-rate 0.000, non-'MISS' rows 1.000 — proof the raw
string encodes the label, so only type keywords may be used.)

Writes data/shot_features_valid2_type.csv (adds stype_* flags) and runs an A/B:
  - current model            vs  + shot type
  - geometry only            vs  geometry + shot type   (standalone signal)

A big, plausible gain (AUC to ~0.70-0.75) = real signal; a jump to >0.9 would
signal leakage (it does not). Note: PBP text is NBA-specific, so this trades away
portability — but most types are in principle recoverable from tracking, which
would restore a portable version.

Run:  python src/experiments/add_shot_type.py
"""
from __future__ import annotations
import os, sys, re
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
import feature_ablation as fa

DATA = "data/shot_features_valid2.csv"
OUT_CSV = "data/shot_features_valid2_type.csv"

# outcome-INDEPENDENT shot-type tokens (never MISS / (PTS))
TYPES = {
    "stype_layup":      r"Layup",
    "stype_dunk":       r"Dunk",
    "stype_hook":       r"Hook",
    "stype_fadeaway":   r"Fadeaway",
    "stype_floating":   r"Floating|Floater",
    "stype_stepback":   r"Step Back|Stepback",
    "stype_pullup":     r"Pullup|Pull-Up",
    "stype_driving":    r"Driving",
    "stype_turnaround": r"Turnaround",
    "stype_bank":       r"Bank",
    "stype_alleyoop":   r"Alley[ -]?Oop",
    "stype_tip":        r"Tip",
    "stype_reverse":    r"Reverse",
    "stype_putback":    r"Putback",
    "stype_cutting":    r"Cutting",
    "stype_running":    r"Running",
    "stype_fingerroll": r"Finger Roll",
    "stype_jumpshot":   r"Jump Shot",
}


def main():
    df = pd.read_csv(DATA)
    df=df.drop(["is_dunk_or_tip"], axis=1)
    df = df[df[fa.TARGET].isin([0, 1])].reset_index(drop=True)
    desc = df["description"].fillna("")

    for col, pat in TYPES.items():
        df[col] = desc.str.contains(pat, case=False, regex=True).astype(int)
    stype_cols = list(TYPES)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(stype_cols)} stype_* flags "
          f"(coverage: {int((df[stype_cols].sum(axis=1) > 0).sum())}/{len(df)} shots typed)\n")

    # game-disjoint split (matches feature_ablation)
    gss = GroupShuffleSplit(1, test_size=0.20, random_state=42)
    trv, te = next(gss.split(df, groups=df[fa.GROUP]))
    trvd = df.iloc[trv]
    gss2 = GroupShuffleSplit(1, test_size=0.15/0.8, random_state=42)
    a, b = next(gss2.split(trvd, groups=trvd[fa.GROUP]))
    trI, vaI, teI = trvd.index[a], trvd.index[b], df.index[te]

    baseline = [c for c in df.columns
                if c not in fa.METADATA + [fa.TARGET] and not c.startswith("stype_")]
    geom = fa.build_groups(df)["geometry"]

    def run(cols, label):
        m = fa.fit_eval(df, cols, trI, vaI, teI, 42)
        print(f"  {label:<34} brier={m['brier']:.5f}  logloss={m['log_loss']:.5f}  auc={m['roc_auc']:.5f}")
        return m

    print("=== MAIN: does PBP shot type add power on top of the current model? ===")
    b0 = run(baseline, "current model (all features)")
    b1 = run(baseline + stype_cols, "  + PBP shot type")
    print(f"  gain: Δbrier {b0['brier']-b1['brier']:+.5f}  Δlogloss {b0['log_loss']-b1['log_loss']:+.5f}  "
          f"Δauc {b1['roc_auc']-b0['roc_auc']:+.5f}")

    print("\n=== STANDALONE: shot type on top of geometry only ===")
    g0 = run(geom, "geometry only")
    g1 = run(geom + stype_cols, "  + PBP shot type")
    print(f"  gain: Δbrier {g0['brier']-g1['brier']:+.5f}  Δauc {g1['roc_auc']-g0['roc_auc']:+.5f}")

    print("\n=== make-rate by type (train, sanity) ===")
    for c in stype_cols:
        mask = df.loc[trI, c] == 1
        if mask.sum() > 200:
            print(f"  {c:<16} n={int(mask.sum()):>6}  make%={df.loc[trI][mask][fa.TARGET].mean():.3f}")


if __name__ == "__main__":
    main()
