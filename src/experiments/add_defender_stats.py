"""
add_defender_stats.py — add prior-season (2014-15) info on the PRIMARY DEFENDER
(closest_def_name) to the shot model, and measure whether it helps.

The model already has the *geometry* of the contest (closest_def_dist/angle,
time_to_contest, def_very_tight/tight/open) but not *who* the defender is. Prior
rim-protection / defensive priors (blocks, size/position, steals, rebounding)
might carry difficulty the geometry misses — especially on interior shots.

Features added (ext_def_*, out-of-sample from a different season):
  blk, stl, drb, pf (per game), mp, position ordinal (PG=1..C=5), is_big (PF/C),
  plus ext_def_has (0 for rookies/unmatched -> NaN, handled by XGBoost).

Run:  python src/experiments/add_defender_stats.py
"""
from __future__ import annotations
import os, sys, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
import feature_ablation as fa

SHOTS = "data/shot_features_valid2.csv"
DEF = "data/def_prior_2014_15.csv"
OUT = "data/shot_features_valid2_def.csv"
POS_ORD = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5}
ALIAS = {"enes kanter": "enes freedom"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    shots = pd.read_csv(SHOTS)
    dpri = pd.read_csv(DEF)
    dpri["key"] = dpri["name"].map(norm)
    dpri = dpri.drop_duplicates("key").set_index("key")
    keys = shots["closest_def_name"].map(norm).map(lambda k: ALIAS.get(k, k))

    def col(src): return keys.map(dpri[src])
    shots["ext_def_blk"] = col("blk")
    shots["ext_def_stl"] = col("stl")
    shots["ext_def_drb"] = col("drb")
    shots["ext_def_pf"]  = col("pf")
    shots["ext_def_mp"]  = col("mp")
    shots["ext_def_pos"] = keys.map(dpri["pos"]).map(POS_ORD)
    shots["ext_def_is_big"] = (shots["ext_def_pos"] >= 4).astype("float")
    shots["ext_def_has"] = keys.isin(dpri.index).astype(int)

    # Impute unmatched defenders (rookies / no 2014-15) with the LEAGUE AVERAGE
    # of each stat instead of NaN; ext_def_has still flags imputed rows.
    for c in [c for c in shots.columns
              if c.startswith("ext_def_") and c != "ext_def_has"]:
        shots[c] = shots[c].fillna(shots[c].mean())

    shots.to_csv(OUT, index=False)

    uniq = shots["closest_def_name"].nunique()
    matched = shots.loc[shots.ext_def_has == 1, "closest_def_name"].nunique()
    print(f"Primary defenders matched: {matched}/{uniq} ({matched/uniq:.1%}) | "
          f"shots covered {shots.ext_def_has.mean():.1%}")
    print(f"Wrote {OUT}\n")

    # game-disjoint split (matches feature_ablation)
    gss = GroupShuffleSplit(1, test_size=0.20, random_state=42)
    trv, te = next(gss.split(shots, groups=shots[fa.GROUP]))
    trvd = shots.iloc[trv]
    gss2 = GroupShuffleSplit(1, test_size=0.15 / 0.8, random_state=42)
    a, b = next(gss2.split(trvd, groups=trvd[fa.GROUP]))
    trI, vaI, teI = trvd.index[a], trvd.index[b], shots.index[te]

    defcols = [c for c in shots.columns if c.startswith("ext_def_")]
    baseline = [c for c in shots.columns
                if c not in fa.METADATA + [fa.TARGET] and not c.startswith("ext_def_")]

    def run(cols, mask, label):
        tr = trI[shots.loc[trI].eval(mask).values] if mask else trI
        va = vaI[shots.loc[vaI].eval(mask).values] if mask else vaI
        te2 = teI[shots.loc[teI].eval(mask).values] if mask else teI
        m = fa.fit_eval(shots, cols, tr, va, te2, 42)
        print(f"    {label:<26} brier={m['brier']:.5f} auc={m['roc_auc']:.5f} logloss={m['log_loss']:.5f} (test n={len(te2)})")
        return m

    print("=== MAIN: does primary-defender prior info help? (full test set) ===")
    b0 = run(baseline, None, "current model")
    b1 = run(baseline + defcols, None, "  + defender priors")
    print(f"    gain: Δbrier {b0['brier']-b1['brier']:+.5f}  Δauc {b1['roc_auc']-b0['roc_auc']:+.5f}")

    print("\n=== INTERIOR shots (dist < 8ft) — where rim protection should matter ===")
    i0 = run(baseline, "dist < 8", "current model")
    i1 = run(baseline + defcols, "dist < 8", "  + defender priors")
    print(f"    gain: Δbrier {i0['brier']-i1['brier']:+.5f}  Δauc {i1['roc_auc']-i0['roc_auc']:+.5f}")

    print("\n=== CONTESTED shots (very tight / tight) ===")
    c0 = run(baseline, "def_very_tight == 1 or def_tight == 1", "current model")
    c1 = run(baseline + defcols, "def_very_tight == 1 or def_tight == 1", "  + defender priors")
    print(f"    gain: Δbrier {c0['brier']-c1['brier']:+.5f}  Δauc {c1['roc_auc']-c0['roc_auc']:+.5f}")


if __name__ == "__main__":
    main()
