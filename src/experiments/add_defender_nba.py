"""
add_defender_nba.py — add NBA.com defender ADVANCED stats (defended FG%, DIFF%)
and an explicit DEFENDER IDENTITY encoding, then A/B test on the shot model.

- ext_ndef_dfg   : opponent FG% when this player is the closest defender (2014-15)
- ext_ndef_diff  : DIFF% = defended FG% minus opponent's normal FG%
                   (negative = shot-suppressing defender; the gold-standard signal)
- ext_ndef_dfga  : defended FGA/game (volume/role)
- ext_defid      : DEFENDER IDENTITY — leakage-safe out-of-fold (GroupKFold on
                   game_id) opponent make-rate for closest_def_name in THIS data,
                   i.e. the defender as an entity/effect (shrunk to league mean).

Run:  python src/experiments/add_defender_nba.py
"""
from __future__ import annotations
import os, sys, re, unicodedata
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(__file__))
import feature_ablation as fa

SHOTS = "data/shot_features_valid2.csv"
NBA = "data/def_nba_2014_15.csv"
OUT = "data/shot_features_valid2_ndef.csv"
ALIAS = {"enes kanter": "enes freedom"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main():
    shots = pd.read_csv(SHOTS)
    nba = pd.read_csv(NBA, sep="|")
    nba["key"] = nba["name"].map(norm)
    nba = nba.drop_duplicates("key").set_index("key")
    keys = shots["closest_def_name"].map(norm).map(lambda k: ALIAS.get(k, k))

    shots["ext_ndef_dfg"]  = keys.map(nba["dfg_pct"])
    shots["ext_ndef_diff"] = keys.map(nba["diff_pct"])
    shots["ext_ndef_dfga"] = keys.map(nba["dfga"])
    shots["ext_ndef_has"]  = keys.isin(nba.index).astype(int)

    # split (game-disjoint), needed both for the OOF identity encoding and the A/B
    gss = GroupShuffleSplit(1, test_size=0.20, random_state=42)
    trv, te = next(gss.split(shots, groups=shots[fa.GROUP]))
    trvd = shots.iloc[trv]
    gss2 = GroupShuffleSplit(1, test_size=0.15 / 0.8, random_state=42)
    a, b = next(gss2.split(trvd, groups=trvd[fa.GROUP]))
    trI, vaI, teI = trvd.index[a], trvd.index[b], shots.index[te]

    # DEFENDER IDENTITY: leakage-safe OOF opponent make-rate for closest_def_name
    shots["ext_defid"] = fa.oof_prior(shots, trI, vaI, teI,
                                      "closest_def_name", fa.TARGET, alpha=200)

    # Impute unmatched defenders (rookies / no 2014-15) with the LEAGUE AVERAGE
    # of each NBA stat instead of NaN; ext_ndef_has still flags imputed rows.
    # (ext_defid is already league-mean-filled by the OOF encoder.)
    for c in ["ext_ndef_dfg", "ext_ndef_diff", "ext_ndef_dfga"]:
        shots[c] = shots[c].fillna(shots[c].mean())

    shots.to_csv(OUT, index=False)

    uniq = shots["closest_def_name"].nunique()
    matched = shots.loc[shots.ext_ndef_has == 1, "closest_def_name"].nunique()
    print(f"NBA defender stats matched: {matched}/{uniq} ({matched/uniq:.1%}), "
          f"shots covered {shots.ext_ndef_has.mean():.1%}")
    print(f"ext_defid (identity) univariate test-AUC: "
          f"{__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(shots.loc[teI, fa.TARGET], -shots.loc[teI,'ext_defid']):.4f}")
    print(f"ext_ndef_diff univariate test-AUC: "
          f"{__import__('sklearn.metrics',fromlist=['roc_auc_score']).roc_auc_score(shots.loc[teI, fa.TARGET], shots.loc[teI,'ext_ndef_diff'].fillna(shots.ext_ndef_diff.mean())):.4f}")
    print(f"Wrote {OUT}\n")

    nbacols = ["ext_ndef_dfg", "ext_ndef_diff", "ext_ndef_dfga"]
    baseline = [c for c in shots.columns
                if c not in fa.METADATA + [fa.TARGET]
                and not c.startswith("ext_ndef_") and c != "ext_defid" and c != "ext_ndef_has"]

    def run(cols, mask, label):
        f = (lambda idx: idx[shots.loc[idx].eval(mask).values]) if mask else (lambda idx: idx)
        m = fa.fit_eval(shots, cols, f(trI), f(vaI), f(teI), 42)
        print(f"    {label:<30} brier={m['brier']:.5f} auc={m['roc_auc']:.5f} logloss={m['log_loss']:.5f} (n={len(f(teI))})")
        return m

    def block(mask, title):
        print(f"=== {title} ===")
        b0 = run(baseline, mask, "current model")
        b1 = run(baseline + nbacols, mask, "  + NBA defended-FG%/DIFF%")
        b2 = run(baseline + ["ext_defid"], mask, "  + defender identity")
        b3 = run(baseline + nbacols + ["ext_defid"], mask, "  + both")
        print(f"    gains vs current — NBA: Δauc {b1['roc_auc']-b0['roc_auc']:+.5f} | "
              f"identity: Δauc {b2['roc_auc']-b0['roc_auc']:+.5f} | both: Δauc {b3['roc_auc']-b0['roc_auc']:+.5f}\n")

    block(None, "FULL test set")
    block("dist < 8", "INTERIOR shots (dist < 8ft)")
    block("def_very_tight == 1 or def_tight == 1", "CONTESTED shots")


if __name__ == "__main__":
    main()
