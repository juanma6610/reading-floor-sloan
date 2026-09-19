"""
add_prior_season_stats.py — merge 2014-15 (prior-season) shooting stats onto the
2015-16 shot table as leakage-free `ext_` features for the ablation harness.

Prior-season skill is the cleanest identity signal: it is measured on a DIFFERENT
season, so it is not derived from the shots we are predicting. Rookies (no
2014-15 data) get NaN — XGBoost handles that natively — plus an `ext_prior_has`
flag.

Input : data/prior_season_2014_15.csv   (scraped via the browser from BBRef)
        data/shot_features_valid2.csv
Output: data/shot_features_valid2_ext.csv   (adds ext_prior_* columns)

Run:    python src/experiments/add_prior_season_stats.py
Then:   python src/experiments/feature_ablation.py --data data/shot_features_valid2_ext.csv --out results/ablation_ext
"""
import re, unicodedata
import numpy as np
import pandas as pd

SHOTS = "data/shot_features_valid2.csv"
PRIOR = "data/prior_season_2014_15.csv"
OUT   = "data/shot_features_valid2_ext.csv"

# players whose BBRef name differs from the SportVU name (normalised form)
ALIAS = {"enes kanter": "enes freedom"}

def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z ]", "", s)             # remove . ' - so "C.J."=="CJ"
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)  # drop generational suffixes
    s = re.sub(r"\s+", " ", s).strip()
    return s

def main():
    shots = pd.read_csv(SHOTS)
    prior = pd.read_csv(PRIOR)

    prior["key"] = prior["name"].map(norm)
    prior = prior.drop_duplicates("key", keep="first").set_index("key")

    keys = shots["player_name"].map(norm).map(lambda k: ALIAS.get(k, k))

    def col(src):
        return keys.map(prior[src]) if src in prior.columns else np.nan

    shots["ext_prior_fg_pct"]  = col("fg_pct")
    shots["ext_prior_fg2_pct"] = col("fg2_pct")
    shots["ext_prior_fg3_pct"] = col("fg3_pct")
    shots["ext_prior_efg_pct"] = col("efg_pct")
    shots["ext_prior_ft_pct"]  = col("ft_pct")
    shots["ext_prior_fga"]     = col("fga")
    fg3a = col("fg3a"); fga = col("fga")
    shots["ext_prior_3par"]    = np.where((fga > 0), fg3a / fga.replace(0, np.nan), np.nan)
    shots["ext_prior_has"]     = keys.isin(prior.index).astype(int)

    # Impute unmatched players (rookies / no 2014-15 season) with the LEAGUE
    # AVERAGE of each stat instead of NaN. The mean is over matched shots (a
    # season aggregate, not the 2015-16 target, so no leakage); ext_prior_has
    # still flags imputed rows so the model can distinguish them.
    for c in [c for c in shots.columns
              if c.startswith("ext_prior_") and c != "ext_prior_has"]:
        shots[c] = shots[c].fillna(shots[c].mean())

    shots.to_csv(OUT, index=False)

    # ── coverage report ──────────────────────────────────────────────────────
    uniq = shots["player_name"].nunique()
    matched_players = keys[keys.isin(prior.index)].nunique() + \
                      keys.map(lambda k: k in prior.index).sum() * 0  # players
    matched_players = shots.loc[shots.ext_prior_has == 1, "player_name"].nunique()
    shot_cov = shots.ext_prior_has.mean()
    print(f"Shooters matched to prior season: {matched_players}/{uniq} "
          f"({matched_players/uniq:.1%})")
    print(f"Shots covered by a prior-season profile: {shot_cov:.1%}")
    print(f"FT% present on {shots.ext_prior_ft_pct.notna().mean():.1%} of shots; "
          f"3P% on {shots.ext_prior_fg3_pct.notna().mean():.1%}")

    unmatched = (shots.loc[shots.ext_prior_has == 0, "player_name"]
                 .value_counts().head(15))
    print("\nTop unmatched shooters (mostly 2015-16 rookies — expected NaN):")
    for name, n in unmatched.items():
        print(f"  {n:>4d}  {name}")
    print(f"\nWrote {OUT}  ({shots.shape[0]} rows, {shots.shape[1]} cols)")


if __name__ == "__main__":
    main()
