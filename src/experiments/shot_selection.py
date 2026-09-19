"""
shot_selection.py — Expected points & shot-selection analysis from the
calibrated make model.

Where POE answers "did the shooter beat expectation?", this answers the
*decision* question: "how good was the shot that was taken?" For every shot,

    xPPS = shot_value * P(make)              (expected points of the shot)

using an OUT-OF-FOLD P(make) (GroupKFold on game_id) so no shot informs its own
probability. From xPPS we build:

  1. Expected-points surface by court zone (where the good shots are).
  2. The 2-vs-3 value frontier, split by defender pressure.
  3. Per-player shot-selection value:
       - xPPS           : average quality of shots taken (decision quality)
       - SSV/shot       : xPPS minus the league mean xPPS for the SAME zone
                          (did they find better-than-average looks?)
       - POE/shot       : actual minus expected (finishing skill; for contrast)
  4. An allocative-efficiency index (Skinner): does a player concentrate volume
     in their own high-xPPS zones, or leave points on the table?

Caveat: xPPS is defined on shots that were TAKEN, so it does not yet value the
pass/reset alternative — that requires possession-level EPV (see
src/experiments/possessions.py and docs/epv_shot_selection.md).

Run:  python src/experiments/shot_selection.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb

sys.path.insert(0, os.path.dirname(__file__))
import feature_ablation as fa  # reuse METADATA/TARGET/GROUP/XGB_PARAMS

DATA = "data/shot_features_valid2_type.csv"
OUT  = "results/shot_selection"
MIN_SHOTS = 100


def oof_make_prob(df, feats, seed=42, n_splits=5):
    """Out-of-fold calibrated P(make) via GroupKFold on game_id."""
    p = np.full(len(df), np.nan)
    params = dict(fa.XGB_PARAMS); params["n_estimators"] = 700  # fixed (no val leak)
    gkf = GroupKFold(n_splits=n_splits)
    for i, (tr, te) in enumerate(gkf.split(df, groups=df[fa.GROUP])):
        m = xgb.XGBClassifier(**params, random_state=seed)
        m.fit(df.iloc[tr][feats], df.iloc[tr][fa.TARGET], verbose=False)
        p[te] = m.predict_proba(df.iloc[te][feats])[:, 1]
        print(f"  fold {i+1}/{n_splits} done")
    return p


def zone_of(row):
    d = row["dist"]; three = row["is_3_pointer"] == 1
    if three:
        if d < 23.2:   return "Corner 3"
        if d < 26.5:   return "Above-break 3"
        return "Deep 3"
    if d < 4:          return "Rim (<4ft)"
    if d < 10:         return "Close (4-10)"
    if d < 16:         return "Mid (10-16)"
    return "Long 2 (16+)"


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA)
    df = df[df[fa.TARGET].isin([0, 1])].reset_index(drop=True)
    feats = [c for c in df.columns if c not in fa.METADATA + [fa.TARGET]]
    print(f"Loaded {len(df):,} shots | computing out-of-fold P(make)...")

    df["p_make"] = oof_make_prob(df, feats)
    df["shot_value"] = np.where(df["is_3_pointer"] == 1, 3.0, 2.0)
    df["xpps"]   = df["shot_value"] * df["p_make"]           # expected points
    df["actual"] = df["shot_value"] * df[fa.TARGET]          # realized points
    df["poe"]    = df["actual"] - df["xpps"]                 # finishing skill
    df["zone"]   = df.apply(zone_of, axis=1)

    league_xpps = df["xpps"].mean()
    print(f"\nLeague mean xPPS (expected points per shot): {league_xpps:.4f}")

    # ── 1) expected-points surface by zone ───────────────────────────────────
    z = (df.groupby("zone")
           .agg(n=("xpps", "size"), share=("xpps", lambda s: len(s)/len(df)),
                mean_xpps=("xpps", "mean"), make_rate=(fa.TARGET, "mean"),
                actual_pps=("actual", "mean"))
           .sort_values("mean_xpps", ascending=False).round(4))
    z.to_csv(f"{OUT}/zone_xpps.csv")
    print("\n=== Expected points by zone (xPPS) ===")
    print(z.to_string())

    # zone baseline for SSV
    zmean = df.groupby("zone")["xpps"].transform("mean")
    df["ssv"] = df["xpps"] - zmean   # shot-selection value vs same-zone average

    # ── 2) 2-vs-3 value frontier by defender pressure ────────────────────────
    df["pressure"] = np.select(
        [df["def_very_tight"] == 1, df["def_tight"] == 1, df["def_open"] == 1],
        ["very tight", "tight", "open"], default="med")
    tvt = (df.assign(kind=np.where(df.is_3_pointer == 1, "3PT", "2PT"))
             .groupby(["pressure", "kind"])
             .agg(n=("xpps", "size"), mean_xpps=("xpps", "mean"),
                  make_rate=(fa.TARGET, "mean")).round(4))
    tvt.to_csv(f"{OUT}/two_vs_three.csv")
    print("\n=== 2-vs-3 expected points by defender pressure ===")
    print(tvt.to_string())

    # ── 3) per-player shot-selection leaderboard ─────────────────────────────
    g = df.groupby("player_name")
    pl = pd.DataFrame({
        "shots":       g.size(),
        "pct3":        g["is_3_pointer"].mean(),
        "xpps":        g["xpps"].mean(),      # quality of shots taken
        "actual_pps":  g["actual"].mean(),
        "poe_per_shot":g["poe"].mean(),       # finishing (contrast)
        "ssv_per_shot":g["ssv"].mean(),       # shot selection vs zone avg
    })
    pl = pl[pl["shots"] >= MIN_SHOTS].round(4)
    pl.sort_values("xpps", ascending=False).to_csv(f"{OUT}/player_shot_selection.csv")
    print(f"\n=== Top 12 shot-selection (xPPS, >= {MIN_SHOTS} shots) ===")
    print(pl.sort_values("xpps", ascending=False).head(12).to_string())
    print("\n=== Top 12 shot-SELECTION value (SSV/shot: better-than-avg looks) ===")
    print(pl.sort_values("ssv_per_shot", ascending=False)
            [["shots","pct3","xpps","ssv_per_shot"]].head(12).to_string())

    # ── 4) allocative efficiency (Skinner): volume vs own-zone value ─────────
    # For each player, correlation between (share of their shots in a zone) and
    # (their xPPS in that zone). Positive => they shoot more where they're good.
    rows = []
    for name, gg in df.groupby("player_name"):
        if len(gg) < MIN_SHOTS:
            continue
        zt = gg.groupby("zone").agg(vol=("xpps", "size"), zx=("xpps", "mean"))
        if len(zt) < 3:
            continue
        vol = zt["vol"] / zt["vol"].sum()
        # points-left-on-table: value if volume shifted to the player's best zone
        gap = (zt["zx"].max() - (vol * zt["zx"]).sum())
        rows.append({"player_name": name, "shots": len(gg),
                     "alloc_corr": float(np.corrcoef(vol, zt["zx"])[0, 1]) if len(zt) > 2 else np.nan,
                     "xpps_gap_to_best_zone": round(float(gap), 4)})
    ae = pd.DataFrame(rows).round(4)
    ae.sort_values("xpps_gap_to_best_zone").to_csv(f"{OUT}/allocative_efficiency.csv", index=False)

    # ── plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        # 2v3 frontier
        piv = tvt.reset_index().pivot(index="pressure", columns="kind", values="mean_xpps")
        piv = piv.reindex(["open", "med", "tight", "very tight"])
        ax = piv.plot(kind="bar", color={"2PT": "#1D428A", "3PT": "#C8102E"}, figsize=(8, 5))
        ax.axhline(league_xpps, ls="--", c="gray", label=f"league xPPS={league_xpps:.2f}")
        ax.set_ylabel("Expected points per shot"); ax.set_xlabel("defender pressure")
        ax.set_title("2-vs-3 shot value by defender pressure"); ax.legend()
        plt.tight_layout(); plt.savefig(f"{OUT}/two_vs_three.png", dpi=150); plt.close()
        # xPPS court hexbin
        fig, a = plt.subplots(figsize=(7, 7))
        hb = a.hexbin(df["x"], df["y"], C=df["xpps"], gridsize=40, cmap="RdYlBu_r", mincnt=15)
        fig.colorbar(hb, label="xPPS"); a.set_title("Expected points per shot (court surface)")
        a.set_aspect("equal"); plt.tight_layout(); plt.savefig(f"{OUT}/xpps_court.png", dpi=150); plt.close()
        print(f"\nSaved plots → {OUT}/two_vs_three.png, xpps_court.png")
    except Exception as e:
        print("plot skipped:", e)

    print(f"\nSaved: {OUT}/zone_xpps.csv, two_vs_three.csv, "
          f"player_shot_selection.csv, allocative_efficiency.csv")


if __name__ == "__main__":
    main()
