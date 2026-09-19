"""
possessions.py — reconstruct possessions from play-by-play (the data foundation
for possession-level EPV).

Two things:
  1. A standard possession estimate + points-per-possession (PPP) check, which
     validates that we can turn PBP into possessions
     (Poss = FGA - OREB + TOV + 0.44*FTA; PPP = points / Poss; NBA ~1.05).
  2. A best-effort event-walk that emits a possessions table
     (offense, start clock, start score-margin -> points scored), the state->outcome
     rows a possession-value / EPV model is trained on.

Works on any directory of stats.nba-style PBP CSVs. Only one sample game ships
in the repo (data/pbp/pbp_example.csv); the full corpus is downloaded per game
by src/game.py from the nba-movement-data mirror — run that over allgames.txt to
build data/pbp/ before training a full EPV model (see docs/epv_shot_selection.md).

EVENTMSGTYPE: 1 made FG, 2 missed FG, 3 free throw, 4 rebound, 5 turnover,
              6 foul, 7 violation, 8 sub, 9 timeout, 10 jump ball, 12/13 period.
Run:  python src/experiments/possessions.py
"""
from __future__ import annotations
import glob, os, re
import numpy as np
import pandas as pd

PBP_GLOB = "data/pbp/*.csv"
OUT = "results/epv"


def _score_total(s):
    if not isinstance(s, str) or "-" not in s:
        return np.nan
    a, b = s.split("-")[:2]
    try:
        return int(a) + int(b)
    except ValueError:
        return np.nan


def load_pbp(path):
    d = pd.read_csv(path)
    d = d.sort_values(["PERIOD", "EVENTNUM"]).reset_index(drop=True)
    d["tot"] = d["SCORE"].map(_score_total).ffill().fillna(0)
    d["pts"] = d["tot"].diff().fillna(0).clip(lower=0)   # points added on this event
    d["sec"] = d["PCTIMESTRING"].map(
        lambda t: int(str(t).split(":")[0]) * 60 + int(str(t).split(":")[1])
        if isinstance(t, str) and ":" in str(t) else np.nan)
    return d


def formula_ppp(d):
    """Standard team-level possession estimate + PPP."""
    fga = d.EVENTMSGTYPE.isin([1, 2]).sum()
    fta = (d.EVENTMSGTYPE == 3).sum()
    tov = (d.EVENTMSGTYPE == 5).sum()
    # offensive rebounds: rebounder team == last shooter team
    oreb = 0
    last_shot_team = None
    for _, r in d.iterrows():
        if r.EVENTMSGTYPE in (1, 2):
            last_shot_team = r.PLAYER1_TEAM_ID
        elif r.EVENTMSGTYPE == 4 and pd.notna(r.PLAYER1_TEAM_ID):
            if r.PLAYER1_TEAM_ID == last_shot_team:
                oreb += 1
    poss = fga - oreb + tov + 0.44 * fta
    pts = d["pts"].sum()
    return poss, pts, pts / poss if poss else np.nan, dict(FGA=fga, FTA=fta, TOV=tov, OREB=oreb)


def event_walk(d):
    """Best-effort possession segmentation -> table of (state -> points)."""
    rows = []
    cur = None
    last_shot_team = None
    prev_type = None

    def close():
        nonlocal cur
        if cur is not None:
            rows.append(cur)
        cur = None

    def start(team, r):
        return {"off_team": team, "period": int(r.PERIOD),
                "start_sec": r.sec, "start_margin": r.get("SCOREMARGIN", np.nan),
                "points": 0.0}

    def is_last_ft(r):
        desc = f"{r.get('HOMEDESCRIPTION','')} {r.get('VISITORDESCRIPTION','')}"
        m = re.search(r"(\d+) of (\d+)", str(desc))
        return bool(m) and m.group(1) == m.group(2)

    for _, r in d.iterrows():
        t = r.EVENTMSGTYPE
        if t in (10, 12, 13):        # jump/period boundaries -> reset
            close(); prev_type = t; continue
        team = r.PLAYER1_TEAM_ID

        # and-1 free throw: the made FG already closed the possession; credit the
        # bonus point to that possession and do NOT open a new one.
        if t == 3 and cur is None and prev_type == 1 and rows:
            rows[-1]["points"] += float(r["pts"]); prev_type = t; continue

        if cur is None and t in (1, 2, 3, 5) and pd.notna(team):
            cur = start(team, r)
        if cur is not None:
            cur["points"] += float(r["pts"])
        if t in (1, 2):
            last_shot_team = team

        if t == 1:                        # made FG -> ends
            close()
        elif t == 5:                      # turnover -> ends
            close()
        elif t == 3 and is_last_ft(r):    # last FT of trip -> ends
            close()
        elif t == 4 and pd.notna(team):   # rebound
            if team != last_shot_team:     # defensive rebound -> ends
                close()
            # offensive rebound -> continue
        prev_type = t
    close()
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    files = glob.glob(PBP_GLOB)
    if not files:
        print("No PBP files found. Run src/game.py over allgames.txt to populate data/pbp/.")
        return
    print(f"PBP files: {len(files)} (only the sample ships in-repo)\n")

    all_poss = []
    for f in files:
        d = load_pbp(f)
        poss, pts, ppp, box = formula_ppp(d)
        pw = event_walk(d)
        ew_ppp = pw["points"].sum() / len(pw) if len(pw) else np.nan
        gid = os.path.basename(f).split("_")[0].split(".")[0]
        print(f"game {gid}: points={pts:.0f} | formula Poss={poss:.1f} PPP={ppp:.3f} "
              f"| event-walk Poss={len(pw)} PPP={ew_ppp:.3f} | {box}")
        pw["game"] = gid
        all_poss.append(pw)

    poss_df = pd.concat(all_poss, ignore_index=True)
    poss_df.to_csv(f"{OUT}/possessions.csv", index=False)
    print(f"\nEvent-walk possessions table: {len(poss_df)} rows -> {OUT}/possessions.csv")
    print("Points-per-possession distribution (event-walk):")
    print(poss_df["points"].value_counts().sort_index().to_string())
    print(f"\nMean PPP (event-walk, all games): {poss_df['points'].mean():.3f} "
          "(NBA 2015-16 ~1.06 — sanity check)")
    # coarse possession-value baseline by start-clock bucket (needs many games to be useful)
    poss_df["clock_bucket"] = pd.cut(poss_df["start_sec"], [-1, 60, 180, 360, 720],
                                     labels=["<1m", "1-3m", "3-6m", "6-12m"])
    print("\nMean points by start-clock bucket (illustrative baseline):")
    print(poss_df.groupby("clock_bucket", observed=True)["points"].agg(["size", "mean"]).round(3).to_string())


if __name__ == "__main__":
    main()
