"""
possessions.py — segment NBA play-by-play into possessions with terminal values.

Implements §2 of docs/epv_m4_true_epv_spec.md. Consumes a standard 33-column PBP
CSV (as in data/pbp/pbp_example.csv; schema in data/pbp/pbpevents.txt) and emits
one row per possession with the offense team, the game-time window, the realized
possession value R (points the offense scored), and the terminal type — the labels
the true-EPV model regresses.

SEGMENTATION MODEL
------------------
A possession is a MAXIMAL RUN of consecutive events controlled by the SAME team,
where "control" is:
  * shot / free-throw / turnover (EVENTMSGTYPE 1,2,3,5) -> the acting team
  * rebound (4)                                         -> the rebounding team
Non-control events (fouls, subs, timeouts, jump balls, period markers) don't flip
control. This makes two things fall out for free:
  * OFFENSIVE REBOUND = same control team -> stays in the possession.
  * AND-1 (made FG + own shooting FTs)   = same control team -> one possession,
    R = 2/3 + made FTs.
Control flips (opponent shoots / defensive rebound / turnover changeover) close the
possession; the terminal type is read off the possession's last control event:
  1 -> made_fg | 3 -> made_ft | 5 -> turnover | 2 (last in run) -> miss_defreb.

VALUE R
-------
R = total points added to the scoreboard during the possession (SCORE column is
"VISITOR - HOME"; we sum both sides and take the increment). Only the offense can
score in its own possession, so the scoreboard delta IS the offense's points — no
description parsing needed for points (and-1s and 3s handled automatically).

TIME
----
t = (period-1)*720 + (720 - gameclock_sec), the SAME formula tracking_graphs.py
uses, so t_start/t_end join directly against the tracking timeline. (OT is left on
the 720 grid on purpose, to stay consistent with the tracking side.)

Run:
  python src/epv/possessions.py                       # validate on the example game
  python src/epv/possessions.py data/pbp/00215XXXX.csv [more.csv ...]
"""
from __future__ import annotations
import sys, os, glob
import numpy as np
import pandas as pd

OUT = "results/epv/possessions_pbp.csv"
CONTROL_TYPES = {1, 2, 3, 4, 5}          # events that define/hold control
QLEN = 720                                # sec per quarter (mirror tracking_graphs)


def _desc(r):
    for c in ("HOMEDESCRIPTION", "VISITORDESCRIPTION", "NEUTRALDESCRIPTION"):
        v = r[c]
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _clock_to_sec(pcstring):
    """'11:42' -> 702 seconds remaining."""
    try:
        m, s = str(pcstring).split(":")
        return int(m) * 60 + float(s)
    except Exception:
        return np.nan


def _score_total(s):
    """'1 - 6' -> 7 ; NaN -> None."""
    if not isinstance(s, str) or "-" not in s:
        return None
    try:
        a, b = s.split("-")
        return int(a) + int(b)
    except Exception:
        return None


def _nickname_to_team(df):
    """Build {team_nickname_upper: team_id} from player rows, for TEAM rebounds."""
    m = {}
    for pfx in ("PLAYER1", "PLAYER2", "PLAYER3"):
        sub = df[[f"{pfx}_TEAM_NICKNAME", f"{pfx}_TEAM_ID"]].dropna()
        for nick, tid in sub.itertuples(index=False):
            if isinstance(nick, str) and nick.strip():
                m[nick.strip().upper()] = int(tid)
    return m


def _control_team(r, nick2team):
    """Team controlling the ball at this event, or None if not a control event."""
    t = int(r.EVENTMSGTYPE)
    if t not in CONTROL_TYPES:
        return None
    tid = r.PLAYER1_TEAM_ID
    if pd.notna(tid) and int(tid) > 1_000_000_000:      # valid 10-digit NBA team id
        return int(tid)
    if t == 4:                                    # TEAM rebound: parse "NICK Rebound"
        d = _desc(r).upper()
        for nick, team in nick2team.items():
            if d.startswith(nick + " ") and "REBOUND" in d:
                return team
    return None


def segment_game(df):
    """df: PBP rows for ONE game. Returns a possession DataFrame."""
    gid = int(df.GAME_ID.iloc[0])
    nick2team = _nickname_to_team(df)
    # the two teams = the two most frequent valid team ids (guards against
    # occasional malformed/truncated PLAYER1_TEAM_ID values in the mirror CSVs)
    vc = df.PLAYER1_TEAM_ID.dropna().astype("int64")
    vc = vc[vc > 1_000_000_000].value_counts()
    teams = list(vc.index[:2])
    other = {teams[0]: teams[1], teams[1]: teams[0]} if len(teams) == 2 else {}

    poss, cur = [], None
    prev_total = 0                                  # SCORE is cumulative for the WHOLE game
    for period in sorted(df.PERIOD.unique()):
        pdf = df[df.PERIOD == period].sort_values("EVENTNUM")
        for _, r in pdf.iterrows():
            total = _score_total(r.SCORE)
            if total is not None:
                dscore = total - prev_total
                prev_total = total
            else:
                dscore = 0

            ctrl = _control_team(r, nick2team)
            gt = (int(period) - 1) * QLEN + (QLEN - _clock_to_sec(r.PCTIMESTRING))

            if ctrl is None:                        # non-control event: only credit
                if cur is not None and dscore:      # a stray score (rare) to current
                    cur["R"] += dscore
                continue

            new_poss = (cur is None or ctrl != cur["off"] or int(period) != cur["period"])
            if new_poss:
                if cur is not None:
                    poss.append(cur)
                cur = dict(gid=gid, period=int(period), off=ctrl,
                           t_start=gt, t_end=gt, R=0,
                           last_type=int(r.EVENTMSGTYPE),
                           start_ev=int(r.EVENTNUM), end_ev=int(r.EVENTNUM))
            cur["R"] += dscore
            cur["t_end"] = gt
            cur["last_type"] = int(r.EVENTMSGTYPE)
            cur["end_ev"] = int(r.EVENTNUM)
        if cur is not None:                         # period boundary closes possession
            cur["term_period_end"] = True
            poss.append(cur); cur = None
    if cur is not None:
        poss.append(cur)

    # Window start = the moment the offense GAINED the ball = previous possession's
    # terminal (same period), so the tracking window spans the whole possession and
    # not just the discrete PBP event timestamps (a fast possession has one event).
    for i, p in enumerate(poss):
        if i > 0 and poss[i - 1]["period"] == p["period"]:
            p["t_gain"] = poss[i - 1]["t_end"]
        else:
            p["t_gain"] = (p["period"] - 1) * QLEN            # period start
        p["t_gain"] = min(p["t_gain"], p["t_end"])           # guard against reversal

    term = {1: "made_fg", 3: "made_ft", 5: "turnover", 2: "miss_defreb", 4: "reb_end"}
    rows = []
    for p in poss:
        rows.append(dict(
            gid=p["gid"], period=p["period"],
            off_team_id=p["off"], def_team_id=other.get(p["off"], 0),
            t_start=round(p["t_gain"], 2), t_end=round(p["t_end"], 2),
            t_terminal=round(p["t_start"], 2),               # first PBP event time (for ref)
            R=int(p["R"]),
            terminal_type=("period_end" if p.get("term_period_end") and p["last_type"] not in (1, 3, 5)
                           else term.get(p["last_type"], "other")),
            start_eventnum=p["start_ev"], end_eventnum=p["end_ev"]))
    return pd.DataFrame(rows)


def load_pbp(path):
    return pd.read_csv(path)


def diagnostics(pos, df=None):
    print(f"  possessions: {len(pos)}")
    print(f"  mean R (points/possession): {pos.R.mean():.3f}   [league PPP ~1.0-1.1]")
    tot = pos.groupby('off_team_id').R.sum()
    print(f"  points by team (sum R): {dict(tot)}")
    if df is not None:
        fs = df.SCORE.dropna().iloc[-1]
        print(f"  final SCORE in PBP: {fs}  (sum={_score_total(fs)}, our R total={pos.R.sum()})")
    print(f"  R distribution: {dict(pos.R.value_counts().sort_index())}")
    print(f"  terminal types: {dict(pos.terminal_type.value_counts())}")
    # alternation: fraction of consecutive possessions that switch offense
    switches = (pos.off_team_id.values[1:] != pos.off_team_id.values[:-1]).mean()
    print(f"  offense-switch rate between consecutive possessions: {switches:.2f}  "
          f"(<1 expected: offensive rebounds keep the same team)")
    dur = pos.t_end - pos.t_start
    print(f"  possession duration sec: mean {dur.mean():.1f}, median {dur.median():.1f}, "
          f"max {dur.max():.1f}")


def main(argv):
    os.makedirs("results/epv", exist_ok=True)
    paths = argv[1:] if len(argv) > 1 else ["data/pbp/pbp_example.csv"]
    all_pos = []
    for path in paths:
        if not os.path.exists(path):
            print(f"[skip] {path} not found"); continue
        df = load_pbp(path)
        pos = segment_game(df)
        print(f"\n=== {os.path.basename(path)} (game {int(df.GAME_ID.iloc[0])}) ===")
        diagnostics(pos, df)
        all_pos.append(pos)
    if all_pos:
        out = pd.concat(all_pos, ignore_index=True)
        out.to_csv(OUT, index=False)
        print(f"\nSaved {OUT}  ({len(out)} possessions from {len(all_pos)} game(s))")


if __name__ == "__main__":
    main(sys.argv)
