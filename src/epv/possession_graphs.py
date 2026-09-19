"""
possession_graphs.py — whole-possession graph sequences for TRUE per-moment EPV.

Implements §3 of docs/epv_m4_true_epv_spec.md. Supersedes the shot-only
tracking_graphs.py (kept as the spatial-block parity reference): instead of one
sample per shot over the 4 s before release, this builds one sample per
POSSESSION over the WHOLE possession window, and carries the possession's terminal
value R + terminal type so the model can learn a value at every frame.

For each possession in results/epv/possessions_pbp.csv:
  * slice the game timeline to [t_start, t_end] (capped to the last SHOTCLOCK=24 s
    to trim dead-ball lead-in),
  * require a stable 10-man lineup across the sampled frames; split 5 offense /
    5 defense by the possession's off_team_id,
  * sample up to KMAX=40 frames evenly, PAD to KMAX + emit a frame mask,
  * node features reuse the tracking_graphs featurization but replace `is_shooter`
    with `is_ballhandler` (offense node nearest the ball each frame) — this is what
    lets ΔEPV(t) be attributed to whoever made the play.

Node order per possession: [ball, off0..off4, def0..def4]  (11 nodes)
Node features (F=10): x/94, y/50, vx, vy, z, is_ball, is_offense, is_ballhandler,
                      dist_to_ball, dist_to_off_hoop

Output: results/epv/epv_possessions.npz
  X:[N,KMAX,11,F] mask:[N,KMAX] R:[N] term:[N] bh:[N,KMAX] tframe:[N,KMAX]
  klen:[N] games:[N] off_team:[N] pids:[N,11]

`pids` maps each node to its NBA player id (0 = ball node), so ΔEPV can be
credited to a named player (epv_actions.py). Shards built before this was added
simply lack the key; every consumer treats it as optional.

Run:  python src/epv/possession_graphs.py
"""
from __future__ import annotations
import os, json, glob
import numpy as np
import pandas as pd

from tracking_graphs import game_timeline, HOOPS, F  # reuse timeline + constants

POSS = "results/epv/possessions_pbp.csv"
TRACK_GLOB = "data/tracking/*.json"
OUT = "results/epv/epv_possessions.npz"
KMAX, KMIN, SHOTCLOCK = 40, 8, 24.0            # max frames, min raw frames, window cap (s)
TERM_CODE = {"made_fg": 0, "made_ft": 1, "miss_defreb": 2,
             "turnover": 3, "reb_end": 4, "period_end": 5, "other": 6}


def build_possession(win, off_pids, def_pids, hoop):
    """win: chronological list of (gt, sclock, players{pid:(x,y,tid)}, ball(x,y,z)).
    Returns (X:[k,11,F], bh:[k] ballhandler node index in 1..5)."""
    order = [None] + list(off_pids) + list(def_pids)          # 11 nodes
    k = len(win)
    X = np.zeros((k, 11, F), dtype=np.float32)
    bh = np.zeros(k, dtype=np.int64)
    xy_prev, gt_prev = None, None
    for t, (gt, sc, players, ball) in enumerate(win):
        xy = np.zeros((11, 2), dtype=np.float32)
        z = np.zeros(11, dtype=np.float32)
        for i, pid in enumerate(order):
            if i == 0:
                xy[i] = ball[0], ball[1]; z[i] = ball[2]
            else:
                px, py, _ = players[pid]; xy[i] = px, py
        vel = ((xy - xy_prev) / (gt - gt_prev)
               if xy_prev is not None and gt - gt_prev > 1e-3
               else np.zeros((11, 2), dtype=np.float32))
        d_ball = np.linalg.norm(xy - xy[0], axis=1)
        d_hoop = np.linalg.norm(xy - hoop, axis=1)
        X[t, :, 0] = xy[:, 0] / 94.0
        X[t, :, 1] = xy[:, 1] / 50.0
        X[t, :, 2] = np.clip(vel[:, 0] / 30.0, -1, 1)
        X[t, :, 3] = np.clip(vel[:, 1] / 30.0, -1, 1)
        X[t, :, 4] = z / 15.0
        X[t, 0, 5] = 1.0                          # is_ball
        X[t, 1:6, 6] = 1.0                        # is_offense (nodes 1..5)
        # ball-handler = offense node nearest the ball this frame
        bidx = 1 + int(np.argmin(d_ball[1:6]))
        X[t, bidx, 7] = 1.0                       # is_ballhandler
        bh[t] = bidx
        X[t, :, 8] = d_ball / 50.0
        X[t, :, 9] = d_hoop / 94.0
        xy_prev, gt_prev = xy, gt
    return X, bh


def records_from_game(tracking_data, gposs, gid):
    """Build possession records from an IN-MEMORY tracking dict + this game's
    possession rows. This is the streaming entry point (no file on disk) — see
    build_epv_dataset.py, which fetches tracking per game a la game.py, feeds it
    here, and discards the raw data. Returns a list of per-possession record dicts."""
    if gposs is None or len(gposs) == 0:
        return []
    tl = game_timeline(tracking_data)
    tl_gt = np.array([r[0] for r in tl])

    out, kept = [], 0
    for _, p in gposs.iterrows():
        t_lo = max(float(p.t_start), float(p.t_end) - SHOTCLOCK)
        lo = np.searchsorted(tl_gt, t_lo)
        hi = np.searchsorted(tl_gt, float(p.t_end) + 0.05)
        win_all = tl[lo:hi]
        if len(win_all) < KMIN:
            continue
        # stable set present in every raw frame of the window
        common = set(win_all[0][2].keys())
        for f in win_all:
            common &= set(f[2].keys())
        off = [pid for pid in common if win_all[-1][2][pid][2] == p.off_team_id]
        dfn = [pid for pid in common if win_all[-1][2][pid][2] == p.def_team_id]
        if len(off) < 5 or len(dfn) < 5:
            continue
        off, dfn = off[:5], dfn[:5]
        # sample up to KMAX frames evenly
        n = min(KMAX, len(win_all))
        idx = np.linspace(0, len(win_all) - 1, n).round().astype(int)
        win = [win_all[i] for i in idx]
        if not all(all(pid in f[2] for pid in off + dfn) for f in win):
            continue
        # offense's target hoop = nearest hoop to the ball at the terminal frame
        ball_last = np.array(win[-1][3][:2])
        hoop = HOOPS[np.argmin(np.linalg.norm(ball_last - HOOPS, axis=1))]

        Xk, bhk = build_possession(win, off, dfn, hoop)
        k = Xk.shape[0]
        X = np.zeros((KMAX, 11, F), dtype=np.float32);  X[:k] = Xk
        mask = np.zeros(KMAX, dtype=np.float32);         mask[:k] = 1.0
        bh = -np.ones(KMAX, dtype=np.int64);             bh[:k] = bhk
        tfr = np.zeros(KMAX, dtype=np.float32);          tfr[:k] = [w[0] for w in win]
        # node -> player id (0 for the ball node). Carrying identity here is what
        # lets dEPV be credited to a NAMED player downstream (epv_actions.py);
        # without it the ball-handler is only a slot index 1..5.
        pids = np.array([0] + [int(q) for q in off] + [int(q) for q in dfn], dtype=np.int64)
        out.append(dict(X=X, mask=mask, bh=bh, tframe=tfr, klen=k, pids=pids,
                        R=float(p.R), term=TERM_CODE.get(p.terminal_type, 6),
                        game=gid, off_team=int(p.off_team_id)))
        kept += 1
    print(f"  {gid}: {kept}/{len(gposs)} possessions -> sequences")
    return out


def process_game(path, poss):
    """File-based wrapper (used by main() for the already-downloaded games)."""
    gid = int(os.path.basename(path).split(".")[0])
    return records_from_game(json.load(open(path)), poss[poss.gid == gid], gid)


def stack_records(recs):
    """Turn a list of record dicts into the npz arrays (shared by builder + merge)."""
    return dict(
        X=np.stack([r["X"] for r in recs]),
        mask=np.stack([r["mask"] for r in recs]),
        bh=np.stack([r["bh"] for r in recs]),
        tframe=np.stack([r["tframe"] for r in recs]),
        pids=np.stack([r["pids"] for r in recs]),
        klen=np.array([r["klen"] for r in recs], dtype=np.int64),
        R=np.array([r["R"] for r in recs], dtype=np.float32),
        term=np.array([r["term"] for r in recs], dtype=np.int64),
        games=np.array([r["game"] for r in recs], dtype=np.int64),
        off_team=np.array([r["off_team"] for r in recs], dtype=np.int64),
    )


def main():
    os.makedirs("results/epv", exist_ok=True)
    poss = pd.read_csv(POSS)
    files = sorted(glob.glob(TRACK_GLOB))
    print(f"Tracking games: {len(files)} | possessions in table: {len(poss)}")
    allrec = []
    for f in files:
        allrec += process_game(f, poss)
    if not allrec:
        print("No possessions matched tracking."); return

    X = np.stack([r["X"] for r in allrec])
    mask = np.stack([r["mask"] for r in allrec])
    bh = np.stack([r["bh"] for r in allrec])
    tframe = np.stack([r["tframe"] for r in allrec])
    pids = np.stack([r["pids"] for r in allrec])
    klen = np.array([r["klen"] for r in allrec], dtype=np.int64)
    R = np.array([r["R"] for r in allrec], dtype=np.float32)
    term = np.array([r["term"] for r in allrec], dtype=np.int64)
    games = np.array([r["game"] for r in allrec], dtype=np.int64)
    off_team = np.array([r["off_team"] for r in allrec], dtype=np.int64)
    np.savez_compressed(OUT, X=X, mask=mask, bh=bh, tframe=tframe, klen=klen,
                        pids=pids, R=R, term=term, games=games, off_team=off_team)

    inv = {v: k for k, v in TERM_CODE.items()}
    print(f"\nSaved {OUT}")
    print(f"X {X.shape}  (N poss, KMAX={KMAX}, 11 nodes, F={F})")
    print(f"possessions: {len(R)} | mean R={R.mean():.3f} | frames/poss: "
          f"mean {klen.mean():.1f}, min {klen.min()}, max {klen.max()}")
    print(f"R dist: {dict(pd.Series(R).astype(int).value_counts().sort_index())}")
    print(f"terminals: { {inv[t]: int((term==t).sum()) for t in sorted(set(term.tolist()))} }")
    print(f"NaNs in X: {int(np.isnan(X).sum())} | mask total frames: {int(mask.sum())}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    main()
