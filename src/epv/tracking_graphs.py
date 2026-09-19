"""
tracking_graphs.py — turn raw SportVU tracking into graph sequences for GNN-EPV.

For each shot in a game we take the ~W seconds of tracking leading up to the
release and build a sequence of K player+ball graphs (11 nodes: ball + 5 offense
+ 5 defense). The label is the points the shot produced (0 / 2 / 3). A GNN that
regresses this label from the pre-shot state learns E[points | state] — i.e.
EPV, restricted for now to the *shot* terminal (the value-regression route to
Route 3; full EPV adds turnover/FT/rebound terminals once PBP is joined).

Node order is fixed within a sequence: [ball, shooter, offense×4, defense×5],
and only windows where the same 10 players are on court throughout are kept, so a
node means the same player across frames (velocities are finite differences).

Node features (F=10): x/94, y/50, vx, vy, z(ball height), is_ball, is_offense,
is_shooter, dist_to_ball, dist_to_off_hoop  (all in court units, scaled).

Input : data/tracking/<gid>.json  +  data/shot_features_valid2_type.csv (labels)
Output: results/epv/epv_graphs.npz  (X:[N,K,11,F], y:[N], meta)

Run:  python src/epv/tracking_graphs.py
"""
from __future__ import annotations
import glob, json, os, re
import numpy as np
import pandas as pd

TRACK_GLOB = "data/tracking/*.json"
SHOTS = "data/shot_features_valid2_type.csv"
OUT = "results/epv/epv_graphs.npz"
W, K, KMIN = 4.0, 12, 25          # window sec, frames sampled, min raw frames
HOOPS = np.array([[5.25, 25.0], [88.75, 25.0]])
F = 10


def norm(s):
    return re.sub(r"[^a-z ]", "", str(s).lower()).strip()


def game_timeline(track):
    """Return sorted unique frames: list of (game_time, shot_clock, {pid:(x,y,z)}, ball(x,y,z))."""
    seen, rows = set(), []
    for e in track["events"]:
        for m in (e["moments"] or []):
            if m[1] in seen:
                continue
            seen.add(m[1])
            q, _, gclock, sclock, _, pos = m
            if not pos:
                continue
            gt = (q - 1) * 720 + (720 - gclock)
            ball = None; players = {}
            for p in pos:
                tid, pid, x, y, z = p
                if pid == -1:
                    ball = (x, y, z)
                else:
                    players[pid] = (x, y, tid)
            if ball is None or len(players) < 10:
                continue
            rows.append((gt, sclock if sclock is not None else np.nan, players, ball))
    rows.sort(key=lambda r: r[0])
    return rows


def build_sequence(win, off_pids, def_pids, shooter_pid, hoop):
    """win: list of (gt, sclock, players{pid:(x,y,tid)}, ball) — chronological, len K."""
    order = [None] + [shooter_pid] + [p for p in off_pids if p != shooter_pid] + list(def_pids)
    gts = np.array([f[0] for f in win])
    X = np.zeros((len(win), 11, F), dtype=np.float32)
    xy_prev = None; gt_prev = None
    for t, (gt, sc, players, ball) in enumerate(win):
        xy = np.zeros((11, 2), dtype=np.float32)
        z = np.zeros(11, dtype=np.float32)
        for i, pid in enumerate(order):
            if i == 0:
                xy[i] = ball[0], ball[1]; z[i] = ball[2]
            else:
                px, py, _ = players[pid]; xy[i] = px, py
        # velocities (ft/s) via finite diff to previous selected frame
        if xy_prev is not None and gt - gt_prev > 1e-3:
            vel = (xy - xy_prev) / (gt - gt_prev)
        else:
            vel = np.zeros((11, 2), dtype=np.float32)
        d_ball = np.linalg.norm(xy - xy[0], axis=1)
        d_hoop = np.linalg.norm(xy - hoop, axis=1)
        X[t, :, 0] = xy[:, 0] / 94.0
        X[t, :, 1] = xy[:, 1] / 50.0
        X[t, :, 2] = np.clip(vel[:, 0] / 30.0, -1, 1)
        X[t, :, 3] = np.clip(vel[:, 1] / 30.0, -1, 1)
        X[t, :, 4] = z / 15.0
        X[t, 0, 5] = 1.0                       # is_ball
        X[t, 1:6, 6] = 1.0                     # is_offense (nodes 1..5)
        X[t, 1, 7] = 1.0                       # is_shooter
        X[t, :, 8] = d_ball / 50.0
        X[t, :, 9] = d_hoop / 94.0
        xy_prev, gt_prev = xy, gt
    return X


def process_game(path, shots):
    gid = int(os.path.basename(path).split(".")[0])
    gshots = shots[shots.game_id == gid]
    if gshots.empty:
        return [], []
    track = json.load(open(path))
    e0 = track["events"][0]
    name2pid, pid2team = {}, {}
    for side in ("home", "visitor"):
        tid = e0[side]["teamid"]
        for p in e0[side]["players"]:
            nm = norm(p["firstname"] + " " + p["lastname"])
            name2pid[nm] = p["playerid"]; pid2team[p["playerid"]] = tid
    tl = game_timeline(track)
    tl_gt = np.array([r[0] for r in tl])

    Xs, ys, kept = [], [], 0
    for _, s in gshots.iterrows():
        pid = name2pid.get(norm(s.player_name))
        if pid is None:
            continue
        ts = s.game_time
        lo = np.searchsorted(tl_gt, ts - W); hi = np.searchsorted(tl_gt, ts + 0.05)
        win_all = tl[lo:hi]
        if len(win_all) < KMIN:
            continue
        # need the shooter + a stable 10-man lineup across the window
        common = set(win_all[0][2].keys())
        for f in win_all:
            common &= set(f[2].keys())
        if pid not in common or len(common) < 10:
            continue
        off_team = pid2team[pid]
        off_pids = [p for p in common if pid2team.get(p) == off_team][:5]
        def_pids = [p for p in common if pid2team.get(p) != off_team][:5]
        if pid not in off_pids or len(off_pids) < 5 or len(def_pids) < 5:
            continue
        # sample K frames evenly across the window
        idx = np.linspace(0, len(win_all) - 1, K).round().astype(int)
        win = [win_all[i] for i in idx]
        # keep only frames where all 10 + ball present with our pid set
        ok = all(all(p in f[2] for p in off_pids + def_pids) for f in win)
        if not ok:
            continue
        hoop = HOOPS[np.argmin(np.linalg.norm(
            np.array(win[-1][2][pid][:2]) - HOOPS, axis=1))]
        Xs.append(build_sequence(win, off_pids, def_pids, pid, hoop))
        pts = float(s.get("is_3_pointer", 0))
        pts = (3.0 if s.is_3_pointer == 1 else 2.0) * float(s.made_shot)
        ys.append(pts); kept += 1
    print(f"  {gid}: {kept}/{len(gshots)} shots -> sequences")
    return Xs, ys


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shots = pd.read_csv(SHOTS)
    files = glob.glob(TRACK_GLOB)
    print(f"Tracking games: {len(files)}")
    allX, ally, allg = [], [], []
    for f in files:
        Xs, ys = process_game(f, shots)
        allX += Xs; ally += ys
        allg += [int(os.path.basename(f).split(".")[0])] * len(Xs)
    X = np.stack(allX); y = np.array(ally, dtype=np.float32); games = np.array(allg)
    np.savez_compressed(OUT, X=X, y=y, games=games)
    print(f"\nSaved {OUT}")
    print(f"X shape: {X.shape}  (N seq, K frames, 11 nodes, {F} feats)")
    print(f"labels: n={len(y)}  mean points/shot={y.mean():.3f}  "
          f"(0={np.mean(y==0):.2f}, 2={np.mean(y==2):.2f}, 3={np.mean(y==3):.2f})")
    print(f"NaNs in X: {np.isnan(X).sum()}")


if __name__ == "__main__":
    main()
