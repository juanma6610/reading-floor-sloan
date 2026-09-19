"""
epv_actions.py — action values from the causal EPV model (§6 of
docs/epv_m4_true_epv_spec.md).

Once V(t) exists at every frame, the value of what happened between two frames is
    dEPV(t) = V(t+1) - V(t)
credited to the BALL-HANDLER at t, and the move in (t, t+1] is classified from the
geometry the dataset already carries:

    pass        the ball-handler node changes
    drive       same handler, his distance to the hoop drops > DRIVE_FT
    dribble     same handler, small positional change
    shot        the transition into the terminal frame of a shooting possession
    turnover    the transition into the terminal frame of a turnover
    other_end   terminal frame of a rebound-end / period-end possession

**Every value used here is out-of-sample.** A leaderboard built from in-sample
predictions would credit players for the model having memorised their
possessions, so V(t) comes from a K-fold GAME-DISJOINT cross-fit: fold k trains
on the other folds' games and predicts only fold k's, so no possession is ever
scored by a model that saw its game.

Player identity: `pids` (node -> NBA player id) is emitted by possession_graphs.py.
The 631 streamed shards predate that field, so the per-PLAYER leaderboard covers
whichever games have pid-carrying shards (rebuild any game with
`build_epv_dataset.py --shard-dir results/epv/shards_pid`). The per-TEAM
leaderboard uses `off_team`, which every shard has, so it covers all 631 games.

Run:
  python src/epv/epv_actions.py                  # 5-fold cross-fit + leaderboards
  python src/epv/epv_actions.py --folds 3
  python src/epv/epv_actions.py --reuse          # reuse a cached cross-fit V
"""
from __future__ import annotations
import os, sys, glob, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epv_data import load_epv, TERM_NAMES

OUT_DIR = "results/epv"
PID_SHARDS = "results/epv/shards_pid"
TRACK_GLOB = "data/tracking/*.json"
VCACHE = os.path.join(OUT_DIR, "epv_crossfit_V.npy")
DRIVE_FT = 1.5              # handler must close this much on the hoop to be a "drive"
ACTIONS = ["pass", "drive", "dribble", "shot", "turnover", "other_end"]
SHOT_TERMS = {0, 1, 2}      # made_fg, made_ft, miss_defreb -> the possession ended on a shot


# ──────────────────────────────────────────────────────────────────────────
def crossfit_values(d, folds=5, seed=0, epochs=40, lam=0.8, gru_h=None,
                    lr=None, refresh=None, verbose=True):
    """Out-of-sample V:[N,K] for every possession, via game-disjoint K-fold."""
    import gnn_epv_causal as G
    games = np.asarray(d["games"]); idx = d["idx"]
    uniq = np.unique(games[idx])
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(uniq), folds)
    V = np.full((len(games), d["mask"].shape[1]), np.nan, dtype=np.float32)
    for k, hold in enumerate(parts, 1):
        te = np.sort(idx[np.isin(games[idx], hold)])   # predict_values returns ascending
        rest = idx[~np.isin(games[idx], hold)]
        # a val slice for early stopping, still disjoint from the held-out games
        rg = np.unique(games[rest]); rng2 = np.random.default_rng(seed + k)
        vg = rng2.permutation(rg)[:max(1, len(rg) // 7)]
        va = rest[np.isin(games[rest], vg)]; tr = rest[~np.isin(games[rest], vg)]
        if verbose:
            print(f"[cross-fit] fold {k}/{folds}: train {len(tr)} / val {len(va)} "
                  f"-> predict {len(te)} possessions ({len(hold)} games)", flush=True)
        kw = dict(lam=lam, epochs=epochs, seed=seed, verbose=False)
        if gru_h: kw["gru_h"] = gru_h
        if lr: kw["lr"] = lr
        if refresh: kw["refresh"] = refresh
        model = G.train(d, tr, va, **kw)
        V[te] = G.predict_values(model, d, te)
        if verbose:
            m = G.evaluate(model, d, te, baseline_mean=float(np.asarray(d["R"][tr]).mean()))
            print(f"            fold term-RMSE {m['term_rmse']:.4f} "
                  f"(baseline {m['baseline_rmse']:.4f}) V(0) {m['first_frame_epv']:.3f}",
                  flush=True)
    assert not np.isnan(V[idx]).any(), "cross-fit left possessions unscored"
    return V


# ──────────────────────────────────────────────────────────────────────────
def action_table(d, V, rows, chunk=4096, verbose=True):
    """One row per frame transition: (row, t, handler node, action, dEPV)."""
    K = V.shape[1]
    out = []
    for s in range(0, len(rows), chunk):
        b = rows[s:s + chunk]
        Xb = np.asarray(d["X"][b]).astype(np.float32)
        bh = np.asarray(d["bh"][b]).astype(np.int16)
        klen = np.asarray(d["klen"][b]).astype(int)
        term = np.asarray(d["term"][b]).astype(int)
        Vb = V[b]
        n = len(b)
        ii = np.arange(n)[:, None]; kk = np.arange(K)[None, :]
        h = np.clip(bh, 1, 5).astype(int)
        dhoop = Xb[..., 9] * 94.0
        bh_dh = dhoop[ii, kk, h]                                  # [n,K]

        tt = np.arange(K - 1)[None, :]
        valid = tt < (klen[:, None] - 1)                          # transitions t -> t+1
        is_last = tt == (klen[:, None] - 2)
        same = bh[:, 1:] == bh[:, :-1]
        closing = bh_dh[:, :-1] - bh_dh[:, 1:]                    # >0 = toward the hoop
        act = np.full((n, K - 1), 2, dtype=np.int8)               # default: dribble
        act[~same] = 0                                            # pass
        act[same & (closing > DRIVE_FT)] = 1                      # drive
        endcode = np.where(np.isin(term, list(SHOT_TERMS)), 3,
                           np.where(term == 3, 4, 5))             # shot / turnover / other
        act = np.where(is_last, endcode[:, None], act)

        dv = Vb[:, 1:] - Vb[:, :-1]
        out.append(np.rec.fromarrays(
            [np.repeat(b[:, None], K - 1, 1)[valid],
             np.repeat(tt, n, 0)[valid],
             bh[:, :-1][valid].astype(np.int16),
             act[valid],
             dv[valid].astype(np.float32)],
            names="row,t,node,action,dv"))
        if verbose and (s // chunk) % 5 == 0:
            print(f"    actions {min(s+chunk, len(rows))}/{len(rows)}", flush=True)
    return np.concatenate(out).view(np.recarray)


def summarise_actions(A, d):
    R = np.asarray(d["R"])
    rows = []
    for a, name in enumerate(ACTIONS):
        s = A[A.action == a]
        if not len(s): continue
        rows.append(dict(action=name, n=len(s), mean_dEPV=float(s.dv.mean()),
                         sd_dEPV=float(s.dv.std()), total_dEPV=float(s.dv.sum()),
                         p_positive=float((s.dv > 0).mean()),
                         mean_R_of_possession=float(R[s.row].mean())))
    return rows


# ──────────────────────────────────────────────────────────────────────────
def load_pid_map():
    """row-key -> pids[11] for the games rebuilt with identity, plus pid -> name."""
    per_game = {}
    for f in sorted(glob.glob(os.path.join(PID_SHARDS, "*.npz"))):
        with np.load(f) as s:
            if "pids" not in s.files: continue
            per_game[int(s["games"][0])] = {k: s[k] for k in
                                            ("pids", "X", "R", "klen", "games")}
    names = {}
    for f in sorted(glob.glob(TRACK_GLOB)):
        try:
            import json as _j
            ev = _j.load(open(f))["events"][0]
            for side in ("home", "visitor"):
                for p in ev[side]["players"]:
                    names[int(p["playerid"])] = f"{p['firstname']} {p['lastname']}"
        except Exception as e:
            print(f"  (name lookup failed for {f}: {e})")
    return per_game, names


def player_leaderboard(d, A, per_game, names, min_frames=50):
    """Credit dEPV to the NAMED handler, for games that carry `pids`.

    The pid rebuild and the main dataset run the same deterministic builder, so
    rows line up game-by-game — but that is VERIFIED against X here rather than
    assumed, and a game whose tensors differ is skipped."""
    games = np.asarray(d["games"])
    credit, touches = {}, {}
    used, skipped = [], []
    for gid, s in per_game.items():
        rows = np.flatnonzero(games == gid)
        if len(rows) != len(s["R"]):
            skipped.append((gid, f"row count {len(rows)} vs {len(s['R'])}")); continue
        Xmain = np.asarray(d["X"][rows]).astype(np.float32)
        if not np.allclose(Xmain, s["X"].astype(np.float32), atol=2e-3):
            skipped.append((gid, "tensor mismatch")); continue
        used.append(gid)
        pos = {r: j for j, r in enumerate(rows)}
        sel = A[np.isin(A.row, rows)]
        for r, node, dv in zip(sel.row, sel.node, sel.dv):
            pid = int(s["pids"][pos[r], int(node)])
            credit[pid] = credit.get(pid, 0.0) + float(dv)
            touches[pid] = touches.get(pid, 0) + 1
    board = [dict(player_id=p, player=names.get(p, str(p)), frames_with_ball=touches[p],
                  total_dEPV=credit[p], dEPV_per_100_frames=100 * credit[p] / touches[p])
             for p in credit if touches[p] >= min_frames]
    board.sort(key=lambda r: -r["total_dEPV"])
    return board, used, skipped


def team_leaderboard(d, A, min_frames=500):
    off = np.asarray(d["off_team"])
    t = off[A.row]
    out = []
    for team in np.unique(t):
        s = A.dv[t == team]
        if len(s) < min_frames: continue
        out.append(dict(team_id=int(team), frames=int(len(s)), total_dEPV=float(s.sum()),
                        dEPV_per_100_frames=float(100 * s.mean())))
    out.sort(key=lambda r: -r["dEPV_per_100_frames"])
    return out


def _csv(path, rows):
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} rows)")


def main(a):
    d = load_epv()
    idx = d["idx"]
    if a.reuse and os.path.exists(VCACHE):
        V = np.load(VCACHE); print(f"reusing cross-fit values from {VCACHE}")
    else:
        V = crossfit_values(d, folds=a.folds, epochs=a.epochs, lam=a.lam,
                            gru_h=a.gru_h, lr=a.lr, refresh=a.refresh)
        np.save(VCACHE, V); print(f"wrote {VCACHE}")

    print("\ncomputing dEPV over all possessions ...")
    A = action_table(d, V, idx)
    print(f"{len(A)} frame transitions over {len(idx)} possessions")

    rows = summarise_actions(A, d)
    print("\n" + "=" * 88)
    print("ACTION VALUES — dEPV(t) = V(t+1) - V(t), credited to the ball-handler at t")
    print("=" * 88)
    print(f"{'action':<12}{'n':>10}{'mean dEPV':>13}{'sd':>9}{'P(dEPV>0)':>12}"
          f"{'total dEPV':>14}{'mean R':>9}")
    for r in rows:
        print(f"{r['action']:<12}{r['n']:>10}{r['mean_dEPV']:>13.5f}{r['sd_dEPV']:>9.4f}"
              f"{r['p_positive']:>12.3f}{r['total_dEPV']:>14.1f}{r['mean_R_of_possession']:>9.3f}")
    print("=" * 88)
    _csv(os.path.join(OUT_DIR, "epv_action_values.csv"), rows)

    tb = team_leaderboard(d, A)
    print(f"\nTEAM added value (all {len(np.unique(np.asarray(d['games'])[idx]))} games, "
          f"top/bottom 5 by dEPV per 100 handled frames):")
    print(f"  {'team_id':>12}{'frames':>10}{'per 100 frames':>17}")
    for r in tb[:5]:
        print(f"  {r['team_id']:>12}{r['frames']:>10}{r['dEPV_per_100_frames']:>17.4f}")
    print(f"  {'...':>12}")
    for r in tb[-5:]:
        print(f"  {r['team_id']:>12}{r['frames']:>10}{r['dEPV_per_100_frames']:>17.4f}")
    _csv(os.path.join(OUT_DIR, "epv_team_added_value.csv"), tb)

    per_game, names = load_pid_map()
    if per_game:
        board, used, skipped = player_leaderboard(d, A, per_game, names)
        print(f"\nPLAYER added value — {len(used)} pid-carrying game(s) "
              f"{used}{', skipped ' + str(skipped) if skipped else ''}")
        print(f"{'player':<24}{'frames w/ ball':>16}{'total dEPV':>13}{'per 100 frames':>16}")
        for r in board[:15]:
            print(f"  {r['player']:<22}{r['frames_with_ball']:>16}"
                  f"{r['total_dEPV']:>13.3f}{r['dEPV_per_100_frames']:>16.3f}")
        if len(board) > 15:
            print("  ...")
            for r in board[-5:]:
                print(f"  {r['player']:<22}{r['frames_with_ball']:>16}"
                      f"{r['total_dEPV']:>13.3f}{r['dEPV_per_100_frames']:>16.3f}")
        _csv(os.path.join(OUT_DIR, "epv_player_added_value.csv"), board)
        json.dump(dict(games_with_identity=used, skipped=skipped,
                       n_players=len(board)),
                  open(os.path.join(OUT_DIR, "epv_player_coverage.json"), "w"), indent=2)
    else:
        print(f"\nNo pid-carrying shards in {PID_SHARDS} — per-player leaderboard skipped. "
              f"Rebuild games with: python src/epv/build_epv_dataset.py --local "
              f"--shard-dir {PID_SHARDS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--gru-h", type=int, default=None, dest="gru_h")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--refresh", type=int, default=None)
    ap.add_argument("--reuse", action="store_true")
    main(ap.parse_args())
