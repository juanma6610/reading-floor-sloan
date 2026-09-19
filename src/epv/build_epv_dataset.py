"""
build_epv_dataset.py — STREAMING builder for the EPV possession dataset.

Answers "can I scale to 631 games without storing 63 GB of raw tracking?" — yes,
by doing exactly what game.py does: fetch ONE game's tracking (+PBP) at a time,
turn it into compact possession tensors, then DISCARD the raw data. Only the
per-game shards (a few hundred KB each) persist. 631 raw tracking JSONs ≈ 63 GB;
631 shards ≈ a few hundred MB total.

Per game:
  fetch tracking+pbp (game.py Game -> downloads to temp, cleans up after load)
    -> possessions.segment_game(pbp)            (§2)
    -> possession_graphs.records_from_game(...)  (§3)
    -> save results/epv/shards/<game_id>.npz     (compact tensors only)
  raw tracking/pbp are gone; peak disk = one game's temp .7z + json.

Shards are the training set (epv_gnn_torch can read them lazily). `--merge`
concatenates them into a single results/epv/epv_possessions.npz if you want one
file. Existing shards are skipped, so runs resume after interruption.

MODES
  --local           build shards from ALREADY-downloaded data/tracking/*.json +
                    data/pbp/<gid>.csv  (no network; for validation / the 3 games)
  --games FILE      stream from the sealneaward mirror; FILE lists one game per
                    line as the tracking 7z name, e.g. 01.13.2016.GSW.at.DEN
  --merge           concat all shards -> results/epv/epv_possessions.npz

NOTE: network streaming (Game()) uses urllib exactly like game.py and must be run
on your machine — this sandbox blocks programmatic web fetches. The --local path
is fully runnable here and exercises the identical segment->graph code.

Run:
  python src/epv/build_epv_dataset.py --local
  python src/epv/build_epv_dataset.py --games games_2015_16.txt
  python src/epv/build_epv_dataset.py --merge
"""
from __future__ import annotations
import os, sys, glob, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))            # src/epv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # src (for game.py)

from possessions import segment_game
from possession_graphs import records_from_game, stack_records

SHARD_DIR = "results/epv/shards"
OUT = "results/epv/epv_possessions.npz"


def save_shard(gid, recs):
    if not recs:
        print(f"  {gid}: 0 possessions, no shard"); return 0
    os.makedirs(SHARD_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(SHARD_DIR, f"{gid}.npz"), **stack_records(recs))
    return len(recs)


def build_one(gid, tracking_data, pbp_df):
    """Shared core: PBP -> possessions -> graph records -> shard. Raw inputs are
    dropped by the caller right after this returns."""
    poss = segment_game(pbp_df)
    recs = records_from_game(tracking_data, poss[poss.gid == gid], gid)
    return save_shard(gid, recs)


# ── mode: local (already-downloaded games) ───────────────────────────────────
def run_local():
    total = 0
    for path in sorted(glob.glob("data/tracking/*.json")):
        gid = int(os.path.basename(path).split(".")[0])
        shard = os.path.join(SHARD_DIR, f"{gid}.npz")
        if os.path.exists(shard):
            print(f"  {gid}: shard exists, skip"); continue
        pbp_path = f"data/pbp/{gid:010d}.csv"
        if not os.path.exists(pbp_path):
            print(f"  {gid}: no PBP at {pbp_path}, skip"); continue
        tracking_data = json.load(open(path))
        pbp_df = pd.read_csv(pbp_path)
        total += build_one(gid, tracking_data, pbp_df)
        del tracking_data, pbp_df                        # free the raw game
    print(f"local build: {total} possessions across shards in {SHARD_DIR}")


# ── mode: stream from mirror (run on your machine) ───────────────────────────
def run_stream(games_file):
    from game import Game                                # heavy + network; import lazily
    specs = [l.strip() for l in open(games_file) if l.strip() and not l.startswith("#")]
    print(f"Streaming {len(specs)} games from the mirror ...")
    total = 0
    for i, spec in enumerate(specs, 1):
        try:
            g = Game(date=None, team1=None, team2=None, game_7z=f"{spec}.7z", verbose=False)
            gid = int(g.game_id)
            shard = os.path.join(SHARD_DIR, f"{gid}.npz")
            if os.path.exists(shard):
                print(f"[{i}/{len(specs)}] {gid}: shard exists, skip"); del g; continue
            n = build_one(gid, g.tracking_data, g.pbp)
            print(f"[{i}/{len(specs)}] {gid}: +{n} possessions")
            total += n
            del g                                        # drop raw tracking for this game
        except Exception as e:
            print(f"[{i}/{len(specs)}] {spec}: FAILED ({e})")
    print(f"stream build: {total} possessions across shards in {SHARD_DIR}")


# ── mode: merge shards -> single npz (streamed, low memory) ──────────────────
def run_merge():
    shards = sorted(glob.glob(os.path.join(SHARD_DIR, "*.npz")))
    if not shards:
        print("no shards to merge"); return
    with np.load(shards[0]) as d0:                 # `pids` is present only in
        keys = list(d0.files)                      # shards built after identity
    acc = {k: [] for k in keys}                    # emission was added
    n = 0
    for s in shards:
        d = np.load(s)
        for k in keys:
            acc[k].append(d[k])
        n += len(d["R"])
    merged = {k: np.concatenate(acc[k], axis=0) for k in keys}
    np.savez_compressed(OUT, **merged)
    sz = os.path.getsize(OUT) / 1e6
    print(f"merged {len(shards)} shards -> {OUT}  ({n} possessions, {sz:.1f} MB, "
          f"mean R={merged['R'].mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--games", metavar="FILE")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--shard-dir", dest="shard_dir",
                    help="write/read shards here instead of results/epv/shards "
                         "(use for a pid-carrying rebuild so existing shards survive)")
    a = ap.parse_args()
    if a.shard_dir:
        global SHARD_DIR
        SHARD_DIR = a.shard_dir
        os.makedirs(SHARD_DIR, exist_ok=True)
    if a.local:  run_local()
    if a.games:  run_stream(a.games)
    if a.merge:  run_merge()
    if not (a.local or a.games or a.merge):
        ap.print_help()


if __name__ == "__main__":
    main()
