"""
epv_data.py — shared loader for the EPV possession dataset.

The merged `results/epv/epv_possessions.npz` is ~1 GB compressed and ~1.75 GB of
float32 `X` once decompressed, which is more than fits comfortably alongside a
training process on a laptop. This module builds a one-off **memmapped cache**
from the per-game shards (streamed shard-by-shard, so peak RAM stays at one
game) with `X` stored as float16 — 875 MB, small enough to hold in RAM, and far
more precision than the features need (x/94 to ~0.02 ft).

It also owns the two things every downstream script must agree on:

  * **label hygiene** — `R` is derived in `possessions.py` from the cumulative
    PBP `SCORE` column, and 0.23 % of rows (229 / 99 450) land outside the
    physically possible [0, 5] points-per-possession range (values from -50 to
    +30) where that column is non-monotone. Those rows are dropped by default;
    the count is always reported. Everything else is clean: 97.7 % of
    miss+defensive-rebound and 97.8 % of turnover possessions have R = 0, and
    made-FG possessions are 2 / 3.
  * **game-disjoint splits** — 70/15/15 over *game ids*, fixed seed, so no
    possession from a test game is ever seen in training.

Run `python src/epv/epv_data.py` for a dataset summary.
"""
from __future__ import annotations
import os, glob
import numpy as np

SHARD_DIR = "results/epv/shards"
NPZ = "results/epv/epv_possessions.npz"
# EPV_CACHE lets you park the ~900 MB cache on a fast local disk when the repo
# lives on a slow mount (e.g. /mnt/c under WSL).
CACHE_DIR = os.environ.get("EPV_CACHE", "results/epv/cache")

KEYS = ("X", "mask", "bh", "tframe", "klen", "R", "term", "games", "off_team")
DTYPES = dict(X=np.float16, mask=np.float32, bh=np.int16, tframe=np.float32,
              klen=np.int16, R=np.float32, term=np.int8, games=np.int64,
              off_team=np.int64)
TERM_NAMES = {0: "made_fg", 1: "made_ft", 2: "miss_defreb", 3: "turnover",
              4: "reb_end", 5: "period_end"}
R_LO, R_HI = 0.0, 5.0                      # physically possible possession value


# ──────────────────────────────────────────────────────────────────────────
def build_cache(cache_dir: str = CACHE_DIR, src: str | None = None, verbose=True):
    """Stream the shards (or the merged npz) into memmapped .npy files."""
    src = src or (SHARD_DIR if os.path.isdir(SHARD_DIR) else NPZ)
    os.makedirs(cache_dir, exist_ok=True)
    if os.path.isdir(src):
        files = sorted(glob.glob(os.path.join(src, "*.npz")))
        if not files:
            raise FileNotFoundError(f"no shards in {src}")
        n = 0
        for f in files:
            with np.load(f) as d:
                n += d["klen"].shape[0]
    else:
        files, n = [src], None

    if verbose:
        print(f"[cache] {len(files)} source file(s) -> {cache_dir}")
    mm = None
    i = 0
    for j, f in enumerate(files):
        with np.load(f) as d:
            k = d["klen"].shape[0]
            if mm is None:
                n = n or k
                shapes = {key: (n,) + d[key].shape[1:] for key in KEYS}
                mm = {key: np.lib.format.open_memmap(
                          os.path.join(cache_dir, key + ".npy"), mode="w+",
                          dtype=DTYPES[key], shape=shapes[key]) for key in KEYS}
            for key in KEYS:
                mm[key][i:i + k] = d[key]
            i += k
        if verbose and (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(files)} files, {i} possessions")
    assert i == n, f"row count mismatch {i} != {n}"
    for key in mm:
        mm[key].flush()
    if verbose:
        print(f"[cache] {n} possessions cached")
    return cache_dir


def load_epv(cache_dir: str | None = None, in_ram: bool = True,
             filter_R: bool = True, verbose: bool = True) -> dict:
    """Return the dataset as a dict of arrays plus `idx` (valid row indices).

    `X` is float16 [N,K,11,F]; cast per batch. Rows are never physically
    dropped (that would copy ~900 MB) — instead `idx` lists the rows that
    survive label hygiene, and every split works on indices into the full
    arrays.
    """
    cache_dir = cache_dir or CACHE_DIR
    if not os.path.exists(os.path.join(cache_dir, "X.npy")):
        build_cache(cache_dir, verbose=verbose)
    load = lambda k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode="r")
    d = {k: load(k) for k in KEYS}
    if in_ram:
        # X is the big one (875 MB as float16); the rest are a few MB each.
        d = {k: np.ascontiguousarray(v) for k, v in d.items()}
    n = len(d["R"])
    R = np.asarray(d["R"])
    if filter_R:
        keep = (R >= R_LO) & (R <= R_HI)
        if verbose and (~keep).any():
            bad = R[~keep]
            print(f"[data] dropping {(~keep).sum()} / {n} possessions "
                  f"({100*(~keep).mean():.3f}%) with R outside [{R_LO:.0f},{R_HI:.0f}] "
                  f"(PBP SCORE non-monotone; range {bad.min():.0f}..{bad.max():.0f})")
    else:
        keep = np.ones(n, bool)
    d["idx"] = np.flatnonzero(keep)
    return d


# ──────────────────────────────────────────────────────────────────────────
def game_splits(games: np.ndarray, idx: np.ndarray, seed: int = 0,
                fracs=(0.70, 0.15, 0.15)):
    """Game-disjoint train/val/test split. Returns three index arrays (into the
    full arrays), guaranteed to share no game id."""
    uniq = np.unique(games[idx])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n = len(perm)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    parts = (perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:])
    out = tuple(idx[np.isin(games[idx], p)] for p in parts)
    assert not (set(games[out[0]].tolist()) & set(games[out[2]].tolist()))
    assert not (set(games[out[0]].tolist()) & set(games[out[1]].tolist()))
    return out


def batches(index: np.ndarray, batch: int, shuffle: bool = False,
            rng: np.random.Generator | None = None):
    """Yield index slices. Shuffled batches are re-sorted so fancy-indexing the
    (possibly memmapped) X stays close to sequential."""
    order = rng.permutation(len(index)) if shuffle else np.arange(len(index))
    for i in range(0, len(order), batch):
        yield np.sort(index[order[i:i + batch]])


# ──────────────────────────────────────────────────────────────────────────
def summarise(d: dict):
    idx = d["idx"]
    R, term, games, klen = (np.asarray(d[k])[idx] for k in ("R", "term", "games", "klen"))
    print(f"possessions {len(idx)} | games {len(np.unique(games))} "
          f"({len(idx)/len(np.unique(games)):.1f} per game)")
    print(f"R: mean {R.mean():.4f}  std {R.std():.3f}  "
          f"dist {dict(sorted(zip(*[x.tolist() for x in np.unique(R.astype(int), return_counts=True)])))}")
    print(f"frames/possession: mean {klen.mean():.1f}  min {klen.min()}  max {klen.max()}")
    print("terminals:")
    for t in sorted(set(term.tolist())):
        s = R[term == t]
        print(f"  {TERM_NAMES[t]:<12} n={len(s):>6}  mean R={s.mean():.3f}")


if __name__ == "__main__":
    d = load_epv()
    summarise(d)
    tr, va, te = game_splits(np.asarray(d["games"]), d["idx"], seed=0)
    print(f"\nsplit(seed=0): train {len(tr)} | val {len(va)} | test {len(te)} possessions, "
          f"games {len(np.unique(np.asarray(d['games'])[tr]))}/"
          f"{len(np.unique(np.asarray(d['games'])[va]))}/"
          f"{len(np.unique(np.asarray(d['games'])[te]))}")
