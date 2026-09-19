"""
epv_baselines.py — the models the causal GNN has to beat (§6 of
docs/epv_m4_true_epv_spec.md, §6 of docs/epv_gnn_experiment_plan.md).

All four baselines predict the SAME quantity as the GNN — a value V(t) at every
tracking frame of a possession — on the SAME game-disjoint 70/15/15 splits, over
the same seeds, so the metrics table is apples-to-apples:

  B0  constant PPP        V(t) = mean R over the training possessions.
  B1  XGBoost value model V(t) = f(hand-crafted features of frames <= t).
                          Ball / handler / defender geometry, spacing summaries,
                          and 5-frame-back deltas (past only). Trained on the MC
                          target (R at every frame), which is what a per-frame
                          regressor can do without bootstrapping.
  B2  GNN spatial-only    node attention, NO temporal encoder (memoryless).
  B3  GNN temporal-only   NO node attention, causal GRU over mean-pooled nodes.
  --  GNN full            node attention + causal GRU + TD(lambda).

B1 is deliberately given NO frame index and no possession-duration feature: under
the fixed-K resampling, frame index partially encodes time-to-terminal, which is
future information. `--leak-probe` quantifies that channel (an XGBoost trained on
the frame index alone) so the size of the advantage a sequence model could steal
from it is on the record rather than assumed away.

Run:
  python src/epv/epv_baselines.py                 # full table, 3 seeds
  python src/epv/epv_baselines.py --seeds 0       # quick
  python src/epv/epv_baselines.py --leak-probe    # frame-index-only diagnostic
"""
from __future__ import annotations
import os, sys, json, csv, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epv_data import load_epv, game_splits, TERM_NAMES

OUT_DIR = "results/epv"
HOOP_3PT = 23.75
LAG = 5                      # frames back for the "how did we get here" deltas

FEATURES = [
    "ball_x", "ball_y", "ball_z", "ball_dhoop", "ball_speed", "ball_closing",
    "bh_x", "bh_y", "bh_dhoop", "bh_speed", "bh_closing", "bh_dball",
    "def1_dist", "def2_dist", "def_min_dhoop", "def_mean_dhoop", "def_cx_dhoop",
    "off_mean_dhoop", "off_min_dhoop", "off_spread_x", "off_spread_y",
    "off_mean_pair", "off_in_arc", "def_between",
    "d5_ball_dhoop", "d5_bh_dhoop", "d5_def1", "bh_switch5",
]


# ──────────────────────────────────────────────────────────────────────────
def frame_features(d, rows, chunk=2048, verbose=False):
    """Hand-crafted per-frame features for possessions `rows`.
    Returns Z:[len(rows), K, len(FEATURES)] float32. Uses ONLY frames <= t."""
    K = d["mask"].shape[1]
    Z = np.zeros((len(rows), K, len(FEATURES)), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        b = rows[s:s + chunk]
        X = np.asarray(d["X"][b]).astype(np.float32)          # [n,K,11,10]
        bh = np.asarray(d["bh"][b]).astype(int)               # [n,K] in 1..5 (-1 pad)
        n = len(b)
        x = X[..., 0] * 94.0; y = X[..., 1] * 50.0; z = X[..., 4] * 15.0
        vx = X[..., 2] * 30.0; vy = X[..., 3] * 30.0
        dball = X[..., 8] * 50.0; dhoop = X[..., 9] * 94.0
        spd = np.hypot(vx, vy)                                # [n,K,11]

        ii, kk = np.arange(n)[:, None], np.arange(K)[None, :]
        h = np.clip(bh, 1, 5)                                 # handler node per frame
        # closing speed toward the hoop: -d/dt of dist_to_hoop, from the hoop unit
        # vector implied by (position, dist). Approximated by the frame-to-frame
        # change in dhoop (past-only: t vs t-1), which is what "driving" means.
        d_prev = np.concatenate([dhoop[:, :1], dhoop[:, :-1]], axis=1)
        closing = d_prev - dhoop                              # >0 = getting closer

        off = slice(1, 6); dfn = slice(6, 11)
        bh_x, bh_y = x[ii, kk, h], y[ii, kk, h]
        bh_dhoop = dhoop[ii, kk, h]; bh_spd = spd[ii, kk, h]
        bh_close = closing[ii, kk, h]; bh_dball = dball[ii, kk, h]
        # defender distances to the ball-handler
        dd = np.hypot(x[:, :, dfn] - bh_x[..., None], y[:, :, dfn] - bh_y[..., None])
        dsort = np.sort(dd, axis=2)
        off_x, off_y = x[:, :, off], y[:, :, off]
        pair = np.hypot(off_x[:, :, :, None] - off_x[:, :, None, :],
                        off_y[:, :, :, None] - off_y[:, :, None, :])
        off_dh, def_dh = dhoop[:, :, off], dhoop[:, :, dfn]

        cols = {
            "ball_x": x[:, :, 0], "ball_y": y[:, :, 0], "ball_z": z[:, :, 0],
            "ball_dhoop": dhoop[:, :, 0], "ball_speed": spd[:, :, 0],
            "ball_closing": closing[:, :, 0],
            "bh_x": bh_x, "bh_y": bh_y, "bh_dhoop": bh_dhoop, "bh_speed": bh_spd,
            "bh_closing": bh_close, "bh_dball": bh_dball,
            "def1_dist": dsort[:, :, 0], "def2_dist": dsort[:, :, 1],
            "def_min_dhoop": def_dh.min(2), "def_mean_dhoop": def_dh.mean(2),
            "def_cx_dhoop": np.hypot(x[:, :, dfn].mean(2) - x[:, :, 0],
                                     y[:, :, dfn].mean(2) - y[:, :, 0]),
            "off_mean_dhoop": off_dh.mean(2), "off_min_dhoop": off_dh.min(2),
            "off_spread_x": off_x.std(2), "off_spread_y": off_y.std(2),
            "off_mean_pair": pair.sum((2, 3)) / 20.0,
            "off_in_arc": (off_dh < HOOP_3PT).sum(2).astype(np.float32),
            "def_between": (def_dh < bh_dhoop[..., None]).sum(2).astype(np.float32),
        }
        # past-only deltas over LAG frames (clamped at the possession start)
        lagi = np.maximum(kk - LAG, 0)
        cols["d5_ball_dhoop"] = dhoop[:, :, 0] - dhoop[ii, lagi, 0]
        cols["d5_bh_dhoop"] = bh_dhoop - bh_dhoop[ii, lagi]
        cols["d5_def1"] = dsort[:, :, 0] - dsort[ii, lagi, 0]
        cols["bh_switch5"] = (bh != bh[ii, lagi]).astype(np.float32)

        for j, name in enumerate(FEATURES):
            Z[s:s + n, :, j] = cols[name]
        if verbose and (s // chunk) % 10 == 0:
            print(f"    features {s + n}/{len(rows)}", flush=True)
    return Z


def _flatten(d, rows, Z, sub=None, rng=None):
    """[n,K,F] -> (design matrix, target R, possession row, frame t) over valid frames."""
    klen = np.asarray(d["klen"][rows]).astype(int)
    R = np.asarray(d["R"][rows]).astype(np.float32)
    K = Z.shape[1]
    valid = np.arange(K)[None, :] < klen[:, None]
    M = Z[valid]
    yy = np.repeat(R[:, None], K, 1)[valid]
    rr = np.repeat(rows[:, None], K, 1)[valid]
    tt = np.repeat(np.arange(K)[None, :], len(rows), 0)[valid]
    if sub is not None and len(M) > sub:
        pick = rng.choice(len(M), sub, replace=False)
        M, yy, rr, tt = M[pick], yy[pick], rr[pick], tt[pick]
    return M, yy, rr, tt


# ──────────────────────────────────────────────────────────────────────────
def eval_per_frame(Vhat, d, rows, baseline_mean, nbins=10):
    """Same metric set as gnn_epv_causal.evaluate, from a [n,K] value matrix."""
    klen = np.asarray(d["klen"][rows]).astype(int)
    R = np.asarray(d["R"][rows]).astype(np.float64)
    m = np.asarray(d["mask"][rows]).astype(bool)
    term = np.asarray(d["term"][rows])
    ar = np.arange(len(rows))
    Vterm = Vhat[ar, klen - 1]
    term_rmse = float(np.sqrt(np.mean((Vterm - R) ** 2)))
    base = float(np.sqrt(np.mean((baseline_mean - R) ** 2)))
    Va = Vhat[m].astype(np.float64); Ra = np.repeat(R[:, None], Vhat.shape[1], 1)[m]
    frame_rmse = float(np.sqrt(np.mean((Va - Ra) ** 2)))
    q = np.unique(np.quantile(Va, np.linspace(0, 1, nbins + 1)))
    bi = np.clip(np.searchsorted(q, Va, side="right") - 1, 0, len(q) - 2)
    cal = [(float(Va[bi == i].mean()), float(Ra[bi == i].mean()), int((bi == i).sum()))
           for i in range(len(q) - 1) if (bi == i).sum() > 50]
    # a constant predictor (B0) collapses every quantile into one bin — calibration
    # is undefined there, not zero.
    ca = np.array([[a, b] for a, b, _ in cal], dtype=float).reshape(-1, 2)
    w = np.array([c for _, _, c in cal], float)
    cal_mae = (float(np.average(np.abs(ca[:, 0] - ca[:, 1]), weights=w))
               if len(cal) else float("nan"))
    cal_slope = float(np.polyfit(ca[:, 0], ca[:, 1], 1)[0]) if len(cal) > 2 else float("nan")
    return dict(term_rmse=term_rmse, baseline_rmse=base,
                skill=1 - (term_rmse / base) ** 2, frame_rmse=frame_rmse,
                term_corr=float(np.corrcoef(Vterm, R)[0, 1]) if Vterm.std() > 0 else 0.0,
                first_frame_epv=float(Vhat[:, 0].mean()),
                first_frame_sd=float(Vhat[:, 0].std()),
                cal_mae=cal_mae, cal_slope=cal_slope, calibration=cal,
                by_term={int(t): (float(Vterm[term == t].mean()), float(R[term == t].mean()),
                                  int((term == t).sum())) for t in sorted(set(term.tolist()))},
                pos_rmse=[float(np.sqrt(np.mean((Vhat[:, t][m[:, t]] - R[m[:, t]]) ** 2)))
                          for t in range(Vhat.shape[1]) if m[:, t].sum() > 50],
                n=len(rows))


def run_b0(d, tr, te):
    bm = float(np.asarray(d["R"][tr]).mean())
    V = np.full((len(te), d["mask"].shape[1]), bm, dtype=np.float32)
    m = eval_per_frame(V, d, te, bm); m["train_mean_R"] = bm
    return m


def run_b1(d, tr, va, te, seed=0, n_train_frames=1_200_000, verbose=True,
           feature_names=None, return_model=False):
    import xgboost as xgb
    rng = np.random.default_rng(seed)
    t0 = time.time()
    Ztr = frame_features(d, tr); Zva = frame_features(d, va); Zte = frame_features(d, te)
    Mtr, ytr, _, _ = _flatten(d, tr, Ztr, sub=n_train_frames, rng=rng)
    Mva, yva, _, _ = _flatten(d, va, Zva, sub=300_000, rng=rng)
    del Ztr, Zva
    if verbose:
        print(f"    B1 features: train {Mtr.shape} val {Mva.shape} ({time.time()-t0:.0f}s)")
    names = feature_names or FEATURES
    model = xgb.XGBRegressor(
        n_estimators=2000, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=50, reg_lambda=1.0,
        tree_method="hist", early_stopping_rounds=50, eval_metric="rmse",
        random_state=seed, n_jobs=8)
    model.fit(Mtr, ytr, eval_set=[(Mva, yva)], verbose=False)
    if verbose:
        print(f"    B1 trees used: {model.best_iteration + 1} ({time.time()-t0:.0f}s)")
    del Mtr, ytr, Mva, yva
    K = Zte.shape[1]
    V = model.predict(Zte.reshape(-1, Zte.shape[-1])).reshape(len(te), K)
    V = V * np.asarray(d["mask"][te])
    bm = float(np.asarray(d["R"][tr]).mean())
    m = eval_per_frame(V, d, te, bm); m["train_mean_R"] = bm
    m["n_trees"] = int(model.best_iteration + 1)
    # the sklearn wrapper labels columns f0..fN; map them back to real names
    gain = model.get_booster().get_score(importance_type="gain")
    named = {names[int(k[1:])] if k[1:].isdigit() and int(k[1:]) < len(names) else k: v
             for k, v in gain.items()}
    m["importance"] = dict(sorted(named.items(), key=lambda kv: -kv[1])[:12])
    return (m, model) if return_model else m


def leak_probe(d, tr, va, te, seed=0):
    """How much of the value signal is available from the FRAME INDEX alone?
    Under fixed-K resampling the index partially encodes time-to-terminal; any
    sequence model can read it. This bounds that channel."""
    import xgboost as xgb
    K = d["mask"].shape[1]
    mk = lambda rows: (np.repeat(np.arange(K)[None, :], len(rows), 0)[
                           np.asarray(d["mask"][rows]).astype(bool)][:, None].astype(np.float32),
                       np.repeat(np.asarray(d["R"][rows])[:, None], K, 1)[
                           np.asarray(d["mask"][rows]).astype(bool)])
    Mtr, ytr = mk(tr); Mva, yva = mk(va)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                         tree_method="hist", random_state=seed, n_jobs=8,
                         early_stopping_rounds=20, eval_metric="rmse")
    m.fit(Mtr, ytr, eval_set=[(Mva, yva)], verbose=False)
    V = m.predict(np.arange(K, dtype=np.float32)[:, None])
    Vte = np.repeat(V[None, :], len(te), 0) * np.asarray(d["mask"][te])
    bm = float(np.asarray(d["R"][tr]).mean())
    r = eval_per_frame(Vte, d, te, bm)
    print(f"[leak-probe] frame-index-only value model: frame-RMSE {r['frame_rmse']:.4f} "
          f"vs constant-PPP {r['baseline_rmse']:.4f}  (V(0)={V[0]:.3f}, V(K-1)={V[-1]:.3f})")
    return r, V


# ──────────────────────────────────────────────────────────────────────────
AGG_KEYS = ["term_rmse", "frame_rmse", "skill", "term_corr", "first_frame_epv",
            "cal_mae", "cal_slope"]


def agg(res):
    return {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res])))
            for k in AGG_KEYS}


def main(a):
    seeds = tuple(int(s) for s in a.seeds.split(","))
    d = load_epv(in_ram=not a.memmap)
    games = np.asarray(d["games"])
    print(f"Dataset: {len(d['idx'])} possessions / {len(np.unique(games[d['idx']]))} games")
    os.makedirs(OUT_DIR, exist_ok=True)

    if a.leak_probe:
        tr, va, te = game_splits(games, d["idx"], seed=seeds[0])
        leak_probe(d, tr, va, te, seed=seeds[0])
        return

    table, raw = {}, {}

    # ── B0 / B1 ───────────────────────────────────────────────────────────
    for name, fn in (("B0 constant PPP", lambda tr, va, te, s: run_b0(d, tr, te)),
                     ("B1 XGBoost value", lambda tr, va, te, s: run_b1(d, tr, va, te, seed=s))):
        res = []
        for s in seeds:
            tr, va, te = game_splits(games, d["idx"], seed=s)
            print(f"\n[{name}] seed {s}: train {len(tr)} / val {len(va)} / test {len(te)}")
            m = fn(tr, va, te, s); m["seed"] = s; res.append(m)
            print(f"  term-RMSE {m['term_rmse']:.4f} | frame-RMSE {m['frame_rmse']:.4f} "
                  f"| skill {m['skill']:+.4f} | V(0) {m['first_frame_epv']:.3f}")
        table[name] = agg(res); raw[name] = res

    # ── B2 / B3 / full GNN ────────────────────────────────────────────────
    import gnn_epv_causal as G
    for name, mode in (("B2 GNN spatial-only", "spatial"),
                       ("B3 GNN temporal-only", "temporal"),
                       ("GNN-EPV causal (full)", "full")):
        print(f"\n[{name}]")
        res, _, _ = G.run_seeds(d, seeds=seeds, mode=mode, lam=a.lam, epochs=a.epochs,
                                verbose=a.verbose)
        table[name] = agg(res); raw[name] = res

    # ── report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"EPV MODEL COMPARISON — game-disjoint 70/15/15, {len(seeds)} seeds, mean ± std")
    print("=" * 96)
    hdr = f"{'model':<24}{'term RMSE':>18}{'frame RMSE':>18}{'skill':>16}{'V(0)':>14}{'cal MAE':>12}"
    print(hdr); print("-" * 96)
    for k, v in table.items():
        print(f"{k:<24}{v['term_rmse'][0]:>10.4f} ±{v['term_rmse'][1]:<6.4f}"
              f"{v['frame_rmse'][0]:>10.4f} ±{v['frame_rmse'][1]:<6.4f}"
              f"{v['skill'][0]:>9.4f} ±{v['skill'][1]:<5.4f}"
              f"{v['first_frame_epv'][0]:>8.3f} ±{v['first_frame_epv'][1]:<4.3f}"
              f"{v['cal_mae'][0]:>11.4f}")
    print("=" * 96)

    with open(os.path.join(OUT_DIR, "epv_baselines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + [f"{k}_{s}" for k in AGG_KEYS for s in ("mean", "std")])
        for k, v in table.items():
            w.writerow([k] + [f"{v[m][i]:.6f}" for m in AGG_KEYS for i in (0, 1)])
    json.dump({"seeds": list(seeds), "lam": a.lam, "epochs": a.epochs, "table": table,
               "per_seed": {k: [{kk: vv for kk, vv in r.items()
                                 if kk not in ("calibration", "by_term", "pos_rmse", "importance")}
                                for r in v] for k, v in raw.items()},
               "calibration_seed0": {k: v[0]["calibration"] for k, v in raw.items()},
               "by_term_seed0": {k: {str(t): x for t, x in v[0]["by_term"].items()}
                                 for k, v in raw.items()},
               "pos_rmse_seed0": {k: v[0]["pos_rmse"] for k, v in raw.items()},
               "b1_importance": raw["B1 XGBoost value"][0].get("importance", {})},
              open(os.path.join(OUT_DIR, "epv_baselines.json"), "w"), indent=2)
    print(f"wrote {OUT_DIR}/epv_baselines.csv and .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--memmap", action="store_true", help="keep X on disk (lower RAM)")
    ap.add_argument("--leak-probe", action="store_true", dest="leak_probe")
    ap.add_argument("--verbose", action="store_true")
    main(ap.parse_args())
