"""
gnn_epv.py — a spatio-temporal GNN that learns EPV from player+ball tracking.

Architecture (per possession = sequence of K graphs, 11 nodes each):
  node encoder (F->d)
  -> per-frame SELF-ATTENTION over the 11 nodes (graph attention on the
     complete player+ball graph) with residual        [spatial]
  -> mean-pool nodes -> per-frame embedding
  -> TEMPORAL ATTENTION pooling over the K frames      [temporal]
  -> MLP head -> EPV (expected points of the possession)

Trained by regressing the shot's realized points (0/2/3): because the label is
the possession outcome, the network learns E[points | state] = EPV directly
(value-regression route to Route 3; full EPV adds turnover/FT terminals once PBP
is joined). Written with `autograd` (pure-NumPy autodiff) so it trains without a
GPU/torch. A PyTorch port is a mechanical rewrite (same math) — see
docs/epv_gnn.md.

Data: results/epv/epv_graphs.npz  (from tracking_graphs.py)
Run:  python src/epv/gnn_epv.py
"""
from __future__ import annotations
import numpy as np
import autograd.numpy as anp
from autograd import grad

NPZ = "results/epv/epv_graphs.npz"
D, H = 24, 16          # node/graph embed dim, head hidden
STEPS, LR = 250, 3e-3
SEED = 0


def init_params(F, rng):
    def r(*s): return rng.standard_normal(s) * (1.0 / np.sqrt(s[0]))
    return {
        "We": r(F, D), "be": np.zeros(D),
        "Wq": r(D, D), "Wk": r(D, D), "Wv": r(D, D), "Wo": r(D, D),
        "wt": r(D),
        "W1": r(D, H), "b1": np.zeros(H), "W2": r(H, 1) * 0.1, "b2": np.zeros(1),
    }


def forward(P, X):                                   # X: [N,K,11,F]
    Hn = anp.maximum(0, anp.matmul(X, P["We"]) + P["be"])          # [N,K,11,D]
    Q, Kk, V = anp.matmul(Hn, P["Wq"]), anp.matmul(Hn, P["Wk"]), anp.matmul(Hn, P["Wv"])
    sc = anp.matmul(Q, anp.swapaxes(Kk, -1, -2)) / anp.sqrt(D)     # [N,K,11,11]
    sc = sc - anp.max(sc, axis=-1, keepdims=True)
    A = anp.exp(sc); A = A / anp.sum(A, axis=-1, keepdims=True)
    ctx = anp.matmul(A, V)                                         # [N,K,11,D]
    H2 = anp.maximum(0, anp.matmul(ctx, P["Wo"])) + Hn            # residual
    frame = anp.mean(H2, axis=2)                                   # [N,K,D]
    ts = anp.matmul(frame, P["wt"])                                # [N,K]
    ts = ts - anp.max(ts, axis=1, keepdims=True)
    al = anp.exp(ts); al = al / anp.sum(al, axis=1, keepdims=True)
    g = anp.sum(anp.expand_dims(al, 2) * frame, axis=1)           # [N,D]
    h = anp.maximum(0, anp.matmul(g, P["W1"]) + P["b1"])
    return anp.matmul(h, P["W2"])[:, 0] + P["b2"][0]              # [N]


def loss_fn(P, X, y):
    return anp.mean((forward(P, X) - y) ** 2)


def adam_train(P, X, y, steps=STEPS, lr=LR, verbose=True):
    g = grad(loss_fn)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        gr = g(P, X, y)
        for k in P:
            m[k] = b1 * m[k] + (1 - b1) * gr[k]
            v[k] = b2 * v[k] + (1 - b2) * gr[k] ** 2
            mh = m[k] / (1 - b1 ** t); vh = v[k] / (1 - b2 ** t)
            P[k] = P[k] - lr * mh / (np.sqrt(vh) + eps)
        if verbose and (t % 50 == 0 or t == 1):
            print(f"    step {t:3d}  train MSE {loss_fn(P, X, y):.4f}")
    return P


def metrics(P, X, y, ybar):
    p = forward(P, X)
    mse = np.mean((p - y) ** 2)
    base = np.mean((ybar - y) ** 2)                 # predict train mean
    corr = np.corrcoef(p, y)[0, 1]
    return dict(rmse=float(np.sqrt(mse)), mse=float(mse),
                baseline_rmse=float(np.sqrt(base)),
                skill=float(1 - mse / base), corr=float(corr),
                pred_mean=float(p.mean()), actual_mean=float(y.mean()))


def epv_curve(P, X_one):
    """EPV using the first t frames (t=1..K) -> the in-possession value curve."""
    K = X_one.shape[0]
    out = []
    for t in range(1, K + 1):
        xt = X_one[:t][None]                        # [1,t,11,F]
        out.append(float(forward(P, xt)[0]))
    return out


def main():
    d = np.load(NPZ)
    X, y, games = d["X"].astype(np.float64), d["y"].astype(np.float64), d["games"]
    F = X.shape[-1]
    print(f"Dataset: X{X.shape}  y(n={len(y)}, mean={y.mean():.3f})  games={sorted(set(games.tolist()))}")

    print("\n=== GAME-HELD-OUT (train on 2 games, test on the 3rd) ===")
    fold = []
    for g in sorted(set(games.tolist())):
        tr, te = games != g, games == g
        rng = np.random.default_rng(SEED)
        P = init_params(F, rng)
        P = adam_train(P, X[tr], y[tr], verbose=False)
        mtr = metrics(P, X[tr], y[tr], y[tr].mean())
        mte = metrics(P, X[te], y[te], y[tr].mean())
        fold.append(mte)
        print(f"  hold-out {g}: test RMSE {mte['rmse']:.4f} vs baseline {mte['baseline_rmse']:.4f} "
              f"| skill {mte['skill']:+.3f} | corr {mte['corr']:+.3f} "
              f"| pred/actual mean {mte['pred_mean']:.2f}/{mte['actual_mean']:.2f}")
    print(f"  MEAN held-out: RMSE {np.mean([f['rmse'] for f in fold]):.4f} "
          f"| skill {np.mean([f['skill'] for f in fold]):+.3f} "
          f"| corr {np.mean([f['corr'] for f in fold]):+.3f}")

    print("\n=== pooled random split (70/30) for a single headline model ===")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    P = init_params(F, rng)
    P = adam_train(P, X[tr], y[tr], verbose=True)
    mte = metrics(P, X[te], y[te], y[tr].mean())
    print(f"  test RMSE {mte['rmse']:.4f} vs baseline {mte['baseline_rmse']:.4f} "
          f"| skill {mte['skill']:+.3f} | corr {mte['corr']:+.3f}")

    # calibration: bucket predicted EPV, show mean actual points
    p = forward(P, X[te])
    q = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    print("  calibration (EPV quintile -> mean actual points):")
    for i in range(5):
        m = (p >= q[i]) & (p <= q[i + 1] if i == 4 else p < q[i + 1])
        if m.sum(): print(f"    Q{i+1}: pred {p[m].mean():.2f}  actual {y[te][m].mean():.2f}  (n={int(m.sum())})")

    # EPV curve within one possession -> action value = delta EPV
    ex = te[np.argmax(y[te])]
    curve = epv_curve(P, X[ex])
    print(f"\n  EPV curve over one possession (actual pts={y[ex]:.0f}): "
          + " ".join(f"{c:.2f}" for c in curve))
    print(f"  (frame-to-frame EPV changes are the on-ball action values; "
          f"max jump {max(np.diff(curve)):+.2f})")


if __name__ == "__main__":
    main()
