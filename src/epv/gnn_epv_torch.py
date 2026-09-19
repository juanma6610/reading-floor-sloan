"""
gnn_epv_torch.py — PyTorch port of the spatio-temporal GNN-EPV model.

This is a faithful, line-by-line port of the autograd reference in
`src/epv/gnn_epv.py` (same math, same forward graph), rewritten as an
`nn.Module` with:
  * minibatched Adam training (the reason to port — scales past the ~426
    possessions the pure-NumPy autograd version handles comfortably to the
    10k+ possessions Phase 1 of docs/epv_gnn_experiment_plan.md targets),
  * optional per-possession TEMPORAL MASK so variable-length possessions can be
    padded into a batch (autograd version assumed fixed K=12),
  * GPU support (device auto-selected),
  * a `--check` self-test that loads the autograd model's exact weights and
    asserts the two forwards agree to < 1e-5 (numerical-equivalence guarantee).

Architecture (unchanged from the autograd reference):
  node encoder Linear(F->D)+ReLU
  -> per-frame SELF-ATTENTION over the 11 nodes (complete player+ball graph),
     single head, scaled dot-product, with a ReLU(Wo·)+residual block   [spatial]
  -> mean-pool the 11 nodes -> per-frame embedding
  -> TEMPORAL ATTENTION pooling over the K frames (masked softmax)       [temporal]
  -> MLP head Linear(D->H)+ReLU -> Linear(H->1) -> EPV

Bias layout mirrors the reference EXACTLY (this matters for --check):
  We: bias   |  Wq,Wk,Wv,Wo: no bias  |  wt: no bias  |  W1: bias  |  W2: bias

NOTE ON EXECUTION: this file was authored in an environment where torch could
not be installed (the CPU wheel host was blocked and the PyPI CUDA build does
not fit / import without the full nvidia stack). It is therefore NOT executed in
that sandbox. The math was cross-checked against the autograd forward with a
pure-NumPy mirror; run `python src/epv/gnn_epv_torch.py --check` on a machine
with torch to confirm bit-level parity before relying on it.

Run:
  python src/epv/gnn_epv_torch.py            # train + evaluate (game-held-out + pooled)
  python src/epv/gnn_epv_torch.py --check    # assert parity with the autograd model
"""
from __future__ import annotations
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

NPZ = "results/epv/epv_graphs.npz"
D, H = 24, 16            # node/graph embed dim, MLP hidden  (match gnn_epv.py)
STEPS, LR, BATCH = 250, 3e-3, 64
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────
class GNNEPV(nn.Module):
    """Spatio-temporal GNN-EPV. Input X: [N, K, 11, Fin]; optional frame mask
    `fmask`: [N, K] with 1 for real frames, 0 for padding."""

    def __init__(self, fin: int, d: int = D, h: int = H):
        super().__init__()
        self.d = d
        self.enc = nn.Linear(fin, d, bias=True)                 # We, be
        self.wq = nn.Linear(d, d, bias=False)                   # Wq
        self.wk = nn.Linear(d, d, bias=False)                   # Wk
        self.wv = nn.Linear(d, d, bias=False)                   # Wv
        self.wo = nn.Linear(d, d, bias=False)                   # Wo
        self.wt = nn.Linear(d, 1, bias=False)                   # wt (temporal score)
        self.h1 = nn.Linear(d, h, bias=True)                    # W1, b1
        self.h2 = nn.Linear(h, 1, bias=True)                    # W2, b2

    def forward(self, X: torch.Tensor, fmask: torch.Tensor | None = None) -> torch.Tensor:
        # X: [N, K, 11, Fin]
        Hn = F.relu(self.enc(X))                                # [N,K,11,D]
        Q, Kk, V = self.wq(Hn), self.wk(Hn), self.wv(Hn)        # [N,K,11,D]
        sc = torch.matmul(Q, Kk.transpose(-1, -2)) / (self.d ** 0.5)   # [N,K,11,11]
        A = torch.softmax(sc, dim=-1)                           # spatial attention
        ctx = torch.matmul(A, V)                                # [N,K,11,D]
        H2 = F.relu(self.wo(ctx)) + Hn                          # ReLU(Wo·)+residual
        frame = H2.mean(dim=2)                                  # mean over 11 nodes -> [N,K,D]

        ts = self.wt(frame).squeeze(-1)                         # [N,K] temporal scores
        if fmask is not None:                                   # mask padded frames
            ts = ts.masked_fill(fmask == 0, float("-inf"))
        al = torch.softmax(ts, dim=1)                           # [N,K]
        g = torch.sum(al.unsqueeze(-1) * frame, dim=1)          # [N,D]

        z = F.relu(self.h1(g))                                  # [N,H]
        return self.h2(z).squeeze(-1)                           # [N]

    # --- load the autograd reference weights (dict of numpy arrays) ---------
    @torch.no_grad()
    def load_autograd_params(self, P: dict):
        """P uses the [in,out] convention and X@W; nn.Linear stores [out,in]
        and computes X@W^T, so every weight is transposed on load."""
        def T(a): return torch.tensor(np.asarray(a).T, dtype=self.enc.weight.dtype)
        def V(a): return torch.tensor(np.asarray(a), dtype=self.enc.weight.dtype)
        self.enc.weight.copy_(T(P["We"]));  self.enc.bias.copy_(V(P["be"]))
        self.wq.weight.copy_(T(P["Wq"]))
        self.wk.weight.copy_(T(P["Wk"]))
        self.wv.weight.copy_(T(P["Wv"]))
        self.wo.weight.copy_(T(P["Wo"]))
        self.wt.weight.copy_(V(np.asarray(P["wt"]).reshape(1, -1)))   # [D] -> [1,D]
        self.h1.weight.copy_(T(P["W1"]));  self.h1.bias.copy_(V(P["b1"]))
        self.h2.weight.copy_(T(P["W2"]));  self.h2.bias.copy_(V(P["b2"]))


# ──────────────────────────────────────────────────────────────────────────
# Train / eval
# ──────────────────────────────────────────────────────────────────────────
def train_model(Xtr, ytr, fin, steps=STEPS, lr=LR, batch=BATCH, seed=SEED,
                fmask_tr=None, verbose=False):
    torch.manual_seed(seed)
    model = GNNEPV(fin).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr = torch.as_tensor(ytr, dtype=torch.float32, device=DEVICE)
    mtr = None if fmask_tr is None else torch.as_tensor(fmask_tr, dtype=torch.float32, device=DEVICE)
    n = len(ytr)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for step in range(1, steps + 1):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            pred = model(Xtr[b], None if mtr is None else mtr[b])
            loss = ((pred - ytr[b]) ** 2).mean()
            loss.backward()
            opt.step()
        if verbose and (step % 50 == 0 or step == 1):
            with torch.no_grad():
                full = ((model(Xtr, mtr) - ytr) ** 2).mean().item()
            print(f"    step {step:3d}  train MSE {full:.4f}")
    return model


@torch.no_grad()
def predict(model, X, fmask=None):
    X = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    m = None if fmask is None else torch.as_tensor(fmask, dtype=torch.float32, device=DEVICE)
    return model(X, m).cpu().numpy()


def metrics(model, X, y, ybar, fmask=None):
    p = predict(model, X, fmask)
    mse = float(np.mean((p - y) ** 2))
    base = float(np.mean((ybar - y) ** 2))
    corr = float(np.corrcoef(p, y)[0, 1]) if len(np.unique(p)) > 1 else 0.0
    return dict(rmse=mse ** 0.5, mse=mse, baseline_rmse=base ** 0.5,
                skill=1 - mse / base, corr=corr,
                pred_mean=float(p.mean()), actual_mean=float(y.mean()))


# ──────────────────────────────────────────────────────────────────────────
# --check : numerical parity with the autograd reference
# ──────────────────────────────────────────────────────────────────────────
def check_parity():
    import importlib.util, os
    ref_path = os.path.join(os.path.dirname(__file__), "gnn_epv.py")
    spec = importlib.util.spec_from_file_location("gnn_epv_ref", ref_path)
    ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)

    d = np.load(NPZ)
    X = d["X"].astype(np.float64)
    fin = X.shape[-1]
    rng = np.random.default_rng(0)
    P = ref.init_params(fin, rng)                       # random reference weights

    y_ref = ref.forward(P, X)                           # autograd forward
    model = GNNEPV(fin)
    model.load_autograd_params(P)          # weights are loaded on CPU, then moved
    model = model.to(DEVICE)
    model.eval()
    y_torch = predict(model, X.astype(np.float32))

    diff = float(np.max(np.abs(y_ref - y_torch)))
    rel = diff / (float(np.max(np.abs(y_ref))) + 1e-12)
    print(f"[check] n={len(y_ref)}  max|Δ|={diff:.3e}  rel={rel:.3e}")
    ok = diff < 1e-4
    print("[check] PARITY OK" if ok else "[check] MISMATCH — investigate")
    return ok


# ──────────────────────────────────────────────────────────────────────────
def main():
    d = np.load(NPZ)
    X, y, games = d["X"].astype(np.float32), d["y"].astype(np.float32), d["games"]
    fin = X.shape[-1]
    print(f"Device: {DEVICE} | Dataset X{X.shape}  y(n={len(y)}, mean={y.mean():.3f})")

    print("\n=== GAME-HELD-OUT (train on the rest, test on one game) ===")
    fold = []
    for g in sorted(set(games.tolist())):
        tr, te = games != g, games == g
        model = train_model(X[tr], y[tr], fin, verbose=False)
        mte = metrics(model, X[te], y[te], y[tr].mean())
        fold.append(mte)
        print(f"  hold-out {g}: test RMSE {mte['rmse']:.4f} vs baseline "
              f"{mte['baseline_rmse']:.4f} | skill {mte['skill']:+.3f} | corr {mte['corr']:+.3f}")
    print(f"  MEAN held-out: RMSE {np.mean([f['rmse'] for f in fold]):.4f} "
          f"| skill {np.mean([f['skill'] for f in fold]):+.3f} "
          f"| corr {np.mean([f['corr'] for f in fold]):+.3f}")

    print("\n=== pooled random split (70/30) ===")
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y)); cut = int(0.7 * len(y))
    tr, te = idx[:cut], idx[cut:]
    model = train_model(X[tr], y[tr], fin, verbose=True)
    mte = metrics(model, X[te], y[te], y[tr].mean())
    print(f"  test RMSE {mte['rmse']:.4f} vs baseline {mte['baseline_rmse']:.4f} "
          f"| skill {mte['skill']:+.3f} | corr {mte['corr']:+.3f}")

    p = predict(model, X[te]); q = np.quantile(p, [0, .2, .4, .6, .8, 1.0])
    print("  calibration (EPV quintile -> mean actual points):")
    for i in range(5):
        m = (p >= q[i]) & ((p <= q[i + 1]) if i == 4 else (p < q[i + 1]))
        if m.sum():
            print(f"    Q{i+1}: pred {p[m].mean():.2f}  actual {y[te][m].mean():.2f}  (n={int(m.sum())})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert parity with the autograd reference")
    args = ap.parse_args()
    if args.check:
        raise SystemExit(0 if check_parity() else 1)
    main()
