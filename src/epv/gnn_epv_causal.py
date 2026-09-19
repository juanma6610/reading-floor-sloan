"""
gnn_epv_causal.py — TRUE per-moment EPV: causal per-frame value head + TD(lambda).

Implements §4-6 of docs/epv_m4_true_epv_spec.md. Reuses the SAME spatial block as
the shot model (node encoder + per-frame self-attention over the 11-node graph +
node mean-pool), but replaces the bidirectional temporal pooling with:
  * a CAUSAL temporal encoder (GRU) so V(t) depends only on frames <= t, and
  * a PER-FRAME value head, giving V(t) for every frame t.
Trained with TD(lambda) targets (terminal frame anchored to the possession value R,
earlier frames bootstrapped through a frozen TARGET network). The value of a
pass/drive is then dEPV(t) = V(t+1) - V(t), attributed to the ball-handler at t.

Data comes through `epv_data.load_epv()` (memmapped float16 cache built from
`results/epv/shards/`, label hygiene applied). Splits are game-disjoint 70/15/15,
model selection is on the VAL split only, and every reported number is
mean +- std over >= 3 seeds.

Ablation modes (baselines B2/B3 of docs/epv_gnn_experiment_plan.md §6) share this
trainer so only the architecture differs:
  full     : node attention + causal GRU          (the model)
  spatial  : node attention, NO GRU               (memoryless: V(t)=f(frame_t))
  temporal : NO node attention, causal GRU        (mean-pooled nodes)

torch is imported LAZILY so the TD(lambda) math (the subtle part) is unit-testable
without torch — run `--selftest`.

Run:
  python src/epv/gnn_epv_causal.py --selftest      # TD(lambda) return checks (no torch)
  python src/epv/gnn_epv_causal.py --causal-test   # assert V(t) ignores frames > t
  python src/epv/gnn_epv_causal.py                 # train + eval, 3 seeds
  python src/epv/gnn_epv_causal.py --sweep         # lambda / hidden / lr / refresh sweep
"""
from __future__ import annotations
import os, sys, glob, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

NPZ = "results/epv/epv_possessions.npz"
SHARD_DIR = "results/epv/shards"
OUT_DIR = "results/epv"
D, Dh, GRU_H = 24, 16, 24          # spatial embed, head hidden, GRU hidden
LAM, LR, EPOCHS, BATCH, REFRESH = 0.8, 3e-3, 40, 512, 2
SEED = 0
SEEDS = (0, 1, 2)


# ──────────────────────────────────────────────────────────────────────────
# TD(lambda) returns — pure numpy reference (also the exact logic used in torch)
# ──────────────────────────────────────────────────────────────────────────
def lambda_returns_np(Vt, R, klen, lam):
    """Forward-view TD(lambda) with zero intra-possession reward and gamma=1.
    Backward recursion:  G_T = R (terminal),  G_t = (1-lam)*Vt[t+1] + lam*G_{t+1}.
    lam=1 -> Monte Carlo (G_t = R for all t);  lam=0 -> one-step (G_t = Vt[t+1]).
    Vt:[N,K] target-net values, R:[N], klen:[N]. Returns G:[N,K] (pad frames = 0)."""
    N, K = Vt.shape
    Vt = Vt.astype(np.float64).copy()
    term = klen - 1
    # bootstrapping onto the terminal frame must use the TRUE terminal value R,
    # not the target net's estimate there (the episode ends with the actual reward)
    Vt[np.arange(N), term] = R
    G = np.zeros((N, K), dtype=np.float64)
    G[np.arange(N), term] = R                                   # terminal target
    for t in range(K - 2, -1, -1):
        rec = (1.0 - lam) * Vt[:, t + 1] + lam * G[:, t + 1]    # bootstrap
        active = t < term                                       # only before terminal
        G[:, t] = np.where(active, rec, G[:, t])                # keep terminal=R, pad=0
    return G


def _forward_view_bruteforce(Vt, R, klen, lam):
    """Explicit truncated lambda-return, as an independent check of the recursion:
       G_t = (1-lam) Σ_{n=1}^{T-t-1} lam^{n-1} Vt[t+n] + lam^{T-t-1} R ."""
    N, K = Vt.shape
    G = np.zeros((N, K))
    for i in range(N):
        T = int(klen[i]) - 1
        for t in range(T + 1):
            if t == T:
                G[i, t] = R[i]; continue
            s, w = 0.0, 1.0                       # w = lam^{n-1}
            for n in range(1, T - t):             # n = 1 .. T-t-1  (bootstraps)
                s += (1 - lam) * w * Vt[i, t + n]
                w *= lam
            s += w * R[i]                         # remaining mass -> terminal R
            G[i, t] = s
    return G


def selftest():
    rng = np.random.default_rng(0)
    N, K = 6, 40
    klen = rng.integers(5, K + 1, size=N)
    Vt = rng.normal(1.0, 0.5, size=(N, K))
    R = rng.choice([0, 2, 3], size=N).astype(float)

    def valid_mask():
        m = np.zeros((N, K), bool)
        for i in range(N): m[i, :klen[i]] = True
        return m
    m = valid_mask()

    # lam=1 -> every valid frame equals R
    g1 = lambda_returns_np(Vt, R, klen, 1.0)
    ok1 = all(np.allclose(g1[i, :klen[i]], R[i]) for i in range(N))
    # lam=0 -> one-step TD: G_t = Vt[t+1] for t<T-1; G_{T-1}=R (terminal bootstrap); G_T=R
    g0 = lambda_returns_np(Vt, R, klen, 0.0)
    def ok0_row(i):
        T = klen[i] - 1
        a = np.allclose(g0[i, :T-1], Vt[i, 1:T]) if T >= 1 else True   # frames 0..T-2
        return a and np.isclose(g0[i, T-1], R[i]) and np.isclose(g0[i, T], R[i])
    ok0 = all(ok0_row(i) for i in range(N))
    # mid lam -> matches brute-force forward view
    gm = lambda_returns_np(Vt, R, klen, 0.8)
    bf = _forward_view_bruteforce(Vt, R, klen, 0.8)
    okm = all(np.allclose(gm[i, :klen[i]], bf[i, :klen[i]]) for i in range(N))
    # pad frames stay exactly zero for every lambda
    okp = all(np.allclose(lambda_returns_np(Vt, R, klen, l)[~m], 0.0)
              for l in (0.0, 0.5, 0.8, 1.0))

    print(f"[selftest] lam=1 -> R everywhere : {ok1}")
    print(f"[selftest] lam=0 -> one-step TD  : {ok0}")
    print(f"[selftest] lam=0.8 == brute force: {okm}  (max|Δ|={np.max(np.abs((gm-bf)*m)):.2e})")
    print(f"[selftest] pad frames stay zero  : {okp}")
    ok = ok1 and ok0 and okm and okp
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


# ──────────────────────────────────────────────────────────────────────────
# torch model + training (lazy import)
# ──────────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


if HAVE_TORCH:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    class SpatialBlock(nn.Module):
        """Per-frame graph encoder — identical math to gnn_epv_torch.GNNEPV's
        spatial half: node encoder + single-head self-attention over the 11 nodes
        (+ ReLU(Wo·) residual) + mean-pool over nodes -> per-frame embedding.
        `attend=False` ablates the graph module (node encoder + mean-pool only),
        which is baseline B3 (temporal-only)."""
        def __init__(self, fin, d=D, attend=True):
            super().__init__()
            self.d, self.attend = d, attend
            self.enc = nn.Linear(fin, d, bias=True)
            if attend:
                self.wq = nn.Linear(d, d, bias=False)
                self.wk = nn.Linear(d, d, bias=False)
                self.wv = nn.Linear(d, d, bias=False)
                self.wo = nn.Linear(d, d, bias=False)

        def forward(self, X):                        # X:[N,K,11,F] -> [N,K,D]
            Hn = F.relu(self.enc(X))
            if not self.attend:
                return Hn.mean(dim=2)
            Q, Kk, V = self.wq(Hn), self.wk(Hn), self.wv(Hn)
            sc = torch.matmul(Q, Kk.transpose(-1, -2)) / (self.d ** 0.5)
            A = torch.softmax(sc, dim=-1)
            H2 = F.relu(self.wo(torch.matmul(A, V))) + Hn
            return H2.mean(dim=2)

    class GNNEPVCausal(nn.Module):
        """Spatial block -> causal GRU over frames -> per-frame value head.

        mode: 'full' (attention + GRU) | 'spatial' (attention, no GRU -> V(t)
        depends on frame t alone) | 'temporal' (no attention, GRU).
        `v_init` biases the value head so an untrained net predicts ~league PPP;
        this matters for TD bootstrapping, where the target net's initial output
        is what every non-terminal target is built from."""
        def __init__(self, fin, d=D, gru_h=GRU_H, head_h=Dh, mode="full", v_init=0.0):
            super().__init__()
            self.mode = mode
            self.spatial = SpatialBlock(fin, d, attend=(mode != "temporal"))
            self.use_gru = mode != "spatial"
            if self.use_gru:
                self.gru = nn.GRU(d, gru_h, batch_first=True)      # causal by construction
            hin = gru_h if self.use_gru else d
            self.head = nn.Sequential(nn.Linear(hin, head_h), nn.ReLU(),
                                      nn.Linear(head_h, 1))
            with torch.no_grad():
                self.head[-1].bias.fill_(float(v_init))

        def forward(self, X, mask, lengths=None):    # X:[N,K,11,F], mask:[N,K] -> V:[N,K]
            frame = self.spatial(X)                  # [N,K,D]
            if not self.use_gru:
                return self.head(frame).squeeze(-1) * mask
            K = frame.shape[1]
            # pack_padded_sequence needs lengths on the CPU. Deriving them from
            # `mask` costs a GPU->CPU sync on EVERY forward (and there are two per
            # step: online net + target net); the caller already has klen on the
            # host, so let it hand them in.
            if lengths is None:
                lengths = mask.sum(1).clamp(min=1).long().cpu()
            packed = pack_padded_sequence(frame, lengths, batch_first=True, enforce_sorted=False)
            h, _ = self.gru(packed)
            h, _ = pad_packed_sequence(h, batch_first=True, total_length=K)
            return self.head(h).squeeze(-1) * mask   # zero the pad frames

    def lambda_returns_torch(Vt, R, klen, lam):
        """torch port of lambda_returns_np (same backward recursion)."""
        N, K = Vt.shape
        Vt = Vt.clone()
        term = (klen - 1).long()
        ar = torch.arange(N, device=Vt.device)
        Vt[ar, term] = R                       # bootstrap terminal frame on true R
        G = torch.zeros_like(Vt)
        G[ar, term] = R
        for t in range(K - 2, -1, -1):
            rec = (1.0 - lam) * Vt[:, t + 1] + lam * G[:, t + 1]
            active = (t < term).float()
            G[:, t] = active * rec + (1 - active) * G[:, t]
            G[t >= klen, t] = 0.0                     # keep pad frames zero
        return G

    # ── batch plumbing ────────────────────────────────────────────────────
    def _batch(d, b, dev):
        """Fetch one minibatch (rows `b` of the full arrays) onto `dev`.
        X is stored float16 to fit in RAM; cast to float32 on device.
        `lengths` stays on the CPU — that is where pack_padded_sequence wants it."""
        X = torch.from_numpy(np.asarray(d["X"][b])).to(dev, non_blocking=True).float()
        mask = torch.from_numpy(np.asarray(d["mask"][b])).to(dev).float()
        R = torch.from_numpy(np.asarray(d["R"][b])).to(dev).float()
        kl = np.asarray(d["klen"][b]).astype(np.int64)
        lengths = torch.from_numpy(kl.clip(1, None))
        return X, mask, R, torch.from_numpy(kl).to(dev), lengths

    def _iter(idx, batch, rng=None):
        order = rng.permutation(len(idx)) if rng is not None else np.arange(len(idx))
        for i in range(0, len(order), batch):
            yield np.sort(idx[order[i:i + batch]])

    @torch.no_grad()
    def predict_values(model, d, idx, batch=1024, dev=None):
        """V:[len(idx), K] for the given rows, in batches (the full test split
        does not fit on a 4 GB GPU in one forward). Rows come back in ASCENDING
        row order — `_iter` sorts within a batch for locality, so returning the
        caller's arbitrary order would silently misalign; sort up front instead."""
        dev = dev or DEVICE
        idx = np.sort(np.asarray(idx))
        model.eval()
        out = np.zeros((len(idx), d["mask"].shape[1]), dtype=np.float32)
        pos = 0
        for b in _iter(idx, batch):
            X, mask, R, klen, lg = _batch(d, b, dev)
            out[pos:pos + len(b)] = model(X, mask, lg).cpu().numpy()
            pos += len(b)
        return out

    # ── training ──────────────────────────────────────────────────────────
    def train(d, tr_idx, va_idx=None, fin=None, lam=LAM, epochs=EPOCHS, lr=LR,
              batch=BATCH, refresh=REFRESH, seed=SEED, mode="full", gru_h=GRU_H,
              dev=None, verbose=True, patience=8):
        """TD(lambda) training with a frozen target network. Model selection is on
        the VAL split's terminal-frame RMSE; the returned model is the best epoch."""
        dev = dev or DEVICE
        fin = fin or d["X"].shape[-1]
        torch.manual_seed(seed); np.random.seed(seed)
        v0 = float(np.asarray(d["R"][tr_idx]).mean())          # ~league PPP
        mk = lambda: GNNEPVCausal(fin, mode=mode, gru_h=gru_h, v_init=v0).to(dev)
        model, target = mk(), mk()
        target.load_state_dict(model.state_dict()); target.eval()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)

        best = (np.inf, None, 0)
        hist = []
        for ep in range(1, epochs + 1):
            model.train()
            tot, nf = 0.0, 0
            for b in _iter(tr_idx, batch, rng):
                X, mask, R, klen, lg = _batch(d, b, dev)
                V = model(X, mask, lg)
                with torch.no_grad():
                    Vt = target(X, mask, lg)
                    tgt = lambda_returns_torch(Vt, R, klen, lam)
                loss = (((V - tgt) ** 2) * mask).sum() / mask.sum()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                nfb = float(mask.sum()); tot += loss.detach().item() * nfb; nf += nfb
            if ep % refresh == 0:
                target.load_state_dict(model.state_dict())
            msg = f"    epoch {ep:3d}  masked-MSE {tot/nf:.4f}"
            if va_idx is not None:
                m = evaluate(model, d, va_idx, dev=dev)
                hist.append((ep, tot / nf, m["term_rmse"]))
                msg += f"  val term-RMSE {m['term_rmse']:.4f}  V0 {m['first_frame_epv']:.3f}"
                if m["term_rmse"] < best[0] - 1e-5:
                    best = (m["term_rmse"], {k: v.detach().clone() for k, v in
                                             model.state_dict().items()}, ep)
                    msg += "  *"
                elif ep - best[2] >= patience:
                    if verbose: print(msg + f"  (early stop, best ep {best[2]})")
                    break
            if verbose and (ep % 5 == 0 or ep == 1 or (va_idx is not None and "*" in msg)):
                print(msg)
        if best[1] is not None:
            model.load_state_dict(best[1])
            if verbose: print(f"    -> restored best epoch {best[2]} "
                              f"(val term-RMSE {best[0]:.4f})")
        model.eval()
        model._hist = hist
        return model

    # ── evaluation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(model, d, idx, dev=None, baseline_mean=None, nbins=10, batch=1024):
        dev = dev or DEVICE
        V = predict_values(model, d, idx, batch=batch, dev=dev)
        idx = np.sort(idx)
        R = np.asarray(d["R"][idx]).astype(np.float64)
        klen = np.asarray(d["klen"][idx]).astype(int)
        term = np.asarray(d["term"][idx])
        m = np.asarray(d["mask"][idx]).astype(bool)
        ar = np.arange(len(R))

        Vterm = V[ar, klen - 1]
        term_rmse = float(np.sqrt(np.mean((Vterm - R) ** 2)))
        bm = R.mean() if baseline_mean is None else float(baseline_mean)
        base = float(np.sqrt(np.mean((bm - R) ** 2)))

        Vt_all = V[m].astype(np.float64)
        R_all = np.repeat(R[:, None], V.shape[1], 1)[m]
        frame_rmse = float(np.sqrt(np.mean((Vt_all - R_all) ** 2)))

        # per-frame value calibration: bin V(t) over valid frames vs realized R
        q = np.unique(np.quantile(Vt_all, np.linspace(0, 1, nbins + 1)))
        bi = np.clip(np.searchsorted(q, Vt_all, side="right") - 1, 0, len(q) - 2)
        cal = []
        for i in range(len(q) - 1):
            s = bi == i
            if s.sum() > 50:
                cal.append((float(Vt_all[s].mean()), float(R_all[s].mean()), int(s.sum())))
        cal_arr = np.array([[a, b] for a, b, _ in cal], dtype=float).reshape(-1, 2)
        w = np.array([c for _, _, c in cal], float)
        cal_mae = (float(np.average(np.abs(cal_arr[:, 0] - cal_arr[:, 1]), weights=w))
                   if len(cal) else float("nan"))
        slope = (float(np.polyfit(cal_arr[:, 0], cal_arr[:, 1], 1)[0])
                 if len(cal) > 2 else float("nan"))

        # value at the terminal frame, split by terminal type (face validity)
        by_term = {int(t): (float(Vterm[term == t].mean()), float(R[term == t].mean()),
                            int((term == t).sum()))
                   for t in sorted(set(term.tolist()))}
        # RMSE by frame position (early frames are genuinely harder)
        pos_rmse = [float(np.sqrt(np.mean((V[:, t][m[:, t]] - R[m[:, t]]) ** 2)))
                    for t in range(V.shape[1]) if m[:, t].sum() > 50]
        corr = float(np.corrcoef(Vterm, R)[0, 1]) if Vterm.std() > 1e-9 else float("nan")
        return dict(term_rmse=term_rmse, baseline_rmse=base,
                    skill=1 - (term_rmse / base) ** 2,
                    frame_rmse=frame_rmse, term_corr=corr,
                    first_frame_epv=float(V[:, 0].mean()),
                    first_frame_sd=float(V[:, 0].std()),
                    cal_mae=cal_mae, cal_slope=slope, calibration=cal,
                    by_term=by_term, pos_rmse=pos_rmse, n=len(R))

    # ── causality guard (spec §9: "unit-test that V(t) is invariant to
    #    zeroing frames > t") ────────────────────────────────────────────────
    @torch.no_grad()
    def causality_test(model, d, idx, dev=None, n=64, tol=1e-5):
        """V(t) must not move when every frame AFTER t is destroyed. We zero the
        future frames (keeping the mask identical, so the only thing that changes
        is future *content*) and compare V[:, :t+1]."""
        dev = dev or DEVICE
        model.eval()
        b = np.sort(idx[:n])
        X, mask, R, klen, lg = _batch(d, b, dev)
        V0 = model(X, mask, lg)
        worst, checks = 0.0, 0
        for t in (0, 1, 5, 10, 20, 30):
            if t >= X.shape[1] - 1: continue
            Xz = X.clone(); Xz[:, t + 1:] = 0.0
            Vz = model(Xz, mask, lg)
            sel = mask[:, :t + 1] > 0
            dmax = float((V0[:, :t + 1] - Vz[:, :t + 1]).abs()[sel].max())
            worst = max(worst, dmax); checks += 1
            print(f"  [causal] zero frames > {t:2d}: max|ΔV(0..t)| = {dmax:.3e}")
        # and a positive control: destroying the PAST must change V(t)
        Xp = X.clone(); Xp[:, :10] = 0.0
        dpast = float((V0[:, 10:] - model(Xp, mask, lg)[:, 10:]).abs().max())
        ok = worst < tol and dpast > tol
        print(f"  [causal] positive control (zero frames < 10 changes V(t>=10)): {dpast:.3e}")
        print(f"[causal-test] {'PASS' if ok else 'FAIL'} "
              f"(worst future-leak {worst:.3e} over {checks} cut points, tol {tol})")
        return ok

    # ── action values ─────────────────────────────────────────────────────
    @torch.no_grad()
    def action_values(model, d, idx=None, dev=None, batch=1024):
        """dEPV(t)=V(t+1)-V(t) over valid frames, tagged with the ball-handler node.
        Returns a structured array: (row, t, bh_t, bh_next, dv)."""
        dev = dev or DEVICE
        idx = np.sort(np.asarray(d["idx"] if idx is None else idx))
        V = predict_values(model, d, idx, batch=batch, dev=dev)
        bh = np.asarray(d["bh"][idx]); klen = np.asarray(d["klen"][idx]).astype(int)
        K = V.shape[1]
        tt = np.arange(K - 1)[None, :]
        valid = tt < (klen[:, None] - 1)
        rows = np.repeat(idx[:, None], K - 1, 1)[valid]
        ts = np.repeat(tt, len(idx), 0)[valid]
        dv = (V[:, 1:] - V[:, :-1])[valid]
        b0 = bh[:, :-1][valid]; b1 = bh[:, 1:][valid]
        return np.rec.fromarrays([rows, ts, b0, b1, dv],
                                 names="row,t,bh,bh_next,dv")


# ──────────────────────────────────────────────────────────────────────────
def load_data(src=None):
    """Back-compat raw loader (whole npz / shard dir into RAM). Prefer
    epv_data.load_epv(), which is memmapped and applies label hygiene."""
    src = src or (NPZ if os.path.exists(NPZ) else SHARD_DIR)
    if os.path.isdir(src):
        keys = ["X", "mask", "bh", "tframe", "klen", "R", "term", "games", "off_team"]
        acc = {k: [] for k in keys}
        for s in sorted(glob.glob(os.path.join(src, "*.npz"))):
            dd = np.load(s)
            for k in keys: acc[k].append(dd[k])
        return {k: np.concatenate(acc[k], 0) for k in keys}
    return dict(np.load(src))


def _fmt(vals, p=4):
    v = np.asarray(vals, float)
    return f"{v.mean():.{p}f} ± {v.std():.{p}f}"


def run_seeds(d, seeds=SEEDS, mode="full", lam=LAM, epochs=EPOCHS, lr=LR,
              gru_h=GRU_H, refresh=REFRESH, verbose=True, save_prefix=None):
    """Train one model per seed on a game-disjoint split and evaluate on test."""
    from epv_data import game_splits
    games = np.asarray(d["games"])
    res, models, splits = [], [], []
    for s in seeds:
        tr, va, te = game_splits(games, d["idx"], seed=s)
        t0 = time.time()
        if verbose:
            print(f"\n  [seed {s}] mode={mode} lam={lam} train {len(tr)} / val {len(va)}"
                  f" / test {len(te)} possessions "
                  f"({len(np.unique(games[tr]))}/{len(np.unique(games[va]))}/"
                  f"{len(np.unique(games[te]))} games)")
        model = train(d, tr, va, lam=lam, epochs=epochs, lr=lr, gru_h=gru_h,
                      refresh=refresh, seed=s, mode=mode, verbose=verbose)
        bm = float(np.asarray(d["R"][tr]).mean())      # constant-PPP baseline from TRAIN
        m = evaluate(model, d, te, baseline_mean=bm)
        m["seed"] = s; m["train_mean_R"] = bm; m["secs"] = time.time() - t0
        res.append(m); models.append(model); splits.append((tr, va, te))
        if verbose:
            print(f"  [seed {s}] test term-RMSE {m['term_rmse']:.4f} "
                  f"(baseline {m['baseline_rmse']:.4f}, skill {m['skill']:+.4f}) | "
                  f"frame-RMSE {m['frame_rmse']:.4f} | V(0) {m['first_frame_epv']:.3f} | "
                  f"{m['secs']:.0f}s")
        if save_prefix:
            torch.save({"state_dict": model.state_dict(), "mode": mode, "lam": lam,
                        "gru_h": gru_h, "fin": d["X"].shape[-1], "seed": s,
                        "train_mean_R": bm},
                       f"{save_prefix}_seed{s}.pt")
    return res, models, splits


def summarise_seeds(res, label=""):
    keys = ["term_rmse", "baseline_rmse", "skill", "frame_rmse", "term_corr",
            "first_frame_epv", "cal_mae", "cal_slope"]
    print(f"\n  === {label} over {len(res)} seeds ===")
    for k in keys:
        print(f"    {k:<18} {_fmt([r[k] for r in res])}")
    return {k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res])))
            for k in keys}


def main(args):
    if not HAVE_TORCH:
        print("torch not available — run --selftest for the TD(lambda) checks."); return
    from epv_data import load_epv, TERM_NAMES
    d = load_epv()
    games = np.asarray(d["games"])
    print(f"Dataset: {len(d['idx'])} possessions / {len(np.unique(games[d['idx']]))} games"
          f" | X{d['X'].shape} | mean R={np.asarray(d['R'][d['idx']]).mean():.4f}"
          f" | device {DEVICE}")

    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    res, models, splits = run_seeds(d, seeds=seeds, mode=args.mode, lam=args.lam,
                                    epochs=args.epochs, lr=args.lr, gru_h=args.gru_h,
                                    refresh=args.refresh,
                                    save_prefix=os.path.join(OUT_DIR, f"epv_causal_{args.mode}"))
    agg = summarise_seeds(res, f"GNN-EPV causal ({args.mode}, lam={args.lam}) TEST")

    m0 = res[0]
    print("\n  per-frame value calibration (seed 0 test split):")
    print("     pred EPV   realized R      n")
    for pv, rv, n in m0["calibration"]:
        print(f"    {pv:9.3f} {rv:12.3f} {n:8d}")
    print("\n  terminal-frame V by terminal type (seed 0):")
    for t, (v, r, n) in sorted(m0["by_term"].items()):
        print(f"    {TERM_NAMES.get(t,t):<12} V={v:6.3f}  actual R={r:6.3f}  n={n}")
    pr = m0["pos_rmse"]
    print(f"\n  RMSE by frame position (seed 0): t=0 {pr[0]:.3f} | "
          f"t=10 {pr[10]:.3f} | t=20 {pr[20]:.3f} | t=39 {pr[-1]:.3f}")

    # causality guard on the held-out split
    print("\n=== CAUSALITY GUARD (V(t) must ignore frames > t) ===")
    ok = causality_test(models[0], d, splits[0][2])

    out = os.path.join(OUT_DIR, f"epv_causal_{args.mode}_metrics.json")
    json.dump({"mode": args.mode, "lam": args.lam, "epochs": args.epochs,
               "lr": args.lr, "gru_h": args.gru_h, "refresh": args.refresh,
               "seeds": list(seeds), "agg": agg, "causal_test_pass": bool(ok),
               "per_seed": [{k: v for k, v in r.items()
                             if k not in ("calibration", "by_term", "pos_rmse")}
                            for r in res],
               "calibration_seed0": m0["calibration"],
               "by_term_seed0": {str(k): v for k, v in m0["by_term"].items()},
               "pos_rmse_seed0": m0["pos_rmse"]}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


def sweep(args):
    """Grid over the knobs that matter (spec §5): lambda, GRU hidden, lr, epochs,
    target-net refresh. Selection is on the VAL split; TEST is reported for the
    winner only, so the sweep cannot leak into the headline number."""
    if not HAVE_TORCH:
        print("torch required"); return
    from epv_data import load_epv, game_splits
    d = load_epv()
    games = np.asarray(d["games"])
    grid = []
    for lam in (0.6, 0.8, 0.9, 1.0):
        grid.append(dict(lam=lam, gru_h=GRU_H, lr=LR, refresh=REFRESH, epochs=args.epochs))
    for gh in (16, 48, 64):
        grid.append(dict(lam=args.lam, gru_h=gh, lr=LR, refresh=REFRESH, epochs=args.epochs))
    for lr in (1e-3, 6e-3):
        grid.append(dict(lam=args.lam, gru_h=GRU_H, lr=lr, refresh=REFRESH, epochs=args.epochs))
    for rf in (1, 5, 10):
        grid.append(dict(lam=args.lam, gru_h=GRU_H, lr=LR, refresh=rf, epochs=args.epochs))
    # the headline run's val curve was still falling at epoch 40, so the epoch
    # budget is a real knob, not a formality (early stopping still restores the
    # best epoch, so a larger budget can only help or tie)
    for ep in (50, 80):
        grid.append(dict(lam=args.lam, gru_h=GRU_H, lr=LR, refresh=REFRESH, epochs=ep))
    # de-duplicate configs the axes share
    seen, uniq = set(), []
    for c in grid:
        k = tuple(sorted(c.items()))
        if k not in seen:
            seen.add(k); uniq.append(c)
    grid = uniq

    seeds = tuple(int(s) for s in args.seeds.split(","))
    rows = []
    for gi, cfg in enumerate(grid, 1):
        vals, tests = [], []
        for s in seeds:
            tr, va, te = game_splits(games, d["idx"], seed=s)
            model = train(d, tr, va, seed=s, verbose=False, **cfg)
            bm = float(np.asarray(d["R"][tr]).mean())
            vals.append(evaluate(model, d, va, baseline_mean=bm)["term_rmse"])
            tests.append(evaluate(model, d, te, baseline_mean=bm)["term_rmse"])
        row = dict(cfg, val_rmse=float(np.mean(vals)), val_sd=float(np.std(vals)),
                   test_rmse=float(np.mean(tests)), test_sd=float(np.std(tests)))
        rows.append(row)
        print(f"[{gi}/{len(grid)}] lam={cfg['lam']:<4} gru_h={cfg['gru_h']:<3} "
              f"lr={cfg['lr']:<6} refresh={cfg['refresh']:<3} epochs={cfg['epochs']:<3} -> "
              f"VAL {row['val_rmse']:.4f}±{row['val_sd']:.4f}  "
              f"TEST {row['test_rmse']:.4f}±{row['test_sd']:.4f}", flush=True)
    rows.sort(key=lambda r: r["val_rmse"])
    print("\nBest by held-out (val) terminal RMSE:")
    for r in rows[:3]:
        print(f"  lam={r['lam']} gru_h={r['gru_h']} lr={r['lr']} refresh={r['refresh']} "
              f"epochs={r['epochs']} -> val {r['val_rmse']:.4f}  test {r['test_rmse']:.4f}")
    out = os.path.join(OUT_DIR, "epv_causal_sweep.json")
    json.dump(rows, open(out, "w"), indent=2)
    import csv
    with open(os.path.join(OUT_DIR, "epv_causal_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} (+ .csv)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--causal-test", action="store_true", dest="causal_test")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--mode", default="full", choices=["full", "spatial", "temporal"])
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--gru-h", type=int, default=GRU_H, dest="gru_h")
    ap.add_argument("--refresh", type=int, default=REFRESH)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    if a.causal_test:
        if not HAVE_TORCH: raise SystemExit("torch required")
        from epv_data import load_epv, game_splits
        d = load_epv()
        tr, va, te = game_splits(np.asarray(d["games"]), d["idx"], seed=0)
        mdl = train(d, tr[:4000], epochs=2, seed=0, verbose=False)
        raise SystemExit(0 if causality_test(mdl, d, te) else 1)
    if a.sweep:
        sweep(a)
    else:
        main(a)
