"""
hist_gbdt.py — histogram-based horizontal federated gradient boosting, with
optional distributed differential privacy.

Why this protocol. Tree bagging (FedXgbBagging) lets every team grow its own
tree on ~2k local shots, so splits are chosen from local noise and the model
plateaus far below centralized XGBoost. Here all teams grow ONE shared tree:

  for each tree:
    server picks the tree's feature subset (colsample, public randomness)
    for each depth level:
      every client bins its shots on the PUBLIC grid (hardening.public_edges),
      and sends per-node, per-feature, per-bin sums of gradients g = p − y and
      hessians h = p(1 − p)
      server SUMS the histograms and picks each node's best split exactly as
      XGBoost's `hist` method does (gain, min_child_weight, missing direction)
    leaves get w = −G/(H + λ), scaled by η

Without DP, the sum of the teams' histograms equals the pooled histogram, so the
trees are the ones centralized XGBoost would grow on the binned features.

Privacy. Clients only ever release histograms (sums over their shots), meant to
be combined with secure aggregation (Flower SecAgg+), so the server sees only
the total over teams. For DP, each client adds Gaussian noise N(0, σ²/K) to its
histograms; the K contributions add up to N(0, σ²) on the aggregate
(distributed DP), so each team pays 1/√K of the noise it would need alone.

  Sensitivity (add/remove one shot): the shot adds g ∈ (−1, 1) to one bin per
  released feature and h ∈ (0, 0.25] likewise. Releasing (G, 4H) jointly, the
  L2 sensitivity is √(2F) for F released features. σ = z·√(2F), with the noise
  multiplier z from the RDP accountant (dp.gaussian_z_for_epsilon) over every
  release a shot takes part in.

  split_mode = "hist":   one release per depth level → trees × max_depth releases,
                         F = features in the tree's subset.
  split_mode = "random": tree structure drawn at random on the public grid (no
                         data involved); only the leaf sums are released →
                         one release per tree, F = 1 (Maddock et al., CCS 2022).

  All split choices, min_child_weight checks and leaf values are computed from
  the noisy aggregate, i.e. post-processing. No amplification from row
  subsampling is claimed, so the guarantee is conservative. The unit protected
  is one shot (event-level DP).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from nba_federated import dp, hardening


@dataclass
class Params:
    n_trees: int = 500
    eta: float = 0.05
    max_depth: int = 5
    reg_lambda: float = 2.0
    min_child_weight: float = 10.0
    colsample_bytree: float = 0.8
    subsample: float = 0.8
    split_mode: str = "hist"       # "hist" or "random"
    bin_stride: int = 1            # keep every k-th public edge (coarser grid → less DP noise per bin)
    dp_epsilon: float = 0.0        # total ε over the whole training run; 0 disables DP
    dp_delta: float = 1e-5
    leaf_clip: float = 1.0         # |w| cap before η, applied only when DP is on
    seed: int = 42


# ──────────────────────────────────────────────────────────────
# Public bin grid
# ──────────────────────────────────────────────────────────────
class BinSpec:
    """Public bin edges per feature; bin k = [e_k, e_{k+1}); slot m_f holds missing values."""

    def __init__(self, feature_names: list[str], bin_stride: int = 1):
        self.features = list(feature_names)
        edges = hardening.public_edges(self.features)
        self.edges = [e[::bin_stride] if bin_stride > 1 and len(e) > 8 else e for e in
                      (edges[f] for f in self.features)]
        self.n_bins = np.array([len(e) for e in self.edges])          # m_f (missing slot excluded)

    def bin(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty(X.shape, dtype=np.int16)
        for f, e in enumerate(self.edges):
            v = X[self.features[f]].to_numpy(float)
            k = np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(e) - 1)
            out[:, f] = np.where(np.isnan(v), len(e), k)
        return out


# ──────────────────────────────────────────────────────────────
# Tree
# ──────────────────────────────────────────────────────────────
class Tree:
    """Binary tree over binned features. Node 0 is the root; children are appended."""

    def __init__(self):
        self.feature, self.thr_bin, self.default_left = [], [], []
        self.left, self.right, self.value = [], [], []

    def add_node(self) -> int:
        for a, v in ((self.feature, -1), (self.thr_bin, 0), (self.default_left, True),
                     (self.left, -1), (self.right, -1), (self.value, 0.0)):
            a.append(v)
        return len(self.left) - 1

    def set_split(self, i, f, j, default_left):
        self.feature[i], self.thr_bin[i], self.default_left[i] = f, j, default_left
        self.left[i], self.right[i] = self.add_node(), self.add_node()

    def route(self, bins: np.ndarray, n_bins: np.ndarray) -> np.ndarray:
        """Leaf node id for each row."""
        feat, thr, dl = np.array(self.feature), np.array(self.thr_bin), np.array(self.default_left)
        left, right = np.array(self.left), np.array(self.right)
        node = np.zeros(len(bins), dtype=np.int64)
        while True:
            internal = left[node] != -1
            if not internal.any():
                return node
            rows = np.flatnonzero(internal)
            n = node[rows]
            b = bins[rows, feat[n]]
            missing = b == n_bins[feat[n]]
            go_left = np.where(missing, dl[n], b < thr[n])
            node[rows] = np.where(go_left, left[n], right[n])

    def predict_bins(self, bins, n_bins):
        return np.array(self.value)[self.route(bins, n_bins)]

    def to_dict(self) -> dict:
        return {"feature": [int(v) for v in self.feature], "thr_bin": [int(v) for v in self.thr_bin],
                "default_left": [bool(v) for v in self.default_left], "left": [int(v) for v in self.left],
                "right": [int(v) for v in self.right], "value": [float(v) for v in self.value]}

    @classmethod
    def from_dict(cls, d: dict) -> "Tree":
        t = cls()
        t.feature, t.thr_bin, t.default_left = list(d["feature"]), list(d["thr_bin"]), list(d["default_left"])
        t.left, t.right, t.value = list(d["left"]), list(d["right"]), list(d["value"])
        return t


# ──────────────────────────────────────────────────────────────
# Client
# ──────────────────────────────────────────────────────────────
class Client:
    """Holds one team's binned shots and its running margin. Releases only sums."""

    def __init__(self, bins: np.ndarray, y: np.ndarray, base_margin: float, spec: BinSpec, seed: int):
        self.bins, self.y, self.spec = bins, np.asarray(y, float), spec
        self.margin = np.full(len(y), base_margin)
        self.rng = np.random.default_rng(seed)
        self.node = None

    def gradients(self):
        p = 1.0 / (1.0 + np.exp(-self.margin))
        self.g, self.h = p - self.y, p * (1.0 - p)

    def begin_tree(self, subsample: float):
        self.gradients()
        sampled = self.rng.random(len(self.y)) < subsample
        self.node = np.where(sampled, 0, -1)

    def histograms(self, n_nodes: int, feats: np.ndarray, noise_sd: float) -> tuple[np.ndarray, np.ndarray]:
        """(G, H) of shape (n_nodes, S) with S = Σ_f (m_f + 1) over `feats`, plus this client's DP share."""
        width = self.spec.n_bins[feats] + 1
        offs = np.concatenate([[0], np.cumsum(width)[:-1]])
        S = int(width.sum())
        rows = np.flatnonzero(self.node >= 0)
        idx = (self.node[rows, None] * S + offs[None, :] + self.bins[np.ix_(rows, feats)]).ravel()
        k = len(feats)
        G = np.bincount(idx, weights=np.repeat(self.g[rows], k), minlength=n_nodes * S).reshape(n_nodes, S)
        H = np.bincount(idx, weights=np.repeat(self.h[rows], k), minlength=n_nodes * S).reshape(n_nodes, S)
        if noise_sd > 0:
            G += self.rng.normal(0.0, noise_sd, G.shape)
            H += self.rng.normal(0.0, noise_sd / 4.0, H.shape)       # released as 4H with the same σ
        return G, H

    def advance(self, feat, thr, dl, left_pos, right_pos):
        """Move sampled rows from frontier positions to the next level's positions (−1 = reached a leaf)."""
        rows = np.flatnonzero(self.node >= 0)
        k = self.node[rows]
        is_split = left_pos[k] >= 0
        b = self.bins[rows, np.maximum(feat[k], 0)]
        missing = b == self.spec.n_bins[np.maximum(feat[k], 0)]
        go_left = np.where(missing, dl[k], b < thr[k])
        self.node[rows] = np.where(is_split, np.where(go_left, left_pos[k], right_pos[k]), -1)

    def leaf_sums(self, tree: Tree, noise_sd: float) -> tuple[np.ndarray, np.ndarray]:
        """Per-node (G, H) of the sampled rows' leaves (random mode), plus the DP share."""
        rows = np.flatnonzero(self.node >= 0)
        leaf = tree.route(self.bins[rows], self.spec.n_bins)
        n = len(tree.left)
        G = np.bincount(leaf, weights=self.g[rows], minlength=n)
        H = np.bincount(leaf, weights=self.h[rows], minlength=n)
        if noise_sd > 0:
            G += self.rng.normal(0.0, noise_sd, n)
            H += self.rng.normal(0.0, noise_sd / 4.0, n)
        return G, H

    def end_tree(self, tree: Tree):
        self.margin += tree.predict_bins(self.bins, self.spec.n_bins)


# ──────────────────────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────────────────────
def noise_sigma(p: Params, n_features: int) -> float:
    """σ of the noise on the AGGREGATE G (4H gets the same σ); 0 if DP is off."""
    if p.dp_epsilon <= 0:
        return 0.0
    z = dp.gaussian_z_for_epsilon(p.dp_epsilon, p.dp_delta, releases_per_shot(p))
    f_released = max(1, int(round(p.colsample_bytree * n_features))) if p.split_mode == "hist" else 1
    return z * np.sqrt(2.0 * f_released)


def releases_per_shot(p: Params) -> int:
    return p.n_trees * (p.max_depth if p.split_mode == "hist" else 1)


class TreeBuilder:
    """Server-side growth of ONE tree from aggregated statistics, one step per message round.

    hist mode:   call `decide_level(G, H)` once per depth level with the summed
                 histograms; it returns what clients need to move their shots to
                 the next level (or None when the tree is finished).
    random mode: `tree` already holds the random structure; call `set_leaves(G, H)`
                 with the summed per-node leaf sums.
    """

    def __init__(self, spec: BinSpec, p: Params, feats: np.ndarray, sigma: float, rng: np.random.Generator):
        self.spec, self.p, self.feats, self.sigma = spec, p, feats, sigma
        self.tree = Tree(); self.tree.add_node()
        self.frontier, self.totals, self.depth, self.done = [0], {}, 0, False
        if p.split_mode == "random":
            self._random_structure(rng)
        else:
            m = spec.n_bins[feats]
            self.m, self.M = m, int(m.max())
            offs = np.concatenate([[0], np.cumsum(m + 1)[:-1]])
            self.pos = np.full((len(feats), self.M), -1)      # padded (F, M) view of a flat histogram row
            for a, (o, mf) in enumerate(zip(offs, m)):
                self.pos[a, :mf] = o + np.arange(mf)
            self.miss_pos = offs + m

    def leaf_value(self, G: float, H: float) -> float:
        w = -G / (max(H, 0.0) + self.p.reg_lambda)
        if self.sigma > 0:
            w = float(np.clip(w, -self.p.leaf_clip, self.p.leaf_clip))
        return self.p.eta * w

    # ── random mode ──
    def _random_structure(self, rng):
        t, frontier = self.tree, [0]
        for _ in range(self.p.max_depth):
            nxt = []
            for i in frontier:
                f = int(rng.choice(self.feats))
                m = self.spec.n_bins[f]
                if m < 2:
                    continue
                t.set_split(i, f, int(rng.integers(1, m)), bool(rng.random() < 0.5))
                nxt += [t.left[i], t.right[i]]
            frontier = nxt

    def set_leaves(self, G: np.ndarray, H: np.ndarray):
        t = self.tree
        for i in range(len(t.left)):
            if t.left[i] == -1:
                t.value[i] = self.leaf_value(G[i], H[i])
        self.done = True

    # ── hist mode ──
    def decide_level(self, G: np.ndarray, H: np.ndarray):
        """Pick splits for the current frontier from summed (n_frontier, S) histograms."""
        p, t, M, m = self.p, self.tree, self.M, self.m
        lam, mcw = p.reg_lambda, p.min_child_weight
        n = len(self.frontier)
        Gz = np.concatenate([G, np.zeros((n, 1))], axis=1); Hz = np.concatenate([H, np.zeros((n, 1))], axis=1)
        Gb, Hb = Gz[:, self.pos], Hz[:, self.pos]            # (n, F, M), padded with 0
        Gm, Hm = G[:, self.miss_pos], H[:, self.miss_pos]    # (n, F)
        Gt, Ht = Gb.sum(2) + Gm, Hb.sum(2) + Hm              # per-feature node totals
        cG, cH = np.cumsum(Gb, 2), np.cumsum(Hb, 2)          # cum[..., j−1] = bins < j go left
        valid = np.arange(M)[None, None, :] < (m - 1)[None, :, None]   # thresholds j = 1..m_f−1

        def gain(GL, HL):
            GR, HR = Gt[..., None] - GL, Ht[..., None] - HL
            g = GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - (Gt ** 2 / (Ht + lam))[..., None]
            return np.where(valid & (HL >= mcw) & (HR >= mcw), g, -np.inf)

        g_r = gain(cG, cH)                                     # missing → right
        g_l = gain(cG + Gm[..., None], cH + Hm[..., None])     # missing → left
        best_l = g_l > g_r
        g_best = np.where(best_l, g_l, g_r)

        feat_arr = np.full(n, -1); thr_arr = np.zeros(n, int); dl_arr = np.zeros(n, bool)
        lpos = np.full(n, -1); rpos = np.full(n, -1)
        nxt = []
        for k, i in enumerate(self.frontier):
            if i not in self.totals:                           # root: average the F noisy totals
                self.totals[i] = (float(Gt[k].mean()), float(Ht[k].mean()))
            a, jm1 = divmod(int(np.argmax(g_best[k])), M)
            if not np.isfinite(g_best[k, a, jm1]) or g_best[k, a, jm1] <= 1e-6:
                t.value[i] = self.leaf_value(*self.totals[i])
                continue
            dl = bool(best_l[k, a, jm1])
            GL = cG[k, a, jm1] + (Gm[k, a] if dl else 0.0)
            HL = cH[k, a, jm1] + (Hm[k, a] if dl else 0.0)
            t.set_split(i, int(self.feats[a]), jm1 + 1, dl)
            self.totals[t.left[i]] = (GL, HL)
            self.totals[t.right[i]] = (Gt[k, a] - GL, Ht[k, a] - HL)
            feat_arr[k], thr_arr[k], dl_arr[k] = self.feats[a], jm1 + 1, dl
            lpos[k], rpos[k] = len(nxt), len(nxt) + 1
            nxt += [t.left[i], t.right[i]]
        self.frontier, self.depth = nxt, self.depth + 1
        if self.depth == p.max_depth or not nxt:
            for i in nxt:                                      # nodes at max depth become leaves
                t.value[i] = self.leaf_value(*self.totals[i])
            self.done = True
            return None
        return feat_arr, thr_arr, dl_arr, lpos, rpos


class Trainer:
    """In-process orchestration (simulation). Every quantity the server reads is a sum over clients."""

    def __init__(self, clients: list[Client], spec: BinSpec, params: Params):
        self.clients, self.spec, self.p = clients, spec, params
        self.rng = np.random.default_rng(params.seed)
        self.trees: list[Tree] = []
        self.sigma = noise_sigma(params, len(spec.features))

    def n_features_per_tree(self) -> int:
        return max(1, int(round(self.p.colsample_bytree * len(self.spec.features))))

    def releases_per_shot(self) -> int:
        return releases_per_shot(self.p)

    def _client_sd(self) -> float:
        return self.sigma / np.sqrt(len(self.clients))

    def grow_tree(self) -> Tree:
        feats = np.sort(self.rng.choice(len(self.spec.features), self.n_features_per_tree(), replace=False))
        for c in self.clients:
            c.begin_tree(self.p.subsample)
        b = TreeBuilder(self.spec, self.p, feats, self.sigma, self.rng)
        sd = self._client_sd()
        if self.p.split_mode == "random":
            b.set_leaves(*map(sum, zip(*(c.leaf_sums(b.tree, sd) for c in self.clients))))
        else:
            while not b.done:
                G, H = map(sum, zip(*(c.histograms(len(b.frontier), feats, sd) for c in self.clients)))
                step = b.decide_level(G, H)
                if step is not None:
                    for c in self.clients:
                        c.advance(*step)
        for c in self.clients:
            c.end_tree(b.tree)
        self.trees.append(b.tree)
        return b.tree

    def fit(self, callback=None):
        for t in range(self.p.n_trees):
            tree = self.grow_tree()
            if callback is not None:
                callback(t + 1, tree)
        return self


# ──────────────────────────────────────────────────────────────
# Export to a standard XGBoost booster (predicts on RAW features)
# ──────────────────────────────────────────────────────────────
def to_xgboost(trees: list[Tree], spec: BinSpec, base_score: float) -> xgb.Booster:
    """Serialise trees as an XGBoost model. Thresholds are public edges, so `x < e_j`
    on raw features is exactly `bin < j` on binned ones. Per-node statistics are 0."""
    F = len(spec.features)
    dummy = pd.DataFrame(np.zeros((2, F)), columns=spec.features)
    template = xgb.train({"objective": "binary:logistic", "base_score": base_score, "max_depth": 1},
                         xgb.DMatrix(dummy, label=[0, 1]), 1)
    model = json.loads(bytes(template.save_raw("json")))
    gb = model["learner"]["gradient_booster"]["model"]
    out = []
    for t_id, t in enumerate(trees):
        n = len(t.left)
        parents = [2147483647] * n
        for i in range(n):
            if t.left[i] != -1:
                parents[t.left[i]] = parents[t.right[i]] = i
        leaf = [t.left[i] == -1 for i in range(n)]
        out.append({
            "base_weights": [t.value[i] if leaf[i] else 0.0 for i in range(n)],
            "categories": [], "categories_nodes": [], "categories_segments": [], "categories_sizes": [],
            "default_left": [int(bool(t.default_left[i]) and not leaf[i]) for i in range(n)],
            "id": t_id,
            "left_children": list(t.left), "right_children": list(t.right), "parents": parents,
            "loss_changes": [0.0] * n, "sum_hessian": [0.0] * n,
            "split_conditions": [t.value[i] if leaf[i] else float(spec.edges[t.feature[i]][t.thr_bin[i]])
                                 for i in range(n)],
            "split_indices": [0 if leaf[i] else int(t.feature[i]) for i in range(n)],
            "split_type": [0] * n,
            "tree_param": {"num_deleted": "0", "num_feature": str(F), "num_nodes": str(n), "size_leaf_vector": "1"},
        })
    gb["trees"] = out
    gb["tree_info"] = [0] * len(out)
    gb["iteration_indptr"] = list(range(len(out) + 1))
    gb["gbtree_model_param"]["num_trees"] = str(len(out))
    bst = xgb.Booster()
    bst.load_model(bytearray(json.dumps(model).encode("utf-8")))
    return bst
