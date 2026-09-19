"""
hist_server.py — Flower ServerApp for the histogram protocol (hist_gbdt.py).

One Flower round = one aggregation: a depth level of the current tree ("hist"
mode) or the leaf sums of a random-structure tree ("random" mode). The strategy
only ever uses the SUM of the clients' vectors; with `hist-secagg = true` that sum
is produced by Flower's SecAgg+ workflow, so the server never sees a single
team's histogram.

After every completed tree, clients report their validation log loss (federated
evaluation). The reported model is the prefix with the lowest pooled validation
loss; training stops after `hist-patience` trees without improvement (the
remaining rounds are no-ops). Global-test metrics are logged per tree for
curves only (simulation convenience; never used for selection).

Run config (pyproject.toml or --run-config):
    hist-trees, hist-split-mode ("hist"|"random"), hist-bin-stride, hist-patience,
    hist-secagg, dp-epsilon, dp-delta, hist-leaf-clip, plus the params.* XGBoost keys.

Outputs (results/federated/): federated_hist_<tag>_curve.csv,
    xgb_federated_hist_<tag>_model_selected.json
"""

from __future__ import annotations

import json
from logging import INFO
from pathlib import Path

import numpy as np
import pandas as pd
from flwr.common import (Context, EvaluateIns, FitIns, Parameters, parameters_to_ndarrays)
from flwr.common.logger import log
from flwr.server import LegacyContext, ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from nba_federated import hist_gbdt as HG
from nba_federated.task import (TARGET_COL, _get_feature_cols, _get_global_split, get_global_make_rate,
                                load_full_dataset)
from sklearn.metrics import roc_auc_score

from nba_federated.task import OUTPUT_DIR  # noqa: E402
SECAGG_CLIP = 8192.0          # |per-team histogram entry| bound before SecAgg+ quantization


def params_from_config(rc: dict) -> HG.Params:
    return HG.Params(
        n_trees=int(rc.get("hist-trees", 3000)), eta=float(rc.get("params.eta", 0.05)),
        max_depth=int(rc.get("params.max_depth", 5)), reg_lambda=float(rc.get("params.reg_lambda", 2.0)),
        min_child_weight=float(rc.get("params.min_child_weight", 10)),
        colsample_bytree=float(rc.get("params.colsample_bytree", 0.8)),
        subsample=float(rc.get("params.subsample", 0.8)), split_mode=str(rc.get("hist-split-mode", "hist")),
        bin_stride=int(rc.get("hist-bin-stride", 1)), dp_epsilon=float(rc.get("dp-epsilon", 0.0)),
        dp_delta=float(rc.get("dp-delta", 1e-5)), leaf_clip=float(rc.get("hist-leaf-clip", 1.0)),
        seed=int(rc.get("seed", 42)))


class HistStrategy(Strategy):
    def __init__(self, p: HG.Params, num_clients: int, secagg: bool, patience: int, tag: str):
        self.p, self.K, self.secagg, self.patience, self.tag = p, num_clients, secagg, patience, tag
        df = load_full_dataset()
        self.spec = HG.BinSpec(_get_feature_cols(df), p.bin_stride)
        self.rng = np.random.default_rng(p.seed)
        self.sigma = HG.noise_sigma(p, len(self.spec.features))
        self.p0 = get_global_make_rate()
        self.trees: list[HG.Tree] = []
        self.builder: HG.TreeBuilder | None = None
        self.feats = None
        self.advance = None
        self.level = 0
        self.best = (np.inf, 0)
        self.curve: list[dict] = []
        self.stopped = False
        _, te_idx = _get_global_split()                  # global test games: curves only
        self.y_te = df.iloc[te_idx][TARGET_COL].to_numpy(float)
        self.b_te = self.spec.bin(df.iloc[te_idx][self.spec.features])
        self.m_te = np.full(len(self.y_te), np.log(self.p0 / (1 - self.p0)))

    # ── helpers ──
    def _new_tree_msg(self):
        return None if not self.trees else {"index": len(self.trees) - 1, "tree": self.trees[-1].to_dict()}

    def _all_clients(self, client_manager):
        client_manager.wait_for(self.K)
        return [client_manager.all()[cid] for cid in sorted(client_manager.all())]

    def initialize_parameters(self, client_manager):
        return Parameters(tensors=[], tensor_type="")

    # ── training ──
    def configure_fit(self, server_round, parameters, client_manager):
        if self.stopped or len(self.trees) >= self.p.n_trees:
            return []
        begin = self.builder is None or self.builder.done
        if begin:
            self.feats = np.sort(self.rng.choice(len(self.spec.features),
                                                 max(1, int(round(self.p.colsample_bytree * len(self.spec.features)))),
                                                 replace=False))
            self.builder = HG.TreeBuilder(self.spec, self.p, self.feats, self.sigma, self.rng)
            self.level, self.advance = 0, None
        instr = {"tree": len(self.trees), "level": self.level, "begin": begin, "subsample": self.p.subsample,
                 "noise_sd": self.sigma / np.sqrt(self.K), "new_tree": self._new_tree_msg(),
                 "advance": None if self.advance is None else [a.tolist() for a in self.advance],
                 "phase": "leaf" if self.p.split_mode == "random" else "hist",
                 "n_nodes": len(self.builder.frontier), "feats": self.feats.tolist(),
                 "structure": self.builder.tree.to_dict() if self.p.split_mode == "random" else None}
        ins = FitIns(Parameters(tensors=[], tensor_type=""), {"instr": json.dumps(instr)})
        return [(c, ins) for c in self._all_clients(client_manager)]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        vecs = [parameters_to_ndarrays(r.parameters)[0] for _, r in results]
        # SecAgg+ hands every result the same unmasked num_examples-weighted mean (all weights = 1).
        total = vecs[0] * len(results) if self.secagg else np.sum(vecs, axis=0)
        G, H = np.split(total, 2)
        b = self.builder
        if self.p.split_mode == "random":
            b.set_leaves(G, H)
        else:
            n = len(b.frontier)
            self.advance = b.decide_level(G.reshape(n, -1), H.reshape(n, -1))
            self.level += 1
        if b.done:
            self.trees.append(b.tree)
            self.m_te += b.tree.predict_bins(self.b_te, self.spec.n_bins)
        return Parameters(tensors=[], tensor_type=""), {}

    # ── federated validation after each completed tree ──
    def configure_evaluate(self, server_round, parameters, client_manager):
        if not (self.builder is not None and self.builder.done and self.trees
                and (not self.curve or self.curve[-1]["trees"] < len(self.trees))):
            return []
        ins = EvaluateIns(Parameters(tensors=[], tensor_type=""),
                          {"instr": json.dumps({"new_tree": self._new_tree_msg()})})
        return [(c, ins) for c in self._all_clients(client_manager)]

    def aggregate_evaluate(self, server_round, results, failures):
        if not results:
            return None, {}
        n = sum(r.num_examples for _, r in results)
        val = sum(r.loss * r.num_examples for _, r in results) / n
        p = 1.0 / (1.0 + np.exp(-self.m_te))
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        row = {"trees": len(self.trees), "round": server_round, "val_logloss": val,
               "test_brier": float(np.mean((p - self.y_te) ** 2)),
               "test_logloss": float(-np.mean(self.y_te * np.log(pc) + (1 - self.y_te) * np.log(1 - pc))),
               "test_auc": float(roc_auc_score(self.y_te, p))}
        self.curve.append(row)
        if val < self.best[0]:
            self.best = (val, len(self.trees))
        elif len(self.trees) - self.best[1] >= self.patience:
            self.stopped = True
        log(INFO, f"[hist/{self.tag}] tree {len(self.trees):4d} | val {val:.4f} | test Brier "
                  f"{row['test_brier']:.4f} AUC {row['test_auc']:.4f}")
        self._save()
        return val, {}

    def evaluate(self, server_round, parameters):
        return None

    def _save(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.curve).to_csv(OUTPUT_DIR / f"federated_hist_{self.tag}_curve.csv", index=False)
        k = self.best[1]
        if k > 0 and (self.stopped or len(self.trees) >= self.p.n_trees or len(self.trees) % 50 == 0):
            HG.to_xgboost(self.trees[:k], self.spec, self.p0).save_model(
                str(OUTPUT_DIR / f"xgb_federated_hist_{self.tag}_model_selected.json"))


def run(grid, context: Context) -> None:
    """Histogram protocol; called from server_app.main when run-config `protocol = "histogram"`."""
    rc = context.run_config
    p = params_from_config(rc)
    K = int(rc.get("num-clients", 30))
    secagg = bool(rc.get("hist-secagg", False))
    tag = str(rc.get("hist-tag", "")) or (f"{rc.get('strategy', 'team')}_{p.split_mode}_seed{p.seed}"
                                           + (f"_eps{p.dp_epsilon:g}" if p.dp_epsilon > 0 else "")
                                           + ("_secagg" if secagg else ""))
    strategy = HistStrategy(p, K, secagg, int(rc.get("hist-patience", 200)), tag)
    rounds = p.n_trees * (p.max_depth if p.split_mode == "hist" else 1)
    log(INFO, f"[hist] {p.split_mode} | trees ≤ {p.n_trees} | rounds ≤ {rounds} | SecAgg+ {secagg} | "
              f"DP ε={p.dp_epsilon} σ={strategy.sigma:.2f}")
    legacy = LegacyContext(context=context, config=ServerConfig(num_rounds=rounds), strategy=strategy)
    # num_shares must be odd: with an even value Flower's workflow bumps its own count
    # after the clients were configured, and key sharing fails with an IndexError.
    shares = max(3, K // 3) | 1
    # max_weight = 1: clients report num_examples = 1, and SecAgg+ scales every vector by
    # num_examples / max_weight BEFORE quantizing. The default (1000) would shrink the
    # histograms 1000x and turn the 2·8192/2^22 ≈ 0.004 quantization step into ≈ 4.
    fit_workflow = (SecAggPlusWorkflow(num_shares=shares, reconstruction_threshold=shares // 2 + 1,
                                       max_weight=1.0, clipping_range=SECAGG_CLIP) if secagg else None)
    DefaultWorkflow(fit_workflow=fit_workflow)(grid, legacy)
    strategy.stopped = True
    strategy._save()
