"""
hist_client.py — Flower ClientApp for the histogram protocol (hist_gbdt.py).

Each message round the server sends one instruction (JSON in FitIns.config["instr"]):

  new_tree   the tree completed in the previous round → added to this team's margins
  begin      start of a tree: row subsample for it (seeded per team and tree)
  advance    splits chosen at the previous level → move shots to their child nodes
  phase      "hist": return per-node, per-feature, per-bin (G, H) histograms
             "leaf": return per-node (G, H) for the given random-structure tree

The reply is ONE flat vector [G, H] with num_examples = 1, so under SecAgg+ (whose
unmasked output is the num_examples-weighted mean) the server recovers the sum
over teams as mean × K. DP noise N(0, noise_sd_g²) / N(0, noise_sd_h²) is added to G / H
before sending; under player-level DP each shooter's per-leaf sums are clipped
first (names never leave the client).

State kept between rounds in context.state: training and validation margins,
the number of trees applied, and each shot's current node. A client whose actor is
restarted loses that state; it reports how many trees it has applied in every reply,
and the server answers a mismatch by resending the whole model ("resync"), because a
client that silently kept fitting stale residuals would corrupt the ensemble.

`optional_secaggplus_mod` runs Flower's SecAgg+ client protocol only when the
server's message carries SecAgg+ configuration, so one ClientApp serves runs with
and without secure aggregation.
"""

from __future__ import annotations

import json
from logging import WARNING

import numpy as np
import xgboost as xgb  # noqa: F401  (keeps import order identical to client_app)
from flwr.client import Client, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import (Code, ConfigRecord, Context, EvaluateIns, EvaluateRes, FitIns, FitRes, Message,
                         MessageType, Status, ndarrays_to_parameters)
from flwr.common.logger import log
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS

from nba_federated import hist_gbdt as HG
from nba_federated.task import PUBLIC_LEAGUE_FG_PCT, partition_frames, teams_per_player

STATE_KEY = "hist"
_DATA: dict[tuple, tuple] = {}          # per-actor cache: binned local train / validation data


def _local_data(pid: int, n_parts: int, strategy: str, seed: int, bin_stride: int, data_path: str | None):
    key = (pid, n_parts, strategy, seed, bin_stride, data_path)
    if key not in _DATA:
        X_tr, X_va, y_tr, y_va, players = partition_frames(pid, n_parts, strategy, random_state=seed,
                                                           data_path=data_path, return_players=True)
        spec = HG.BinSpec(list(X_tr.columns), bin_stride)
        _DATA[key] = (spec, spec.bin(X_tr), y_tr.to_numpy(float), spec.bin(X_va), y_va.to_numpy(float),
                      players, teams_per_player(data_path))
    return _DATA[key]


class HistClient(Client):
    def __init__(self, context: Context):
        rc = context.run_config
        self.pid = int(context.node_config["partition-id"])
        self.seed = int(rc.get("seed", 42))
        self.spec, self.b_tr, self.y_tr, self.b_va, self.y_va, self.players, self.player_teams = _local_data(
            self.pid, int(context.node_config["num-partitions"]), str(rc.get("strategy", "team")),
            self.seed, int(rc.get("hist-bin-stride", 1)), rc.get("data-path") or None)
        m0 = float(np.log(PUBLIC_LEAGUE_FG_PCT / (1 - PUBLIC_LEAGUE_FG_PCT)))
        self.state = context.state
        if STATE_KEY not in self.state.config_records:
            self.state.config_records[STATE_KEY] = ConfigRecord({
                "applied": 0, "margin": np.full(len(self.y_tr), m0).tobytes(),
                "margin_va": np.full(len(self.y_va), m0).tobytes(),
                "node": np.full(len(self.y_tr), -1, dtype=np.int64).tobytes()})
        st = self.state.config_records[STATE_KEY]
        self.applied = int(st["applied"])
        self.margin = np.frombuffer(st["margin"], dtype=np.float64).copy()
        self.margin_va = np.frombuffer(st["margin_va"], dtype=np.float64).copy()
        self.node = np.frombuffer(st["node"], dtype=np.int64).copy()

    def _save(self):
        self.state.config_records[STATE_KEY] = ConfigRecord({
            "applied": self.applied, "margin": self.margin.tobytes(),
            "margin_va": self.margin_va.tobytes(), "node": self.node.tobytes()})

    def _apply_new_tree(self, instr: dict):
        """Apply the tree just completed, or rebuild both margins from a full resync."""
        resync = instr.get("resync_trees")
        if resync is not None:
            m0 = float(np.log(PUBLIC_LEAGUE_FG_PCT / (1 - PUBLIC_LEAGUE_FG_PCT)))
            self.margin = np.full(len(self.y_tr), m0)
            self.margin_va = np.full(len(self.y_va), m0)
            for d in resync:
                tree = HG.Tree.from_dict(d)
                self.margin += tree.predict_bins(self.b_tr, self.spec.n_bins)
                self.margin_va += tree.predict_bins(self.b_va, self.spec.n_bins)
            self.applied = len(resync)
            log(WARNING, f"[hist client {self.pid}] resynced to {self.applied} trees")
            return
        nt = instr.get("new_tree")
        if nt is not None and int(nt["index"]) == self.applied:
            tree = HG.Tree.from_dict(nt["tree"])
            self.margin += tree.predict_bins(self.b_tr, self.spec.n_bins)
            self.margin_va += tree.predict_bins(self.b_va, self.spec.n_bins)
            self.applied += 1

    def fit(self, ins: FitIns) -> FitRes:
        instr = json.loads(str(ins.config["instr"]))
        self._apply_new_tree(instr)
        # A fresh engine client per round; its RNG is seeded per (team, tree, level) → reproducible.
        c = HG.Client(self.b_tr, self.y_tr, 0.0, self.spec,
                      seed=[self.seed, self.pid, int(instr["tree"]), int(instr.get("level", 0))],
                      players=self.players, player_teams=self.player_teams,
                      discrete=bool(instr.get("discrete")), lattice=float(instr.get("lattice", 0.0)))
        c.margin, c.node = self.margin, self.node
        if instr.get("begin"):
            c.begin_tree(float(instr["subsample"]), bool(instr.get("by_player")))
        else:
            c.gradients()
        if instr.get("advance") is not None:
            c.advance(*(np.asarray(a) for a in instr["advance"]))
        sd_g, sd_h = float(instr["noise_sd_g"]), float(instr["noise_sd_h"])
        if instr["phase"] == "leaf":
            clip = tuple(instr["clip"]) if instr.get("clip") else None
            G, H = c.leaf_sums(HG.Tree.from_dict(instr["structure"]), sd_g, sd_h, clip)
        else:
            G, H = c.histograms(int(instr["n_nodes"]), np.asarray(instr["feats"]), sd_g, sd_h)
        self.node = c.node
        self._save()
        vec = np.concatenate([G.ravel(), H.ravel()])
        return FitRes(status=Status(code=Code.OK, message="OK"),
                      parameters=ndarrays_to_parameters([vec]), num_examples=1,
                      metrics={"applied": self.applied})

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Federated validation: log loss of the current ensemble on this team's held-out games."""
        self._apply_new_tree(json.loads(str(ins.config["instr"])))
        self._save()
        p = np.clip(1.0 / (1.0 + np.exp(-self.margin_va)), 1e-15, 1 - 1e-15)
        ll = float(-np.mean(self.y_va * np.log(p) + (1 - self.y_va) * np.log(1 - p)))
        return EvaluateRes(status=Status(code=Code.OK, message="OK"), loss=ll,
                           num_examples=len(self.y_va), metrics={"trees": self.applied})


def client_fn(context: Context):
    return HistClient(context)


def optional_secaggplus_mod(msg: Message, ctxt: Context, call_next) -> Message:
    """SecAgg+ for training messages that carry its configuration; pass-through otherwise."""
    if msg.metadata.message_type == MessageType.TRAIN and RECORD_KEY_CONFIGS in msg.content.config_records:
        return secaggplus_mod(msg, ctxt, call_next)
    return call_next(msg, ctxt)


app = ClientApp(client_fn=client_fn, mods=[optional_secaggplus_mod])
