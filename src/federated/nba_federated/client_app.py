"""
client_app.py — Flower XGBoost ClientApp for NBA Shot Prediction

Each client represents one NBA team. On every FL round:
  1. Receive the current global model (serialized JSON bytes) from the server.
  2. Train exactly 1 new tree on local (private) team data using the global
     model as the starting point — this is the core of the bagging approach.
  3. Return the updated model back to the server.

The client also evaluates the global model on its local test split so the
server can track per-team performance without ever seeing raw team data.

Hyperparameters come from pyproject.toml ([tool.flwr.app.config]) via the
server's `on_fit_config_fn`: max_depth=5, eta=0.05, min_child_weight=10,
reg_lambda=2.0, subsample/colsample=0.8, NO `scale_pos_weight` (calibration
matters downstream for POE/EPV). The defaults below mirror those values as a
safety net in case the config is empty.
"""

import gc
from logging import INFO

import xgboost as xgb
from flwr.client import ClientApp
from flwr.common import (
    Context,
    Parameters,
)
from flwr.common.logger import log

from nba_federated.task import partition_frames
from nba_federated import dp, hardening


# ──────────────────────────────────────────────────────────────
# XGBoost model serialisation helpers
# ──────────────────────────────────────────────────────────────

def _params_to_model(parameters: Parameters) -> xgb.Booster | None:
    """Deserialise Flower Parameters → XGBoost Booster (or None for round 0)."""
    if not parameters.tensors or parameters.tensors[0] == b"":
        return None
    booster = xgb.Booster()
    booster.load_model(bytearray(parameters.tensors[0]))
    return booster


def _model_to_params(booster: xgb.Booster) -> Parameters:
    """Serialise XGBoost Booster → Flower Parameters (JSON bytes)."""
    return Parameters(tensors=[bytes(booster.save_raw("json"))], tensor_type="json")


# ──────────────────────────────────────────────────────────────
# XGBoost hyperparameters
# ──────────────────────────────────────────────────────────────

def _build_xgb_params(config: dict) -> dict:
    """
    Build XGBoost training params for a single client step.

    NOTE: We DO NOT use `scale_pos_weight`. The dataset is roughly balanced
    (~45% makes) and the centralized baseline (train_xgboost.py) intentionally
    leaves it out so the model produces calibrated probabilities. Using a
    per-client scale_pos_weight here was the main cause of round-over-round
    degradation: every team has a slightly different make-rate, so each tree
    was biased differently and the inconsistencies compounded across rounds.
    """
    return {
        "objective":        config.get("params.objective",        "binary:logistic"),
        "eval_metric":      config.get("params.eval_metric",      "logloss"),
        "tree_method":      config.get("params.tree_method",      "hist"),
        "max_depth":        int(config.get("params.max_depth",     5)),
        "eta":              float(config.get("params.eta",         0.05)),
        "subsample":        float(config.get("params.subsample",   0.8)),
        "colsample_bytree": float(config.get("params.colsample_bytree", 0.8)),
        "min_child_weight": int(config.get("params.min_child_weight", 10)),
        "reg_lambda":       float(config.get("params.reg_lambda",  2.0)),
        "base_score":       float(config.get("params.base_score",  0.45)),
        "seed":             int(config.get("seed",                  42)),
        "nthread":          1,
    }


# ──────────────────────────────────────────────────────────────
# Flower Client
# ──────────────────────────────────────────────────────────────

def client_fn(context: Context):
    """Factory function: create a Flower client for a given partition."""
    from flwr.client import Client
    from flwr.common import FitIns, FitRes, EvaluateIns, EvaluateRes, Status, Code
    from sklearn.metrics import roc_auc_score, accuracy_score
    import numpy as np

    # Read partition config from context
    partition_id   = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    strategy       = context.run_config.get("strategy", "team")
    data_path      = context.run_config.get("data-path", None)
    # Master seed for IID partitioning + local train/eval split. The global
    # held-out test set is intentionally NOT seeded by this — it stays fixed
    # across seeds so multi-seed runs measure training variance on the SAME
    # test set rather than bootstrapping the test set itself.
    seed           = int(context.run_config.get("seed", 42))

    log(INFO, f"[Client {partition_id}] Loading partition (strategy={strategy}, seed={seed})")

    X_train, X_test, y_train, y_test = partition_frames(
        partition_id=partition_id,
        num_partitions=num_partitions,
        strategy=strategy,
        data_path=data_path,
        random_state=seed,
    )
    num_train, num_test = len(X_train), len(X_test)

    # Layer-1 hardening (nba_federated/hardening.py). public-bins: train on
    # features rounded to a public grid, then snap thresholds to it, so no split
    # value comes from this team's data. harden-strip: zero the per-node stats
    # (sum_hessian, loss_changes, internal weights) before sending.
    public_bins  = bool(context.run_config.get("public-bins", False))
    harden_strip = bool(context.run_config.get("harden-strip", False))
    edges = hardening.public_edges(list(X_train.columns)) if public_bins else None
    X_fit = hardening.bin_to_public_edges(X_train, edges) if public_bins else X_train

    train_dmatrix = xgb.DMatrix(X_fit, label=y_train)
    test_dmatrix  = xgb.DMatrix(X_test, label=y_test)    # raw: snapped trees predict identically

    log(INFO, f"[Client {partition_id}] "
              f"Train={num_train} shots | Test={num_test} shots")

    # Per-client RNG for the (optional) differential-privacy leaf noise.
    # Seeded per partition so runs are reproducible; distinct across clients.
    dp_rng = np.random.default_rng(1000 + partition_id)

    class XGBClient(Client):

        def fit(self, ins: FitIns) -> FitRes:
            """
            Receive global model, add `local-epochs` new tree(s) on local data,
            and return ONLY the new tree(s).

            Why "only the new tree(s)": Flower's FedXgbBagging aggregator
            extracts the client's contribution as `trees_curr[0..local_epochs]`
            — the FIRST trees of the booster the client returns. If the client
            returns the FULL updated model (global + new), `trees_curr[0]` is
            actually the oldest global tree, not the new one — so each round
            ends up appending duplicates of round-1 trees instead of new
            information. That's the bug that caused calibration to collapse.

            Fix (matches the official Flower XGBoost example): slice out the
            new trees with `booster[n_old:n_new]` before serialising.
            """
            global_model    = _params_to_model(ins.parameters)
            xgb_params      = _build_xgb_params(ins.config)
            num_local_round = int(ins.config.get("local-epochs", 1))
            aggregation     = str(ins.config.get("aggregation", "bagging")).lower()

            if global_model is None:
                # Round 1 — train from scratch; every tree is new.
                local_model = xgb.train(
                    params=xgb_params,
                    dtrain=train_dmatrix,
                    num_boost_round=num_local_round,
                    evals=[(train_dmatrix, "train")],
                    verbose_eval=False,
                )
                return_bst = local_model
            else:
                # Round N>1 — continue training on top of the global model.
                n_old = global_model.num_boosted_rounds()
                local_model = xgb.train(
                    params=xgb_params,
                    dtrain=train_dmatrix,
                    num_boost_round=num_local_round,
                    xgb_model=global_model,
                    evals=[(train_dmatrix, "train")],
                    verbose_eval=False,
                )
                if aggregation == "cyclic":
                    # FedXgbCyclic: exactly one client per round, and its full
                    # booster BECOMES the next round's global model. Return
                    # the full updated model.
                    return_bst = local_model
                else:
                    # FedXgbBagging: server appends our new trees onto the
                    # global. Return ONLY the new tree(s) — see protocol notes
                    # above; otherwise the aggregator duplicates old trees.
                    n_new = local_model.num_boosted_rounds()
                    return_bst = local_model[n_old:n_new]

            if public_bins:
                return_bst = hardening.snap_thresholds(return_bst, edges)

            # Optional local differential privacy: clip + noise the leaves of
            # the tree(s) we are about to transmit. No-op if dp-epsilon <= 0
            # (default), so the baseline protocol is unchanged.
            return_bst = dp.maybe_dp(return_bst, num_local_round, ins.config, dp_rng)

            if harden_strip:
                return_bst = hardening.strip_bookkeeping(return_bst)

            serialised_model = _model_to_params(return_bst)
            del local_model, return_bst
            gc.collect()

            return FitRes(
                status=Status(code=Code.OK, message="OK"),
                parameters=serialised_model,
                num_examples=num_train,
                metrics={},
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            """Evaluate the global model on this team's local test split."""
            global_model = _params_to_model(ins.parameters)
            if global_model is None:
                return EvaluateRes(
                    status=Status(code=Code.OK, message="No model yet"),
                    loss=1.0, num_examples=num_test, metrics={}
                )

            y_probs = global_model.predict(test_dmatrix)
            y_true  = test_dmatrix.get_label()
            y_preds = (y_probs > 0.5).astype(int)

            auc      = float(roc_auc_score(y_true, y_probs))
            accuracy = float(accuracy_score(y_true, y_preds))
            loss     = float(-np.mean(
                y_true * np.log(y_probs + 1e-7) +
                (1 - y_true) * np.log(1 - y_probs + 1e-7)
            ))

            log(INFO, f"[Client {partition_id}] Eval → AUC={auc:.4f} | Acc={accuracy:.4f}")

            return EvaluateRes(
                status=Status(code=Code.OK, message="OK"),
                loss=loss,
                num_examples=num_test,
                metrics={"auc": auc, "accuracy": accuracy},
            )

    return XGBClient()


def dispatch_client_fn(context: Context):
    if str(context.run_config.get("protocol", "bagging")) == "histogram":
        from nba_federated.hist_client import HistClient
        return HistClient(context)
    return client_fn(context)


# Flower ClientApp entry point. The optional SecAgg+ mod is a pass-through unless the
# server runs the SecAgg+ workflow (histogram protocol with hist-secagg = true).
from nba_federated.hist_client import optional_secaggplus_mod  # noqa: E402

app = ClientApp(client_fn=dispatch_client_fn, mods=[optional_secaggplus_mod])
